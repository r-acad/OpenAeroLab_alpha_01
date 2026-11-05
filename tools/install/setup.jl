module OpenAeroLabSetup
using Pkg, TOML, Dates, SHA, UUIDs
using FileWatching: Pidfile

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const VERSION_REQUIRED = VersionNumber(strip(read(joinpath(@__DIR__, "julia_version.txt"), String)))
const PROJECTS = [
    (name="Web workbenches and JXfoil", path="third_party/JXfoil", cuda=false),
    (name="CPU octree D3Q19 solver (experimental)", path="src/octree_q19", cuda=false),
    (name="Main LBM and C-mesh solvers", path="src", cuda=true),
    (name="Pinned LBM backend", path="solvers/lbm_lbmtau/src", cuda=true),
    (name="Pressure solver", path="solvers/pbcav_s1030/src/pbcav", cuda=true),
    (name="Makie reports, charts and renders", path="tools/postprocess", cuda=false),
]
private_root() = abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
selected_projects(; platform=Sys.KERNEL) = filter(p -> platform != :Darwin || !p.cuda, PROJECTS)
function fingerprint(project; root=ROOT)
    bytes2hex(sha256(join((read(joinpath(root,project,file),String) for file in ("Project.toml","Manifest.toml")),"\n")))
end

# Never print proxy passwords or signed-download query strings in our diagnostics.
function safe_error(err)
    text=sprint(showerror,err)
    text=replace(text,r"(?i)((?:https?|socks5h?)://)[^\s/@]+@"=>s"\1[redacted]@")
    replace(text,r"(?i)((?:https?|socks5h?)://[^\s?#]+)[?#][^\s]*"=>s"\1?[redacted]")
end

function transient_download_error(err)
    err isa InterruptException && return false
    message=lowercase(safe_error(err))
    # Do not keep retrying security/policy, integrity, or local filesystem failures.
    occursin(r"certificate|ssl.*(?:verif|ca)|checksum|hash mismatch|permission denied|access is denied|read-only|proxy authentication|(?:status|http[^\n]*)[ :]*(?:401|403|407)\b|no space left|disk full|operation not permitted",message) && return false
    occursin(r"timed?\s*out|timeout|connection (?:reset|refused|closed)|unexpected eof|temporarily unavailable|could(?: not|n't) resolve (?:host|proxy|address|name)|temporary failure in name resolution|(?:status|http[^\n]*)[ :]*(?:408|429|500|502|503|504)\b",message)
end

"""Retry interrupted transfers, reusing Pkg's existing cache and verified artifacts."""
function instantiate_with_retry(action; offline=false, log=stderr, console=stdout, sleepfn=sleep, attempts=3)
    attempts>=1 || throw(ArgumentError("attempts must be positive"))
    for attempt in 1:(offline ? 1 : attempts)
        try
            return action()
        catch err
            retry=!offline && attempt<attempts && transient_download_error(err)
            println(log,"Package installation attempt $attempt failed: ",safe_error(err));flush(log)
            retry || rethrow()
            delay=2^(attempt-1)*2
            println(console,"  Transfer interrupted; retrying in $delay seconds ($attempt/$attempts). Cached downloads are retained.");flush(console)
            sleepfn(delay)
        end
    end
end

function network_guidance(io; offline=false)
    if offline
        println(io,"Offline setup needs the matching Julia packages and artifacts already cached. Prepare the depot on an approved connected machine with the same OS/architecture, copy it, then set JULIA_DEPOT_PATH before setup.")
    else
        println(io,"For company networks: use your approved HTTPS_PROXY / HTTP_PROXY and NO_PROXY settings. Julia package downloads also honor JULIA_PKG_SERVER; ask IT for an approved package mirror or allow access to pkg.julialang.org and the artifact hosts named in the log.")
        println(io,"If your company inspects TLS, set JULIA_SSL_CA_ROOTS_PATH to its IT-approved PEM CA bundle before setup. Keep certificate and checksum verification enabled. A 407 response requires proxy authentication; a 403 response may require an IT allow-list change.")
        println(io,"If downloads are prohibited, use an IT-provided populated Julia depot with JULIA_DEPOT_PATH and rerun setup --offline. Rerunning normal setup reuses completed downloads.")
    end
end

function failure_guidance(io, stage; offline=false)
    if stage=="download"
        network_guidance(io;offline)
    else
        println(io,"Package downloads finished for this environment. Check the log for the compilation or import failure, and ensure Julia, its child processes and the package depot are permitted by company policy. --no-precompile --no-verify prepares downloads only; it does not repair a blocked runtime.")
    end
end

function atomic_receipt(path, value)
    mkpath(dirname(path))
    temporary=path*"."*string(uuid4())*".tmp"
    try
        open(io->TOML.print(io,value),temporary,"w")
        mv(temporary,path;force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
end

function verify_project(project, log; root=ROOT)
    command=`$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(root,project.path)) $(joinpath(@__DIR__,"verify.jl")) $(project.path)`
    run(pipeline(addenv(command,"GKSwstype"=>"100"),stdout=log,stderr=log))
end

"""Install recorded environments; callbacks permit failure-path tests without networking."""
function install_projects!(selected; opts=Set{String}(), root=ROOT, private=private_root(), console=stdout,
    activate=(path,log)->Pkg.activate(path;io=log),
    instantiate=log->Pkg.instantiate(;verbose=true,allow_autoprecomp=false,io=log),
    precompile=log->Pkg.precompile(;io=log), verify=(project,log)->verify_project(project,log;root), sleepfn=sleep)
    # Logs and progress are useful diagnostics, so they must survive clearing scratch data.
    runtime=joinpath(private,"runtime");mkpath(runtime)
    logs=joinpath(runtime,"install_logs");mkpath(logs)
    stamp=Dates.format(now(UTC),"yyyymmdd_HHMMSS")*"_"*string(uuid4())
    runlogs=joinpath(logs,stamp);mkpath(runlogs)
    logpath=joinpath(runlogs,"setup.log")
    receipt=joinpath(runtime,"installation.toml")
    attempt_receipt=joinpath(runtime,"installation_attempt.toml")
    progress=Dict{String,Any}("status"=>"running","julia"=>string(VERSION),"platform"=>string(Sys.KERNEL),
        "repository"=>root,"started_utc"=>string(now(UTC)),"log"=>logpath,"projects"=>Dict{String,Any}[])
    save_progress()=(atomic_receipt(joinpath(runlogs,"progress.toml"),progress);atomic_receipt(attempt_receipt,progress))
    println(console,"Detailed log: $logpath");flush(console)
    # One installer owns the environment mutations. The lock is outside disposable data.
    Pidfile.mkpidlock(joinpath(runtime,"setup.lock");wait=false,stale_age=180,refresh=30) do
        open(logpath,"w") do log
            save_progress()
            println(log,"OpenAeroLab installation started at ",progress["started_utc"]," UTC")
            try
                for (index,p) in enumerate(selected)
                    progress["current_project"]=p.path;progress["stage"]="download";save_progress()
                    println(console,"[$index/$(length(selected))] Installing $(p.name)…");flush(console)
                    println(log,"\n=== $(p.name) ===");flush(log)
                    activate(joinpath(root,p.path),log)
                    instantiate_with_retry(()->instantiate(log);offline="--offline" in opts,log,console,sleepfn)
                    if !("--no-precompile" in opts)
                        progress["stage"]="precompile";save_progress()
                        println(console,"  Precompiling; this can take several minutes. Details: $logpath");flush(console)
                        precompile(log)
                    end
                    if !("--no-verify" in opts)
                        progress["stage"]="verify";save_progress()
                        println(console,"  Checking package imports and runtime…");flush(console)
                        verify(p,log)
                    end
                    push!(progress["projects"],Dict("name"=>p.name,"project"=>p.path,"fingerprint"=>fingerprint(p.path;root)))
                    save_progress()
                end
                progress["status"]="complete";progress["stage"]="complete"
                progress["completed_utc"]=string(now(UTC));progress["verified"]=!("--no-verify" in opts)
                save_progress()
                # A failed new attempt leaves the last successful installation receipt intact.
                atomic_receipt(receipt,progress)
            catch err
                progress["status"]=err isa InterruptException ? "interrupted" : "failed"
                progress["failed_utc"]=string(now(UTC));progress["error"]=safe_error(err)
                save_progress()
                println(log,"\nSetup failed in ",get(progress,"stage","startup"),": ",safe_error(err))
                failure_guidance(log,get(progress,"stage","startup");offline="--offline" in opts);flush(log)
                println(console,"Setup stopped in ",get(progress,"current_project","startup")," (",get(progress,"stage","startup"),"). Details: $logpath")
                failure_guidance(console,get(progress,"stage","startup");offline="--offline" in opts)
                rethrow()
            end
        end
    end
    receipt
end
function options(args)
    allowed=Set(["--no-precompile","--no-verify","--offline","--dry-run","--help","-h"])
    for arg in args; arg in allowed || error("Unknown setup option: $arg. Use --help."); end
    Set(args)
end
function main(args=ARGS)
    opts=options(args)
    if "--help" in opts || "-h" in opts
        println("""
        OpenAeroLab setup: setup.cmd / bash setup.sh / bash setup.command
          --dry-run        Show environments and destinations, without changing files
          --no-precompile  Download packages without explicit precompilation
          --no-verify      Skip runtime import and application startup checks
          --offline        Use cached packages/artifacts only (no Julia download)
        Default: instantiate all supported environments, precompile, verify.
        Run setup again after updating the repository. No Pkg.update is performed.
        Julia $(VERSION_REQUIRED) is required by the pinned pressure solver.
        """)
        return
    end
    VERSION == VERSION_REQUIRED || error("Use Julia $VERSION_REQUIRED; detected $VERSION. Run the platform setup launcher.")
    selected=selected_projects()
    println("OpenAeroLab · Julia $VERSION · $(Sys.KERNEL) / $(Sys.ARCH)")
    println("Repository: $ROOT\nPrivate data: $(private_root())\nPackage cache: $(first(DEPOT_PATH))")
    println("Setup installs packages and binary artifacts automatically; keep this console open.")
    Sys.isapple() && println("macOS: installing the workbenches, JXfoil and Makie. CUDA solver environments are unsupported on macOS.")
    for (index,p) in enumerate(selected); println("  $index/$(length(selected)) $(p.name) [$(p.path)]"); end
    "--dry-run" in opts && return
    scratch=joinpath(private_root(),"TEMP_DELETEABLE","setup");mkpath(scratch)
    withenv("JULIA_PKG_PRECOMPILE_AUTO"=>"0",
        "JULIA_NUM_PRECOMPILE_TASKS"=>get(ENV,"JULIA_NUM_PRECOMPILE_TASKS","2"),
        "TEMP"=>scratch,"TMP"=>scratch,"TMPDIR"=>scratch) do
        Pkg.offline("--offline" in opts)
        install_projects!(selected;opts)
    end
    println("Setup complete. Launch OpenAeroLab from __Start here!__/Windows, Linux or Mac.")
    println("The NVIDIA display driver is a system prerequisite; CUDA libraries are installed by Julia.")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    try OpenAeroLabSetup.main() catch err
        println(stderr,OpenAeroLabSetup.safe_error(err),"\nSetup failed. Check the printed log; rerun setup to resume cached downloads.");exit(1)
    end
end
