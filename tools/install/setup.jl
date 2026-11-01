module OpenAeroLabSetup
using Pkg, TOML, Dates, SHA
using FileWatching: Pidfile

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const VERSION_REQUIRED = VersionNumber(strip(read(joinpath(@__DIR__, "julia_version.txt"), String)))
const PROJECTS = [
    (name="Web workbenches and JXfoil", path="third_party/JXfoil", cuda=false),
    (name="Main LBM and C-mesh solvers", path="src", cuda=true),
    (name="Pinned LBM backend", path="solvers/lbm_lbmtau/src", cuda=true),
    (name="Pressure solver", path="solvers/pbcav_s1030/src/pbcav", cuda=true),
    (name="Makie reports, charts and renders", path="tools/postprocess", cuda=false),
]
private_root() = abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
selected_projects(; platform=Sys.KERNEL) = filter(p -> platform != :Darwin || !p.cuda, PROJECTS)
function fingerprint(project)
    bytes2hex(sha256(join((read(joinpath(ROOT,project,file),String) for file in ("Project.toml","Manifest.toml")),"\n")))
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
    logs=joinpath(private_root(),"TEMP_DELETEABLE","setup");mkpath(logs)
    receipt=joinpath(private_root(),"runtime","installation.toml");mkpath(dirname(receipt))
    logpath=joinpath(logs,"setup_"*Dates.format(now(UTC),"yyyymmdd_HHMMSS")*".log")
    println("Detailed log: $logpath")
    # A second setup must not change environments while one is already installing.
    Pidfile.mkpidlock(joinpath(logs,"setup.lock");wait=false,stale_age=180,refresh=30) do
        open(logpath,"w") do log
            ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"
            ENV["JULIA_NUM_PRECOMPILE_TASKS"]=get(ENV,"JULIA_NUM_PRECOMPILE_TASKS","2")
            Pkg.offline("--offline" in opts)
            completed=Dict{String,Any}[]
            for (index,p) in enumerate(selected)
                println("[$index/$(length(selected))] Installing $(p.name)…");flush(stdout)
                println(log,"\n=== $(p.name) ===");flush(log)
                Pkg.activate(joinpath(ROOT,p.path);io=log)
                Pkg.instantiate(;verbose=true,allow_autoprecomp=false,io=log)
                if !("--no-precompile" in opts)
                    println("  Precompiling; this can take several minutes. Details: $logpath");flush(stdout)
                    Pkg.precompile(;io=log)
                end
                if !("--no-verify" in opts)
                    println("  Checking package imports and runtime…");flush(stdout)
                    command=`$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(ROOT,p.path)) $(joinpath(@__DIR__,"verify.jl")) $(p.path)`
                    run(pipeline(addenv(command,"GKSwstype"=>"100"),stdout=log,stderr=log))
                end
                push!(completed,Dict("name"=>p.name,"project"=>p.path,"fingerprint"=>fingerprint(p.path)))
            end
            result=Dict("julia"=>string(VERSION),"platform"=>string(Sys.KERNEL),"repository"=>ROOT,
                "completed_utc"=>string(now(UTC)),"verified"=>!("--no-verify" in opts),"projects"=>completed)
            temporary=receipt*".tmp";open(io->TOML.print(io,result),temporary,"w");mv(temporary,receipt;force=true)
        end
    end
    println("Setup complete. Launch OpenAeroLab from __Start here!__/Windows, Linux or Mac.")
    println("The NVIDIA display driver is a system prerequisite; CUDA libraries are installed by Julia.")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    try OpenAeroLabSetup.main() catch err
        showerror(stderr,err,catch_backtrace());println(stderr,"\nSetup failed. Check the printed log; rerun setup to resume cached downloads.");exit(1)
    end
end
