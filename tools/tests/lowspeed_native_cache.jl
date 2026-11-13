using Test,TOML,Serialization,Dates
const ROOT=dirname(dirname(@__DIR__))
const WS=get(ENV,"OAL_WORKSPACE_ROOT",dirname(ROOT))
include(joinpath(ROOT,"apps","airfoil_web","src","job_control.jl"))
const TEMP=mktempdir(joinpath(WS,"02_PRIVATE_PROJECT_DEVELOPMENT","TEMP_DELETEABLE");prefix="native_host_cache_tests_",cleanup=false)
const REPORT=isempty(ARGS) ? TEMP : abspath(ARGS[1]);mkpath(REPORT)
const FIXTURE=joinpath(TEMP,"native");const LEGACY=joinpath(TEMP,"legacy")
const LOADER=joinpath(ROOT,"apps","airfoil_web","src","lowspeed_native_loader.jl")
for family in ("pbcav","cmesh")
    source=joinpath(ROOT,"solvers","pbcav_s1030","src",family)
    for (dir,_,names) in walkdir(source),name in names
        (endswith(name,".jl")||name in ("Project.toml","Manifest.toml","LocalPreferences.toml"))||continue
        dest=joinpath(FIXTURE,"src",family,relpath(joinpath(dir,name),source))
        mkpath(dirname(dest));cp(joinpath(dir,name),dest)
    end
end
cp(FIXTURE,LEGACY)
legacyproject=joinpath(LEGACY,"src","pbcav","Project.toml")
legacy=TOML.parsefile(legacyproject)
for k in ("name","uuid","version");delete!(legacy,k);end
open(io->TOML.print(io,legacy;sorted=true),legacyproject,"w")
# Remove only the test fixture's copied wrapper entry, preserving its workload
# so direct/cached/legacy arms execute the exact same numerical function body.
legacyentry=realpath(joinpath(LEGACY,"src","pbcav","src","PBCavNativeCache.jl"))
startswith(legacyentry,realpath(TEMP)*(Sys.iswindows() ? "\\" : "/"))||error("Fixture confinement failed")
rm(legacyentry)
const DEPOT=join(vcat(joinpath(TEMP,"depot"),DEPOT_PATH),Sys.iswindows() ? ';' : ':')
function launch(mode,snapshot;limit=120.)
    output=joinpath(REPORT,mode);log=output*".log"
    cmd=`$(joinpath(Sys.BINDIR,Base.julia_exename())) --startup-file=no --depwarn=error --threads=2 --project=$(joinpath(snapshot,"src","pbcav")) $(joinpath(@__DIR__,"fixtures","lowspeed_native_cache_probe.jl")) $mode $snapshot $LOADER $output $ROOT`
    cmd=addenv(cmd,"JULIA_DEPOT_PATH"=>DEPOT,"JULIA_NUM_PRECOMPILE_TASKS"=>"1","JULIA_PKG_PRECOMPILE_AUTO"=>"0",
        "LUDWIG_PRIVATE_ROOT"=>joinpath(TEMP,"private"))
    cmd=AirfoilJobControl.process_command(cmd)
    started=time();timedout=false
    child=open(log,"w") do io;run(pipeline(cmd;stdout=io,stderr=io);wait=false);end
    pid=getpid(child);owned=AirfoilJobControl.track!(child,nothing)
    try
        while process_running(child)
            if time()-started>limit;timedout=true;break;end
            sleep(.1)
        end
    finally
        AirfoilJobControl.finish!(owned)
    end
    wait(child)
    open(io->TOML.print(io,Dict("mode"=>mode,"seconds"=>time()-started,"timed_out"=>timedout,
        "exit_code"=>child.exitcode,"owned_child_pid"=>pid,"limit_seconds"=>limit);sorted=true),output*"_launch.toml","w")
    println(mode," exit ",child.exitcode," after ",round(time()-started;digits=2),"s; output ",output);flush(stdout)
    if child.exitcode!=0
        println(join(last(split(read(log,String),'\n'),18),'\n'))
    end
    @test !timedout
    @test child.exitcode==0
    child.exitcode==0||error("Host-cache probe failed: $mode")
    TOML.parsefile(output*".toml"),deserialize(output*".jls")
end
@testset "Native host cache provenance and numerical parity" begin
    cached,a=launch("cached",FIXTURE)
    direct,b=launch("direct",FIXTURE)
    legacy,c=launch("legacy",LEGACY)
    invalidation,_=launch("invalidate",FIXTURE)
    @test cached["loader_mode"]=="named_package"
    @test cached["cache_depot"]==joinpath(TEMP,"private","TEMP_DELETEABLE","airfoil_native_host_cache")
    @test isdir(cached["cache_depot"])
    @test direct["loader_mode"]=="direct_include"
    @test legacy["loader_mode"]=="legacy_include"
    @test cached["cells"]==64
    @test all(!x["cuda_loaded"] for x in (cached,direct,legacy,invalidation))
    @test all(x["source_sha256"]==cached["source_sha256"] for x in (direct,legacy,invalidation))
    @test cached["state_sha256"]==direct["state_sha256"]==legacy["state_sha256"]
    @test cached["diagnostics"]["steps"]==1
    @test cached["diagnostics"]["outcome"]=="REACHED"
    @test keys(a.state)==keys(b.state)==keys(c.state)
    for name in keys(a.state)
        @test isequal(a.state[name],b.state[name])
        @test isequal(a.state[name],c.state[name])
    end
    @test isequal(a.diagnostics,b.diagnostics)
    @test isequal(a.diagnostics,c.diagnostics)
    for name in ("native_source","generator_source","locked_environment","restored_source_accepted")
        @test invalidation["invalidation"][name]
    end
    for mode in ("bootstrap_sharp","bootstrap_ch")
        meta,data=launch(mode,FIXTURE)
        @test meta["loader_mode"]=="named_package"
        @test !meta["cuda_loaded"]
        @test meta["state_sha256"]==cached["state_sha256"]
        @test isequal(data.state,a.state)
        @test isequal(data.diagnostics,a.diagnostics)
    end
end
println("Private fixture retained at ",TEMP)
