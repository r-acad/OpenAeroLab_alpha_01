using TOML,Serialization,SHA,LinearAlgebra
BLAS.set_num_threads(1)
const MODE=ARGS[1];const SNAP=abspath(ARGS[2]);const LOADER=ARGS[3];const OUT=ARGS[4]
const NATIVE=joinpath(SNAP,"src","pbcav")
const CLOCK=time()
const LOAD=if MODE=="direct"
    (native=include(joinpath(NATIVE,"PBCav.jl")),mode="direct_include",provenance=nothing)
elseif startswith(MODE,"bootstrap_")
    worker=MODE=="bootstrap_ch" ? "lowspeed_ch_worker.jl" : "lowspeed_worker.jl"
    path=joinpath(ARGS[5],"apps","airfoil_web","src","workers",worker)
    text=read(path,String)
    beginline="include(joinpath(@__DIR__, \"..\", \"lowspeed_native_loader.jl\"))"
    block=split(split(text,beginline;limit=2)[2],"flush(stdout)";limit=2)[1]
    Base.include_string(Main,beginline*block*"flush(stdout)",path)
    Base.invokelatest(()->getglobal(Main,:NATIVE_LOAD))
else
    include(LOADER)
    AirfoilLowSpeedNative.load_native(SNAP)
end
const LOAD_SECONDS=time()-CLOCK
const P=LOAD.native
const WORKLOAD=if MODE=="direct"||MODE=="legacy"
    # The same function body is evaluated against the direct native module.
    @eval const PBCav=$P
    include(joinpath(NATIVE,"src","host_precompile_workload.jl"))
    _cpu_sst_workload
else
    Base.require(Base.PkgId(Base.UUID(AirfoilLowSpeedNative.PACKAGE_UUID),AirfoilLowSpeedNative.PACKAGE_NAME))._cpu_sst_workload
end
const START_WORK=time()
const SAMPLE=Base.invokelatest(WORKLOAD)
const WORK_SECONDS=time()-START_WORK
function plain(x)
    x isa Symbol&&return String(x)
    x isa NamedTuple&&return Dict(string(k)=>plain(v) for(k,v)in pairs(x) if v!==nothing)
    x isa AbstractDict&&return Dict(string(k)=>plain(v) for(k,v)in pairs(x) if v!==nothing)
    x isa Tuple&&return collect(x)
    x isa AbstractArray&&return x
    x
end
state=Dict(string(n)=>Array(getfield(SAMPLE.state,n)) for n in P.STATE_FIELDS)
io=IOBuffer()
for n in sort!(collect(keys(state)));write(io,n);write(io,reinterpret(UInt8,vec(state[n])));end
for f in (:t,:dt,:dt_prev,:nstep);write(io,string(f));write(io,repr(getfield(SAMPLE.state,f)));end
diagnostics=Dict("projection"=>plain(SAMPLE.projection),"outcome"=>string(SAMPLE.result.outcome),
    "steps"=>SAMPLE.result.nstep,"turbulence"=>collect(SAMPLE.turbulence),
    "mass_error"=>SAMPLE.result.ledger.step_err_max,"rejections"=>plain(SAMPLE.result.tally),
    "wall_cf"=>SAMPLE.wall.cf,"wall_yplus"=>SAMPLE.wall.yplus,"t"=>SAMPLE.state.t,
    "dt"=>SAMPLE.state.dt,"dt_prev"=>SAMPLE.state.dt_prev)
const CUDA_LOADED=any(m->nameof(m)===:CUDA,values(Base.loaded_modules))
meta=Dict("mode"=>MODE,"loader_mode"=>LOAD.mode,"source_sha256"=>P.freeair_source_signature(),
    "cache_depot"=>haskey(LOAD,:cache_depot) ? LOAD.cache_depot : "",
    "state_sha256"=>bytes2hex(sha256(take!(io))),"diagnostics"=>diagnostics,"load_seconds"=>LOAD_SECONDS,
    "workload_seconds"=>WORK_SECONDS,"cuda_loaded"=>CUDA_LOADED,"threads"=>Threads.nthreads(),
    "cells"=>SAMPLE.mesh.nc,"purpose"=>"native host-cache component parity only; not an airfoil convergence/speed qualification")
if MODE=="invalidate"
    package=Base.require(Base.PkgId(Base.UUID(AirfoilLowSpeedNative.PACKAGE_UUID),AirfoilLowSpeedNative.PACKAGE_NAME))
    checks=Dict{String,Bool}()
    for (key,file) in (("native_source",joinpath(NATIVE,"freeair.jl")),
                       ("generator_source",joinpath(NATIVE,"..","cmesh","cmesh_gen.jl")),
                       ("locked_environment",joinpath(NATIVE,"Project.toml")))
        original=read(file)
        try
            open(file,"a") do io;println(io,"\n# Private cache invalidation check");end
            checks[key]=try
                Base.invokelatest(package.validate_provenance!;native_dir=NATIVE)
                false
            catch e
                e isa ArgumentError
            end
        finally
            write(file,original)
        end
    end
    checks["restored_source_accepted"]=try
        Base.invokelatest(package.validate_provenance!;native_dir=NATIVE);true
    catch;false;end
    meta["invalidation"]=checks
end
serialize(OUT*".jls",(state=state,diagnostics=diagnostics))
open(io->TOML.print(io,meta;sorted=true),OUT*".toml","w")
println("Probe ",MODE," ",LOAD.mode,"; ",SAMPLE.result.outcome," steps",SAMPLE.result.nstep,
    " source ",meta["source_sha256"]," state ",meta["state_sha256"],"; CUDA loaded ",CUDA_LOADED)
