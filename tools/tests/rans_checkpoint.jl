using Test, SHA, Serialization
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const SRC=joinpath(ROOT,"src")
include(joinpath(ROOT,"apps/airfoil_web/src/backends/rans_checkpoint.jl"))
_rans_restart_bind_source_snapshot!(_rans_restart_capture_source_snapshot())

mutable struct RestartFixture{T}
    NI::Int; NJ::Int; g::Int; sst::Bool; gam::T
    rho::Matrix{T}; ux::Matrix{T}; uy::Matrix{T}; T::Matrix{T}
    tk::Matrix{T}; tom::Matrix{T}; nut::Matrix{T}; tnl::Matrix{T}
end
function fixture(;sst=true,scalar_type=Float32)
    s=RestartFixture(2,2,1,sst,zero(scalar_type),(ones(scalar_type,4,4) for _ in 1:8)...)
    s.tk[:,1].=-1f0 # reflected wall ghosts are intentionally negative
    sst || (s.tom.=0)
    s
end

scratch=normpath(joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","rans_restart_tests"))
mkpath(scratch)
folder=mktempdir(scratch)
mesh=joinpath(folder,"mesh.bin");write(mesh,"test mesh identity")
cfg=Dict{String,Any}("outdir"=>folder,"alpha_deg"=>0.,"freestream_temperature_k"=>288.15)
source=fixture();blocks=[(name="C",solver=source)]
path=_rans_save_restart(cfg,mesh,"test",blocks)
good=open(deserialize,path)
cfg["warm_start"]=path
write_saved(saved)=open(io->serialize(io,saved),path,"w")

@testset "Restart identity records the effective SST metric" begin
    default_identity=_rans_restart_identity(cfg,mesh,"test")
    explicit_true=merge(cfg,Dict("sst_wall_metric_correction"=>true))
    explicit_false=merge(cfg,Dict("sst_wall_metric_correction"=>false))
    @test default_identity["settings"]["sst_wall_metric_correction"]===true
    @test _rans_restart_identity(explicit_true,mesh,"test")==default_identity
    @test _rans_restart_identity(explicit_false,mesh,"test")!=default_identity
    destination=fixture();destination.ux.=2
    @test isempty(_rans_load_restart!(explicit_false,mesh,"test",[(name="C",solver=destination)]))
    @test all(==(2f0),destination.ux)
end

@testset "Restart identity covers reconstruction and implicit helper sources" begin
    identity=_rans_restart_identity(cfg,mesh,"test")
    helpers=("cmesh_geometric_reconstruction_ka.jl","cmesh_steady_globalization_ka.jl",
        "cmesh_steady_diagnostics_ka.jl","cmesh_steady_block_line_ka.jl",
        "cmesh_steady_kinetic_blocks_ka.jl","cmesh_steady_multilevel_ka.jl",
        "cmesh_steady_packed_cgs2_ka.jl")
    for name in helpers
        @test identity["solver_sha256"][name]==bytes2hex(open(sha256,joinpath(SRC,"compressible",name)))
        changed=deepcopy(good);changed["identity"]["solver_sha256"][name]="changed"
        write_saved(changed);destination=fixture();destination.ux.=2
        @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)]))
        @test all(==(2f0),destination.ux)
    end
    for name in ("cmesh/cmesh_metrics.jl","collision/lattice.jl")
        @test identity["geometry_sha256"][name]==bytes2hex(open(sha256,joinpath(SRC,split(name,'/')...)))
        changed=deepcopy(good);changed["identity"]["geometry_sha256"][name]="changed"
        write_saved(changed);destination=fixture();destination.ux.=2
        @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)]))
        @test all(==(2f0),destination.ux)
    end
    # Discover a future helper without editing the production source tree.
    source_fixture=joinpath(folder,"source_fixture");mkpath(joinpath(source_fixture,"compressible"))
    for name in keys(identity["solver_sha256"])
        cp(joinpath(SRC,"compressible",name),joinpath(source_fixture,"compressible",name))
    end
    before=_rans_restart_capture_solver_sources(source_fixture)
    future=joinpath(source_fixture,"compressible","cmesh_steady_future_helper.jl")
    write(future,"# future numerical helper\n")
    added=_rans_restart_capture_solver_sources(source_fixture)
    @test before!=added
    @test haskey(added,basename(future))
    write(future,"# changed future numerical helper\n")
    @test added!=_rans_restart_capture_solver_sources(source_fixture)
    for missing in (:geometry,:helper)
        older=deepcopy(good)
        missing===:geometry ? delete!(older["identity"],"geometry_sha256") :
            delete!(older["identity"]["solver_sha256"],first(helpers))
        write_saved(older);destination=fixture();destination.ux.=2
        @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)]))
        @test all(==(2f0),destination.ux)
    end
    write_saved(good)
end

worker_call(worker,name,args...;kwargs...)=Base.invokelatest(Base.invokelatest(getproperty,worker,name),args...;kwargs...)
function source_worker(src,native_dir;bind=true,during_load! = ()->nothing)
    worker=Module(gensym(:RestartWorker))
    # Match the backend: checkpoint helper definitions precede SRC and solver
    # includes. These fixture includes execute distinct, observable code.
    Base.include(worker,joinpath(ROOT,"apps/airfoil_web/src/backends/rans_checkpoint.jl"))
    Core.eval(worker,:(const SRC=$src))
    if bind
        before=worker_call(worker,:_rans_restart_capture_source_snapshot;native_dir)
        Base.include(worker,joinpath(src,"compressible","cmesh_solver.jl"))
        Base.include(worker,joinpath(native_dir,"gas_strip_sst_steady.jl"))
        during_load!()
        worker_call(worker,:_rans_restart_bind_source_snapshot!,before;native_dir)
    end
    worker
end

@testset "Verified worker-load provenance survives later disk edits" begin
    root=mktempdir(scratch);src=joinpath(root,"src");native_dir=joinpath(root,"runtime")
    current=_rans_restart_capture_source_snapshot()
    for (category,entries) in current, name in keys(entries)
        source_path=category=="solver_sha256" ? joinpath(SRC,"compressible",name) :
            category=="geometry_sha256" ? joinpath(SRC,split(name,'/')...) :
            category=="flux_sha256" ? joinpath(SRC,"cmesh",name) :
            joinpath(ROOT,"apps/airfoil_web/src/runtime",name)
        target=category=="solver_sha256" ? joinpath(src,"compressible",name) :
            category=="geometry_sha256" ? joinpath(src,split(name,'/')...) :
            category=="flux_sha256" ? joinpath(src,"cmesh",name) : joinpath(native_dir,name)
        mkpath(dirname(target));cp(source_path,target)
    end
    solver_file=joinpath(src,"compressible","cmesh_solver.jl")
    native_file=joinpath(native_dir,"gas_strip_sst_steady.jl")
    write(solver_file,"fixture_solver_version()=1\n")
    write(native_file,"fixture_native_version()=11\n")
    old=source_worker(src,native_dir)
    @test worker_call(old,:fixture_solver_version)==1
    @test worker_call(old,:fixture_native_version)==11
    before=worker_call(old,:_rans_restart_identity,cfg,mesh,"test")
    @test before["source_provenance"]=="worker_file_load_snapshot_v1"
    @test_throws ErrorException worker_call(old,:_rans_restart_bind_source_snapshot!,current;native_dir)

    # Mutate every protected category and add a future numerical helper after
    # loading. The old worker still runs its original methods and saves those
    # startup hashes, rather than relabelling state with any of these edits.
    write(solver_file,"fixture_solver_version()=2\n")
    write(native_file,"fixture_native_version()=22\n")
    open(io->println(io,"\n# changed after the old worker loaded"),joinpath(src,"cmesh","cmesh_metrics.jl"),"a")
    open(io->println(io,"\n# changed after the old worker loaded"),joinpath(src,"cmesh","cmesh_flux_core.jl"),"a")
    write(joinpath(src,"compressible","cmesh_steady_future_loaded_helper.jl"),"# new helper\n")
    @test worker_call(old,:fixture_solver_version)==1
    @test worker_call(old,:fixture_native_version)==11
    @test worker_call(old,:_rans_restart_identity,cfg,mesh,"test")==before
    @test worker_call(old,:_rans_restart_solver_sources)==before["solver_sha256"]
    identity_copy=worker_call(old,:_rans_restart_identity,cfg,mesh,"test")
    identity_copy["solver_sha256"]["cmesh_solver.jl"]="caller mutation"
    @test worker_call(old,:_rans_restart_identity,cfg,mesh,"test")==before

    localcfg=copy(cfg);localcfg["outdir"]=mktempdir(scratch)
    saved_path=worker_call(old,:_rans_save_restart,localcfg,mesh,"test",[(name="C",solver=fixture())])
    saved=open(deserialize,saved_path)
    @test saved["identity"]==before
    @test !isfile(saved_path*".tmp")
    localcfg["warm_start"]=saved_path
    destination=fixture();destination.ux.=2
    @test worker_call(old,:_rans_load_restart!,localcfg,mesh,"test",[(name="C",solver=destination)])==saved_path
    @test all(==(1f0),destination.ux)

    fresh=source_worker(src,native_dir)
    @test worker_call(fresh,:fixture_solver_version)==2
    @test worker_call(fresh,:fixture_native_version)==22
    after=worker_call(fresh,:_rans_restart_identity,cfg,mesh,"test")
    for category in ("solver_sha256","geometry_sha256","flux_sha256","native_sha256")
        @test after[category]!=before[category]
    end
    @test haskey(after["solver_sha256"],"cmesh_steady_future_loaded_helper.jl")
    destination=fixture();destination.ux.=2;destination.gam=.25f0
    @test isempty(worker_call(fresh,:_rans_load_restart!,localcfg,mesh,"test",[(name="C",solver=destination)]))
    @test all(==(2f0),destination.ux)&&destination.gam==.25f0

    # Old unproven checkpoints cold-fallback instead of acquiring a loaded-code
    # claim. Normal field/result formats are not involved in this check.
    delete!(saved["identity"],"source_provenance")
    open(io->serialize(io,saved),saved_path,"w")
    destination=fixture();destination.ux.=2
    @test isempty(worker_call(old,:_rans_load_restart!,localcfg,mesh,"test",[(name="C",solver=destination)]))
    @test all(==(2f0),destination.ux)

    unbound=source_worker(src,native_dir;bind=false)
    @test_throws ErrorException worker_call(unbound,:_rans_restart_identity,cfg,mesh,"test")
    @test_throws ErrorException worker_call(unbound,:_rans_save_restart,localcfg,mesh,"test",[(name="C",solver=fixture())])
    @test isempty(worker_call(unbound,:_rans_load_restart!,localcfg,mesh,"test",[(name="C",solver=destination)]))
    @test all(==(2f0),destination.ux)
    @test_throws ErrorException source_worker(src,native_dir;during_load! = ()->open(io->println(io,"# load race"),native_file,"a"))
end

@testset "Backend snapshot boundaries enclose early and late numerical includes" begin
    text=read(joinpath(ROOT,"apps/airfoil_web/src/backends/cfd_backend.jl"),String)
    start=first(findfirst("const _RANS_RESTART_SOURCE_LOAD_BEFORE",text))
    geometry=first(findfirst("include(joinpath(SRC, \"cmesh\", \"cmesh_metrics.jl\"))",text))
    solver=first(findfirst("include(joinpath(SRC, \"compressible\", \"cmesh_solver_ka.jl\"))",text))
    late=first(findfirst("include(joinpath(@__DIR__,\"..\",\"workers\",\"ch_transonic_worker.jl\"))",text))
    seal=first(findfirst("IS_WORKER && _rans_restart_bind_source_snapshot!",text))
    @test start<geometry<solver<late<seal
    @test occursin("IS_WORKER ? _rans_restart_capture_source_snapshot() : nothing",text)
end

@testset "Restart rejects invalid states before modifying the solver" begin
    destination=fixture();destination.ux.=2
    @test _rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)])==path
    @test all(==(1f0),destination.ux)
    @test _rans_restart_identity(cfg,mesh,"test")["format"]==2
    for (field,value) in (("rho",0f0),("T",-1f0),("tk",-1f0),("tom",0f0),("nut",-1f0),("tnl",-1f0))
        saved=deepcopy(good);saved["arrays"]["C/"*field][2,2]=value;write_saved(saved)
        destination=fixture();destination.ux.=2
        @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)]))
        @test all(==(2f0),destination.ux)
    end
    for kind in (:element_type,:circulation,:temperature,:source)
        saved=deepcopy(good)
        kind===:element_type && (saved["arrays"]["C/rho"]=Float64.(saved["arrays"]["C/rho"]))
        kind===:circulation && (saved["circulation"]["C"]=1e300)
        kind===:temperature && (saved["identity"]["settings"]["freestream_temperature_k"]=300.)
        kind===:source && (saved["identity"]["flux_sha256"]["cmesh_flux_core.jl"]="changed")
        write_saved(saved);destination=fixture();destination.ux.=2
        @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=destination)]))
        @test all(==(2f0),destination.ux)
    end
    @test _rans_restart_physical(fixture(sst=false).tom,:tom,fixture(sst=false))
    source=fixture()
    extras=(rk=ones(Float32,2,2),rw=ones(Float32,2,2))
    _rans_save_restart(cfg,mesh,"test",[(name="C",solver=source,extras)])
    @test _rans_load_restart!(cfg,mesh,"test",[(name="C",solver=source,extras)])==path
    saved=open(deserialize,path);saved["arrays"]["C/rw"][1,1]=2;write_saved(saved)
    @test isempty(_rans_load_restart!(cfg,mesh,"test",[(name="C",solver=source,extras)]))
end

@testset "Float64 checkpoints retain values below Float32 resolution" begin
    double_cfg=copy(cfg);double_cfg["outdir"]=mktempdir(scratch)
    source=fixture(scalar_type=Float64)
    source.ux[2,2]=1+2.0^-40;source.gam=1+2.0^-40
    double_path=_rans_save_restart(double_cfg,mesh,"test",[(name="C",solver=source)])
    double_cfg["warm_start"]=double_path
    destination=fixture(scalar_type=Float64);destination.ux.=2
    @test _rans_load_restart!(double_cfg,mesh,"test",[(name="C",solver=destination)])==double_path
    @test destination.ux==source.ux
    @test destination.ux[2,2]!=Float64(Float32(source.ux[2,2]))
    @test destination.gam==source.gam
    single=fixture();single.ux.=2
    @test isempty(_rans_load_restart!(double_cfg,mesh,"test",[(name="C",solver=single)]))
    @test all(==(2f0),single.ux)
end
