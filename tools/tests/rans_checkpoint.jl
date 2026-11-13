using Test, SHA, Serialization
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const SRC=joinpath(ROOT,"src")
include(joinpath(ROOT,"apps/airfoil_web/src/backends/rans_checkpoint.jl"))

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

@testset "Restart identity covers reconstruction and implicit helper sources" begin
    identity=_rans_restart_identity(cfg,mesh,"test")
    helpers=("cmesh_geometric_reconstruction_ka.jl","cmesh_steady_globalization_ka.jl",
        "cmesh_steady_diagnostics_ka.jl","cmesh_steady_block_line_ka.jl",
        "cmesh_steady_kinetic_blocks_ka.jl","cmesh_steady_multilevel_ka.jl")
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
    before=_rans_restart_solver_sources(source_fixture)
    future=joinpath(source_fixture,"compressible","cmesh_steady_future_helper.jl")
    write(future,"# future numerical helper\n")
    added=_rans_restart_solver_sources(source_fixture)
    @test before!=added
    @test haskey(added,basename(future))
    write(future,"# changed future numerical helper\n")
    @test added!=_rans_restart_solver_sources(source_fixture)
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
