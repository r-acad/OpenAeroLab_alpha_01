using Test, SHA, Serialization
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const SRC=joinpath(ROOT,"src")
include(joinpath(ROOT,"apps/airfoil_web/src/backends/rans_checkpoint.jl"))

mutable struct RestartFixture
    NI::Int; NJ::Int; g::Int; sst::Bool; gam::Float32
    rho::Matrix{Float32}; ux::Matrix{Float32}; uy::Matrix{Float32}; T::Matrix{Float32}
    tk::Matrix{Float32}; tom::Matrix{Float32}; nut::Matrix{Float32}; tnl::Matrix{Float32}
end
function fixture(;sst=true)
    s=RestartFixture(2,2,1,sst,0f0,(ones(Float32,4,4) for _ in 1:8)...)
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
