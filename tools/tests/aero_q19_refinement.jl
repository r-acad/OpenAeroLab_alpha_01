# Unit-level admission and argv tests; never launches a solver.
using Test, JSON3, UUIDs, HTTP
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","aero_q19_refinement_tests")
mkpath(SCRATCH)
const FIXTURE=mktempdir(SCRATCH;prefix="admission_")
ENV["LUDWIG_PRIVATE_ROOT"]=joinpath(FIXTURE,"app")
include(joinpath(ROOT,"apps","ludwig_web","src","aero_backend.jl"))
const A=AeroBackend
const SID=string(uuid4());const TID=string(uuid4())
const SRC=joinpath(FIXTURE,"source case; not a shell command");const MESH=joinpath(FIXTURE,"source mesh")
mkpath(joinpath(SRC,"RESULTS","checkpoints"));mkpath(MESH)
write(joinpath(SRC,"config.yaml"),"basic:\n  reference_length_for_meshing: 17.088\n");write(joinpath(SRC,"model.stl"),"geometry")
write(joinpath(MESH,"mesh.oct"),"existing mesh")
const TMESH=joinpath(A.CASES_ROOT,"$(TID)_mesh");mkpath(TMESH);write(joinpath(TMESH,"model.stl"),"geometry")
const CHECKPOINT=joinpath(SRC,"RESULTS","checkpoints","checkpoint_000000100.bin")
open(CHECKPOINT,"w") do io
    write(io,codeunits("OCTQ19C\0"));write(io,Int64(1));write(io,Int64(4));write(io,zeros(UInt8,128));write(io,Int64(100))
end
const FLOW=Dict("velocity"=>35.,"alpha"=>14.,"beta"=>0.)
const TRANSFORM=Dict("scale"=>[1.,1.,1.])
const DOMAIN=Dict("boundary"=>"symmetry")
case(levels,steps)=Dict{String,Any}("solver"=>"q19_octree","flow"=>deepcopy(FLOW),"transform"=>deepcopy(TRANSFORM),"domain"=>deepcopy(DOMAIN),"octree"=>Dict("seal_cavities"=>true,"threads"=>2,"stage_reference_length_m"=>17.088),"steps"=>steps,"levels"=>levels)
actual(levels)=Dict{String,Any}("num_levels"=>levels,"levels"=>[Dict("dx_m"=>.6144)],"domain_size_m"=>[46.6944,15.9744,33.1776],"mesh_offset_m"=>[17.58,0.,17.32],"estimated_run_bytes_total"=>1_000_000_000,"octree"=>Dict("cavity_seal_policy_version"=>2,"symmetry_boundary_refinement"=>1))
source=Dict{String,Any}("job_id"=>SID,"status"=>"completed","case"=>case(5,100),"actual"=>actual(5),"run_case_dir"=>SRC,"mesh_case_dir"=>MESH)
target=Dict{String,Any}("job_id"=>TID,"case"=>case(6,200),"mesh_case_dir"=>TMESH,"run_case_dir"=>joinpath(FIXTURE,"target run"))
const UPDATES=Dict("refinement_source_job_id"=>SID,"accept_nonconservative_restart"=>true)
loader(id)=id==SID ? source : error("unexpected ID")
admit(state=target,ta=actual(6),updates=UPDATES;ram=40e9)=A._q19_refinement_source(TID,state,ta,updates;load_job=loader,available_ram=ram,total_ram=64e9)

@testset "Explicit staged restart admission" begin
    @test admit(target,actual(6),Dict())===nothing
    info=admit();@test info["source_step"]==100 && info["accepted_nonconservative"]
    @test info["source_case_dir"]==realpath(SRC) && isfile(info["source_checkpoint"])
    @test_throws ArgumentError admit(target,actual(6),Dict("refinement_source_job_id"=>SID))
    @test_throws ArgumentError admit(target,actual(6),Dict("refinement_source_job_id"=>TID,"accept_nonconservative_restart"=>true))
    source["status"]="running";@test_throws ArgumentError admit();source["status"]="completed"
    bad=deepcopy(target);bad["case"]["steps"]=100;@test_throws ArgumentError admit(bad)
    bad=deepcopy(target);bad["case"]["flow"]["velocity"]=36.;@test_throws ArgumentError admit(bad)
    @test_throws ArgumentError admit(target,actual(5))
    bad=actual(6);bad["domain_size_m"][1]+=.1;@test_throws ArgumentError admit(target,bad)
    bad=actual(6);bad["mesh_offset_m"][1]+=1e-8;@test_throws ArgumentError admit(target,bad)
    bad=actual(6);bad["levels"][1]["dx_m"]=.7;@test_throws ArgumentError admit(target,bad)
    bad=deepcopy(target);bad["case"]["octree"]["stage_reference_length_m"]=17.0688;@test_throws ArgumentError admit(bad)
    @test_throws ArgumentError admit(ram=9e9)
    write(joinpath(TMESH,"model.stl"),"different");@test_throws ArgumentError admit();write(joinpath(TMESH,"model.stl"),"geometry")
    old=read(CHECKPOINT);write(CHECKPOINT,"not a checkpoint");@test_throws ArgumentError admit();write(CHECKPOINT,old)
    @test read(CHECKPOINT)==old # Admission has not changed source data.
    target["refinement_source"]=info
    cmd=A._solver_command(target,"run")
    @test cmd.exec[end-7:end]==["--from-case",realpath(SRC),"--from-checkpoint",info["source_checkpoint"],"--source-checkpoint-bytes",string(info["source_checkpoint_bytes"]),"--source-checkpoint-mtime",string(info["source_checkpoint_mtime"])]
    @test !any(==("--from-case"),A._solver_command(target,"mesh").exec)
end
