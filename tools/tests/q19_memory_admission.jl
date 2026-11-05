# CPU octrees use available RAM, including construction scratch and wake cells.
# julia --project=third_party/JXfoil tools/tests/q19_memory_admission.jl
using Test
include(joinpath(@__DIR__,"..","..","apps","ludwig_web","src","aero_backend.jl"))
const A=AeroBackend

function aircraft_request(dx=.0125;levels=7)
    case=Dict{String,Any}(
        "solver"=>"q19_octree", "surface_dx"=>dx,"levels"=>levels,
        "bounds"=>Dict("min"=>[0.,-10.,-3.],"max"=>[17.,10.,3.]),
        "surface_area_m2"=>420.,
        "domain"=>Dict{String,Any}("boundary"=>"symmetry","symmetry_center_y"=>true,
            "upstream"=>29.9,"downstream"=>29.9,"lateral"=>15.6,"height"=>22.6),
        "octree"=>A._q19_settings(Dict{String,Any}()))
    case
end

@testset "RAM admission for a large CPU octree" begin
    case=aircraft_request()
    rich=A._mesh_limits(case;available_ram_bytes=110e9)
    @test rich["fine_envelope_cells"]>30_000_000
    # Resource-rich hosts may build more than the old fixed 30M-cell ceiling.
    # On smaller hosts this same request must be refused with a RAM explanation.
    @test rich["valid"] == (Sys.total_memory()>=rich["build_memory_estimate_gb"]*1e9/.8)
    poor=A._mesh_limits(case;available_ram_bytes=8e9)
    @test !poor["valid"]
    @test any(occursin("available RAM",s) for s in poor["reasons"])
    no_wake=deepcopy(case);no_wake["octree"]["wake_length"]=0.
    @test A._mesh_limits(no_wake;available_ram_bytes=110e9)["build_memory_estimate_gb"] < rich["build_memory_estimate_gb"]
    # Dense construction masks can dominate even without a large body surface.
    deep=deepcopy(case);deep["levels"]=10
    deep["surface_dx"]=case["surface_dx"]/8
    deep["surface_area_m2"]=0.;deep["octree"]["wake_length"]=0.
    @test !A._mesh_limits(deep;available_ram_bytes=110e9)["valid"]
    gpu=deepcopy(case);gpu["solver"]="main"
    @test !A._mesh_limits(gpu;available_ram_bytes=110e9)["valid"]
    @test !haskey(A._mesh_limits(gpu),"build_ram_budget_gb")
    # Old symmetry meshes must be rebuilt before trusting their smaller RAM count.
    @test_throws ArgumentError A._require_current_q19_mesh(case,Dict("octree"=>Dict()))
    @test A._require_current_q19_mesh(case,Dict("octree"=>Dict("symmetry_boundary_refinement"=>1)))===nothing
    @test A._require_current_q19_mesh(gpu,Dict())===nothing
    free=deepcopy(case);free["domain"]["boundary"]="free_air"
    @test A._require_current_q19_mesh(free,Dict())===nothing
end
