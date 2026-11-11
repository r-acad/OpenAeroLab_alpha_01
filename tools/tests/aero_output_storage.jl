using Test, Printf
const OUTPUT_DEFAULTS=Dict("surface_vtk"=>true,"surface_lbmp"=>true,"volume_vtk"=>false)
include(joinpath(@__DIR__,"..","..","apps","ludwig_web","src","aero","storage.jl"))
include(joinpath(@__DIR__,"..","..","apps","ludwig_web","src","aero","q19.jl"))

@testset "D3Q19 output storage" begin
    case=Dict("steps"=>40_000,"display_every"=>100,"effective_display_every"=>100,
              "outputs"=>copy(OUTPUT_DEFAULTS),"octree"=>Dict("checkpoint_keep"=>2))
    actual=Dict("total_cells"=>197_000_000,"num_levels"=>8,"octree"=>Dict("stored_slots"=>220_000_000,"precision"=>"Float32","reports"=>Dict("wall_cells"=>6_400_000,"wall_sample_count"=>7_414_049)))
    forecast=_q19_storage_forecast(case,actual;mesh_bytes=100_000_000,available_bytes=70_000_000_000)
    @test forecast["sample_count_exact"]
    @test forecast["vtk_bytes_per_frame"] == 39*7_414_049+2048
    @test forecast["native_bytes_per_frame"] == 4*7_414_049+128
    @test forecast["frames"] == 400
    @test forecast["blocked"]
    @test forecast["checkpoint_size_exact"]
    @test forecast["checkpoint_bytes"] == 184*220_000_000+4*6_400_000+264+16*8
    @test forecast["checkpoint_peak_bytes"] > 100_000_000_000
    case["effective_display_every"]=1000
    @test !_q19_storage_forecast(case,actual;available_bytes=70_000_000_000)["blocked"]
    # Forty surface frames plus reserve fit in 50 GB, but the mandatory
    # final checkpoint cannot: fail before creating a run folder or results.
    with_checkpoint=_q19_storage_forecast(case,actual;available_bytes=50_000_000_000)
    @test with_checkpoint["core_bytes"]+with_checkpoint["reserve_bytes"] < 50_000_000_000
    @test with_checkpoint["blocked"]
    # Free space already accounts for existing files; the run is created in a
    # fresh time-stamped directory, so historical checkpoints are not counted twice.
    @test with_checkpoint["minimum_bytes"] == with_checkpoint["core_bytes"]+with_checkpoint["checkpoint_bytes"]+with_checkpoint["reserve_bytes"]
    delete!(actual["octree"],"stored_slots")
    unknown_checkpoint=_q19_storage_forecast(case,actual;available_bytes=50_000_000_000)
    @test !unknown_checkpoint["checkpoint_size_exact"] && !unknown_checkpoint["blocked"]
    delete!(actual["octree"]["reports"],"wall_sample_count")
    old=_q19_storage_forecast(case,actual;available_bytes=1)
    @test !old["sample_count_exact"] && !old["blocked"]
    @test _q19_storage_forecast(case,actual;available_bytes=nothing)["available_bytes"]===nothing
end

@testset "D3Q19 native surface choice reaches solver configuration" begin
    octree=Dict("threads"=>1,"band_fine"=>8,"band"=>6,"wake_length"=>0.,"wake_level_offset"=>1)
    case=Dict("octree"=>octree,"levels"=>3,"surface_dx"=>.01,"outputs"=>copy(OUTPUT_DEFAULTS))
    yaml=join(_q19_yaml_lines(case),"\n")
    @test occursin("surface_lbmp: true",yaml)
    case["outputs"]["surface_lbmp"]=false
    case["outputs"]["surface_vtk"]=false
    yaml=join(_q19_yaml_lines(case),"\n")
    @test occursin("surface_lbmp: false",yaml)
    @test occursin("surface_vtk: false",yaml)
end

@testset "old cavity sealing meshes cannot start new runs" begin
    case=Dict("solver"=>"q19_octree","domain"=>Dict("boundary"=>"free_air"),"octree"=>Dict("seal_cavities"=>true))
    @test_throws ArgumentError _require_current_q19_mesh(case,Dict("octree"=>Dict()))
    @test _require_current_q19_mesh(case,Dict("octree"=>Dict("cavity_seal_policy_version"=>2)))===nothing
    case["octree"]["seal_cavities"]=false
    @test _require_current_q19_mesh(case,Dict("octree"=>Dict()))===nothing
end
