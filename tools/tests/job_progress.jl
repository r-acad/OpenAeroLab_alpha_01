using Test
include(joinpath(@__DIR__,"..","..","apps","airfoil_web","src","api","job_progress.jl"))
const JP=AirfoilJobProgress
update(message,progress)=Dict{String,Any}("message"=>message,"progress"=>progress)
event(kind;kw...)=Dict{String,Any}("event"=>kind,(String(k)=>v for (k,v) in kw)...)

@testset "Worker heartbeat retains iteration progress for its own point" begin
    latest=Dict{Int,Dict{String,Any}}()
    first=update("iteration 430/2000, CL=1.4, CD=0.06",0.215)
    @test JP.preserve_iteration!(latest,1,event("iteration"),first)==first
    heartbeat=update("Running lowspeed after 3600 s",0.0)
    for elapsed in (3600,3610,3620)
        @test JP.preserve_iteration!(latest,1,event("running";elapsed_seconds=elapsed),heartbeat)==first
    end
    next=update("iteration 431/2000, CL=1.4, CD=0.06",0.2155)
    @test JP.preserve_iteration!(latest,1,event("iteration"),next)==next
    @test JP.preserve_iteration!(latest,1,event("running"),heartbeat)==next
    other=update("Running next angle",0.5)
    @test JP.preserve_iteration!(latest,2,event("running"),other)==other
    @test JP.preserve_iteration!(latest,1,event("point_complete"),update("point complete",0.5))["progress"]==0.5
    @test !haskey(latest,1)
end

@testset "Worker and GPU transitions retain their own messages" begin
    for transition in (event("started"),event("gpu_wait"),event("gpu_acquired"),
                       event("running";gpu_wait="another owner"),event("point_failed"))
        latest=Dict(1=>update("iteration 1/20",0.05))
        waiting=update("waiting, starting or failed",0.0)
        @test JP.preserve_iteration!(latest,1,transition,waiting)==waiting
        @test isempty(latest)
        @test JP.preserve_iteration!(latest,1,event("running"),update("running",0.0))["progress"]==0.0
    end
end
