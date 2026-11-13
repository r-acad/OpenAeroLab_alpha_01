using Test, UUIDs
const REPOSITORY=dirname(dirname(@__DIR__))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","airfoil_flow_qualification_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
ENV["AIRFOIL_WEB_WARM_PREVIEW"]="0"
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb;const Q=A.CFDBackend.CFDResultQualification
function point(;qualified=false)
    raw=Dict{String,Any}("available"=>true,"qualified_stopping_limits"=>qualified,
        "checks_passed"=>qualified,"final_values"=>Dict{String,Any}(name=>1e-9 for name in Q.RAW_REQUIRED_METRICS),
        "stopping_limits"=>Dict{String,Any}(name=>1e-7 for name in Q.RAW_REQUIRED_METRICS))
    Dict{String,Any}("backend"=>"transonic","status"=>"completed","converged"=>true,
        "alpha_deg"=>4.,"cl"=>.7,"cd"=>.02,"cm"=>-.1,"convergence"=>"old monitor passed",
        "fields"=>Dict("full_artifact"=>"/api/jobs/example/files/field.msgpack"),
        "diagnostics"=>Dict{String,Any}("rans_method"=>"jfnk_native","time_mode"=>"steady",
            "physical_sanity"=>true,"implicit_stop_reason"=>"legacy-stop-reason",
            "convergence_guard"=>Dict("status"=>"early_stop_eligible","reason"=>"legacy engine guard passed"),
            "steady_assessment"=>Dict("converged"=>true),"implicit_raw_sst_residuals"=>raw))
end
case()=Dict{String,Any}("airfoil"=>Dict("kind"=>"naca4","id"=>"2412"),"backends"=>["transonic"],
    "flow"=>Dict("mach"=>.3,"reynolds"=>3e6,"alpha_deg"=>4.),"alphas"=>[4.])
job(p;id=string(uuid4()))=Dict{String,Any}("id"=>id,"status"=>"completed","message"=>"Analysis complete",
    "case"=>case(),"result"=>Dict{String,Any}("results"=>[p]))
function api(path)
    response=A.app_handler(A.HTTP.Request("GET",path,["Accept"=>"application/msgpack"]))
    response.status,A.AirfoilWire.decode(response.body,:msgpack)
end
@testset "Implicit raw SST qualification is independent of old stopping monitors" begin
    old=point();unchanged=deepcopy(old);p=Q.qualify_result_point(old)
    @test old==unchanged
    @test p["status"]=="solver_stop_unverified" && p["converged"]===false
    @test p["flow_checks_qualified"]===false
    @test p["solver_reported_status"]=="completed"
    @test p["diagnostics"]["engine_guard"]==old["diagnostics"]["convergence_guard"]
    @test p["diagnostics"]["engine_stop_reason"]=="legacy-stop-reason"
    @test p["fields"]==old["fields"]
    @test occursin("Full flow checks unverified",p["convergence"])
    @test Q.unverified_implicit_point(old)
    @test A._point_outcome([old])[1]=="partial"
    @test occursin("unverified",A._point_outcome([old])[2])
    @test A._point_outcome([p])[1]=="partial"
    summary=A._job_summary(job(old))
    @test summary["status"]=="partial"
    @test summary["completed_points"]==0 && summary["unverified_flow_points"]==1
    public=A._public_job(job(old))
    @test public["status"]=="partial" && public["result"]["results"][1]["converged"]===false
    positive=point(qualified=true);ready=Q.qualify_result_point(positive)
    @test ready["status"]=="completed" && ready["flow_checks_qualified"] && ready["converged"]
    @test A._point_outcome([ready])[1]=="completed"
    goodsummary=A._job_summary(job(positive))
    @test goodsummary["completed_points"]==1 && goodsummary["unverified_flow_points"]==0
    for mutate in (
        raw->(raw["checks_passed"]=false),
        raw->(raw["qualified_stopping_limits"]=false),
        raw->(raw["available"]=false),
        raw->empty!(raw["final_values"]),
        raw->delete!(raw["stopping_limits"],"turb_omega_wall_constraint_relative_linf"),
        raw->(raw["final_values"]["turb_k_equation_l2"]=1.),
        raw->(raw["final_values"]["turb_omega_equation_l2"]=NaN),
        raw->(raw["final_values"]["turb_omega_wall_constraint_relative_linf"]=Inf),
        raw->(raw["stopping_limits"]["turb_k_equation_linf"]=true))
        candidate=point(qualified=true);mutate(candidate["diagnostics"]["implicit_raw_sst_residuals"])
        @test !Q.qualify_result_point(candidate)["flow_checks_qualified"]
        @test A._point_outcome([candidate])[1]=="partial"
    end
    missing=point();delete!(missing["diagnostics"],"implicit_raw_sst_residuals")
    @test Q.unverified_implicit_point(missing)
    failedguard=point(qualified=true);failedguard["diagnostics"]["convergence_guard"]["status"]="max_steps"
    @test !Q.qualify_result_point(failedguard)["flow_checks_qualified"]
    for backend in ("lowspeed","cavitation","lbm","jxfoil")
        other=point();other["backend"]=backend
        @test Q.qualify_result_point(other)===other
        @test A._point_outcome([other])[1]=="completed"
    end
    standard=point();standard["diagnostics"]["rans_method"]="standard";standard["diagnostics"]["implicit_raw_sst_residuals"]["available"]=false
    @test Q.qualify_result_point(standard)===standard
    physical=point();physical["diagnostics"]["time_mode"]="physical"
    @test Q.qualify_result_point(physical)===physical
end
@testset "Older saved completed records and recent summaries are qualified on read" begin
    id=string(uuid4());directory=A.AirfoilResultLocations.create_directory(A.RUN_DIR,id)
    record=job(point();id);record["request"]=A._job_request(pop!(record,"case"))
    write(joinpath(directory,A.JOB_RECORD_FILE),A.AirfoilWire.encode(record,:msgpack))
    write(joinpath(directory,A.JOB_SUMMARY_FILE),A.AirfoilWire.encode(Dict("id"=>id,"status"=>"completed","message"=>"Analysis complete","backends"=>["transonic"],"completed_points"=>1),:msgpack))
    original=read(joinpath(directory,A.JOB_RECORD_FILE))
    code,visible=api("/api/jobs/"*id)
    @test code==200 && visible["status"]=="partial"
    @test visible["result"]["results"][1]["status"]=="solver_stop_unverified"
    @test visible["result"]["results"][1]["fields"]["full_artifact"]==record["result"]["results"][1]["fields"]["full_artifact"]
    @test read(joinpath(directory,A.JOB_RECORD_FILE))==original
    code,recent=api("/api/recent")
    @test code==200
    summary=only(filter(item->item["id"]==id,recent["items"]))
    @test summary["status"]=="partial" && summary["completed_points"]==0
    @test summary["unverified_flow_points"]==1
end
