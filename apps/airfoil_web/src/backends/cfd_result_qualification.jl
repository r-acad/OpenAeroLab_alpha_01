module CFDResultQualification

export implicit_flow_qualification, unverified_implicit_point, qualify_result_point

const RAW_REQUIRED_METRICS=("turb_k_equation_l2","turb_omega_equation_l2",
    "turb_k_equation_linf","turb_omega_equation_linf",
    "turb_omega_wall_constraint_relative_linf")
const COMPLETED_STATUSES=("completed","converged","ok")
_dict(value)=value isa AbstractDict ? value : Dict{String,Any}()
_finite_nonnegative(value)=value isa Real && !(value isa Bool) && isfinite(value) && value>=0

"Qualification is separate from the engine's historical stopping monitor."
function implicit_flow_qualification(data::AbstractDict)
    raw=_dict(get(data,"implicit_raw_sst_residuals",nothing))
    required=get(data,"time_mode","steady")!="physical" &&
        (get(data,"implicit_method",false)===true ||
         get(data,"rans_method","") in ("line_implicit","jfnk_native","jfnk_krylov") ||
         get(raw,"available",false)===true)
    guard=_dict(get(data,"convergence_guard",nothing))
    stopped=get(guard,"status","") in ("early_stop_eligible","six_equation_guard_passed")
    available=get(raw,"available",false)===true
    limits_qualified=get(raw,"qualified_stopping_limits",false)===true
    checks_passed=get(raw,"checks_passed",false)===true
    values=_dict(get(raw,"final_values",nothing));limits=_dict(get(raw,"stopping_limits",nothing))
    values_pass=all(RAW_REQUIRED_METRICS) do name
        value=get(values,name,nothing);limit=get(limits,name,nothing)
        _finite_nonnegative(value) && _finite_nonnegative(limit) && limit>0 && value<=limit
    end
    qualified=required && stopped && available && limits_qualified && checks_passed && values_pass
    reason=!required ? "Raw implicit SST qualification does not apply to this solver/time mode." :
        qualified ? "Flow equation checks passed under qualified raw turbulence and wall-constraint limits; mesh independence and physical validation remain unverified." :
        !available ? "Full flow checks unverified: raw turbulence equation diagnostics are unavailable." :
        !limits_qualified ? "Full flow checks unverified: raw turbulence and wall-constraint stopping limits are not qualified." :
        !checks_passed || !values_pass ? "Full flow checks unverified: raw turbulence equations and the prescribed wall constraint have not all passed finite qualified limits." :
        "Full flow checks unverified: the solver flow, mass and load stop guard has not passed."
    Dict{String,Any}("required"=>required,"qualified"=>qualified,
        "solver_stop_reached"=>stopped,"raw_available"=>available,
        "raw_limits_qualified"=>limits_qualified,"raw_checks_passed"=>checks_passed,
        "raw_values_pass_limits"=>values_pass,"required_metrics"=>collect(RAW_REQUIRED_METRICS),
        "status"=>!required ? "not_applicable" : qualified ? "flow_checks_passed" : "flow_checks_unverified",
        "reason"=>reason)
end

function unverified_implicit_point(point::AbstractDict)
    get(point,"backend","")=="transonic" || return false
    qualification=implicit_flow_qualification(_dict(get(point,"diagnostics",nothing)))
    qualification["required"] && !qualification["qualified"] &&
        lowercase(string(get(point,"status",""))) in (COMPLETED_STATUSES...,"solver_stop_unverified")
end

"Return a view of a result without promoting the old monitor to full convergence."
function qualify_result_point(point::AbstractDict)
    get(point,"backend","")=="transonic" || return point
    original=_dict(get(point,"diagnostics",nothing))
    qualification=implicit_flow_qualification(original)
    qualification["required"] || return point
    result=Dict{String,Any}(string(k)=>v for (k,v) in point)
    diagnostics=Dict{String,Any}(string(k)=>v for (k,v) in original)
    diagnostics["flow_qualification"]=qualification
    diagnostics["flow_checks_qualified"]=qualification["qualified"]
    diagnostics["engine_guard"]=get(original,"engine_guard",deepcopy(get(original,"convergence_guard",Dict{String,Any}())))
    stop_reason=get(original,"implicit_stop_reason","")
    isempty(string(stop_reason)) && (stop_reason=get(_dict(get(original,"convergence_guard",nothing)),"reason",""))
    diagnostics["engine_stop_reason"]=get(original,"engine_stop_reason",stop_reason)
    result["diagnostics"]=diagnostics
    result["flow_checks_qualified"]=qualification["qualified"]
    status=lowercase(string(get(point,"status","")))
    result["converged"]=qualification["qualified"] && status in COMPLETED_STATUSES
    if unverified_implicit_point(point)
        result["solver_reported_status"]=get(point,"solver_reported_status",get(point,"status",""))
        diagnostics["engine_reported_convergence"]=get(original,"engine_reported_convergence",get(point,"convergence",""))
        result["status"]="solver_stop_unverified"
        result["convergence"]=(qualification["solver_stop_reached"] ? "Solver stop reached; " : "")*qualification["reason"]
    end
    result
end

end
