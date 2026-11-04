"""Shared steady-RANS reporting from measured evidence, independent of iteration cost.

No solver equations are changed here. Residual keys and normalization-specific
limits are supplied by the backend; values from different equations are never
combined into an unlabelled residual. Window sizes count diagnostic samples, so
an implicit Newton iteration need not satisfy an explicit solver's 50k-step rule.
"""
module RANSConvergence
using Statistics: mean
export Config, assess

Base.@kwdef struct Config
    window_samples::Int = 20
    mass_limit::Float64 = 1e-3
    mass_failure_limit::Float64 = .01
    cl_span_abs::Float64 = .0035
    cl_span_rel::Float64 = .0025
    cl_drift_abs::Float64 = .0015
    cl_drift_rel::Float64 = .001
    cd_span_abs::Float64 = .0005
    cd_span_rel::Float64 = .004
    cd_drift_abs::Float64 = .0001
    cd_drift_rel::Float64 = .0015
    cm_span_abs::Float64 = .0005
    cm_span_rel::Float64 = .002
    cm_drift_abs::Float64 = .0002
    cm_drift_rel::Float64 = .001
    drag_min::Float64 = -1e-8
    drag_max::Float64 = 1.
    lift_max::Float64 = 3.
end

_get(row,key,default=nothing)=row isa AbstractDict ? get(row,String(key),get(row,Symbol(key),default)) :
    hasproperty(row,Symbol(key)) ? getproperty(row,Symbol(key)) : default
function _number(value)
    value isa Bool && return nothing
    value isa Real && return Float64(value)
    value isa AbstractString && return tryparse(Float64,value)
    nothing
end
_value(row,key)=_number(_get(row,key))
_limits(raw)=Dict(String(k)=>Float64(v) for (k,v) in pairs(raw))
_iteration(row)=something(_number(_get(row,:iteration,_get(row,:step))),0.)

function _validate(cfg,limits,relative,ceilings)
    cfg.window_samples>=2 || throw(ArgumentError("window_samples must be at least two"))
    0<cfg.mass_limit<=cfg.mass_failure_limit && isfinite(cfg.mass_failure_limit) ||
        throw(ArgumentError("mass limits must be finite, positive and ordered"))
    for name in fieldnames(Config)
        occursin(r"_(span|drift)_",String(name)) || continue
        value=getfield(cfg,name)
        isfinite(value) && value>=0 || throw(ArgumentError("$name must be finite and nonnegative"))
    end
    isfinite(cfg.drag_min) && isfinite(cfg.drag_max) && cfg.drag_min<cfg.drag_max &&
        isfinite(cfg.lift_max) && cfg.lift_max>0 || throw(ArgumentError("Invalid physical load bounds"))
    for (key,v) in limits
        isfinite(v)&&v>0 || throw(ArgumentError("Residual limit $key must be positive and finite"))
    end
    for (key,v) in relative
        haskey(limits,key) && isfinite(v) && 0<v<1 || throw(ArgumentError("Relative limit $key needs an absolute limit and must be in (0,1)"))
        ceiling=get(ceilings,key,NaN)
        isfinite(ceiling) && ceiling>=limits[key] || throw(ArgumentError("Relative acceptance for $key needs a finite absolute ceiling"))
    end
    nothing
end

"""
    assess(rows; residual_limits, config=Config(), ...)

Return a JSON/TOML-safe report with `status`, `reason`, independent gates and
metrics. Status is `converged`, `loads_stable_flow_unsettled`, `incomplete`, or
`failed`. Missing diagnostics never count as zero or as passed.

`residual_limits` maps measured column names to absolute limits in their stated
normalization. Optional relative acceptance additionally requires an explicit
reference and an `absolute_ceilings` bound. Absolute acceptance does not require
relative reduction, including when the reference is exactly zero.

`equation_coverage_complete` must only be true when the backend measures all
equations needed for its stop. Fixed-clock turbulence update defects may be
included as diagnostics, but are not automatically labelled raw SST residuals.
`solver_steady_passed` can preserve a native backend's complete steady stop
(e.g. pressure, momentum, SST and circulation); its final residual and global
mass gates are still rechecked. It must not be set from a force-only plateau.

`phase_start` excludes samples from a previous solver/damping phase. Load fields
default to CL/CD/CM; use `load_fields=("cl","cd")` for legacy histories which
never recorded CM. Optional update limits are checked in addition to equations.
"""
function assess(rows; residual_limits, config::Config=Config(), relative_limits=Dict(),
                residual_references=Dict(), absolute_ceilings=Dict(), update_limits=Dict(),
                mass_key="mass_net_frac", load_fields=("cl","cd","cm"), phase_start=0,
                equation_coverage_complete::Bool=false, solver_steady_passed::Bool=false,
                physical_state_valid::Bool=true, failed::Bool=false,
                scope="backend-defined steady equations; mesh independence and experimental validation are separate")
    limits=_limits(residual_limits);relative=_limits(relative_limits);ceilings=_limits(absolute_ceilings)
    updates=_limits(update_limits);_validate(config,merge(limits,updates),relative,ceilings)
    isempty(intersect(keys(limits),keys(updates))) || throw(ArgumentError("Equation residuals and update defects must use distinct column names"))
    references=Dict(String(k)=>v for (k,v) in pairs(residual_references))
    fields=String.(collect(load_fields));all(k->k in ("cl","cd","cm"),fields) && !isempty(fields) ||
        throw(ArgumentError("load_fields must select cl, cd and/or cm"))
    samples=[row for row in rows if _iteration(row)>=phase_start]
    n=length(samples);w=config.window_samples;reasons=String[];invalid=String[];missing=String[]
    report=Dict{String,Any}("status"=>"incomplete","reason"=>"Insufficient diagnostic history",
        "scope"=>String(scope),"converged"=>false,"criteria_passed"=>false,
        "window_samples"=>w,"available_samples"=>n,"phase_start"=>phase_start,
        "equation_coverage_complete"=>equation_coverage_complete,
        "native_solver_steady_passed"=>solver_steady_passed,"metrics"=>Dict{String,Any}())
    metrics=report["metrics"]
    failed && push!(invalid,"the solver reported failure")
    physical_state_valid || push!(invalid,"the final physical state failed positivity, finiteness or backend bounds")
    if isempty(samples)
        push!(missing,"history")
        report["status"]=isempty(invalid) ? "incomplete" : "failed"
        report["reason"]=isempty(invalid) ? "No diagnostic samples are available" : join(invalid,"; ")
        report["missing"]=missing;report["invalid"]=invalid;return report
    end
    iterations=_iteration.(samples)
    all(isfinite,iterations) && all(>(0),diff(iterations)) || push!(invalid,"diagnostic sample iterations are not finite and strictly increasing")
    report["last_iteration"]=last(iterations)
    recent=samples[max(1,n-w+1):n]
    prior=n>=2w ? samples[n-2w+1:n-w] : eltype(samples)[]
    current=last(samples)
    for (key,lo,hi) in (("cl",-config.lift_max,config.lift_max),("cd",config.drag_min,config.drag_max))
        value=_value(current,key)
        value===nothing ? push!(missing,key) : (!isfinite(value)||!(lo<=value<=hi)) && push!(invalid,"$key is outside physical sanity limits")
    end
    load_ok=n>=2w
    load_metrics=Dict{String,Any}()
    for key in fields
        av=[_value(row,key) for row in recent];pv=[_value(row,key) for row in prior]
        if any(isnothing,av)||any(isnothing,pv)
            push!(missing,key);load_ok=false;continue
        elseif any(x->!isfinite(x),av)||any(x->!isfinite(x),pv)
            push!(invalid,"$key history is nonfinite");load_ok=false;continue
        elseif isempty(pv)
            continue
        end
        a=Float64.(av);p=Float64.(pv);scale=max(abs(mean(a)),abs(mean(p)))
        span_limit=getfield(config,Symbol(key*"_span_abs"))+scale*getfield(config,Symbol(key*"_span_rel"))
        drift_limit=getfield(config,Symbol(key*"_drift_abs"))+scale*getfield(config,Symbol(key*"_drift_rel"))
        span=max(maximum(a)-minimum(a),maximum(p)-minimum(p));drift=abs(mean(a)-mean(p))
        passed=span<=span_limit && drift<=drift_limit
        load_ok &= passed
        load_metrics[key]=Dict("span"=>span,"span_limit"=>span_limit,"mean_drift"=>drift,"drift_limit"=>drift_limit,"passed"=>passed)
    end
    metrics["loads"]=load_metrics
    mass_values=[_value(row,mass_key) for row in recent]
    mass_final=_value(current,mass_key);mass_ok=false;mass_final_ok=false
    if any(isnothing,mass_values)
        push!(missing,String(mass_key))
    elseif any(x->!isfinite(x),mass_values)
        push!(invalid,"global mass balance is nonfinite")
    else
        mass_max=maximum(abs,Float64.(mass_values));mass_final_ok=abs(mass_final)<=config.mass_limit
        mass_ok=mass_max<=config.mass_limit
        metrics["mass"]=Dict("recent_max_abs"=>mass_max,"final_abs"=>abs(mass_final),"limit"=>config.mass_limit,"passed"=>mass_ok)
        abs(mass_final)>config.mass_failure_limit && push!(invalid,"global mass imbalance exceeds $(100config.mass_failure_limit)% of the reference mass flux")
    end
    equation_ok=!isempty(limits);final_equation_ok=!isempty(limits);update_ok=true
    residual_metrics=Dict{String,Any}()
    for (key,limit) in merge(limits,updates)
        values=[_value(row,key) for row in recent];is_update=haskey(updates,key)
        if any(isnothing,values)
            push!(missing,key);is_update ? (update_ok=false) : (equation_ok=false;final_equation_ok=false);continue
        elseif any(x->!isfinite(x)||x<0,values)
            push!(invalid,"$key is nonfinite or negative");is_update ? (update_ok=false) : (equation_ok=false;final_equation_ok=false);continue
        end
        peak=maximum(Float64.(values));final=Float64(last(values));absolute=peak<=limit;passed=absolute
        metric=Dict{String,Any}("recent_max"=>peak,"final"=>final,"absolute_limit"=>limit,"absolute_passed"=>absolute,
            "kind"=>is_update ? "normalized update defect" : "backend-defined equation residual")
        ref=_number(get(references,key,nothing))
        if ref!==nothing && isfinite(ref) && ref>0
            metric["reference"]=ref;metric["relative_ratio"]=peak/ref
            if !is_update && haskey(relative,key)
                relative_passed=peak/ref<=relative[key] && peak<=ceilings[key]
                metric["relative_limit"]=relative[key];metric["absolute_ceiling"]=ceilings[key]
                metric["relative_passed"]=relative_passed;passed |= relative_passed
            end
        end
        metric["passed"]=passed;residual_metrics[key]=metric
        if is_update;update_ok &= passed
        else;equation_ok &= passed;final_equation_ok &= final<=limit;end
    end
    metrics["residuals"]=residual_metrics
    history_ready=n>=2w
    criteria=history_ready && load_ok && mass_ok && equation_ok && update_ok && isempty(invalid) && isempty(missing)
    native=solver_steady_passed && final_equation_ok && mass_final_ok && update_ok && isempty(invalid) && isempty(missing)
    converged=native || criteria && equation_coverage_complete
    report["criteria_passed"]=criteria;report["converged"]=converged
    report["gates"]=Dict("loads_stable"=>load_ok,"history_ready"=>history_ready,"global_mass"=>mass_ok,
        "equations"=>equation_ok,"updates"=>update_ok,"physical_state"=>isempty(invalid),"native_steady_stop_rechecked"=>native)
    history_ready || push!(reasons,"need $(2w) diagnostic samples in the current solver phase")
    load_ok || push!(reasons,"load stability has not passed")
    equation_ok || push!(reasons,"equation residual limits have not passed")
    mass_ok || push!(reasons,"global mass balance is unavailable or above its limit")
    update_ok || push!(reasons,"normalized update limits have not passed")
    equation_coverage_complete || solver_steady_passed || push!(reasons,"complete equation convergence evidence is unavailable")
    report["missing"]=unique(missing);report["invalid"]=invalid
    if !isempty(invalid)
        report["status"]="failed";report["reason"]=join(invalid,"; ")
    elseif converged
        report["status"]="converged"
        report["reason"]=native ? "Native steady stop passed; final equation and signed mass-balance limits rechecked" :
            "Load windows, measured equation limits, signed mass balance and configured update checks passed"
    else
        report["status"]=load_ok ? "loads_stable_flow_unsettled" : "incomplete"
        report["reason"]=criteria && !equation_coverage_complete ?
            "All monitored equation, mass-balance, load and configured update checks passed; complete raw-equation coverage is unavailable" :
            join(reasons,"; ")
    end
    report
end
end
