module CFDTimeStatistics

export time_statistics

_value(row,key) = row isa AbstractDict ? Float64(row[key]) : Float64(getproperty(row,Symbol(key)))
_unavailable(reason) = Dict{String,Any}("available"=>false,"reason"=>reason,
    "sampling_adequate"=>false,"stationarity_passed"=>false,
    "physical_mean_eligible"=>false,"status"=>"insufficient_sampling")

"""Physical-time force means with explicit sampling and timestep qualification.

Trapezoidal integration uses the actual nonuniform physical clock. Block
uncertainties are conditional on the reported stationarity/correlation checks;
they exclude grid, turbulence model and experimental uncertainty. Setting the
timestep flag records an externally established refinement result, not a new
temporal validation performed by this routine.
"""
function time_statistics(history; burn_in_ctu=nothing, block_duration_ctu::Real=5.0,
                         min_blocks::Int=10, min_samples_per_block::Int=8,
                         max_sample_gap_ctu::Real=0.25,
                         time_step_refined::Bool=false,
                         requested_duration_ctu=nothing)
    min_blocks>=10 && min_samples_per_block>=2 && block_duration_ctu>0 &&
        max_sample_gap_ctu>0 || throw(ArgumentError("invalid physical force sampling settings"))
    length(history)>=2 || return _unavailable("too few physical-time samples")
    tt=[_value(row,"t_ctu") for row in history]
    all(isfinite,tt) && all(>(0),diff(tt)) ||
        return _unavailable("physical clock must be finite and strictly increasing")
    finish=last(tt);start=burn_in_ctu===nothing ? .5finish : Float64(burn_in_ctu)
    isfinite(start) && 0<=start<finish && first(tt)<=start ||
        return _unavailable("physical history does not cover a positive post-burn-in interval")
    keys=("cl","cd","cm")
    values=Dict(key=>[_value(row,key) for row in history] for key in keys)
    all(all(isfinite,values[key]) for key in keys) ||
        return _unavailable("physical force history contains nonfinite coefficients")
    function mean_interval(key,lo,hi)
        integral=0.0;covered=0.0;vv=values[key]
        for k in 2:length(tt)
            ta,tb=tt[k-1],tt[k];l=max(ta,lo);r=min(tb,hi)
            r>l || continue
            slope=(vv[k]-vv[k-1])/(tb-ta)
            integral+=.5*((vv[k-1]+slope*(l-ta))+(vv[k-1]+slope*(r-ta)))*(r-l)
            covered+=r-l
        end
        covered>=.999999(hi-lo) || error("physical force history integration coverage gap")
        integral/covered
    end
    duration=finish-start
    blocks=max(min_blocks,min(1000,floor(Int,duration/block_duration_ctu)))
    width=duration/blocks
    count_retained=count(>=(start),tt)
    sample_gap=maximum(tt[k]-tt[k-1] for k in 2:length(tt) if tt[k]>start)
    block_counts=[count(t->start+(b-1)*width<=t<=start+b*width,tt) for b in 1:blocks]
    duration_complete=requested_duration_ctu===nothing ||
        finish>=Float64(requested_duration_ctu)*(1-1e-7)
    sampling_adequate=width>=block_duration_ctu*(1-1e-9) &&
        minimum(block_counts)>=min_samples_per_block &&
        sample_gap<=max_sample_gap_ctu && duration_complete
    stats=Dict{String,Any}("available"=>true,
        "average_start_ctu"=>start,"average_end_ctu"=>finish,
        "average_duration_ctu"=>duration,"burn_in_fraction"=>start/finish,
        "block_count"=>blocks,"block_duration_ctu"=>width,
        "minimum_block_samples"=>minimum(block_counts),"retained_samples"=>count_retained,
        "maximum_sample_gap_ctu"=>sample_gap,"sample_gap_limit_ctu"=>Float64(max_sample_gap_ctu),
        "duration_complete"=>duration_complete,"sampling_adequate"=>sampling_adequate,
        "time_step_refinement_confirmed"=>time_step_refined,
        "method"=>"trapezoidal integration on actual physical clock; equal-duration block means",
        "uncertainty_scope"=>"conditional block standard error and approximate 95% interval; no model/grid uncertainty; external timestep refinement flag",
        "validation_scope"=>"sampling qualification only; equation conservation, turbulence/model fidelity and stall validation require separate evidence")
    stationary=true;independent=true
    abs_limits=Dict("cl"=>.005,"cd"=>.0005,"cm"=>.001)
    for key in keys
        means=[mean_interval(key,start+(b-1)*width,start+b*width) for b in 1:blocks]
        avg=mean_interval(key,start,finish);bar=sum(means)/blocks
        variance=sum((m-bar)^2 for m in means)/(blocks-1)
        stderr=sqrt(variance/blocks)
        denominator=sum((m-bar)^2 for m in means)
        lag1=denominator<=eps(Float64)*max(1.,bar^2) ? 0.0 :
            sum((means[k]-bar)*(means[k-1]-bar) for k in 2:blocks)/denominator
        firsthalf=mean_interval(key,start,.5(start+finish))
        secondhalf=mean_interval(key,.5(start+finish),finish)
        drift=abs(secondhalf-firsthalf)
        # These are provisional reporting gates, not calibrated stall tolerances.
        limit=max(abs_limits[key],2stderr)
        stationary &= drift<=limit
        independent &= abs(lag1)<=.3
        stats[key*"_mean"]=avg;stats[key*"_block_means"]=means
        stats[key*"_block_standard_error"]=stderr
        stats[key*"_confidence95_halfwidth"]=2.262stderr
        stats[key*"_block_lag1_correlation"]=lag1
        stats[key*"_first_half_mean"]=firsthalf
        stats[key*"_second_half_mean"]=secondhalf
        stats[key*"_half_mean_drift"]=drift
        stats[key*"_half_mean_drift_limit"]=limit
    end
    stats["stationarity_passed"]=stationary
    stats["block_independence_screen_passed"]=independent
    stats["physical_mean_eligible"]=sampling_adequate && stationary && independent && time_step_refined
    stats["status"]=!sampling_adequate ? "insufficient_sampling" :
        !stationary ? "unsettled_physical_mean" : !independent ? "correlated_blocks" :
        !time_step_refined ? "time_step_refinement_unverified" : "physical_sampling_qualified"
    stats
end

end
