module CFDConvergence

using Statistics: median

export Config, Monitor, observe!, convergence_report

"""
Conservative *early-stop* guard for the steady CMESH2D worker. The measured
residuals cover continuity only. Passing this guard is not a certificate that
momentum, energy, turbulence, or experimental validation have converged.

`residual_scaled_limit`, `residual_l2_limit`, and `mass_net_limit` are
deliberately explicit: a force plateau at a large equation residual or signed
global mass imbalance is a stalled solve. A separately confirmed stagnation
stop can end wasted iterations with an explicitly nonconverged status. The
defaults are provisional guardrails, not universal physical tolerances;
calibrate them on mesh studies.
"""
Base.@kwdef struct Config
    sfd_start::Int = 150_000
    max_steps::Int = 500_000
    min_post_sfd::Int = 50_000
    window_iterations::Int = 20_000
    min_samples_per_window::Int = 40
    confirmation_iterations::Int = 10_000
    confirmation_checks::Int = 10
    margin::Float64 = 0.75
    residual_scaled_limit::Float64 = 0.1
    residual_l2_limit::Float64 = 0.02
    mass_net_limit::Float64 = 0.001
    residual_ratio_limit::Float64 = 0.7
    plateau_fraction::Float64 = 0.25
    residual_spike_factor::Float64 = 2.0
    cl_span_abs::Float64 = 0.0035
    cl_span_rel::Float64 = 0.0025
    cd_span_abs::Float64 = 0.0005
    cd_span_rel::Float64 = 0.004
    cl_drift_abs::Float64 = 0.0015
    cl_drift_rel::Float64 = 0.001
    cd_drift_abs::Float64 = 0.0001
    cd_drift_rel::Float64 = 0.0015
    divergence_ratio::Float64 = 100.0
    divergence_checks::Int = 5
    stop_on_stall::Bool = true
    stall_min_post_sfd::Int = 100_000
    stall_window_iterations::Int = 40_000
    stall_min_samples::Int = 75
    stall_confirmation_iterations::Int = 20_000
    stall_confirmation_checks::Int = 20
    stall_progress_fraction::Float64 = 0.02
end

const Sample = NamedTuple{(:iteration, :residual_scaled, :residual_l2, :cl, :cd,
                           :mass_net_frac),
                          Tuple{Int, Float64, Float64, Float64, Float64,
                                Union{Nothing,Float64}}}

mutable struct Monitor
    config::Config
    samples::Vector{Sample}
    reference::Union{Nothing,Sample}
    candidate_since::Int
    candidate_checks::Int
    divergence_checks::Int
    stall_since::Int
    stall_checks::Int
    last_status::String
    last_reason::String
    last_metrics::Dict{String,Float64}
end

function Monitor(config::Config=Config())
    0 < config.margin < 1 || throw(ArgumentError("margin must be between 0 and 1"))
    config.window_iterations > 0 || throw(ArgumentError("window_iterations must be positive"))
    config.min_post_sfd >= 2 * config.window_iterations ||
        throw(ArgumentError("min_post_sfd must cover two complete windows"))
    config.max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    config.sfd_start >= 0 || throw(ArgumentError("sfd_start must be nonnegative"))
    config.confirmation_iterations > 0 ||
        throw(ArgumentError("confirmation_iterations must be positive"))
    config.confirmation_checks > 0 ||
        throw(ArgumentError("confirmation_checks must be positive"))
    config.min_samples_per_window >= 2 ||
        throw(ArgumentError("min_samples_per_window must be at least two"))
    config.stall_window_iterations > 0 &&
        config.stall_min_post_sfd >= 2 * config.stall_window_iterations ||
        throw(ArgumentError("stall warmup must cover two positive history windows"))
    config.stall_min_samples >= 2 && config.stall_confirmation_checks > 0 &&
        config.stall_confirmation_iterations > 0 ||
        throw(ArgumentError("stall confirmation and sample counts must be positive"))
    0 < config.stall_progress_fraction < 1 ||
        throw(ArgumentError("stall_progress_fraction must be between 0 and 1"))
    for field in (:residual_scaled_limit, :residual_l2_limit, :mass_net_limit,
                  :residual_ratio_limit, :plateau_fraction,
                  :residual_spike_factor, :divergence_ratio)
        value = getfield(config, field)
        isfinite(value) && value > 0 ||
            throw(ArgumentError("$field must be finite and positive"))
    end
    Monitor(config, Sample[], nothing, 0, 0, 0, 0, 0, "warming",
            "waiting for selective frequency damping and full history windows",
            Dict{String,Float64}())
end

function _global_mass_ok(m::Monitor, prior::Vector{Sample}, recent::Vector{Sample})
    limit = m.config.margin*m.config.mass_net_limit
    current = last(recent).mass_net_frac
    m.last_metrics["mass_net_limit_effective"] = limit
    if any(r -> r.mass_net_frac === nothing, prior) ||
       any(r -> r.mass_net_frac === nothing, recent)
        m.last_metrics["mass_net_available"] = 0.0
        return false
    end
    prior_max = maximum(abs(r.mass_net_frac::Float64) for r in prior)
    recent_max = maximum(abs(r.mass_net_frac::Float64) for r in recent)
    m.last_metrics["mass_net_available"] = 1.0
    m.last_metrics["mass_net_current"] = current::Float64
    m.last_metrics["mass_net_abs_prior_max"] = prior_max
    m.last_metrics["mass_net_abs_recent_max"] = recent_max
    prior_max <= limit && recent_max <= limit
end

_decision(m::Monitor; stop=false, abort=false) =
    (stop=stop, abort=abort, status=m.last_status,
     reason=m.last_reason, metrics=copy(m.last_metrics))

function _set!(m::Monitor, status::String, reason::String;
               stop=false, abort=false)
    m.last_status = status
    m.last_reason = reason
    _decision(m; stop, abort)
end

_mean(v) = sum(v) / length(v)
_span(v) = maximum(v) - minimum(v)

function _window_slope(rows::Vector{Sample}, property::Symbol, width::Int)
    tbar = _mean(r.iteration for r in rows)
    ybar = _mean(getproperty(r, property) for r in rows)
    den = sum((r.iteration-tbar)^2 for r in rows)
    den > 0 || return Inf
    width * sum((r.iteration-tbar)*(getproperty(r, property)-ybar)
                for r in rows) / den
end

function _force_ok(m::Monitor, prior::Vector{Sample}, recent::Vector{Sample},
                   property::Symbol, span_abs::Float64, span_rel::Float64,
                   drift_abs::Float64, drift_rel::Float64;
                   width::Int=m.config.window_iterations,
                   metric_prefix::String=string(property))
    cfg = m.config
    av = _mean(getproperty(r, property) for r in recent)
    pv = _mean(getproperty(r, property) for r in prior)
    span_limit = cfg.margin * (span_abs + span_rel * max(abs(av), abs(pv)))
    drift_limit = cfg.margin * (drift_abs + drift_rel * max(abs(av), abs(pv)))
    span_recent = _span(getproperty(r, property) for r in recent)
    span_prior = _span(getproperty(r, property) for r in prior)
    drift = abs(av-pv)
    slope_recent = abs(_window_slope(recent, property, width))
    slope_prior = abs(_window_slope(prior, property, width))
    prefix = metric_prefix
    merge!(m.last_metrics, Dict(
        "$(prefix)_span_recent"=>span_recent,
        "$(prefix)_span_prior"=>span_prior,
        "$(prefix)_mean_drift"=>drift,
        "$(prefix)_slope_recent"=>slope_recent,
        "$(prefix)_slope_prior"=>slope_prior,
        "$(prefix)_span_limit"=>span_limit,
        "$(prefix)_drift_limit"=>drift_limit))
    span_recent <= span_limit && span_prior <= span_limit &&
        drift <= drift_limit && slope_recent <= drift_limit &&
        slope_prior <= drift_limit
end

function _residual_ok(m::Monitor, prior::Vector{Sample}, recent::Vector{Sample},
                      property::Symbol, ceiling::Float64)
    cfg = m.config
    p = median(getproperty(r, property) for r in prior)
    r = median(getproperty(row, property) for row in recent)
    ref = max(getproperty(m.reference::Sample, property), eps(Float64))
    ratio = r/ref
    plateau = r/max(p, eps(Float64))
    spike = maximum(getproperty(row, property) for row in recent) /
            max(r, eps(Float64))
    recent_max = maximum(getproperty(row, property) for row in recent)
    prefix = string(property)
    merge!(m.last_metrics, Dict(
        "$(prefix)_median"=>r,
        "$(prefix)_reference"=>ref,
        "$(prefix)_ratio"=>ratio,
        "$(prefix)_window_ratio"=>plateau,
        "$(prefix)_spike_factor"=>spike,
        "$(prefix)_recent_max"=>recent_max,
        "$(prefix)_limit"=>cfg.margin*ceiling))
    # A small absolute residual is already satisfactory. Relative reduction,
    # plateau and spike ratios become ill-conditioned near zero and must not
    # veto a warm start or roundoff-level residual. Keep them as diagnostics;
    # bound both the median (with margin) and every recent sample (hard ceiling).
    # Loads, signed global mass balance and confirmation windows remain required.
    r <= cfg.margin*ceiling && recent_max <= ceiling
end

# A plateau above the equation tolerances is a failed solve. Stop wasting
# iterations only after longer windows AND an additional confirmation margin;
# never promote this decision to early_stop_eligible or a valid polar point.
function _stall_decision!(m::Monitor, n::Int)
    cfg = m.config
    cfg.stop_on_stall && n >= cfg.sfd_start+cfg.stall_min_post_sfd || return nothing
    w = cfg.stall_window_iterations
    prior = [r for r in m.samples if n-2w < r.iteration <= n-w]
    recent = [r for r in m.samples if n-w < r.iteration <= n]
    if min(length(prior),length(recent)) < cfg.stall_min_samples
        return nothing
    end
    stationary = _force_ok(m,prior,recent,:cl,cfg.cl_span_abs,cfg.cl_span_rel,
        cfg.cl_drift_abs,cfg.cl_drift_rel;width=w,metric_prefix="stall_cl") &&
        _force_ok(m,prior,recent,:cd,cfg.cd_span_abs,cfg.cd_span_rel,
        cfg.cd_drift_abs,cfg.cd_drift_rel;width=w,metric_prefix="stall_cd")
    stagnant = true
    for property in (:residual_scaled,:residual_l2)
        p = median(getproperty(r,property) for r in prior)
        r = median(getproperty(row,property) for row in recent)
        ratio = r/max(p,eps(Float64))
        spike = maximum(getproperty(row,property) for row in recent)/max(r,eps(Float64))
        relative_slope = _window_slope(recent,property,w)/max(r,eps(Float64))
        m.last_metrics["stall_$(property)_window_ratio"] = ratio
        m.last_metrics["stall_$(property)_relative_slope"] = relative_slope
        stagnant &= 1-cfg.stall_progress_fraction <= ratio <= 1+cfg.stall_progress_fraction &&
                    relative_slope >= -cfg.stall_progress_fraction &&
                    spike <= cfg.residual_spike_factor
    end
    mass_available = all(r->r.mass_net_frac !== nothing,prior) &&
                     all(r->r.mass_net_frac !== nothing,recent)
    if mass_available
        p = median(abs(r.mass_net_frac::Float64) for r in prior)
        r = median(abs(row.mass_net_frac::Float64) for row in recent)
        mass_ratio = r/max(p,eps(Float64))
        mass_slope = _window_slope(recent,:mass_net_frac,w) *
            sign(median(row.mass_net_frac::Float64 for row in recent))/max(r,eps(Float64))
        m.last_metrics["stall_mass_net_window_ratio"] = mass_ratio
        m.last_metrics["stall_mass_net_relative_slope"] = mass_slope
        # Retain useful progress towards closing a global mass imbalance.
        stagnant &= max(p,r) <= cfg.margin*cfg.mass_net_limit ||
                    (mass_ratio >= 1-cfg.stall_progress_fraction &&
                     mass_slope >= -cfg.stall_progress_fraction)
    else
        stagnant = false
    end
    if stationary && stagnant
        if m.stall_since == 0
            m.stall_since = n
            m.stall_checks = 1
        else
            m.stall_checks += 1
        end
        m.last_metrics["stall_confirmation_iterations"] = n-m.stall_since
        m.last_metrics["stall_confirmation_checks"] = m.stall_checks
        if n-m.stall_since >= cfg.stall_confirmation_iterations &&
           m.stall_checks >= cfg.stall_confirmation_checks
            return _set!(m,"stalled_stop",
                "stopped without convergence: CL/CD are stationary and continuity and signed mass balance have made less than $(100cfg.stall_progress_fraction)% progress across two $(w)-iteration windows plus a $(cfg.stall_confirmation_iterations)-iteration confirmation margin";stop=true)
        end
    else
        m.stall_since = 0
        m.stall_checks = 0
    end
    nothing
end

"""
Record one strictly increasing solver-history row and return a decision:
`(stop, abort, status, reason, metrics)`. Call after each CSV/stream sample.
`stop=true` means either the conservative early-stop gates passed or a
confirmed, explicitly nonconverged `stalled_stop` ended an unproductive run;
`abort=true`
means sustained numerical divergence. The worker's maximum step count remains
the hard fallback. The returned status is descriptive, not a CFD validity flag.
"""
function observe!(m::Monitor, row)
    n = Int(row.iteration)
    isempty(m.samples) || n > last(m.samples).iteration ||
        throw(ArgumentError("convergence samples must have increasing iterations"))
    cfg = m.config
    1 <= n <= cfg.max_steps ||
        throw(ArgumentError("iteration must be in 1:max_steps"))
    mass_raw = hasproperty(row, :mass_net_frac) ? row.mass_net_frac : nothing
    mass_net = mass_raw === nothing ? nothing : Float64(mass_raw)
    sample = Sample((n, Float64(row.residual_scaled), Float64(row.residual_l2),
                     Float64(row.cl), Float64(row.cd), mass_net))
    if !all(isfinite, (sample.residual_scaled, sample.residual_l2,
                       sample.cl, sample.cd)) ||
       (mass_net !== nothing && !isfinite(mass_net)) ||
       sample.residual_scaled < 0 || sample.residual_l2 < 0
        return _set!(m, "diverging", "nonfinite or negative convergence metric";
                     abort=true)
    end
    push!(m.samples, sample)
    if mass_net === nothing
        m.last_metrics["mass_net_available"] = 0.0
    else
        m.last_metrics["mass_net_available"] = 1.0
        m.last_metrics["mass_net_current"] = mass_net
    end
    if m.reference === nothing && n >= cfg.sfd_start
        m.reference = sample
    end
    ref = m.reference
    if ref !== nothing && n >= cfg.sfd_start+10_000
        ratio = max(sample.residual_scaled/max(ref.residual_scaled, eps(Float64)),
                    sample.residual_l2/max(ref.residual_l2, eps(Float64)))
        if (ratio > cfg.divergence_ratio &&
            (sample.residual_scaled > cfg.residual_scaled_limit ||
             sample.residual_l2 > cfg.residual_l2_limit)) || abs(sample.cl) > 20 ||
           abs(sample.cd) > 20
            m.divergence_checks += 1
        else
            m.divergence_checks = 0
        end
        if m.divergence_checks >= cfg.divergence_checks
            return _set!(m, "diverging",
                         "residual or force magnitude grew persistently after damping";
                         abort=true)
        end
    end
    if n < cfg.sfd_start+cfg.min_post_sfd || ref === nothing
        return _set!(m, "warming",
                     "waiting for post-damping warmup and two complete history windows")
    end
    w = cfg.window_iterations
    prior = [r for r in m.samples if n-2w < r.iteration <= n-w]
    recent = [r for r in m.samples if n-w < r.iteration <= n]
    if length(prior) < cfg.min_samples_per_window ||
       length(recent) < cfg.min_samples_per_window
        return _set!(m, "warming", "insufficient samples in full history windows")
    end
    empty!(m.last_metrics)
    cl_ok = _force_ok(m, prior, recent, :cl,
                      cfg.cl_span_abs, cfg.cl_span_rel,
                      cfg.cl_drift_abs, cfg.cl_drift_rel)
    cd_ok = _force_ok(m, prior, recent, :cd,
                      cfg.cd_span_abs, cfg.cd_span_rel,
                      cfg.cd_drift_abs, cfg.cd_drift_rel)
    mass_ok = _residual_ok(m, prior, recent, :residual_scaled,
                           cfg.residual_scaled_limit)
    mass_l2_ok = _residual_ok(m, prior, recent, :residual_l2,
                              cfg.residual_l2_limit)
    global_mass_ok = _global_mass_ok(m, prior, recent)
    m.last_metrics["forces_stationary"] = Float64(cl_ok && cd_ok)
    m.last_metrics["local_continuity_passed"] = Float64(mass_ok && mass_l2_ok)
    m.last_metrics["global_continuity_passed"] = Float64(global_mass_ok)
    plausible = sample.cd >= 0 && sample.cd <= 1 && abs(sample.cl) <= 3
    passed = cl_ok && cd_ok && mass_ok && mass_l2_ok &&
             global_mass_ok && plausible
    if passed
        m.stall_since = 0
        m.stall_checks = 0
        if m.candidate_since == 0
            m.candidate_since = n
            m.candidate_checks = 1
        else
            m.candidate_checks += 1
        end
        m.last_metrics["confirmation_iterations"] = n-m.candidate_since
        m.last_metrics["confirmation_checks"] = m.candidate_checks
        if n-m.candidate_since >= cfg.confirmation_iterations &&
           m.candidate_checks >= cfg.confirmation_checks
            return _set!(m, "early_stop_eligible",
                         "local and signed global continuity metrics and CL/CD passed two-window gates with confirmation margin; other equations and physical parity remain unverified";
                         stop=true)
        end
        return _set!(m, "candidate",
                     "all gates currently pass; waiting for confirmation margin")
    end
    m.candidate_since = 0
    m.candidate_checks = 0
    stalled = _stall_decision!(m,n)
    stalled === nothing || return stalled
    if cl_ok && cd_ok && mass_ok && mass_l2_ok && !global_mass_ok
        return _set!(m, "forces_stationary",
                     "CL/CD and local continuity are stationary but signed global mass balance is unavailable or above its limit")
    end
    if cl_ok && cd_ok && (!mass_ok || !mass_l2_ok)
        return _set!(m, "forces_stationary",
                     "CL/CD are stationary but continuity residual remains above its absolute median or maximum limit")
    end
    _set!(m, "unsettled",
          "continuity residual or signed mass balance, force variation, or physical sanity gate has not passed")
end

"""TOML-safe metadata for the latest assessment and configured guardrails."""
function convergence_report(m::Monitor)
    cfg = m.config
    Dict{String,Any}(
        "status"=>m.last_status,
        "reason"=>m.last_reason,
        "scope"=>"local and signed global continuity metrics and CL/CD stationarity only; not full equation convergence or physical validation",
        "last_iteration"=>isempty(m.samples) ? 0 : last(m.samples).iteration,
        "reference_iteration"=>m.reference === nothing ? -1 : m.reference.iteration,
        "sfd_start"=>cfg.sfd_start,
        "max_steps"=>cfg.max_steps,
        "window_iterations"=>cfg.window_iterations,
        "min_post_sfd"=>cfg.min_post_sfd,
        "confirmation_iterations"=>cfg.confirmation_iterations,
        "margin"=>cfg.margin,
        "residual_scaled_limit"=>cfg.residual_scaled_limit,
        "residual_l2_limit"=>cfg.residual_l2_limit,
        "mass_net_limit"=>cfg.mass_net_limit,
        "mass_net_scope"=>"signed sum of final-state raw continuity divergence divided by freestream rho*U*chord; both history windows use max(abs(.))",
        "residual_ratio_limit"=>cfg.residual_ratio_limit,
        "residual_acceptance"=>"median below margin times absolute limit and every recent sample below hard absolute limit; relative reduction and plateau ratios are diagnostic only",
        "plateau_fraction"=>cfg.plateau_fraction,
        "early_stop_eligible"=>m.last_status == "early_stop_eligible",
        "stop_on_stall"=>cfg.stop_on_stall,
        "stopped_without_convergence"=>m.last_status == "stalled_stop",
        "stall_window_iterations"=>cfg.stall_window_iterations,
        "stall_min_post_sfd"=>cfg.stall_min_post_sfd,
        "stall_confirmation_iterations"=>cfg.stall_confirmation_iterations,
        "stall_progress_fraction"=>cfg.stall_progress_fraction,
        "metrics"=>copy(m.last_metrics))
end

end # module CFDConvergence
