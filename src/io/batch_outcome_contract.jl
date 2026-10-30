module BatchOutcomeContract

using Dates

export CaseMetrics, CaseOutcome, BatchOutcome, BatchFailedError,
       lbm_case_outcome, pond_case_outcome, failed_case_outcome,
       require_case_outcome, require_batch_outcome, build_batch_outcome, batch_passed,
       case_completed, finalize_batch!

@enum CaseState CASE_COMPLETED CASE_FAILED
@enum BatchState BATCH_PASS BATCH_FAIL

const MaybeFloat = Union{Nothing,Float64}

"""Validated, solver-neutral data rendered by the batch summary."""
struct CaseMetrics
    solver_kind::Symbol
    re_number::MaybeFloat
    u_physical::MaybeFloat
    alpha_deg::MaybeFloat
    beta_deg::MaybeFloat
    total_cells::Int
    wall_time::Float64
    mlups::MaybeFloat
    steps::Int
    start_time::DateTime
    end_time::DateTime
    Cd::MaybeFloat
    Cl::MaybeFloat
    Cs::MaybeFloat
    Cdp::MaybeFloat
    Cdv::MaybeFloat
    Cmx::MaybeFloat
    Cmy::MaybeFloat
    Cmz::MaybeFloat
    avg_Cd::MaybeFloat
    avg_Cl::MaybeFloat
    avg_Cs::MaybeFloat
    avg_Cdp::MaybeFloat
    avg_Cdv::MaybeFloat
    avg_Cmx::MaybeFloat
    avg_Cmy::MaybeFloat
    avg_Cmz::MaybeFloat
    n_samples::Int
    min_Cd::MaybeFloat
    max_Cd::MaybeFloat
    min_Cl::MaybeFloat
    max_Cl::MaybeFloat
    min_Cs::MaybeFloat
    max_Cs::MaybeFloat
    force_divergence_flag::Bool
    force_peak_Cd::MaybeFloat
    force_peak_Cl::MaybeFloat
    force_peak_Cs::MaybeFloat
    force_ratio_Cd::MaybeFloat
    force_ratio_Cl::MaybeFloat
    force_ratio_Cs::MaybeFloat
    collision_operator::Symbol
    ma_physical::MaybeFloat
    compressibility_correction::Bool
    output_dir::String
end

"""The only value accepted from a case runner by the batch loop."""
struct CaseOutcome
    case_name::String
    state::CaseState
    error_msg::String
    metrics::CaseMetrics
end

function Base.getproperty(outcome::CaseOutcome, name::Symbol)
    if name in (:case_name, :state, :error_msg, :metrics)
        return getfield(outcome, name)
    elseif name === :status
        return getfield(outcome, :state) == CASE_COMPLETED ? :completed : :failed
    elseif name in fieldnames(CaseMetrics)
        return getfield(getfield(outcome, :metrics), name)
    end
    throw(ArgumentError("CaseOutcome has no property $(repr(name))"))
end

Base.propertynames(::CaseOutcome, private::Bool=false) =
    private ? (:case_name, :state, :status, :error_msg, :metrics, fieldnames(CaseMetrics)...) :
              (:case_name, :status, :error_msg, fieldnames(CaseMetrics)...)

"""Immutable, validated outcome of the complete selected case queue."""
struct BatchOutcome
    cases::Tuple
    start_time::DateTime
    end_time::DateTime
    state::BatchState
end

struct BatchFailedError <: Exception
    failed_cases::Tuple
end

function Base.showerror(io::IO, err::BatchFailedError)
    print(io, "batch failed for ", length(err.failed_cases), " case(s): ",
          join(err.failed_cases, ", "))
end

_reject_bool_number(value, label) =
    value isa Bool ? throw(ArgumentError("$label must be numeric, not Bool")) : value

function _finite_float(value, label::AbstractString)
    _reject_bool_number(value, label)
    value isa Real || throw(ArgumentError("$label must be a real number"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$label must be finite"))
    return result
end

function _maybe_float(value, label::AbstractString)
    value === nothing && return nothing
    return _finite_float(value, label)
end

function _exact_int(value, label::AbstractString)
    value isa Bool && throw(ArgumentError("$label must be an integer, not Bool"))
    value isa Integer || throw(ArgumentError("$label must be an integer"))
    return Int(value)
end

function _validate_metrics(metrics::CaseMetrics; completed::Bool)
    metrics.solver_kind in (:lbm, :pond_fv, :unknown) ||
        throw(ArgumentError("unsupported solver_kind $(repr(metrics.solver_kind))"))
    for field in fieldnames(CaseMetrics)
        value = getfield(metrics, field)
        value isa Float64 && isfinite(value) || !(value isa Float64) ||
            throw(ArgumentError("CaseMetrics.$field must be finite"))
    end
    metrics.total_cells >= 0 || throw(ArgumentError("total_cells must be nonnegative"))
    metrics.wall_time >= 0 || throw(ArgumentError("wall_time must be nonnegative"))
    metrics.steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    metrics.n_samples >= 0 || throw(ArgumentError("n_samples must be nonnegative"))
    metrics.end_time >= metrics.start_time ||
        throw(ArgumentError("end_time must not precede start_time"))
    completed && isempty(metrics.output_dir) &&
        throw(ArgumentError("completed outcome must name its output directory"))
    completed && metrics.solver_kind == :unknown &&
        throw(ArgumentError("completed outcome must identify its solver kind"))
    metrics.solver_kind == :pond_fv && metrics.collision_operator != :pond_fv &&
        throw(ArgumentError("PonD outcome must use collision_operator=:pond_fv"))
    return metrics
end

function _validate_outcome(outcome::CaseOutcome)
    isempty(outcome.case_name) && throw(ArgumentError("case_name must not be empty"))
    any(iscntrl, outcome.case_name) && throw(ArgumentError("case_name contains control characters"))
    completed = outcome.state == CASE_COMPLETED
    completed == isempty(outcome.error_msg) ||
        throw(ArgumentError(completed ?
            "completed outcome must not carry an error" :
            "failed outcome must carry a nonempty error"))
    _validate_metrics(outcome.metrics; completed=completed)
    return outcome
end

case_completed(outcome::CaseOutcome) = outcome.state == CASE_COMPLETED
batch_passed(batch::BatchOutcome) = batch.state == BATCH_PASS

function require_batch_outcome(batch::BatchOutcome)
    batch.end_time >= batch.start_time ||
        throw(ArgumentError("batch end_time must not precede start_time"))
    isempty(batch.cases) && throw(ArgumentError("batch must contain at least one case outcome"))
    foreach(_validate_outcome, batch.cases)
    names = Tuple(case.case_name for case in batch.cases)
    length(unique(names)) == length(names) ||
        throw(ArgumentError("batch contains duplicate case names"))
    derived = all(case_completed, batch.cases) ? BATCH_PASS : BATCH_FAIL
    batch.state == derived ||
        throw(ArgumentError("batch state disagrees with its case outcomes"))
    return batch
end

const LBM_FIELDS = (
    :case_name, :status, :error_msg, :re_number, :u_physical, :alpha_deg, :beta_deg,
    :total_cells, :wall_time, :mlups, :steps, :start_time, :end_time,
    :Cd, :Cl, :Cs, :Cdp, :Cdv, :Cmx, :Cmy, :Cmz,
    :avg_Cd, :avg_Cl, :avg_Cs, :avg_Cdp, :avg_Cdv, :avg_Cmx, :avg_Cmy, :avg_Cmz,
    :n_samples, :min_Cd, :max_Cd, :min_Cl, :max_Cl, :min_Cs, :max_Cs,
    :force_divergence_flag, :force_peak_Cd, :force_peak_Cl, :force_peak_Cs,
    :force_ratio_Cd, :force_ratio_Cl, :force_ratio_Cs, :collision_operator,
    :ma_physical, :compressibility_correction, :output_dir,
)

function _require_exact_fields(raw::NamedTuple, required::Tuple, label::AbstractString)
    actual = propertynames(raw)
    length(actual) == length(required) && Set(actual) == Set(required) ||
        throw(ArgumentError("$label schema mismatch; expected exactly $(join(required, ", "))"))
end

"""Normalize the exact LBM result schema into the shared case outcome type."""
function lbm_case_outcome(raw)
    raw isa NamedTuple || throw(ArgumentError("LBM result must be a NamedTuple"))
    _require_exact_fields(raw, LBM_FIELDS, "LBM result")
    raw.case_name isa AbstractString || throw(ArgumentError("LBM case_name must be a string"))
    raw.status === :completed || throw(ArgumentError("LBM result status must be :completed"))
    raw.error_msg isa AbstractString || throw(ArgumentError("LBM error_msg must be a string"))
    isempty(raw.error_msg) || throw(ArgumentError("completed LBM result must not carry an error"))
    raw.start_time isa DateTime || throw(ArgumentError("LBM start_time must be DateTime"))
    raw.end_time isa DateTime || throw(ArgumentError("LBM end_time must be DateTime"))
    raw.collision_operator isa Symbol || throw(ArgumentError("LBM collision_operator must be Symbol"))
    raw.collision_operator === :pond_fv &&
        throw(ArgumentError("LBM result must not claim the reserved :pond_fv operator"))
    raw.force_divergence_flag isa Bool || throw(ArgumentError("LBM force_divergence_flag must be Bool"))
    raw.compressibility_correction isa Bool ||
        throw(ArgumentError("LBM compressibility_correction must be Bool"))
    raw.output_dir isa AbstractString || throw(ArgumentError("LBM output_dir must be a string"))

    f(name) = _finite_float(getproperty(raw, name), "LBM.$name")
    i(name) = _exact_int(getproperty(raw, name), "LBM.$name")
    metrics = CaseMetrics(
        :lbm, f(:re_number), f(:u_physical), f(:alpha_deg), f(:beta_deg),
        i(:total_cells), f(:wall_time), f(:mlups), i(:steps), raw.start_time, raw.end_time,
        f(:Cd), f(:Cl), f(:Cs), f(:Cdp), f(:Cdv), f(:Cmx), f(:Cmy), f(:Cmz),
        f(:avg_Cd), f(:avg_Cl), f(:avg_Cs), f(:avg_Cdp), f(:avg_Cdv),
        f(:avg_Cmx), f(:avg_Cmy), f(:avg_Cmz), i(:n_samples),
        f(:min_Cd), f(:max_Cd), f(:min_Cl), f(:max_Cl), f(:min_Cs), f(:max_Cs),
        raw.force_divergence_flag, f(:force_peak_Cd), f(:force_peak_Cl),
        f(:force_peak_Cs), f(:force_ratio_Cd), f(:force_ratio_Cl), f(:force_ratio_Cs),
        raw.collision_operator, f(:ma_physical), raw.compressibility_correction,
        String(raw.output_dir),
    )
    return _validate_outcome(CaseOutcome(String(raw.case_name), CASE_COMPLETED, "", metrics))
end

function _pond_metrics(; re_number, ma_physical, total_cells, wall_time, steps,
                       start_time, end_time, output_dir, cd_cv=nothing, n_samples=0)
    start_time isa DateTime || throw(ArgumentError("PonD start_time must be DateTime"))
    end_time isa DateTime || throw(ArgumentError("PonD end_time must be DateTime"))
    cells = _exact_int(total_cells, "PonD.total_cells")
    elapsed = _finite_float(wall_time, "PonD.wall_time")
    completed_steps = _exact_int(steps, "PonD.steps")
    samples = _exact_int(n_samples, "PonD.n_samples")
    cd = _maybe_float(cd_cv, "PonD.Cd_cv")
    samples > 0 && cd === nothing &&
        throw(ArgumentError("PonD n_samples > 0 requires a finite Cd_cv"))
    mlups = elapsed > 0 ? cells * completed_steps / (elapsed * 1.0e6) : 0.0
    return CaseMetrics(
        :pond_fv, _finite_float(re_number, "PonD.re_number"), nothing, nothing, nothing,
        cells, elapsed, mlups, completed_steps, start_time, end_time,
        cd, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        samples > 0 ? cd : nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        samples, nothing, nothing, nothing, nothing, nothing, nothing,
        false, nothing, nothing, nothing, nothing, nothing, nothing,
        :pond_fv, _finite_float(ma_physical, "PonD.ma_physical"), false,
        String(output_dir),
    )
end

"""The sole explicit Bool adapter; raw Bool is rejected everywhere else."""
function pond_case_outcome(success::Bool; case_name, error_msg="", kwargs...)
    case_name isa AbstractString || throw(ArgumentError("PonD case_name must be a string"))
    error_msg isa AbstractString || throw(ArgumentError("PonD error_msg must be a string"))
    metrics = _pond_metrics(; kwargs...)
    if success
        isempty(error_msg) || throw(ArgumentError("successful PonD outcome must not carry an error"))
        return _validate_outcome(CaseOutcome(String(case_name), CASE_COMPLETED, "", metrics))
    end
    isempty(error_msg) && throw(ArgumentError("failed PonD outcome must carry an error"))
    return _validate_outcome(CaseOutcome(String(case_name), CASE_FAILED, String(error_msg), metrics))
end

function _failed_metrics(start_time::DateTime, end_time::DateTime, output_dir::AbstractString)
    return CaseMetrics(
        :unknown,             # solver_kind
        nothing, nothing, nothing, nothing,
        0, 0.0, nothing, 0,
        start_time, end_time,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        0,
        nothing, nothing, nothing, nothing, nothing, nothing,
        false,
        nothing, nothing, nothing, nothing, nothing, nothing,
        :unknown, nothing, false, String(output_dir),
    )
end

"""Convert a caught per-case exception into an explicit shared failed outcome."""
function failed_case_outcome(case_name::AbstractString, error_value;
                             start_time::DateTime=Dates.now(),
                             end_time::DateTime=Dates.now(), output_dir::AbstractString="")
    message = sprint(showerror, error_value)
    isempty(message) && (message = string(typeof(error_value)))
    return _validate_outcome(CaseOutcome(String(case_name), CASE_FAILED, message,
        _failed_metrics(start_time, max(start_time, end_time), output_dir)))
end

function require_case_outcome(raw, expected_case::AbstractString)
    raw isa CaseOutcome ||
        throw(ArgumentError("case runner returned $(typeof(raw)); expected CaseOutcome"))
    outcome = _validate_outcome(raw)
    outcome.case_name == expected_case ||
        throw(ArgumentError("case outcome name $(repr(outcome.case_name)) does not match $(repr(expected_case))"))
    return outcome
end

function build_batch_outcome(raw_cases, start_time::DateTime, end_time::DateTime;
                             expected_count::Integer=length(raw_cases))
    expected_count isa Bool && throw(ArgumentError("expected_count must not be Bool"))
    expected_count >= 0 || throw(ArgumentError("expected_count must be nonnegative"))
    expected_count > 0 || throw(ArgumentError("batch must expect at least one case outcome"))
    end_time >= start_time || throw(ArgumentError("batch end_time must not precede start_time"))
    length(raw_cases) == expected_count ||
        throw(ArgumentError("batch contains $(length(raw_cases)) outcomes; expected $expected_count"))
    cases = Tuple(_validate_outcome(case) for case in raw_cases)
    length(unique(case.case_name for case in cases)) == length(cases) ||
        throw(ArgumentError("batch contains duplicate case names"))
    batch = all(case_completed, cases) ?
        BatchOutcome(cases, start_time, end_time, BATCH_PASS) :
        BatchOutcome(cases, start_time, end_time, BATCH_FAIL)
    return require_batch_outcome(batch)
end

"""Publish first; only then announce and propagate a failed batch."""
function finalize_batch!(publish::Function, batch::BatchOutcome; announce::Function=println)
    require_batch_outcome(batch)
    published = publish(batch)  # publication errors are deliberately fatal
    if batch_passed(batch)
        announce("BATCH FINISHED: PASS")
        return published
    end
    announce("BATCH FINISHED: FAIL")
    failed = Tuple(case.case_name for case in batch.cases if !case_completed(case))
    throw(BatchFailedError(failed))
end

end # module
