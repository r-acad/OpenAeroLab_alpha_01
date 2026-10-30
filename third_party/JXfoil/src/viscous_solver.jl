using LinearAlgebra
using ForwardDiff

# Module-level flag: when true, _eval_interval uses _bldif_analytic (Float64)
# instead of bldif_residuals, matching SETBL's equation/clamping path.
# Set to true during VISCAL Newton remarching for march/SETBL consistency.
const _USE_BLDIF_ANALYTIC = Ref(false)

struct SurfaceStations
    side::Symbol
    x::Vector{Float64}
    xc::Vector{Float64}
    ue::Vector{Float64}
end

struct BLStagnationPoint
    ist::Int
    sst::Float64
    sst_go::Float64
    sst_gp::Float64
end

struct BLSideTopology
    side::Symbol
    ipan::Vector{Int}
    vti::Vector{Float64}
    xssi::Vector{Float64}
end

struct BLTopology
    s::Vector{Float64}
    stagnation::BLStagnationPoint
    upper::BLSideTopology
    lower::BLSideTopology
end

struct BLSystemTopology
    upper::Vector{Int}
    lower::Vector{Int}
    nsys::Int
end

struct BLIntervalSystem
    mode::Symbol
    residual::Vector{Float64}
    vs1::Matrix{Float64}
    vs2::Matrix{Float64}
    vsrez::Vector{Float64}
    vsm::Vector{Float64}
    vsr::Vector{Float64}
    vsx::Vector{Float64}
    p1::BLPrimaryState
    p2::BLPrimaryState
    s1::BLSecondaryState
    s2::BLSecondaryState
    transition_x::Float64
    transition_ok::Bool
end

struct SurfaceMarchResult
    side::Symbol
    x::Vector{Float64}
    xc::Vector{Float64}
    ue::Vector{Float64}
    mode::Vector{Symbol}
    primary::Vector{BLPrimaryState}
    secondary::Vector{BLSecondaryState}
    transition_x::Float64
    solved_similarity_station::Bool
    converged_intervals::BitVector
end

struct WakeStations
    x::Vector{Float64}
    ue::Vector{Float64}
    dw::Vector{Float64}
    xw::Vector{Float64}
    yw::Vector{Float64}
    apanel::Vector{Float64}
end

struct WakeMarchResult
    x::Vector{Float64}
    ue::Vector{Float64}
    dw::Vector{Float64}
    xw::Vector{Float64}
    yw::Vector{Float64}
    apanel::Vector{Float64}
    primary::Vector{BLPrimaryState}
    secondary::Vector{BLSecondaryState}
    converged_intervals::BitVector
end

"""
Metadata describing which pure-Julia solver backend produced a viscous result
and whether the returned state is the final converged state or an explicit
fallback selection.

`selected_iteration` / `selected_residual` may be `0` / `NaN` when a backend
does not expose a comparable global residual.
"""
struct SolverProvenance
    backend::Symbol
    selection::Symbol
    converged::Bool
    status::Symbol
    failure_message::String
    iterations::Int
    selected_iteration::Int
    selected_residual::Float64
end

struct CoupledIterationSample
    global_iteration::Int
    segment::Symbol
    segment_iteration::Int
    residual::Float64
    cl::Float64
    cd::Float64
    cm_c4::Float64
    top_xtr::Float64
    bot_xtr::Float64
    upper_transition_index::Int
    lower_transition_index::Int
    mrchdu_active::Bool
    transition_recheck_active::Bool
end

struct ViscousCaseResult
    alpha_deg::Float64
    reynolds::Float64
    ncrit::Float64
    cl::Float64
    cd::Float64
    cm_c4::Float64
    top_xtr::Float64
    bot_xtr::Float64
    upper::SurfaceMarchResult
    lower::SurfaceMarchResult
    inviscid::CaseResult
    provenance::SolverProvenance
end

struct CoupledCaseResult
    alpha_deg::Float64
    reynolds::Float64
    ncrit::Float64
    cl::Float64
    cd::Float64
    cm_c4::Float64
    top_xtr::Float64
    bot_xtr::Float64
    coupling_iterations::Int
    coupling_residual::Float64
    upper::SurfaceMarchResult
    lower::SurfaceMarchResult
    inviscid::CaseResult
    provenance::SolverProvenance
end

struct CoupledSolveTrace
    airfoil_name::String
    alpha_deg::Float64
    reynolds::Float64
    ncrit::Float64
    result::CoupledCaseResult
    samples::Vector{CoupledIterationSample}
end

@inline function _solver_provenance(;
    backend::Symbol,
    selection::Symbol = :final_state,
    converged::Bool,
    status::Symbol,
    failure_message::AbstractString = "",
    iterations::Integer = 0,
    selected_iteration::Integer = 0,
    selected_residual::Real = NaN,
)
    return SolverProvenance(
        backend,
        selection,
        converged,
        status,
        String(failure_message),
        Int(iterations),
        Int(selected_iteration),
        float(selected_residual),
    )
end

function Base.show(io::IO, provenance::SolverProvenance)
    print(
        io,
        "SolverProvenance(",
        "backend=", provenance.backend,
        ", selection=", provenance.selection,
        ", status=", provenance.status,
        ", converged=", provenance.converged,
        ", iterations=", provenance.iterations,
    )
    if provenance.selected_iteration > 0
        print(io, ", selected_iteration=", provenance.selected_iteration)
    end
    if isfinite(provenance.selected_residual)
        print(io, ", selected_residual=", provenance.selected_residual)
    end
    if !isempty(provenance.failure_message)
        print(io, ", failure_message=", repr(provenance.failure_message))
    end
    print(io, ")")
end

@inline function _as_viscous_case_result(res::CoupledCaseResult)
    return ViscousCaseResult(
        res.alpha_deg,
        res.reynolds,
        res.ncrit,
        res.cl,
        res.cd,
        res.cm_c4,
        res.top_xtr,
        res.bot_xtr,
        res.upper,
        res.lower,
        res.inviscid,
        res.provenance,
    )
end

@inline function _with_provenance(res::CoupledCaseResult, provenance::SolverProvenance)
    return CoupledCaseResult(
        res.alpha_deg,
        res.reynolds,
        res.ncrit,
        res.cl,
        res.cd,
        res.cm_c4,
        res.top_xtr,
        res.bot_xtr,
        res.coupling_iterations,
        res.coupling_residual,
        res.upper,
        res.lower,
        res.inviscid,
        provenance,
    )
end

@inline function _public_output_sanity_score(
    cl::Real,
    cd::Real,
    cm_c4::Real,
    top_xtr::Real,
    bot_xtr::Real,
    ;
    alpha_deg::Real = NaN,
    inviscid_cl::Real = NaN,
    abs_cl_soft_limit::Real = 2.25,
    abs_cl_hard_limit::Real = 3.00,
    cd_soft_limit::Real = 0.20,
    cd_hard_limit::Real = 0.60,
    abs_cm_soft_limit::Real = 0.80,
    abs_cm_hard_limit::Real = 1.50,
    cl_excess_soft_limit::Real = 0.90,
    cl_excess_hard_limit::Real = 1.80,
)
    score = 0.0
    clf = float(cl)
    cdf = float(cd)
    cmf = float(cm_c4)
    alphaf = float(alpha_deg)
    invclf = float(inviscid_cl)

    if !isfinite(clf)
        score += 1.0e6
    else
        abs_cl = abs(clf)
        if abs_cl > float(abs_cl_soft_limit)
            score += 2.0 * (abs_cl - float(abs_cl_soft_limit))
        end
        if abs_cl > float(abs_cl_hard_limit)
            score += 20.0 * (abs_cl - float(abs_cl_hard_limit))
        end

        if isfinite(alphaf) && abs(alphaf) <= 12.0 && abs_cl > 2.0
            score += 2.0 * (abs_cl - 2.0)
        end

        if isfinite(invclf)
            cl_excess = abs_cl - abs(invclf)
            if cl_excess > float(cl_excess_soft_limit)
                score += 3.0 * (cl_excess - float(cl_excess_soft_limit))
            end
            if cl_excess > float(cl_excess_hard_limit)
                score += 15.0 * (cl_excess - float(cl_excess_hard_limit))
            end
        end
    end

    if !isfinite(cdf)
        score += 1.0e6
    elseif cdf < 0.0
        score += abs(cdf)
    else
        if cdf > float(cd_soft_limit)
            score += 5.0 * (cdf - float(cd_soft_limit))
        end
        if cdf > float(cd_hard_limit)
            score += 25.0 * (cdf - float(cd_hard_limit))
        end
    end

    if !isfinite(cmf)
        score += 1.0e6
    else
        abs_cm = abs(cmf)
        if abs_cm > float(abs_cm_soft_limit)
            score += 2.0 * (abs_cm - float(abs_cm_soft_limit))
        end
        if abs_cm > float(abs_cm_hard_limit)
            score += 10.0 * (abs_cm - float(abs_cm_hard_limit))
        end
    end

    for xtr in (top_xtr, bot_xtr)
        if !isfinite(float(xtr))
            score += 100.0
        elseif float(xtr) < 0.0
            score += abs(float(xtr))
        elseif float(xtr) > 1.0
            score += float(xtr) - 1.0
        end
    end

    return score
end

@inline function _coupled_result_sanity_score(res::CoupledCaseResult)
    return _public_output_sanity_score(
        res.cl,
        res.cd,
        res.cm_c4,
        res.top_xtr,
        res.bot_xtr,
        alpha_deg = res.alpha_deg,
        inviscid_cl = res.inviscid.cl,
    )
end

@inline function _continuation_seed_surface(
    seed::SurfaceMarchResult,
    fallback::SurfaceMarchResult;
    flow::Union{Nothing,BLFlowParams} = nothing,
    model::Union{Nothing,BLModelParams} = nothing,
    min_ratio::Real = 0.60,
    max_ratio::Real = 1.40,
)
    seed.side == fallback.side || return fallback
    return _continuation_seed_surface(
        seed,
        SurfaceStations(fallback.side, fallback.x, fallback.xc, fallback.ue);
        flow = flow,
        model = model,
        min_ratio = min_ratio,
        max_ratio = max_ratio,
    )
end

@inline function _continuation_seed_surface(
    seed::SurfaceMarchResult,
    inv_surface::SurfaceStations;
    flow::Union{Nothing,BLFlowParams} = nothing,
    model::Union{Nothing,BLModelParams} = nothing,
    min_ratio::Real = 0.60,
    max_ratio::Real = 1.40,
)
    seed.side == inv_surface.side || return seed

    seeded_ue = _continuation_seed_ue(seed, inv_surface; min_ratio = min_ratio, max_ratio = max_ratio)
    fallback_flow = isnothing(flow) ? _xfoil_bl_flow_params(1.0e5; model = BLModelParams()) : flow
    fallback_model = isnothing(model) ? BLModelParams() : model
    target_n = length(inv_surface.x)

    if length(seed.primary) == target_n &&
       length(seed.secondary) == target_n &&
       length(seed.mode) == target_n &&
       length(seed.converged_intervals) == max(target_n - 1, 0)
        return _stmove_surface_remap(
            seed,
            SurfaceStations(inv_surface.side, inv_surface.x, inv_surface.xc, seeded_ue),
            0;
            flow = fallback_flow,
            model = fallback_model,
        )
    end

    n = length(inv_surface.x)
    n >= 1 || return seed

    mode = Vector{Symbol}(undef, n)
    primary = Vector{BLPrimaryState}(undef, n)
    secondary = Vector{BLSecondaryState}(undef, n)
    conv = falses(max(n - 1, 0))
    transition_x = _remap_seed_transition_x(seed, inv_surface)

    for i in 1:n
        src = _nearest_monotone_index(seed.xc, inv_surface.xc[i])
        src_mode = _surface_seed_remap_mode(seed.mode[src])
        mode[i] = src_mode
        primary[i] = _surface_primary_with_station(
            seed.primary[src],
            src_mode,
            inv_surface.x[i],
            seeded_ue[i],
        )
        secondary[i] = blsecondary_state(
            primary[i],
            _mode_closure(src_mode);
            flow = fallback_flow,
            model = fallback_model,
        )
    end

    if !isempty(conv) && !isempty(seed.converged_intervals)
        src_n = min(length(seed.converged_intervals), max(length(seed.xc) - 1, 0))
        src_n >= 1 || return SurfaceMarchResult(
            inv_surface.side,
            copy(inv_surface.x),
            copy(inv_surface.xc),
            seeded_ue,
            mode,
            primary,
            secondary,
            transition_x,
            seed.solved_similarity_station,
            conv,
        )
        src_mid = [
            0.5 * (float(seed.xc[i]) + float(seed.xc[i + 1]))
            for i in 1:src_n
        ]
        for i in eachindex(conv)
            target_mid = 0.5 * (float(inv_surface.xc[i]) + float(inv_surface.xc[i + 1]))
            src = _nearest_monotone_index(src_mid, target_mid)
            conv[i] = seed.converged_intervals[src]
        end
    end

    return SurfaceMarchResult(
        inv_surface.side,
        copy(inv_surface.x),
        copy(inv_surface.xc),
        seeded_ue,
        mode,
        primary,
        secondary,
        transition_x,
        seed.solved_similarity_station,
        conv,
    )
end

@inline function _surface_seed_remap_mode(mode::Symbol)
    return mode == :transition ? :turbulent : mode
end

@inline function _nearest_monotone_index(values::AbstractVector{<:Real}, target::Real)
    n = length(values)
    n >= 1 || throw(ArgumentError("values must contain at least 1 element"))
    targetf = float(target)
    if n == 1
        return 1
    elseif targetf <= float(values[1])
        return 1
    elseif targetf >= float(values[end])
        return n
    end

    j = clamp(searchsortedlast(values, targetf), 1, n - 1)
    left = abs(targetf - float(values[j]))
    right = abs(float(values[j + 1]) - targetf)
    return left <= right ? j : (j + 1)
end

@inline function _remap_seed_transition_x(
    seed::SurfaceMarchResult,
    inv_surface::SurfaceStations,
)
    xtr = float(seed.transition_x)
    isfinite(xtr) || return NaN

    transition_xc = _surface_transition_xc(seed)
    if isfinite(transition_xc)
        return _interp_clamped(inv_surface.xc, inv_surface.x, transition_xc)
    end

    x0 = float(seed.x[1])
    x1 = float(seed.x[end])
    if x1 > x0
        ratio = clamp((xtr - x0) / (x1 - x0), 0.0, 1.0)
        return float(inv_surface.x[1]) + ratio * (float(inv_surface.x[end]) - float(inv_surface.x[1]))
    end

    return clamp(xtr, float(inv_surface.x[1]), float(inv_surface.x[end]))
end

@inline function _viscal_candidate_sanity_score(
    candidate,
    inv_cl::Real,
    alpha_deg::Real,
)
    return _public_output_sanity_score(
        candidate.cl,
        candidate.cd,
        candidate.cm,
        candidate.top_xtr,
        candidate.bot_xtr,
        alpha_deg = alpha_deg,
        inviscid_cl = inv_cl,
    )
end

@inline function _normalized_candidate_mismatch(
    lhs::Real,
    rhs::Real;
    floor::Real = 1.0e-3,
    invalid_penalty::Real = 100.0,
)
    lhsf = float(lhs)
    rhsf = float(rhs)
    isfinite(lhsf) && isfinite(rhsf) || return float(invalid_penalty)
    scale = max(abs(lhsf), abs(rhsf), float(floor))
    return abs(lhsf - rhsf) / scale
end

@inline function _viscal_candidate_closure_score(
    candidate;
    te_gap_ante::Real = NaN,
)
    hasproperty(candidate, :upper) || return 0.0
    hasproperty(candidate, :lower) || return 0.0
    hasproperty(candidate, :wake) || return 0.0

    upper = candidate.upper
    lower = candidate.lower
    wake = candidate.wake
    length(upper.primary) >= 1 || return 100.0
    length(lower.primary) >= 1 || return 100.0
    length(wake.primary) >= 1 || return 100.0

    p_u = upper.primary[end]
    p_l = lower.primary[end]
    p_w = wake.primary[1]
    ct_u = _surface_transition_aware_ctau(upper, length(upper.primary))
    ct_l = _surface_transition_aware_ctau(lower, length(lower.primary))
    tte = float(p_u.t + p_l.t)
    cte = (float(ct_u) * float(p_u.t) + float(ct_l) * float(p_l.t)) / max(tte, 1.0e-12)

    score = max(
        _normalized_candidate_mismatch(cte, p_w.s),
        _normalized_candidate_mismatch(tte, p_w.t),
    )

    if isfinite(float(te_gap_ante))
        dte = float(p_u.d + p_l.d + te_gap_ante)
        score = max(score, _normalized_candidate_mismatch(dte, p_w.d))
    end

    return score
end

@inline function _viscal_candidate_acceptable(
    candidate,
    inv_cl::Real,
    alpha_deg::Real;
    max_sanity_score::Real = 0.75,
    max_residual::Real = 5.0e-4,
)
    candidate.converged || return false
    residual = float(candidate.residual)
    isfinite(residual) || return false
    residual <= float(max_residual) || return false
    return _viscal_candidate_sanity_score(candidate, inv_cl, alpha_deg) <= float(max_sanity_score)
end

@inline function _viscal_low_residual_branch_protected(
    candidate,
    inv_cl::Real,
    alpha_deg::Real,
    viscal_tol::Real;
    max_residual::Real = 7.5e-5,
    max_sanity_score::Real = 0.25,
    max_cd::Real = 0.18,
    near_zero_lift::Real = 0.10,
    near_zero_lift_min_cd::Real = 0.13,
    high_lift_late_lower_xtr::Real = 0.95,
    high_lift_limit::Real = 0.50,
)
    _viscal_candidate_acceptable(candidate, inv_cl, alpha_deg) || return false
    residual = float(candidate.residual)
    isfinite(residual) || return false
    residual <= min(float(viscal_tol), float(max_residual)) || return false
    _viscal_candidate_sanity_score(candidate, inv_cl, alpha_deg) <= float(max_sanity_score) || return false

    # This path overrides a failed replay audit, so keep it narrow. The extra
    # gates reject low-residual but wrong high-lift/lower-laminar or
    # low-drag near-zero-lift branches seen in low-Re UIUC edge cases.
    clf = float(candidate.cl)
    cdf = float(candidate.cd)
    bot = float(candidate.bot_xtr)
    isfinite(clf) && isfinite(cdf) && isfinite(bot) || return false
    cdf <= float(max_cd) || return false
    abs(clf) <= float(near_zero_lift) && cdf < float(near_zero_lift_min_cd) && return false
    clf > float(high_lift_limit) && bot >= float(high_lift_late_lower_xtr) && return false
    return true
end

@inline function _viscal_candidate_plateau_converged(
    candidate,
    inv_cl::Real,
    alpha_deg::Real;
    max_sanity_score::Real = 0.10,
    max_residual::Real = 5.0e-3,
)
    candidate.converged && return false
    residual = float(candidate.residual)
    isfinite(residual) || return false
    residual <= float(max_residual) || return false
    cdf = float(candidate.cd)
    isfinite(cdf) || return false
    cdf >= 0.0 || return false
    return _viscal_candidate_sanity_score(candidate, inv_cl, alpha_deg) <= float(max_sanity_score)
end

@inline function _viscal_separated_fallback_score(
    candidate;
    alpha_deg::Real = NaN,
    branch_kind::Symbol = :generic,
    max_residual::Real = 0.25,
    min_cd::Real = 0.095,
    max_cd::Real = 0.22,
    max_abs_cl::Real = 0.95,
    min_top_xtr::Real = 0.45,
    max_top_xtr::Real = 0.90,
    min_bot_xtr::Real = 0.04,
    max_bot_xtr::Real = 0.55,
)
    residual = float(candidate.residual)
    cl = float(candidate.cl)
    cd = float(candidate.cd)
    top = float(candidate.top_xtr)
    bot = float(candidate.bot_xtr)
    isfinite(residual) && isfinite(cl) && isfinite(cd) && isfinite(top) && isfinite(bot) ||
        return Inf
    residual <= float(max_residual) || return Inf
    float(min_cd) <= cd <= float(max_cd) || return Inf
    abs(cl) <= float(max_abs_cl) || return Inf
    float(min_top_xtr) <= top <= float(max_top_xtr) || return Inf
    float(min_bot_xtr) <= bot <= float(max_bot_xtr) || return Inf

    alpha = float(alpha_deg)
    if isfinite(alpha) && abs(alpha) >= 1.0
        signed_cl = sign(alpha) * cl
        signed_cl >= -0.05 || return Inf
        lift_floor_penalty = 3.0 * max(0.12 - signed_cl, 0.0)
    else
        lift_floor_penalty = 0.0
    end

    if branch_kind === :near_zero_low_drag
        signed_cl = isfinite(alpha) && abs(alpha) >= 1.0 ? sign(alpha) * cl : abs(cl)
        residual_weight = 0.12
        cd_target = 0.118
        top_target = 0.733
        bot_target = 0.40
        lift_target_penalty = 1.0 * abs(signed_cl - 0.16)
        bot_weight = 0.55
    else
        residual_weight = 0.35
        cd_target = 0.14
        top_target = 0.72
        bot_target = 0.18
        lift_target_penalty = 0.50 * max(abs(cl) - 0.75, 0.0)
        bot_weight = 0.35
    end

    residual_term = clamp(log10(max(residual, 1.0e-12)) + 2.0, 0.0, 4.0)
    return residual_weight * residual_term +
           4.0 * abs(cd - cd_target) +
           lift_target_penalty +
           0.75 * abs(top - top_target) +
           bot_weight * abs(bot - bot_target) +
           lift_floor_penalty
end

@inline function _viscal_separated_fallback_preferred(
    candidate,
    incumbent;
    alpha_deg::Real = NaN,
    branch_kind::Symbol = :generic,
    max_score::Real = 1.0,
    allow_high_lift::Bool = true,
    allow_near_zero::Bool = true,
    high_lift_limit::Real = 0.50,
    high_lift_late_lower_xtr::Real = 0.90,
    min_high_lift_cl_drop::Real = 0.25,
    near_zero_lift::Real = 0.12,
    near_zero_lift_max_cd::Real = 0.13,
    min_near_zero_cd_gain::Real = 0.005,
    min_near_zero_cl_gain::Real = 0.05,
)
    score = _viscal_separated_fallback_score(
        candidate;
        alpha_deg = alpha_deg,
        branch_kind = branch_kind,
    )
    isfinite(score) && score <= float(max_score) || return false

    inc_cl = float(incumbent.cl)
    inc_cd = float(incumbent.cd)
    inc_bot = float(incumbent.bot_xtr)
    cand_cl = float(candidate.cl)
    cand_cd = float(candidate.cd)
    cand_bot = float(candidate.bot_xtr)
    isfinite(inc_cl) && isfinite(inc_cd) && isfinite(inc_bot) &&
        isfinite(cand_cl) && isfinite(cand_cd) && isfinite(cand_bot) || return false

    high_lift_laminar_branch =
        inc_cl > float(high_lift_limit) &&
        inc_bot >= float(high_lift_late_lower_xtr) &&
        cand_cl <= inc_cl - float(min_high_lift_cl_drop) &&
        cand_bot <= 0.55
    allow_high_lift && high_lift_laminar_branch && return true

    near_zero_lift_gain =
        inc_cl >= 0.0 ?
        cand_cl >= inc_cl + float(min_near_zero_cl_gain) :
        cand_cl <= inc_cl - float(min_near_zero_cl_gain)
    near_zero_low_drag_branch =
        abs(inc_cl) <= float(near_zero_lift) &&
        inc_cd < float(near_zero_lift_max_cd) &&
        cand_cd >= inc_cd + float(min_near_zero_cd_gain) &&
        near_zero_lift_gain
    return allow_near_zero && near_zero_low_drag_branch
end

@inline function _viscal_separated_fallback_probe_kind(
    incumbent;
    high_lift_limit::Real = 0.50,
    high_lift_late_lower_xtr::Real = 0.90,
    near_zero_lift::Real = 0.12,
    near_zero_lift_max_cd::Real = 0.13,
)
    inc_cl = float(incumbent.cl)
    inc_cd = float(incumbent.cd)
    inc_bot = float(incumbent.bot_xtr)
    isfinite(inc_cl) && isfinite(inc_cd) && isfinite(inc_bot) || return false
    high_lift_laminar_branch =
        inc_cl > float(high_lift_limit) &&
        inc_bot >= float(high_lift_late_lower_xtr)
    high_lift_laminar_branch && return :high_lift_late_lower_xtr
    near_zero_low_drag_branch =
        abs(inc_cl) <= float(near_zero_lift) &&
        inc_cd < float(near_zero_lift_max_cd)
    near_zero_low_drag_branch && return :near_zero_low_drag
    return :none
end

@inline function _viscal_separated_fallback_probe_needed(incumbent)
    return _viscal_separated_fallback_probe_kind(incumbent) !== :none
end

@inline function _result_cm_value(result)
    if hasproperty(result, :cm_c4)
        return float(getproperty(result, :cm_c4))
    elseif hasproperty(result, :cm)
        return float(getproperty(result, :cm))
    end
    return NaN
end

@inline function _result_selected_residual(result)
    if hasproperty(result, :provenance)
        provenance = getproperty(result, :provenance)
        if hasproperty(provenance, :selected_residual)
            return float(getproperty(provenance, :selected_residual))
        end
    end
    if hasproperty(result, :selected_residual)
        return float(getproperty(result, :selected_residual))
    elseif hasproperty(result, :residual)
        return float(getproperty(result, :residual))
    end
    return NaN
end

@inline function _viscal_candidate_from_result(result)
    provenance = getproperty(result, :provenance)
    return (
        upper = result.upper,
        lower = result.lower,
        cl = result.cl,
        cd = result.cd,
        cm = result.cm_c4,
        top_xtr = result.top_xtr,
        bot_xtr = result.bot_xtr,
        residual = _result_selected_residual(result),
        converged = provenance.converged,
        status = provenance.status,
        selected_iteration = provenance.selected_iteration,
        selection = provenance.selection,
    )
end

@inline function _result_output_mismatch(reference, candidate)
    dcl = abs(float(reference.cl) - float(candidate.cl))
    dcd = abs(float(reference.cd) - float(candidate.cd))
    dcm = abs(_result_cm_value(reference) - _result_cm_value(candidate))
    dxtr = max(
        abs(float(reference.top_xtr) - float(candidate.top_xtr)),
        abs(float(reference.bot_xtr) - float(candidate.bot_xtr)),
    )
    return (dcl = dcl, dcd = dcd, dcm = dcm, dxtr = dxtr)
end

@inline function _plateau_output_matches(
    reference,
    candidate;
    max_dcl::Real = 5.0e-3,
    max_dcd::Real = 5.0e-4,
    max_dcm::Real = 5.0e-3,
    max_dxtr::Real = 5.0e-3,
)
    mismatch = _result_output_mismatch(reference, candidate)
    return mismatch.dcl <= float(max_dcl) &&
           mismatch.dcd <= float(max_dcd) &&
           mismatch.dcm <= float(max_dcm) &&
           mismatch.dxtr <= float(max_dxtr)
end

@inline function _startup_family_output_matches(
    reference,
    candidate;
    max_dcl::Real = 1.0e-2,
    max_dcd::Real = 1.0e-3,
    max_dcm::Real = 1.0e-2,
    max_dxtr::Real = 2.0e-2,
)
    mismatch = _result_output_mismatch(reference, candidate)
    return mismatch.dcl <= float(max_dcl) &&
           mismatch.dcd <= float(max_dcd) &&
           mismatch.dcm <= float(max_dcm) &&
           mismatch.dxtr <= float(max_dxtr)
end

@inline function _plateau_verification_consistent(
    reference,
    verification;
    max_dcl::Real = 4.0e-2,
    max_dcd::Real = 5.0e-3,
    max_dcm::Real = 2.0e-2,
    max_dxtr::Real = 5.0e-2,
    max_residual::Real = 2.0e-2,
    residual_factor::Real = 8.0,
)
    _plateau_output_matches(
        reference,
        verification;
        max_dcl = max_dcl,
        max_dcd = max_dcd,
        max_dcm = max_dcm,
        max_dxtr = max_dxtr,
    ) || return false

    reference_residual = _result_selected_residual(reference)
    verification_residual = _result_selected_residual(verification)
    isfinite(verification_residual) || return false

    residual_limit = float(max_residual)
    if isfinite(reference_residual)
        residual_limit = max(residual_limit, float(residual_factor) * max(reference_residual, 1.0e-12))
    end
    return verification_residual <= residual_limit
end

@inline function _converged_verification_severe(
    reference,
    verification;
    max_dcl::Real = 1.0e-1,
    max_dcd::Real = 1.0e-2,
    max_dcm::Real = 5.0e-2,
    max_dxtr::Real = 1.0e-1,
)
    mismatch = _result_output_mismatch(reference, verification)
    return mismatch.dcl > float(max_dcl) ||
           mismatch.dcd > float(max_dcd) ||
           mismatch.dcm > float(max_dcm) ||
           mismatch.dxtr > float(max_dxtr)
end

@inline function _result_replay_candidate(result; force_converged::Bool = true)
    provenance = getproperty(result, :provenance)
    return (
        upper = result.upper,
        lower = result.lower,
        cl = result.cl,
        cd = result.cd,
        cm = result.cm_c4,
        top_xtr = result.top_xtr,
        bot_xtr = result.bot_xtr,
        residual = _result_selected_residual(result),
        converged = force_converged,
        status = force_converged ? :converged : provenance.status,
        selected_iteration = provenance.selected_iteration,
        selection = provenance.selection,
    )
end

@inline function _result_physically_acceptable(
    result,
    inv_cl::Real,
    alpha_deg::Real;
    max_sanity_score::Real = 0.75,
    max_residual::Real = 5.0e-4,
)
    candidate = _result_replay_candidate(result; force_converged = true)
    return _viscal_candidate_acceptable(
        candidate,
        inv_cl,
        alpha_deg;
        max_sanity_score = max_sanity_score,
        max_residual = max_residual,
    )
end

@inline function _result_mismatch_scale(
    mismatch;
    dcl_scale::Real = 1.0e-1,
    dcd_scale::Real = 1.0e-2,
    dcm_scale::Real = 5.0e-2,
    dxtr_scale::Real = 1.0e-1,
)
    return max(
        mismatch.dcl / float(dcl_scale),
        mismatch.dcd / float(dcd_scale),
        mismatch.dcm / float(dcm_scale),
        mismatch.dxtr / float(dxtr_scale),
    )
end

@inline function _converged_replay_contracts(
    reference,
    verification,
    verification2;
    contraction_factor::Real = 4.0,
    max_followup_dcl::Real = 5.0e-2,
    max_followup_dcd::Real = 1.0e-2,
    max_followup_dcm::Real = 2.0e-2,
    max_followup_dxtr::Real = 5.0e-2,
    max_followup_residual::Real = 5.0e-2,
    residual_factor::Real = 12.0,
)
    mismatch01 = _result_output_mismatch(reference, verification)
    mismatch12 = _result_output_mismatch(verification, verification2)
    mismatch12.dcl <= float(max_followup_dcl) || return false
    mismatch12.dcd <= float(max_followup_dcd) || return false
    mismatch12.dcm <= float(max_followup_dcm) || return false
    mismatch12.dxtr <= float(max_followup_dxtr) || return false

    scale01 = _result_mismatch_scale(mismatch01)
    scale12 = _result_mismatch_scale(mismatch12)
    isfinite(scale01) && isfinite(scale12) || return false
    scale01 >= float(contraction_factor) * max(scale12, 1.0e-12) || return false

    verification_residual = _result_selected_residual(verification)
    verification2_residual = _result_selected_residual(verification2)
    isfinite(verification2_residual) || return false
    residual_limit = float(max_followup_residual)
    if isfinite(verification_residual)
        residual_limit = max(
            residual_limit,
            float(residual_factor) * max(verification_residual, 1.0e-12),
        )
    end
    return verification2_residual <= residual_limit
end

@inline function _replay_orbit_kind(
    reference,
    verification,
    verification2;
    verification3 = nothing,
)
    if _plateau_verification_consistent(verification, verification2)
        return :fixed_point
    elseif _converged_replay_contracts(reference, verification, verification2)
        return :contracting
    elseif !isnothing(verification3) &&
           _plateau_verification_consistent(reference, verification2) &&
           _plateau_verification_consistent(verification, verification3)
        return :two_cycle
    end
    return :unresolved
end

@inline function _plateau_trace_cycle_detected(
    samples::AbstractVector{<:CoupledIterationSample},
    reference;
    residual_match_factor::Real = 2.0,
    residual_match_abs::Real = 2.0e-3,
    residual_jump_factor::Real = 20.0,
    residual_jump_abs::Real = 5.0e-2,
    max_dcl::Real = 5.0e-3,
    max_dcd::Real = 5.0e-4,
    max_dcm::Real = 5.0e-3,
    max_dxtr::Real = 5.0e-3,
)
    target_residual = _result_selected_residual(reference)
    isfinite(target_residual) || return false

    match_limit = max(float(residual_match_abs), float(residual_match_factor) * max(target_residual, 1.0e-12))
    match_indices = Int[]
    for (idx, sample) in pairs(samples)
        residual = float(sample.residual)
        isfinite(residual) || continue
        residual <= match_limit || continue
        _plateau_output_matches(
            reference,
            sample;
            max_dcl = max_dcl,
            max_dcd = max_dcd,
            max_dcm = max_dcm,
            max_dxtr = max_dxtr,
        ) || continue
        push!(match_indices, idx)
    end

    length(match_indices) >= 2 || return false
    jump_limit = max(float(residual_jump_abs), float(residual_jump_factor) * max(target_residual, 1.0e-12))

    for left_idx in 1:(length(match_indices) - 1)
        left = match_indices[left_idx]
        for right_idx in (left_idx + 1):length(match_indices)
            right = match_indices[right_idx]
            right - left >= 2 || continue

            peak_residual = -Inf
            for probe in (left + 1):(right - 1)
                residual = float(samples[probe].residual)
                isfinite(residual) || continue
                peak_residual = max(peak_residual, residual)
            end
            peak_residual >= jump_limit || continue
            return true
        end
    end

    return false
end

@inline function _viscal_candidate_merit(
    candidate,
    inv_cl::Real,
    alpha_deg::Real;
    te_gap_ante::Real = NaN,
    closure_weight::Real = 1.0,
    converged_bonus::Real = 0.35,
    nonconverged_penalty::Real = 0.75,
)
    score = _viscal_candidate_sanity_score(candidate, inv_cl, alpha_deg)
    score += float(closure_weight) * _viscal_candidate_closure_score(candidate; te_gap_ante = te_gap_ante)
    score += candidate.converged ? -float(converged_bonus) : float(nonconverged_penalty)

    residual = float(candidate.residual)
    if !isfinite(residual)
        score += 100.0
    else
        score += clamp(log10(max(residual, 1.0e-12)) + 4.0, 0.0, 6.0)
    end

    return score
end

@inline function _prefer_viscal_candidate(
    candidate,
    incumbent,
    inv_cl::Real,
    alpha_deg::Real;
    te_gap_ante::Real = NaN,
)
    cand_ok = _viscal_candidate_acceptable(candidate, inv_cl, alpha_deg)
    inc_ok = _viscal_candidate_acceptable(incumbent, inv_cl, alpha_deg)
    cand_ok == inc_ok || return cand_ok

    cand_merit = _viscal_candidate_merit(candidate, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
    inc_merit = _viscal_candidate_merit(incumbent, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
    if cand_merit + 1.0e-12 < inc_merit
        return true
    elseif inc_merit + 1.0e-12 < cand_merit
        return false
    end

    cand_res = float(candidate.residual)
    inc_res = float(incumbent.residual)
    isfinite(cand_res) && isfinite(inc_res) || return false
    return cand_res + 1.0e-12 < inc_res
end

@inline function _should_probe_literal_transition_branch(
    cl::Real,
    top_xtr::Real,
    inv_cl::Real;
    min_top_xtr::Real = 0.995,
    min_cl_gap::Real = 0.1,
)
    isfinite(float(cl)) && isfinite(float(top_xtr)) && isfinite(float(inv_cl)) || return false
    float(top_xtr) >= float(min_top_xtr) || return false
    return abs(float(cl) - float(inv_cl)) >= float(min_cl_gap)
end

@inline function _prefer_literal_transition_branch(
    candidate,
    incumbent,
    inv_cl::Real;
    min_top_xtr_gain::Real = 0.05,
    max_cd_increase::Real = 0.05,
    min_cl_error_improvement::Real = 0.05,
)
    candidate.converged || return false
    isfinite(float(candidate.top_xtr)) && isfinite(float(incumbent.top_xtr)) || return false
    float(incumbent.top_xtr) - float(candidate.top_xtr) >= float(min_top_xtr_gain) || return false

    cand_cd = float(candidate.cd)
    inc_cd = float(incumbent.cd)
    isfinite(cand_cd) && isfinite(inc_cd) || return false
    cand_cd <= inc_cd + float(max_cd_increase) || return false

    inv_cl_f = float(inv_cl)
    cand_cl_err = abs(float(candidate.cl) - inv_cl_f)
    inc_cl_err = abs(float(incumbent.cl) - inv_cl_f)
    return cand_cl_err + float(min_cl_error_improvement) <= inc_cl_err
end

@inline function _prefer_literal_followup_branch(
    candidate,
    incumbent,
    literal_anchor,
    inv_cl::Real;
    min_anchor_top_xtr_gain::Real = 0.02,
    max_cl_delta_from_anchor::Real = 0.12,
    max_cd_delta_from_anchor::Real = 0.01,
    max_cm_delta_from_anchor::Real = 0.03,
    max_top_xtr_delta_from_anchor::Real = 0.05,
)
    candidate.converged || return false
    literal_anchor === nothing && return false

    isfinite(float(incumbent.top_xtr)) && isfinite(float(literal_anchor.top_xtr)) || return false
    float(incumbent.top_xtr) - float(literal_anchor.top_xtr) >= float(min_anchor_top_xtr_gain) || return false

    abs(float(candidate.cl - literal_anchor.cl)) <= float(max_cl_delta_from_anchor) || return false
    abs(float(candidate.cd - literal_anchor.cd)) <= float(max_cd_delta_from_anchor) || return false
    abs(float(candidate.cm - literal_anchor.cm)) <= float(max_cm_delta_from_anchor) || return false
    abs(float(candidate.top_xtr - literal_anchor.top_xtr)) <= float(max_top_xtr_delta_from_anchor) || return false

    cand_cl_err = abs(float(candidate.cl) - float(inv_cl))
    inc_cl_err = abs(float(incumbent.cl) - float(inv_cl))
    return cand_cl_err + 1.0e-3 < inc_cl_err || float(candidate.cd) > float(incumbent.cd) + 0.01
end

@inline function _should_probe_aggressive_transition_startup_branch(
    candidate,
    inv_cl::Real;
    max_top_xtr::Real = 0.5,
    min_cl_excess::Real = 0.01,
)
    candidate.converged || return false
    isfinite(float(candidate.cl)) && isfinite(float(candidate.top_xtr)) && isfinite(float(inv_cl)) || return false
    float(candidate.top_xtr) <= float(max_top_xtr) || return false
    return float(candidate.cl) >= float(inv_cl) + float(min_cl_excess)
end

@inline function _prefer_aggressive_transition_startup_branch(
    candidate,
    incumbent,
    inv_cl::Real;
    min_top_xtr_recovery::Real = 0.2,
    max_cl_excess_over_inviscid::Real = 0.01,
    nonconverged_max_top_xtr_loss::Real = 0.02,
    nonconverged_residual_factor::Real = 0.2,
    nonconverged_max_cl_error_growth::Real = 0.0,
)
    candidate.converged || return false
    isfinite(float(candidate.cl)) && isfinite(float(candidate.top_xtr)) || return false
    isfinite(float(incumbent.top_xtr)) && isfinite(float(inv_cl)) || return false
    if !incumbent.converged
        isfinite(float(candidate.residual)) && isfinite(float(incumbent.residual)) || return false
        float(candidate.residual) <= float(nonconverged_residual_factor) * float(incumbent.residual) || return false
        float(candidate.top_xtr) + float(nonconverged_max_top_xtr_loss) >= float(incumbent.top_xtr) || return false
        cand_cl_err = abs(float(candidate.cl) - float(inv_cl))
        inc_cl_err = abs(float(incumbent.cl) - float(inv_cl))
        cand_cl_err <= inc_cl_err + float(nonconverged_max_cl_error_growth) || return false
        return true
    end
    float(candidate.top_xtr) >= float(incumbent.top_xtr) + float(min_top_xtr_recovery) || return false
    float(candidate.cl) <= float(inv_cl) + float(max_cl_excess_over_inviscid) || return false
    return true
end

@inline function _append_restart_note!(notes::Vector{String}, note::AbstractString)
    isempty(note) || push!(notes, String(note))
    return notes
end

@inline _restart_notes_message(notes::Vector{String}) = isempty(notes) ? "" : join(notes, " | ")

@inline function _startup_transition_recheck_note(
    incumbent,
    aggressive,
    branch_preferred::Bool,
    candidate_preferred::Bool;
    te_gap_ante::Real = NaN,
)
    decision = branch_preferred && candidate_preferred ? "selected" : "rejected"
    direct_closure = _viscal_candidate_closure_score(incumbent; te_gap_ante = te_gap_ante)
    aggressive_closure = _viscal_candidate_closure_score(aggressive; te_gap_ante = te_gap_ante)
    return (
        "startup_transition_recheck=aggressive " *
        "decision=$(decision) " *
        "prefer_branch=$(branch_preferred) " *
        "prefer_candidate=$(candidate_preferred) " *
        "direct_residual=$(round(incumbent.residual; digits = 6)) " *
        "aggressive_residual=$(round(aggressive.residual; digits = 6)) " *
        "direct_cl=$(round(incumbent.cl; digits = 6)) " *
        "direct_top_xtr=$(round(incumbent.top_xtr; digits = 6)) " *
        "direct_closure=$(round(direct_closure; digits = 6)) " *
        "aggressive_cl=$(round(aggressive.cl; digits = 6)) " *
        "aggressive_top_xtr=$(round(aggressive.top_xtr; digits = 6)) " *
        "aggressive_closure=$(round(aggressive_closure; digits = 6))"
    )
end

@inline function _viscal_alpha_continuation_schedule(alpha_deg::Real; step_deg::Real = 1.0)
    target = float(alpha_deg)
    abs(target) <= 1.0e-9 && return Float64[target]
    nsteps = max(1, Int(ceil(abs(target) / float(step_deg))))
    schedule = collect(range(0.0, target; length = nsteps + 1))
    return unique(schedule)
end

@inline function _viscal_seed_surface(
    seed::Union{Nothing,SurfaceMarchResult},
    init::SurfaceMarchResult,
)
    isnothing(seed) && return init
    return seed
end

function _viscal_alpha_continuation_rescue(
    airfoil::Airfoil,
    alpha_deg::Real,
    direct_candidate,
    model::BLModelParams,
    flow::BLFlowParams;
    reynolds::Real,
    ncrit::Real,
    repanel::Bool,
    max_iter::Integer,
    tol::Real,
    viscal_max_iter::Integer,
    viscal_tol::Real,
    xfoil_load_normalize_digits::Union{Nothing,Int},
    transition_recheck::Bool,
    capture_trace::Bool = false,
    trace_offset::Int = 0,
)
    target_alpha = float(alpha_deg)
    abs(target_alpha) >= 1.5 || return (
        accepted = false,
        candidate = direct_candidate,
        iterations = 0,
        note = "",
        trace = CoupledIterationSample[],
    )

    schedule = _viscal_alpha_continuation_schedule(target_alpha)
    length(schedule) >= 3 || return (
        accepted = false,
        candidate = direct_candidate,
        iterations = 0,
        note = "",
        trace = CoupledIterationSample[],
    )

    total_iterations = 0
    local_trace_offset = Int(trace_offset)
    trace_samples = CoupledIterationSample[]
    prev_upper = nothing
    prev_lower = nothing
    rescue_candidate = direct_candidate
    separated_candidate = nothing
    separated_score = Inf
    fallback_kind = _viscal_separated_fallback_probe_kind(direct_candidate)
    capture_fallback = fallback_kind !== :none
    final_ctx = nothing
    early_fallback_stop = false
    function consider_separated_fallback!(fallback)
        capture_fallback || return
        isnothing(fallback) && return
        score = _viscal_separated_fallback_score(
            fallback;
            alpha_deg = target_alpha,
            branch_kind = fallback_kind,
        )
        if score + 1.0e-12 < separated_score
            separated_candidate = fallback
            separated_score = score
        end
        return
    end
    try
        for (step_idx, alpha_step) in enumerate(schedule)
            inv_step = solve_inviscid_case(
                airfoil,
                alpha_step;
                repanel = repanel,
                xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            )
            topo_step = _xfoil_bl_topology_f32(inv_step)
            upper_init, lower_init, wake_init, wake_trace = _mrchue_initial_march_with_wake(
                inv_step,
                topo_step;
                flow = flow,
                model = model,
                ncrit = float(ncrit),
            )
            dij_step = _qdcalc_full(inv_step, wake_trace)

            if isnothing(prev_upper)
                seed_upper = upper_init
            else
                remapped_upper = try
                    _continuation_seed_surface(prev_upper, upper_init; flow = flow, model = model)
                catch
                    upper_init
                end
                seed_upper = _march_surface(
                    _surface_stations(remapped_upper);
                    flow = flow,
                    model = model,
                    ncrit = float(ncrit),
                    tol = float(tol),
                    max_iter = Int(max_iter),
                    seed = remapped_upper,
                    preserve_seed_transition = true,
                    solve_similarity_station = true,
                )
            end
            if isnothing(prev_lower)
                seed_lower = lower_init
            else
                remapped_lower = try
                    _continuation_seed_surface(prev_lower, lower_init; flow = flow, model = model)
                catch
                    lower_init
                end
                seed_lower = _march_surface(
                    _surface_stations(remapped_lower);
                    flow = flow,
                    model = model,
                    ncrit = float(ncrit),
                    tol = float(tol),
                    max_iter = Int(max_iter),
                    seed = remapped_lower,
                    preserve_seed_transition = true,
                    solve_similarity_station = true,
                )
            end

            fallback_ref = Ref{Any}(nothing)
            rescue_candidate = _viscal_solve(
                inv_step,
                topo_step,
                dij_step,
                seed_upper,
                seed_lower;
                flow = flow,
                model = model,
                ncrit = float(ncrit),
                max_iter = Int(viscal_max_iter),
                tol = Float64(viscal_tol),
                wake_trace = wake_trace,
                wake_march = wake_init,
                transition_recheck = transition_recheck,
                transition_recheck_upper_mode = :literal,
                transition_recheck_lower_mode = :literal,
                capture_trace = capture_trace,
                trace_segment = Symbol("alpha_continuation_step_$(step_idx)"),
                trace_offset = local_trace_offset,
                capture_separated_fallback = capture_fallback,
                separated_fallback_kind = fallback_kind,
                separated_fallback_ref = fallback_ref,
            )
            consider_separated_fallback!(fallback_ref[])
            total_iterations += rescue_candidate.iterations
            if capture_trace
                append!(trace_samples, rescue_candidate.trace)
                local_trace_offset += length(rescue_candidate.trace)
            end
            prev_upper = rescue_candidate.upper
            prev_lower = rescue_candidate.lower
            final_ctx = (inv = inv_step, topo = topo_step, dij = dij_step, wake_init = wake_init, wake_trace = wake_trace)
            if !isnothing(separated_candidate) &&
               separated_score <= 0.5 &&
               fallback_kind === :high_lift_late_lower_xtr &&
               _viscal_separated_fallback_preferred(
                   separated_candidate,
                   direct_candidate;
                   alpha_deg = target_alpha,
                   branch_kind = fallback_kind,
                   allow_near_zero = false,
               )
                early_fallback_stop = true
                break
            end
        end

        if !isnothing(final_ctx) && !early_fallback_stop
            fallback_ref = Ref{Any}(nothing)
            refine = _viscal_solve(
                final_ctx.inv,
                final_ctx.topo,
                final_ctx.dij,
                rescue_candidate.upper,
                rescue_candidate.lower;
                flow = flow,
                model = model,
                ncrit = float(ncrit),
                max_iter = max(Int(viscal_max_iter), 160),
                tol = Float64(viscal_tol),
                wake_trace = final_ctx.wake_trace,
                wake_march = final_ctx.wake_init,
                transition_recheck = transition_recheck,
                transition_recheck_upper_mode = :literal,
                transition_recheck_lower_mode = :literal,
                capture_trace = capture_trace,
                trace_segment = :alpha_continuation_refine,
                trace_offset = local_trace_offset,
                capture_separated_fallback = capture_fallback,
                separated_fallback_kind = fallback_kind,
                separated_fallback_ref = fallback_ref,
            )
            consider_separated_fallback!(fallback_ref[])
            total_iterations += refine.iterations
            if capture_trace
                append!(trace_samples, refine.trace)
                local_trace_offset += length(refine.trace)
            end
            if float(refine.residual) + 1.0e-12 < float(rescue_candidate.residual)
                rescue_candidate = refine
            end
        end

        fallback_selected =
            !isnothing(separated_candidate) &&
            _viscal_separated_fallback_preferred(
                separated_candidate,
                direct_candidate;
                alpha_deg = target_alpha,
                branch_kind = fallback_kind,
            )
        if fallback_selected
            rescue_candidate = separated_candidate
        end
        residual_improved = float(rescue_candidate.residual) + 1.0e-12 < float(direct_candidate.residual)
        accepted = rescue_candidate.converged || residual_improved
        note =
            "restart_seed=exact_alpha_continuation " *
            "steps=$(length(schedule)) " *
            "initial_residual=$(round(direct_candidate.residual; digits = 6)) " *
            "restart_residual=$(round(rescue_candidate.residual; digits = 6))" *
            (fallback_selected ?
                " branch=separated_fallback fallback_score=$(round(separated_score; digits = 6))" :
                "")
        return (
            accepted = accepted || fallback_selected,
            candidate = rescue_candidate,
            iterations = total_iterations,
            note = note,
            trace = trace_samples,
        )
    catch err
        message = replace(sprint(showerror, err), '\n' => ' ')
        note =
            "restart_seed=exact_alpha_continuation " *
            "error=$(message)"
        return (
            accepted = false,
            candidate = direct_candidate,
            iterations = total_iterations,
            note = note,
            trace = trace_samples,
        )
    end
end

@inline function _plausible_converged_retry_candidate(
    candidate::CoupledCaseResult;
    max_sanity_score::Real = 1.0,
)
    candidate.provenance.converged || return false
    return _coupled_result_sanity_score(candidate) <= float(max_sanity_score)
end

@inline function _is_meaningfully_better_residual(
    candidate::SolverProvenance,
    incumbent::SolverProvenance;
    ratio::Real = 0.5,
)
    cand = float(candidate.selected_residual)
    inc = float(incumbent.selected_residual)

    if !isfinite(cand)
        return false
    elseif !isfinite(inc)
        return true
    end

    return cand <= float(ratio) * max(inc, 1.0e-12)
end

@inline function _is_early_stalled_nonconverged_result(
    provenance::SolverProvenance;
    max_selected_iteration::Integer = 5,
    min_selected_residual::Real = 0.5,
)
    provenance.converged && return false
    provenance.selected_iteration > 0 || return false
    provenance.selected_iteration <= Int(max_selected_iteration) || return false
    isfinite(provenance.selected_residual) || return false
    return provenance.selected_residual >= float(min_selected_residual)
end

@inline function _prefer_nonconverged_legacy_retry_candidate(
    candidate::CoupledCaseResult,
    incumbent::CoupledCaseResult;
    residual_ratio::Real = 0.7,
)
    if candidate.provenance.converged && !incumbent.provenance.converged
        return _plausible_converged_retry_candidate(candidate)
    end
    candidate_score = _coupled_result_sanity_score(candidate)
    incumbent_score = _coupled_result_sanity_score(incumbent)
    candidate_score <= incumbent_score + 1.0e-12 || return false
    return _is_meaningfully_better_residual(
        candidate.provenance,
        incumbent.provenance;
        ratio = residual_ratio,
    )
end

@inline function _same_drag_regime(
    candidate_cd::Real,
    incumbent_cd::Real;
    split_cd::Real = 1.0,
)
    cand = float(candidate_cd)
    inc = float(incumbent_cd)
    isfinite(cand) && isfinite(inc) || return false
    split = float(split_cd)
    return (cand >= split) == (inc >= split)
end

@inline function _allow_implausible_drag_regime_escape(
    candidate::CoupledCaseResult,
    incumbent::CoupledCaseResult;
    max_candidate_cd::Real = 1.0,
    min_incumbent_cd::Real = 100.0,
)
    candidate.provenance.converged || return false
    cand_cd = float(candidate.cd)
    inc_cd = float(incumbent.cd)
    isfinite(cand_cd) && isfinite(inc_cd) || return false
    cand_cd <= float(max_candidate_cd) || return false
    inc_cd >= float(min_incumbent_cd) || return false
    return _coupled_result_sanity_score(candidate) + 1.0e-12 <
           _coupled_result_sanity_score(incumbent)
end

@inline function _allow_plausible_nonconverged_branch_refinement(
    candidate::CoupledCaseResult,
    incumbent::CoupledCaseResult;
    max_candidate_cd::Real = 0.1,
    min_incumbent_cd::Real = 1.0,
    max_cl_delta::Real = 0.1,
    max_cm_delta::Real = 0.1,
    max_xtr_delta::Real = 0.1,
)
    candidate.provenance.converged && return false
    incumbent.provenance.converged && return false
    cand_cd = float(candidate.cd)
    inc_cd = float(incumbent.cd)
    isfinite(cand_cd) && isfinite(inc_cd) || return false
    cand_cd <= float(max_candidate_cd) || return false
    inc_cd >= float(min_incumbent_cd) || return false
    abs(float(candidate.cl - incumbent.cl)) <= float(max_cl_delta) || return false
    abs(float(candidate.cm_c4 - incumbent.cm_c4)) <= float(max_cm_delta) || return false
    abs(float(candidate.top_xtr - incumbent.top_xtr)) <= float(max_xtr_delta) || return false
    abs(float(candidate.bot_xtr - incumbent.bot_xtr)) <= float(max_xtr_delta) || return false
    candidate_score = _coupled_result_sanity_score(candidate)
    incumbent_score = _coupled_result_sanity_score(incumbent)
    candidate_score <= incumbent_score + 1.0e-12 || return false
    cand_res = float(candidate.provenance.selected_residual)
    inc_res = float(incumbent.provenance.selected_residual)
    isfinite(cand_res) && isfinite(inc_res) || return false
    return cand_res + 1.0e-6 < inc_res
end

@inline function _legacy_hardcase_retry_presets()
    return (
        (
            label = "soft_no_disp",
            coupling_relax = 0.10,
            coupling_gain = 2.0,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = nothing,
        ),
        (
            label = "soft_grad_no_disp",
            coupling_relax = 0.10,
            coupling_gain = 4.0,
            coupling_grad_gain = 0.4,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = nothing,
        ),
        (
            label = "low_gain_no_disp",
            coupling_relax = 0.35,
            coupling_gain = 1.0,
            coupling_grad_gain = 0.4,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = nothing,
        ),
        (
            label = "soft_no_disp_long",
            coupling_relax = 0.10,
            coupling_gain = 2.0,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = 50,
        ),
        (
            label = "hypersoft_no_disp_long",
            coupling_relax = 0.03,
            coupling_gain = 2.0,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = 250,
        ),
        (
            label = "ultrasoft_no_disp_300",
            coupling_relax = 0.02,
            coupling_gain = 2.0,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = 300,
        ),
        (
            label = "ultrasoft_low_gain_no_disp_300",
            coupling_relax = 0.02,
            coupling_gain = 1.0,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = 300,
        ),
        (
            label = "ultrasoft_half_gain_no_disp_600",
            coupling_relax = 0.005,
            coupling_gain = 0.5,
            coupling_grad_gain = 0.0,
            coupling_smooth = 0.0,
            coupling_disp_scale = 0.0,
            coupling_disp_relax = 0.0,
            max_coupling_iter = 600,
        ),
    )
end

@inline function _continuation_seed_ue(
    seed::SurfaceMarchResult,
    inv_surface::SurfaceStations;
    min_ratio::Real = 0.5,
    max_ratio::Real = 1.5,
)
    base = max.(Float64.(inv_surface.ue), 1.0e-6)
    seed_ue =
        if length(seed.ue) == length(inv_surface.ue)
            Float64.(seed.ue)
        elseif length(seed.ue) >= 2
            [_interp_clamped(seed.xc, seed.ue, xc) for xc in inv_surface.xc]
        else
            copy(inv_surface.ue)
        end
    all(isfinite, seed_ue) || return copy(inv_surface.ue)
    return clamp.(seed_ue, float(min_ratio) .* base, float(max_ratio) .* base)
end

@inline function _validated_continuation_seed(
    seed::Union{Nothing,CoupledCaseResult},
    upper_inv::SurfaceStations,
    lower_inv::SurfaceStations,
)
    isnothing(seed) && return nothing
    seed.upper.side == :upper || return nothing
    seed.lower.side == :lower || return nothing
    length(seed.upper.primary) >= 2 || return nothing
    length(seed.lower.primary) >= 2 || return nothing
    total_seed = length(seed.upper.primary) + length(seed.lower.primary)
    total_target = length(upper_inv.ue) + length(lower_inv.ue)
    total_seed == total_target || return nothing
    return seed
end

@inline function _continuation_seed_pair(
    seed::CoupledCaseResult,
    upper_fallback::SurfaceMarchResult,
    lower_fallback::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
)
    upper_target = _surface_stations(upper_fallback)
    lower_target = _surface_stations(lower_fallback)
    upper_shift = length(upper_target.ue) - length(seed.upper.primary)
    lower_shift = length(lower_target.ue) - length(seed.lower.primary)

    if upper_shift == 0 && lower_shift == 0
        upper_seed = _continuation_seed_surface(seed.upper, upper_fallback; flow = flow, model = model)
        lower_seed = _continuation_seed_surface(seed.lower, lower_fallback; flow = flow, model = model)
        return upper_seed, lower_seed
    end

    if upper_shift == -lower_shift
        upper_seed = _stmove_surface_remap(
            seed.upper,
            SurfaceStations(:upper, upper_target.x, upper_target.xc, _continuation_seed_ue(seed.upper, upper_target)),
            upper_shift;
            flow = flow,
            model = model,
        )
        lower_seed = _stmove_surface_remap(
            seed.lower,
            SurfaceStations(:lower, lower_target.x, lower_target.xc, _continuation_seed_ue(seed.lower, lower_target)),
            lower_shift;
            flow = flow,
            model = model,
        )
        return upper_seed, lower_seed
    end

    upper_seed = _continuation_seed_surface(seed.upper, upper_fallback; flow = flow, model = model)
    lower_seed = _continuation_seed_surface(seed.lower, lower_fallback; flow = flow, model = model)
    return upper_seed, lower_seed
end

@inline function _legacy_continuation_seed_specs(
    alpha_deg::Real,
    reynolds::Real,
    ncrit::Real,
)
    α = float(alpha_deg)
    re = float(reynolds)
    nc = float(ncrit)
    specs = NamedTuple[]

    for factor in (2.0, 4.0)
        seed_re = re * factor
        seed_re <= 5.0e6 || continue
        push!(specs, (
            label = "same_alpha_re_x$(Int(round(factor)))",
            alpha = α,
            reynolds = seed_re,
            ncrit = nc,
        ))
    end

    if abs(α) >= 1.0
        for delta in (2.0, 4.0)
            seed_alpha = α - copysign(delta, α)
            push!(specs, (
                label = "toward_zero_alpha_$(Int(round(delta)))",
                alpha = seed_alpha,
                reynolds = re,
                ncrit = nc,
            ))
        end
    end

    return specs
end

function _retry_nonconverged_legacy_coupled_case(
    incumbent::CoupledCaseResult,
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real,
    ncrit::Real,
    repanel::Bool,
    model::BLModelParams,
    flow::Union{Nothing,BLFlowParams},
    tol::Real,
    max_iter::Integer,
    coupling_tol::Real,
    max_coupling_iter::Integer,
    viscal_max_iter::Integer,
    viscal_tol::Real,
    xfoil_load_normalize_digits::Union{Nothing,Int},
    continuation_retry::Bool,
)
    incumbent.provenance.converged && return incumbent
    best = incumbent
    best_label = ""
    best_seed_label = ""

    for preset in _legacy_hardcase_retry_presets()
        retry_max_coupling_iter =
            isnothing(preset.max_coupling_iter) ?
            max_coupling_iter :
            max(Int(preset.max_coupling_iter), Int(max_coupling_iter))
        candidate = solve_coupled_case(
            airfoil,
            alpha_deg;
            reynolds = reynolds,
            ncrit = ncrit,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            coupling_gain = preset.coupling_gain,
            coupling_grad_gain = preset.coupling_grad_gain,
            coupling_smooth = preset.coupling_smooth,
            coupling_relax = preset.coupling_relax,
            coupling_tol = coupling_tol,
            max_coupling_iter = retry_max_coupling_iter,
            coupling_disp_scale = preset.coupling_disp_scale,
            coupling_disp_clip = 0.03,
            coupling_disp_relax = preset.coupling_disp_relax,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            legacy = true,
        )
        if (
            candidate.provenance.converged &&
            !best.provenance.converged &&
            _prefer_nonconverged_legacy_retry_candidate(candidate, best)
        ) || (
            (
                _same_drag_regime(candidate.cd, best.cd) ||
                _allow_implausible_drag_regime_escape(candidate, best)
            ) &&
            _prefer_nonconverged_legacy_retry_candidate(candidate, best)
        ) || _allow_plausible_nonconverged_branch_refinement(candidate, best)
            best = candidate
            best_label = preset.label
            best_seed_label = ""
        end
    end

    if continuation_retry
        for seed_spec in _legacy_continuation_seed_specs(alpha_deg, reynolds, ncrit)
            seed_case = solve_coupled_case(
                airfoil,
                seed_spec.alpha;
                reynolds = seed_spec.reynolds,
                ncrit = seed_spec.ncrit,
                repanel = repanel,
                model = model,
                flow = flow,
                tol = tol,
                max_iter = max_iter,
                coupling_gain = 4.0,
                coupling_grad_gain = 0.8,
                coupling_smooth = 0.2,
                coupling_relax = 0.35,
                coupling_tol = coupling_tol,
                max_coupling_iter = max_coupling_iter,
                coupling_disp_scale = 0.2,
                coupling_disp_clip = 0.03,
                coupling_disp_relax = 0.15,
                viscal_max_iter = viscal_max_iter,
                viscal_tol = viscal_tol,
                xfoil_load_normalize_digits = xfoil_load_normalize_digits,
                continuation_retry = false,
            )
            _plausible_converged_retry_candidate(seed_case) || continue

            for preset in _legacy_hardcase_retry_presets()
                retry_max_coupling_iter =
                    isnothing(preset.max_coupling_iter) ?
                    max_coupling_iter :
                    max(Int(preset.max_coupling_iter), Int(max_coupling_iter))
                candidate = solve_coupled_case(
                    airfoil,
                    alpha_deg;
                    reynolds = reynolds,
                    ncrit = ncrit,
                    repanel = repanel,
                    model = model,
                    flow = flow,
                    tol = tol,
                    max_iter = max_iter,
                    coupling_gain = preset.coupling_gain,
                    coupling_grad_gain = preset.coupling_grad_gain,
                    coupling_smooth = preset.coupling_smooth,
                    coupling_relax = preset.coupling_relax,
                    coupling_tol = coupling_tol,
                    max_coupling_iter = retry_max_coupling_iter,
                    coupling_disp_scale = preset.coupling_disp_scale,
                    coupling_disp_clip = 0.03,
                    coupling_disp_relax = preset.coupling_disp_relax,
                    viscal_max_iter = viscal_max_iter,
                    viscal_tol = viscal_tol,
                    xfoil_load_normalize_digits = xfoil_load_normalize_digits,
                    legacy = true,
                    continuation_seed = seed_case,
                    continuation_retry = false,
                )
                if (
                    candidate.provenance.converged &&
                    !best.provenance.converged &&
                    _prefer_nonconverged_legacy_retry_candidate(candidate, best)
                ) || (
                    (
                        _same_drag_regime(candidate.cd, best.cd) ||
                        _allow_implausible_drag_regime_escape(candidate, best)
                    ) &&
                    _prefer_nonconverged_legacy_retry_candidate(candidate, best)
                ) || _allow_plausible_nonconverged_branch_refinement(candidate, best)
                    best = candidate
                    best_label = preset.label
                    best_seed_label = seed_spec.label
                end
            end
        end
    end

    if isempty(best_label)
        return incumbent
    end

    retry_provenance = _solver_provenance(
        backend = :legacy_coupled_hardcase_retry,
        selection = :legacy_nonconverged_retry,
        converged = best.provenance.converged,
        status = best.provenance.status,
        failure_message =
            "retry_preset=$(best_label)" *
            (isempty(best_seed_label) ? "" : " continuation_seed=$(best_seed_label)") * " " *
            "base_residual=$(round(incumbent.provenance.selected_residual; digits = 6)) " *
            "retry_residual=$(round(best.provenance.selected_residual; digits = 6))",
        iterations = best.provenance.iterations,
        selected_iteration = best.provenance.selected_iteration,
        selected_residual = best.provenance.selected_residual,
    )
    return _with_provenance(best, retry_provenance)
end

function _stagnation_split_indices(inv::CaseResult)
    m = length(inv.xcp)
    m == length(inv.vt) || throw(ArgumentError("CaseResult vt length mismatch"))
    m >= 6 || throw(ArgumentError("need at least 6 panels for stagnation split"))

    ile = argmin(inv.xcp)
    band = max(3, m ÷ 10)
    window_lo = max(1, ile - band)
    window_hi = min(m, ile + band)
    istag_local = argmin(abs.(inv.vt[window_lo:window_hi]))
    istag = clamp(window_lo + istag_local - 1, 2, m - 2)

    upper_idx = reverse(collect(1:istag))
    lower_idx = collect((istag + 1):m)
    return upper_idx, lower_idx
end

function _arc_coordinates(inv::CaseResult, idx::AbstractVector{<:Integer})
    n = length(idx)
    n >= 1 || return Float64[]
    s = zeros(Float64, n)
    for k in 2:n
        i1 = idx[k - 1]
        i2 = idx[k]
        s[k] = s[k - 1] + hypot(inv.xcp[i2] - inv.xcp[i1], inv.ycp[i2] - inv.ycp[i1])
    end
    return s
end

function _setexp_ratio(ds1::Real, smax::Real, nn::Integer)
    nn >= 2 || throw(ArgumentError("SETEXP requires at least 2 points"))

    T = promote_type(typeof(ds1), typeof(smax))
    T <: AbstractFloat || (T = Float64)

    nex = nn - 1
    nex == 1 && throw(ArgumentError("SETEXP cannot fill array with one interval"))

    sigma = T(smax) / T(ds1)
    rnex = T(nex)
    rni = one(T) / rnex

    aaa = rnex * (rnex - one(T)) * (rnex - T(2.0)) / T(6.0)
    bbb = rnex * (rnex - one(T)) / T(2.0)
    ccc = rnex - sigma

    disc = max(bbb^2 - T(4.0) * aaa * ccc, zero(T))
    ratio =
        if nex == 2
            -ccc / bbb + one(T)
        else
            (-bbb + sqrt(disc)) / (T(2.0) * aaa) + one(T)
        end

    if ratio != one(T)
        for _ in 1:100
            sigman = (ratio^nex - one(T)) / (ratio - one(T))
            res = sigman^rni - sigma^rni
            dresdr =
                rni *
                sigman^rni *
                (rnex * ratio^(nex - 1) - sigman) /
                (ratio^nex - one(T))
            dratio = -res / dresdr
            ratio += dratio
            abs(dratio) < T(1.0e-5) && break
        end
    end
    return ratio
end

function _setexp_residual(ratio::T, ds1::T, smax::T, nn::Integer) where {T<:AbstractFloat}
    nex = nn - 1
    sigma = smax / ds1
    rnex = T(nex)
    rni = one(T) / rnex
    sigman = (ratio^nex - one(T)) / (ratio - one(T))
    return sigman^rni - sigma^rni
end

function _setexp_array(ds1::T, ratio::T, nn::Integer) where {T<:AbstractFloat}
    s = zeros(T, nn)
    ds = ds1
    for i in 2:nn
        s[i] = s[i - 1] + ds
        ds *= ratio
    end
    return s
end

function _setexp(ds1::Real, smax::Real, nn::Integer)
    ratio = _setexp_ratio(ds1, smax, nn)
    T = typeof(ratio)
    return _setexp_array(T(ds1), ratio, nn)
end

function _xfoil_stfind(inv::CaseResult, q::AbstractVector{<:Real} = inv.qnode)
    n = length(inv.xcp)
    n == length(inv.ycp) || throw(ArgumentError("CaseResult xcp/ycp length mismatch"))
    n == length(q) || throw(ArgumentError("stagnation q length mismatch"))
    n >= 3 || throw(ArgumentError("need at least 3 airfoil nodes for stagnation point"))

    s = _scalc(inv.xcp, inv.ycp)

    ist = 0
    for i in 1:(n - 1)
        if float(q[i]) >= 0.0 && float(q[i + 1]) < 0.0
            ist = i
            break
        end
    end
    if ist == 0
        ist = n ÷ 2
    end

    dgam = float(q[ist + 1]) - float(q[ist])
    abs(dgam) > 1.0e-30 || throw(ArgumentError("degenerate stagnation-point circulation jump"))
    ds = s[ist + 1] - s[ist]

    sst =
        if float(q[ist]) < -float(q[ist + 1])
            s[ist] - ds * (float(q[ist]) / dgam)
        else
            s[ist + 1] - ds * (float(q[ist + 1]) / dgam)
        end

    if sst <= s[ist]
        sst = s[ist] + 1.0e-7
    end
    if sst >= s[ist + 1]
        sst = s[ist + 1] - 1.0e-7
    end

    sst_go = (sst - s[ist + 1]) / dgam
    sst_gp = (s[ist] - sst) / dgam
    return BLStagnationPoint(ist, sst, sst_go, sst_gp), s
end

function _xfoil_stfind_f32(inv::CaseResult, q::AbstractVector{<:Real} = inv.qnode)
    n = length(inv.xcp)
    n == length(inv.ycp) || throw(ArgumentError("CaseResult xcp/ycp length mismatch"))
    n == length(q) || throw(ArgumentError("stagnation q length mismatch"))
    n >= 3 || throw(ArgumentError("need at least 3 airfoil nodes for stagnation point"))

    s = _scalc(Float32.(inv.xcp), Float32.(inv.ycp))
    qf = Float32.(q)

    ist = 0
    for i in 1:(n - 1)
        if qf[i] >= 0.0f0 && qf[i + 1] < 0.0f0
            ist = i
            break
        end
    end
    if ist == 0
        ist = div(n, 2)
    end

    dgam = qf[ist + 1] - qf[ist]
    abs(dgam) > 1.0f-30 || throw(ArgumentError("degenerate stagnation-point circulation jump"))
    ds = s[ist + 1] - s[ist]

    sst =
        if qf[ist] < -qf[ist + 1]
            s[ist] - ds * (qf[ist] / dgam)
        else
            s[ist + 1] - ds * (qf[ist + 1] / dgam)
        end

    if sst <= s[ist]
        sst = s[ist] + 1.0f-7
    end
    if sst >= s[ist + 1]
        sst = s[ist + 1] - 1.0f-7
    end

    sst_go = (sst - s[ist + 1]) / dgam
    sst_gp = (s[ist] - sst) / dgam
    return BLStagnationPoint(ist, Float64(sst), Float64(sst_go), Float64(sst_gp)), s
end

function _xfoil_iblpan(inv::CaseResult, stag::BLStagnationPoint)
    n = length(inv.xcp)

    upper_n = stag.ist + 1
    upper_ipan = zeros(Int, upper_n)
    upper_vti = zeros(Float64, upper_n)
    ibl = 1
    for i in stag.ist:-1:1
        ibl += 1
        upper_ipan[ibl] = i
        upper_vti[ibl] = 1.0
    end

    lower_n = n - stag.ist + 1
    lower_ipan = zeros(Int, lower_n)
    lower_vti = zeros(Float64, lower_n)
    ibl = 1
    for i in (stag.ist + 1):n
        ibl += 1
        lower_ipan[ibl] = i
        lower_vti[ibl] = -1.0
    end

    upper = BLSideTopology(:upper, upper_ipan, upper_vti, zeros(Float64, upper_n))
    lower = BLSideTopology(:lower, lower_ipan, lower_vti, zeros(Float64, lower_n))
    return upper, lower
end

function _xfoil_xicalc(
    s::AbstractVector{<:Real},
    stag::BLStagnationPoint,
    side::BLSideTopology,
)
    xeps = 1.0e-7 * (float(s[end]) - float(s[1]))
    xssi = zeros(Float64, length(side.ipan))
    xssi[1] = 0.0
    for ibl in 2:length(side.ipan)
        i = side.ipan[ibl]
        xssi[ibl] =
            if side.side == :upper
                max(stag.sst - float(s[i]), xeps)
            else
                max(float(s[i]) - stag.sst, xeps)
            end
    end
    return BLSideTopology(side.side, copy(side.ipan), copy(side.vti), xssi)
end

function _xfoil_xicalc_f32(
    s::AbstractVector{Float32},
    stag::BLStagnationPoint,
    side::BLSideTopology,
)
    xeps = 1.0f-7 * (s[end] - s[1])
    sst = Float32(stag.sst)
    xssi = zeros(Float64, length(side.ipan))
    xssi[1] = 0.0
    for ibl in 2:length(side.ipan)
        i = side.ipan[ibl]
        xssi[ibl] =
            if side.side == :upper
                Float64(max(sst - s[i], xeps))
            else
                Float64(max(s[i] - sst, xeps))
            end
    end
    return BLSideTopology(side.side, copy(side.ipan), copy(side.vti), xssi)
end

function _xfoil_bl_topology(inv::CaseResult; q::AbstractVector{<:Real} = inv.qnode)
    stag, s = _xfoil_stfind(inv, q)
    upper0, lower0 = _xfoil_iblpan(inv, stag)
    upper = _xfoil_xicalc(s, stag, upper0)
    lower = _xfoil_xicalc(s, stag, lower0)
    return BLTopology(s, stag, upper, lower)
end

function _xfoil_bl_topology_f32(inv::CaseResult; q::AbstractVector{<:Real} = inv.qnode)
    stag, s = _xfoil_stfind_f32(inv, q)
    upper0, lower0 = _xfoil_iblpan(inv, stag)
    upper = _xfoil_xicalc_f32(s, stag, upper0)
    lower = _xfoil_xicalc_f32(s, stag, lower0)
    return BLTopology(Float64.(s), stag, upper, lower)
end

function _bl_side(topo::BLTopology, side::Symbol)
    side == :upper && return topo.upper
    side == :lower && return topo.lower
    throw(ArgumentError("side must be :upper or :lower"))
end

function _xfoil_iblsys(topo::BLTopology)
    upper = zeros(Int, length(topo.upper.ipan))
    lower = zeros(Int, length(topo.lower.ipan))
    iv = 0

    for ibl in 2:length(upper)
        iv += 1
        upper[ibl] = iv
    end
    for ibl in 2:length(lower)
        iv += 1
        lower[ibl] = iv
    end

    return BLSystemTopology(upper, lower, iv)
end

function _surface_stations(
    inv::CaseResult,
    side::Symbol;
    uemin::Float64 = 1.0e-4,
    ue_panel::Union{Nothing,AbstractVector{<:Real}} = nothing,
    topology::Union{Nothing,BLTopology} = nothing,
)
    side in (:upper, :lower) || throw(ArgumentError("side must be :upper or :lower"))
    m = length(inv.xcp)
    m == length(inv.cp) || throw(ArgumentError("CaseResult xcp/cp length mismatch"))
    m == length(inv.qnode) || throw(ArgumentError("CaseResult qnode length mismatch"))

    topo =
        isnothing(topology) ?
        _xfoil_bl_topology(inv; q = isnothing(ue_panel) ? inv.qnode : ue_panel) :
        topology
    st = _bl_side(topo, side)
    length(st.ipan) >= 3 || throw(ArgumentError("Too few BL stations on $side surface"))

    idx = st.ipan[2:end]
    x = copy(st.xssi[2:end])
    xc = [float(inv.xcp[i]) for i in idx]
    ue =
        if isnothing(ue_panel)
            [max(float(st.vti[k + 1] * inv.qnode[i]), uemin) for (k, i) in enumerate(idx)]
        else
            length(ue_panel) == m || throw(ArgumentError("ue_panel length mismatch"))
            [max(float(st.vti[k + 1] * ue_panel[i]), uemin) for (k, i) in enumerate(idx)]
        end

    return SurfaceStations(side, x, xc, ue)
end

function _surface_stations(surface::SurfaceMarchResult)
    return SurfaceStations(surface.side, copy(surface.x), copy(surface.xc), copy(surface.ue))
end

function _interp_clamped(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    xi::Real,
)
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y length mismatch"))
    n >= 2 || throw(ArgumentError("x/y must contain at least 2 points"))

    xif = float(xi)
    x1 = float(x[1])
    xN = float(x[end])
    if xif <= x1
        return float(y[1])
    elseif xif >= xN
        return float(y[end])
    end

    j = searchsortedlast(x, xif)
    j = clamp(j, 1, n - 1)
    xa = float(x[j])
    xb = float(x[j + 1])
    ya = float(y[j])
    yb = float(y[j + 1])
    t = (xif - xa) / max(xb - xa, 1.0e-12)
    return (1.0 - t) * ya + t * yb
end

function _max_relative_change(
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real};
    floor::Float64 = 1.0e-7,
    ref_floor_frac::Float64 = 0.1,
)
    length(a) == length(b) || throw(ArgumentError("relative-change length mismatch"))
    bref = 0.0
    for bi in b
        bref = max(bref, abs(float(bi)))
    end
    denfloor = max(float(ref_floor_frac) * bref, floor)
    rmax = 0.0
    for i in eachindex(a, b)
        ai = float(a[i])
        bi = float(b[i])
        den = max(abs(bi), denfloor)
        rmax = max(rmax, abs(ai - bi) / den)
    end
    return rmax
end

function _panel_ue_from_surfaces(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    ;
    topology::Union{Nothing,BLTopology} = nothing,
)
    m = length(inv.xcp)
    topo = isnothing(topology) ? _xfoil_bl_topology(inv) : topology
    # Initialize with inviscid velocity for panels not covered by BL stations
    ue = [abs(Float64(inv.qnode[i])) for i in 1:m]
    # Direct mapping: each BL station maps to its panel via ipan,
    # matching XFoil's QVFUE (QVIS(I) = VTI(IBL,IS) * UEDG(IBL,IS)).
    # The surface march Ue corresponds to BL stations 2:end.
    n_upper = min(length(topo.upper.ipan) - 1, length(upper.ue))
    for k in 1:n_upper
        i = topo.upper.ipan[k + 1]
        if 1 <= i <= m && k <= length(upper.ue)
            ue[i] = abs(Float64(upper.ue[k]))
        end
    end
    n_lower = min(length(topo.lower.ipan) - 1, length(lower.ue))
    for k in 1:n_lower
        i = topo.lower.ipan[k + 1]
        if 1 <= i <= m && k <= length(lower.ue)
            ue[i] = abs(Float64(lower.ue[k]))
        end
    end
    return ue
end

function _panel_dstar_from_surfaces(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    ;
    topology::Union{Nothing,BLTopology} = nothing,
)
    m = length(inv.xcp)
    topo = isnothing(topology) ? _xfoil_bl_topology(inv) : topology
    dup = [max(p.d, 0.0) for p in upper.primary]
    dlo = [max(p.d, 0.0) for p in lower.primary]

    dstar = Vector{Float64}(undef, m)
    for (k, i) in enumerate(topo.upper.ipan[2:end])
        dstar[i] = max(0.0, _interp_clamped(upper.x, dup, topo.upper.xssi[k + 1]))
    end
    for (k, i) in enumerate(topo.lower.ipan[2:end])
        dstar[i] = max(0.0, _interp_clamped(lower.x, dlo, topo.lower.xssi[k + 1]))
    end
    return dstar
end

function _displaced_airfoil(
    airfoil::Airfoil,
    d_panel::AbstractVector{<:Real};
    scale::Float64 = 0.7,
    max_frac::Float64 = 0.03,
)
    geom = _panel_geometry(_xfoil_panel_airfoil(airfoil))
    m = geom.m
    chord = max(geom.chord, 1.0e-8)
    dmax = max(0.0, max_frac) * chord

    node_d =
        if length(d_panel) == m
            dp = [clamp(scale * float(d_panel[i]), 0.0, dmax) for i in 1:m]
            nd = Vector{Float64}(undef, m + 1)
            for i in 1:m
                im = i == 1 ? m : i - 1
                nd[i] = 0.5 * (dp[im] + dp[i])
            end
            nd[m + 1] = nd[1]
            nd
        elseif length(d_panel) == m + 1
            nd = [clamp(scale * float(d_panel[i]), 0.0, dmax) for i in 1:(m + 1)]
            nd[end] = nd[1]
            nd
        else
            throw(ArgumentError("d_panel length mismatch"))
        end
    nxn = Vector{Float64}(undef, m + 1)
    nyn = Vector{Float64}(undef, m + 1)

    for i in 1:m
        im = i == 1 ? m : i - 1
        nx = geom.nx[im] + geom.nx[i]
        ny = geom.ny[im] + geom.ny[i]
        nrm = hypot(nx, ny)
        if nrm <= 1.0e-12
            nx = geom.nx[i]
            ny = geom.ny[i]
            nrm = max(hypot(nx, ny), 1.0e-12)
        end
        nxn[i] = nx / nrm
        nyn[i] = ny / nrm
    end
    nxn[m + 1] = nxn[1]
    nyn[m + 1] = nyn[1]

    xnew = Vector{Float64}(undef, m + 1)
    ynew = Vector{Float64}(undef, m + 1)
    for i in 1:(m + 1)
        xnew[i] = geom.x[i] + node_d[i] * nxn[i]
        ynew[i] = geom.y[i] + node_d[i] * nyn[i]
    end
    # Keep closure exact.
    xnew[end] = xnew[1]
    ynew[end] = ynew[1]
    return Airfoil(string(airfoil.name, "_disp"), xnew, ynew)
end

function _resampled_inviscid_ue(
    ref::SurfaceStations,
    inv::CaseResult,
    side::Symbol;
    uemin::Float64 = 1.0e-4,
    topology::Union{Nothing,BLTopology} = nothing,
)
    st = _surface_stations(inv, side; uemin = uemin, topology = topology)
    ue = Vector{Float64}(undef, length(ref.x))
    for i in eachindex(ref.x)
        xi = clamp(ref.x[i], st.x[1], st.x[end])
        ue[i] = max(uemin, _interp_clamped(st.x, st.ue, xi))
    end
    return ue
end

function _surface_mass_by_node(
    inv::CaseResult,
    topo::BLTopology,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    ueps::Float64 = 1.0e-7,
)
    n = length(inv.qnode)
    mass = zeros(Float64, n)
    vti = zeros(Float64, n)

    length(upper.primary) == length(topo.upper.ipan) - 1 ||
        throw(ArgumentError("upper topology/state length mismatch"))
    length(lower.primary) == length(topo.lower.ipan) - 1 ||
        throw(ArgumentError("lower topology/state length mismatch"))

    for (k, i) in enumerate(topo.upper.ipan[2:end])
        mass[i] = max(upper.primary[k].d, 0.0) * max(upper.ue[k], ueps)
        vti[i] = topo.upper.vti[k + 1]
    end
    for (k, i) in enumerate(topo.lower.ipan[2:end])
        mass[i] = max(lower.primary[k].d, 0.0) * max(lower.ue[k], ueps)
        vti[i] = topo.lower.vti[k + 1]
    end
    return mass, vti
end

function _surface_qnode_from_surfaces(
    inv::CaseResult,
    topo::BLTopology,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    ueps::Float64 = 1.0e-7,
)
    n = length(inv.qnode)
    q = zeros(Float64, n)

    length(upper.ue) == length(topo.upper.ipan) - 1 ||
        throw(ArgumentError("upper topology/state length mismatch"))
    length(lower.ue) == length(topo.lower.ipan) - 1 ||
        throw(ArgumentError("lower topology/state length mismatch"))

    for (k, i) in enumerate(topo.upper.ipan[2:end])
        q[i] = topo.upper.vti[k + 1] * max(upper.ue[k], ueps)
    end
    for (k, i) in enumerate(topo.lower.ipan[2:end])
        q[i] = topo.lower.vti[k + 1] * max(lower.ue[k], ueps)
    end
    return q
end

function _surface_primary_with_station(
    p::BLPrimaryState,
    mode::Symbol,
    x::Real,
    u::Real;
    ueps::Float64 = 1.0e-7,
)
    xf = float(x)
    uf = max(float(u), ueps)
    if mode in (:laminar, :similarity)
        return BLPrimaryState(
            xf,
            uf,
            p.t,
            p.d;
            s = 0.0,
            ampl = max(p.ampl, 0.0),
            dw = p.dw,
        )
    elseif mode in (:turbulent, :transition, :wake)
        return BLPrimaryState(
            xf,
            uf,
            p.t,
            p.d;
            s = max(p.s, 1.0e-12),
            ampl = 0.0,
            dw = p.dw,
        )
    end
    throw(ArgumentError("invalid BL mode: $mode"))
end

function _stmove_surface_remap(
    old::SurfaceMarchResult,
    new_stations::SurfaceStations,
    shift::Integer;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    nold = length(old.x)
    nnew = length(new_stations.x)
    length(old.primary) == nold == length(old.secondary) == length(old.mode) ||
        throw(ArgumentError("old surface state length mismatch"))
    length(new_stations.xc) == nnew == length(new_stations.ue) ||
        throw(ArgumentError("new station length mismatch"))

    if shift > 0
        nnew == nold + shift || throw(ArgumentError("surface gain remap size mismatch"))
    elseif shift < 0
        nnew == nold + shift || throw(ArgumentError("surface loss remap size mismatch"))
    else
        nnew == nold || throw(ArgumentError("surface no-shift remap size mismatch"))
    end

    primary = Vector{BLPrimaryState}(undef, nnew)
    secondary = Vector{BLSecondaryState}(undef, nnew)
    mode = Vector{Symbol}(undef, nnew)
    conv = falses(max(nnew - 1, 0))

    if shift > 0
        for j in (shift + 1):nnew
            src = j - shift
            mode[j] = old.mode[src]
            primary[j] = _surface_primary_with_station(
                old.primary[src],
                mode[j],
                new_stations.x[j],
                new_stations.ue[j];
                ueps = ueps,
            )
            secondary[j] = blsecondary_state(
                primary[j],
                _mode_closure(mode[j]);
                flow = flow,
                model = model,
            )
        end

        template = primary[shift + 1]
        template_mode = mode[shift + 1]
        dudx = template.u / max(new_stations.x[shift + 1], 1.0e-12)
        for j in shift:-1:1
            uj = max(dudx * new_stations.x[j], ueps)
            mode[j] = template_mode
            primary[j] = _surface_primary_with_station(
                template,
                template_mode,
                new_stations.x[j],
                uj;
                ueps = ueps,
            )
            secondary[j] = blsecondary_state(
                primary[j],
                _mode_closure(template_mode);
                flow = flow,
                model = model,
            )
        end

        for j in (shift + 1):(nnew - 1)
            conv[j] = old.converged_intervals[j - shift]
        end
    elseif shift < 0
        ishift = -shift
        for j in 1:nnew
            src = j + ishift
            mode[j] = old.mode[src]
            primary[j] = _surface_primary_with_station(
                old.primary[src],
                mode[j],
                new_stations.x[j],
                new_stations.ue[j];
                ueps = ueps,
            )
            secondary[j] = blsecondary_state(
                primary[j],
                _mode_closure(mode[j]);
                flow = flow,
                model = model,
            )
        end

        for j in 1:(nnew - 1)
            conv[j] = old.converged_intervals[j + ishift]
        end
    else
        for j in 1:nnew
            mode[j] = old.mode[j]
            primary[j] = _surface_primary_with_station(
                old.primary[j],
                mode[j],
                new_stations.x[j],
                new_stations.ue[j];
                ueps = ueps,
            )
            secondary[j] = blsecondary_state(
                primary[j],
                _mode_closure(mode[j]);
                flow = flow,
                model = model,
            )
        end
        conv .= old.converged_intervals
    end

    ue = [p.u for p in primary]
    return SurfaceMarchResult(
        new_stations.side,
        copy(new_stations.x),
        copy(new_stations.xc),
        ue,
        mode,
        primary,
        secondary,
        old.transition_x,
        old.solved_similarity_station,
        conv,
    )
end

function _xfoil_stmove_results(
    old_topo::BLTopology,
    new_topo::BLTopology,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    upper_stations::SurfaceStations,
    lower_stations::SurfaceStations;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    idif = new_topo.stagnation.ist - old_topo.stagnation.ist
    if idif > 0
        upper_new = _stmove_surface_remap(
            upper,
            upper_stations,
            idif;
            flow = flow,
            model = model,
            ueps = ueps,
        )
        lower_new = _stmove_surface_remap(
            lower,
            lower_stations,
            -idif;
            flow = flow,
            model = model,
            ueps = ueps,
        )
    elseif idif < 0
        upper_new = _stmove_surface_remap(
            upper,
            upper_stations,
            idif;
            flow = flow,
            model = model,
            ueps = ueps,
        )
        lower_new = _stmove_surface_remap(
            lower,
            lower_stations,
            -idif;
            flow = flow,
            model = model,
            ueps = ueps,
        )
    else
        upper_new = _stmove_surface_remap(
            upper,
            upper_stations,
            0;
            flow = flow,
            model = model,
            ueps = ueps,
        )
        lower_new = _stmove_surface_remap(
            lower,
            lower_stations,
            0;
            flow = flow,
            model = model,
            ueps = ueps,
        )
    end
    return upper_new, lower_new
end

function _surface_gamma_factorization(inv::CaseResult)
    T = _XFOIL_GGCALC_PRECISION
    x = T.(inv.xnode)
    y = T.(inv.ynode)
    n = length(x)
    n == length(y) == length(inv.qnode) || throw(ArgumentError("CaseResult node geometry mismatch"))

    s = _scalc(x, y)
    nx, ny, xp, yp = _ncalc(x, y, s)
    te = _tecalc(x, y, xp, yp)
    apanel = _apcalc(x, y, nx, ny; sharp = te.sharp)

    aij = zeros(T, n + 1, n + 1)
    for i in 1:n
        dzdg, _, _, _ = _psilin_source_vortex_derivatives(
            i,
            x[i],
            y[i],
            nx[i],
            ny[i],
            x,
            y,
            s,
            apanel;
            scs = te.scs,
            sds = te.sds,
        )
        for j in 1:n
            aij[i, j] = dzdg[j]
        end
        aij[i, n + 1] = -one(T)
    end

    aij[n + 1, :] .= zero(T)
    aij[n + 1, 1] = one(T)
    aij[n + 1, n] = one(T)

    if te.sharp
        bwt = T(0.1)
        ag1 = _xfoil_atan2(-yp[1], -xp[1])
        ag2 = _atanc(yp[n], xp[n], ag1)
        abis = T(0.5) * (ag1 + ag2)
        cbis = cos(abis)
        sbis = sin(abis)

        ds1 = sqrt((x[1] - x[2])^2 + (y[1] - y[2])^2)
        ds2 = sqrt((x[n] - x[n - 1])^2 + (y[n] - y[n - 1])^2)
        dsmin = min(ds1, ds2)

        xbis = te.xte - bwt * dsmin * cbis
        ybis = te.yte - bwt * dsmin * sbis
        _, dqdg = _psilin_vortex_derivatives(
            0,
            xbis,
            ybis,
            -sbis,
            cbis,
            x,
            y,
            s,
            apanel;
            scs = te.scs,
            sds = te.sds,
        )
        for j in 1:n
            aij[n, j] = dqdg[j]
        end
        aij[n, n + 1] = zero(T)
    end

    return (
        x = x,
        y = y,
        n = n,
        s = s,
        nx = nx,
        ny = ny,
        xp = xp,
        yp = yp,
        te = te,
        apanel = apanel,
        fac = _xfoil_lu(aij; precision = T),
    )
end

function _wake_apanel(
    cache,
    xw::AbstractVector{<:Real},
    yw::AbstractVector{<:Real},
)
    n = length(xw)
    n == length(yw) || throw(ArgumentError("wake geometry length mismatch"))
    n >= 2 || throw(ArgumentError("wake must contain at least two nodes"))

    T = _XFOIL_GGCALC_PRECISION
    ap = Vector{T}(undef, n)
    for i in 1:(n - 1)
        psi_x, psi_y = _wake_streamfunction_gradient(cache, length(cache.x) + i, xw[i], yw[i])
        ap[i] = _xfoil_atan2(T(psi_y), T(psi_x))
    end
    ap[n] = ap[n - 1]
    return ap
end

function _pswlin_source_derivatives(
    io::Int,
    xi::Real,
    yi::Real,
    nxi::Real,
    nyi::Real,
    xw::AbstractVector{<:Real},
    yw::AbstractVector{<:Real},
    apanel::AbstractVector{<:Real};
    eval_on_wake::Bool = false,
)
    nw = length(xw)
    nw == length(yw) == length(apanel) || throw(ArgumentError("wake derivative length mismatch"))
    nw >= 2 || throw(ArgumentError("wake must contain at least two nodes"))

    T = _XFOIL_GGCALC_PRECISION
    qopi = T(0.25) / T(pi)
    pi_t = T(pi)
    dzdm = zeros(T, nw)
    dqdm = zeros(T, nw)

    for jo in 1:(nw - 1)
        jp = jo + 1
        jm = jo == 1 ? jo : jo - 1
        jq = jo == nw - 1 ? jp : jp + 1

        dso = sqrt((T(xw[jo]) - T(xw[jp]))^2 + (T(yw[jo]) - T(yw[jp]))^2)
        dso <= 0.0 && continue
        dsio = inv(dso)
        apan = T(apanel[jo])

        rx1 = T(xi) - T(xw[jo])
        ry1 = T(yi) - T(yw[jo])
        rx2 = T(xi) - T(xw[jp])
        ry2 = T(yi) - T(yw[jp])

        sx = (T(xw[jp]) - T(xw[jo])) * dsio
        sy = (T(yw[jp]) - T(yw[jo])) * dsio

        x1 = sx * rx1 + sy * ry1
        x2 = sx * rx2 + sy * ry2
        yy = sx * ry1 - sy * rx1

        rs1 = rx1 * rx1 + ry1 * ry1
        rs2 = rx2 * rx2 + ry2 * ry2
        sgn = eval_on_wake ? one(T) : copysign(one(T), yy)

        if io != jo && rs1 > zero(T)
            g1 = _xfoil_log(rs1)
            t1 = _xfoil_atan2(sgn * x1, sgn * yy) - (T(0.5) - T(0.5) * sgn) * pi_t
        else
            g1 = zero(T)
            t1 = zero(T)
        end

        if io != jp && rs2 > zero(T)
            g2 = _xfoil_log(rs2)
            t2 = _xfoil_atan2(sgn * x2, sgn * yy) - (T(0.5) - T(0.5) * sgn) * pi_t
        else
            g2 = zero(T)
            t2 = zero(T)
        end

        x1i = sx * T(nxi) + sy * T(nyi)
        x2i = x1i
        yyi = sx * T(nyi) - sy * T(nxi)

        x0 = T(0.5) * (x1 + x2)
        rs0 = x0 * x0 + yy * yy
        g0 = _xfoil_log(rs0)
        t0 = _xfoil_atan2(sgn * x0, sgn * yy) - (T(0.5) - T(0.5) * sgn) * pi_t

        dx = x1 - x0
        if abs(dx) <= T(1.0e-15)
            dx = dx >= zero(T) ? T(1.0e-15) : -T(1.0e-15)
        end
        dxinv = inv(dx)
        psum = x0 * (t0 - apan) - x1 * (t1 - apan) + T(0.5) * yy * (g1 - g0)
        pdif =
            (
                (x1 + x0) * psum +
                rs1 * (t1 - apan) -
                rs0 * (t0 - apan) +
                (x0 - x1) * yy
            ) * dxinv

        psx1 = -(t1 - apan)
        psx0 = t0 - apan
        psyy = T(0.5) * (g1 - g0)
        pdx1 = ((x1 + x0) * psx1 + psum + T(2) * x1 * (t1 - apan) - pdif) * dxinv
        pdx0 = ((x1 + x0) * psx0 + psum - T(2) * x0 * (t0 - apan) + pdif) * dxinv
        pdyy = ((x1 + x0) * psyy + T(2) * (x0 - x1 + yy * (t1 - t0))) * dxinv

        dsm = sqrt((T(xw[jp]) - T(xw[jm]))^2 + (T(yw[jp]) - T(yw[jm]))^2)
        dsim = inv(dsm)

        dzdm[jm] += qopi * (-psum * dsim + pdif * dsim)
        dzdm[jo] += qopi * (-psum * dsio - pdif * dsio)
        dzdm[jp] += qopi * (psum * (dsio + dsim) + pdif * (dsio - dsim))

        psni = psx1 * x1i + psx0 * (x1i + x2i) * T(0.5) + psyy * yyi
        pdni = pdx1 * x1i + pdx0 * (x1i + x2i) * T(0.5) + pdyy * yyi
        dqdm[jm] += qopi * (-psni * dsim + pdni * dsim)
        dqdm[jo] += qopi * (-psni * dsio - pdni * dsio)
        dqdm[jp] += qopi * (psni * (dsio + dsim) + pdni * (dsio - dsim))

        dx = x0 - x2
        if abs(dx) <= T(1.0e-15)
            dx = dx >= zero(T) ? T(1.0e-15) : -T(1.0e-15)
        end
        dxinv = inv(dx)
        psum = x2 * (t2 - apan) - x0 * (t0 - apan) + T(0.5) * yy * (g0 - g2)
        pdif =
            (
                (x0 + x2) * psum +
                rs0 * (t0 - apan) -
                rs2 * (t2 - apan) +
                (x2 - x0) * yy
            ) * dxinv

        psx0 = -(t0 - apan)
        psx2 = t2 - apan
        psyy = T(0.5) * (g0 - g2)
        pdx0 = ((x0 + x2) * psx0 + psum + T(2) * x0 * (t0 - apan) - pdif) * dxinv
        pdx2 = ((x0 + x2) * psx2 + psum - T(2) * x2 * (t2 - apan) + pdif) * dxinv
        pdyy = ((x0 + x2) * psyy + T(2) * (x2 - x0 + yy * (t0 - t2))) * dxinv

        dsp = sqrt((T(xw[jq]) - T(xw[jo]))^2 + (T(yw[jq]) - T(yw[jo]))^2)
        dsip = inv(dsp)

        dzdm[jo] += qopi * (-psum * (dsip + dsio) - pdif * (dsip - dsio))
        dzdm[jp] += qopi * (psum * dsio - pdif * dsio)
        dzdm[jq] += qopi * (psum * dsip + pdif * dsip)

        psni = psx0 * (x1i + x2i) * T(0.5) + psx2 * x2i + psyy * yyi
        pdni = pdx0 * (x1i + x2i) * T(0.5) + pdx2 * x2i + pdyy * yyi
        dqdm[jo] += qopi * (-psni * (dsip + dsio) - pdni * (dsip - dsio))
        dqdm[jp] += qopi * (psni * dsio - pdni * dsio)
        dqdm[jq] += qopi * (psni * dsip + pdni * dsip)
    end

    return dzdm, dqdm
end

function _surface_wake_source_influence(
    inv::CaseResult,
    wake,
)
    T = _XFOIL_GGCALC_PRECISION
    geom = _surface_gamma_factorization(inv)
    cache = _wake_geometry_cache(inv)
    xw = T.(wake.xw)
    yw = T.(wake.yw)
    nw = length(xw)
    wake_ap =
        hasproperty(wake, :apanel) && length(getproperty(wake, :apanel)) == nw ?
        T.(getproperty(wake, :apanel)) : _wake_apanel(cache, xw, yw)
    b = zeros(T, geom.n, nw)

    for i in 1:geom.n
        dzdm, _ = _pswlin_source_derivatives(
            0,
            geom.x[i],
            geom.y[i],
            geom.nx[i],
            geom.ny[i],
            xw,
            yw,
            wake_ap;
            eval_on_wake = false,
        )
        b[i, :] .= -dzdm
    end

    if geom.te.sharp
        b[geom.n, :] .= zero(T)
    end

    dij = Matrix{Float64}(undef, geom.n, nw)
    rhs = zeros(T, geom.n + 1)
    for j in 1:nw
        rhs .= zero(T)
        rhs[1:geom.n] .= b[:, j]
        sol = _xfoil_lu_solve!(geom.fac, rhs)
        dij[:, j] .= Float64.(sol[1:geom.n])
    end
    return dij
end

"""
    _qdcalc_full(inv, wake_trace) -> Matrix{Float64}

Build the full (N+NW)×(N+NW) DIJ influence matrix following XFoil's QDCALC.
Returns `dij` where `dij[i,j]` = ∂Qtan(i)/∂σ(j), combining direct source
influence and indirect influence through vorticity.

Rows 1:N    = airfoil surface nodes
Rows N+1:N+NW = wake nodes
Cols 1:N    = airfoil surface sources
Cols N+1:N+NW = wake sources
"""
function _qdcalc_full(inv::CaseResult, wake_trace)
    T = _XFOIL_GGCALC_PRECISION
    geom = _surface_gamma_factorization(inv)
    n = geom.n
    xw = T.(wake_trace.xw)
    yw = T.(wake_trace.yw)
    nxw = T.(wake_trace.nxw)
    nyw = T.(wake_trace.nyw)
    nw = length(xw)
    wake_ap = T.(wake_trace.apw)

    ntot = n + nw
    dij = zeros(T, ntot, ntot)

    # ---- Block (1:N, 1:N): airfoil source influence on airfoil velocity ----
    # Reuse existing _surface_source_influence computation
    bij_surf = zeros(T, n + 1, n)
    for i in 1:n
        _, _, dzdm, _ = _psilin_source_vortex_derivatives(
            i,
            geom.x[i], geom.y[i], geom.nx[i], geom.ny[i],
            geom.x, geom.y, geom.s, geom.apanel;
            scs = geom.te.scs, sds = geom.te.sds,
        )
        for j in 1:n
            bij_surf[i, j] = -dzdm[j]
        end
    end
    # Sharp TE: override row n using bisector point
    if geom.te.sharp
        bwt = T(0.1)
        ag1 = _xfoil_atan2(-geom.yp[1], -geom.xp[1])
        ag2 = _atanc(geom.yp[n], geom.xp[n], ag1)
        abis = T(0.5) * (ag1 + ag2)
        cbis = cos(abis)
        sbis = sin(abis)
        ds1 = sqrt((geom.x[1] - geom.x[2])^2 + (geom.y[1] - geom.y[2])^2)
        ds2 = sqrt((geom.x[n] - geom.x[n - 1])^2 + (geom.y[n] - geom.y[n - 1])^2)
        dsmin = min(ds1, ds2)
        xbis = geom.te.xte - bwt * dsmin * cbis
        ybis = geom.te.yte - bwt * dsmin * sbis
        _, _, _, dqdm_bis = _psilin_source_vortex_derivatives(
            0, xbis, ybis, -sbis, cbis,
            geom.x, geom.y, geom.s, geom.apanel;
            scs = geom.te.scs, sds = geom.te.sds,
        )
        for j in 1:n
            bij_surf[n, j] = -dqdm_bis[j]
        end
    end
    # Back-substitute to get dij(1:N, 1:N)
    rhs = zeros(T, n + 1)
    for j in 1:n
        rhs .= bij_surf[:, j]
        sol = _xfoil_lu_solve!(geom.fac, rhs)
        for i in 1:n
            dij[i, j] = sol[i]
        end
    end

    # ---- Block (1:N, N+1:N+NW): wake source influence on airfoil velocity ----
    # PSWLIN at each airfoil node for wake panels
    bij_wake = zeros(T, n + 1, nw)
    for i in 1:n
        dzdm_w, _ = _pswlin_source_derivatives(
            0, geom.x[i], geom.y[i], geom.nx[i], geom.ny[i],
            xw, yw, wake_ap; eval_on_wake = false,
        )
        for j in 1:nw
            bij_wake[i, j] = -dzdm_w[j]
        end
    end
    # Kutta row: zero wake source influence
    bij_wake[n + 1, :] .= zero(T)
    # Sharp TE: zero wake source influence on gamma-extrapolation row
    if geom.te.sharp
        bij_wake[n, :] .= zero(T)
    end
    # Back-substitute
    for j in 1:nw
        rhs .= bij_wake[:, j]
        sol = _xfoil_lu_solve!(geom.fac, rhs)
        for i in 1:n
            dij[i, n + j] = sol[i]
        end
    end

    # ---- Block (N+1:N+NW, 1:N) and CIJ: wake node velocities ----
    # For each wake node, compute direct influence of airfoil vorticity (CIJ)
    # and direct source influence (DIJ rows), then add indirect path.
    cij = zeros(T, nw, n)
    for iw in 1:nw
        # PSILIN at wake node for airfoil panels (SIGLIN=.TRUE. equivalent)
        _, dqdg_w, _, dqdm_w = _psilin_source_vortex_derivatives(
            0,
            Float64(xw[iw]), Float64(yw[iw]),
            Float64(nxw[iw]), Float64(nyw[iw]),
            geom.x, geom.y, geom.s, geom.apanel;
            scs = geom.te.scs, sds = geom.te.sds,
        )
        for j in 1:n
            cij[iw, j] = dqdg_w[j]     # dQtan(wake)/dGam(airfoil)
            dij[n + iw, j] = dqdm_w[j]  # direct source influence
        end

        # ---- Block (N+iw, N+1:N+NW): wake source on wake velocity ----
        # PSWLIN at wake node for wake panels
        _, dqdm_ww = _pswlin_source_derivatives(
            iw, xw[iw], yw[iw], nxw[iw], nyw[iw],
            xw, yw, wake_ap; eval_on_wake = true,
        )
        for j in 1:nw
            dij[n + iw, n + j] = dqdm_ww[j]
        end
    end

    # ---- Add indirect effect: sources → vorticity → wake velocity ----
    # DIJ(N+iw, j) += Σ_k CIJ(iw,k) * DIJ(k,j)   for airfoil sources j=1:N
    # DIJ(N+iw, N+j) += Σ_k CIJ(iw,k) * BIJ(k,N+j) for wake sources j=1:NW
    # where BIJ(:,N+j) is bij_wake (before back-substitution) — NO, after back-sub
    # Actually: DIJ(K,J) for K=1:N is already dGam(K)/dSig(J) after back-sub
    for iw in 1:nw
        # Airfoil sources: indirect via vorticity
        for j in 1:n
            s = zero(T)
            for k in 1:n
                s += cij[iw, k] * dij[k, j]
            end
            dij[n + iw, j] += s
        end
        # Wake sources: indirect via vorticity
        for j in 1:nw
            s = zero(T)
            for k in 1:n
                s += cij[iw, k] * dij[k, n + j]
            end
            dij[n + iw, n + j] += s
        end
    end

    # ---- First wake node copies TE row ----
    for j in 1:ntot
        dij[n + 1, j] = dij[n, j]
    end

    return Float64.(dij)
end

function _wake_mass_vector(
    wake;
    ueps::Float64 = 1.0e-7,
)
    return [max(p.d, 0.0) * max(p.u, ueps) for p in wake.primary]
end

function _ueset_surface_qvis(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    wake = nothing,
    wake_dij::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    ueps::Float64 = 1.0e-7,
    include_wake::Bool = false,
)
    n = length(inv.qnode)
    size(dij, 1) == n == size(dij, 2) || throw(ArgumentError("surface influence size mismatch"))

    mass, vti = _surface_mass_by_node(inv, topo, upper, lower; ueps = ueps)
    wake_mass = nothing
    if include_wake
        isnothing(wake) && throw(ArgumentError("wake state required when wake_dij is provided"))
        isnothing(wake_dij) && throw(ArgumentError("wake_dij required when wake state is provided"))
        size(wake_dij, 1) == n || throw(ArgumentError("wake influence row mismatch"))
        wake_mass = _wake_mass_vector(wake; ueps = ueps)
        size(wake_dij, 2) == length(wake_mass) ||
            throw(ArgumentError("wake influence column mismatch"))
    end

    qvis = copy(inv.qnode)
    for i in 1:n
        dui = 0.0
        for j in 1:n
            dui += -vti[j] * float(dij[i, j]) * mass[j]
        end
        if !isnothing(wake_mass)
            @inbounds for j in eachindex(wake_mass)
                # XFoil stores all wake source stations on the lower-side BL
                # branch with VTI = -1, so the signed QVIS contribution is +DIJ*MASS.
                dui += float(wake_dij[i, j]) * wake_mass[j]
            end
        end
        qvis[i] = inv.qnode[i] + dui
    end
    return qvis
end

function _force_coefficients_from_cp(airfoil::Airfoil, alpha_deg::Real, cp::AbstractVector{<:Real})
    return _force_coefficients_from_cp(
        solve_inviscid_case(airfoil, alpha_deg),
        alpha_deg,
        cp,
    )
end

function _cyclic_panel_to_node_speed(v_panel::AbstractVector{<:Real})
    m = length(v_panel)
    m >= 2 || throw(ArgumentError("need at least two panel speeds"))
    q = Vector{Float64}(undef, m)
    q[1] = 0.5 * (abs(float(v_panel[m])) + abs(float(v_panel[1])))
    for i in 2:m
        q[i] = 0.5 * (abs(float(v_panel[i - 1])) + abs(float(v_panel[i])))
    end
    return q
end

function _force_coefficients_from_cp(inv::CaseResult, alpha_deg::Real, cp::AbstractVector{<:Real})
    m = length(cp)
    nnode = length(inv.xnode)
    length(inv.ynode) == nnode || throw(ArgumentError("inviscid node array mismatch"))
    m >= 2 || throw(ArgumentError("cp profile needs at least 2 points"))

    q_panel = [sqrt(max(1.0 - float(cpi), 0.0)) for cpi in cp]
    q_nodes =
        if m == nnode
            copy(q_panel)
        elseif m + 1 == nnode
            _panel_to_node_speed(q_panel)[1:nnode]
        else
            throw(ArgumentError("cp length mismatch with inviscid node geometry"))
        end

    cl, cm, _ = _clcalc_from_surface_speed(
        inv.xnode,
        inv.ynode,
        q_nodes,
        alpha_deg;
        xref = 0.25,
        yref = 0.0,
        qinf = 1.0,
        minf = 0.0,
    )
    return cl, cm
end

function _coupled_ue_target(
    inv_surface::SurfaceStations,
    march::SurfaceMarchResult;
    gain::Float64,
    grad_gain::Float64 = 0.8,
    smooth::Float64 = 0.2,
    uemin::Float64 = 1.0e-4,
)
    n = length(inv_surface.x)
    n == length(march.primary) || throw(ArgumentError("coupling profile length mismatch"))
    n >= 2 || throw(ArgumentError("coupling profile needs at least 2 points"))
    raw = Vector{Float64}(undef, n)

    for i in 1:n
        # Avoid over-damping the first few stagnation-adjacent stations where
        # chordwise x/c collapses much faster than the BL march coordinate.
        x = max(max(inv_surface.xc[i], 0.4 * inv_surface.x[i]), 1.0e-5)
        d = max(march.primary[i].d, 0.0)
        block = clamp(d / x, 0.0, 0.9)
        factor = 1.0 / (1.0 + max(gain, 0.0) * block)
        raw[i] = max(uemin, inv_surface.ue[i] * factor)
    end

    # Apply blockage-gradient forcing so outer profile responds to local d* growth/decay.
    gg = max(grad_gain, 0.0)
    if gg > 0.0 && n >= 3
        for i in 2:(n - 1)
            dx = max(inv_surface.x[i + 1] - inv_surface.x[i - 1], 1.0e-8)
            ddx = (march.primary[i + 1].d - march.primary[i - 1].d) / dx
            corr = exp(-gg * ddx)
            corr = clamp(corr, 0.6, 1.4)
            raw[i] = max(uemin, raw[i] * corr)
        end
    end

    sigma = clamp(float(smooth), 0.0, 0.8)
    sigma <= 0.0 && return raw

    a = zeros(Float64, n)
    b = ones(Float64, n)
    c = zeros(Float64, n)
    rhs = copy(raw)

    for i in 2:(n - 1)
        a[i] = -sigma
        b[i] = 1.0 + 2.0 * sigma
        c[i] = -sigma
    end

    # Thomas algorithm for tridiagonal solve.
    for i in 2:n
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        rhs[i] -= w * rhs[i - 1]
    end
    ue = similar(raw)
    ue[n] = rhs[n] / b[n]
    for i in (n - 1):-1:1
        ue[i] = (rhs[i] - c[i] * ue[i + 1]) / b[i]
    end
    for i in 1:n
        ue[i] = max(ue[i], uemin)
    end
    return ue
end

function _interp_u_linear(x1::Real, u1::Real, x2::Real, u2::Real, x::Real)
    dx = float(x2) - float(x1)
    if abs(dx) <= 1.0e-12
        return max(float(u2), 1.0e-4)
    end
    t = clamp((float(x) - float(x1)) / dx, 0.0, 1.0)
    return max((1.0 - t) * float(u1) + t * float(u2), 1.0e-4)
end

function _transition_shear_seed(
    u::Real,
    t::Real,
    d::Real;
    flow::BLFlowParams,
    model::BLModelParams,
)
    tur = blvar_closure(u, t, d; ityp = :turbulent, flow = flow, model = model)
    ctr = initial_ctau(tur.kin.hk2, model)
    return max(ctr * tur.cq2, 1.0e-12)
end

_transition_ctau_seed() = 0.03

"""
XFoil-exact CTR seed at transition: CTR = CTRCON * exp(-CTRCEX/(HK-1)) * CQ.
Uses turbulent closure at the transition point to get CQ (equilibrium shear).
Falls back to constant 0.03 if closure fails.
"""
function _transition_ctau_seed_xfoil(
    u::Real, t::Real, d::Real;
    flow::BLFlowParams, model::BLModelParams,
)
    cl = try
        blvar_closure(u, t, d; ityp = :turbulent, flow = flow, model = model)
    catch
        return 0.03
    end
    hk = cl.kin.hk2
    hk <= 1.0 && return 0.03
    ctr = model.ctrcon * exp(-model.ctrcex / (hk - 1.0))
    st = ctr * cl.cq2
    return max(st, 1.0e-6)
end

function _transition_interp_state(p1::BLPrimaryState, p2::BLPrimaryState, xt::Real)
    x1 = p1.x
    x2 = p2.x
    dx = max(x2 - x1, 1.0e-12)
    w = clamp((float(xt) - x1) / dx, 0.0, 1.0)
    ut = _interp_u_linear(x1, p1.u, x2, p2.u, xt)
    tt = max((1.0 - w) * p1.t + w * p2.t, 1.0e-9)
    dt = max((1.0 - w) * p1.d + w * p2.d, 1.0e-9)
    return ut, tt, dt
end

function _transition_laminar_kinematics(
    x::Real,
    u::Real,
    t::Real,
    d::Real;
    flow::BLFlowParams,
    model::BLModelParams,
)
    p = BLPrimaryState(x, u, t, d; s = 0.0, ampl = 0.0, dw = 0.0)
    s = blsecondary_state(p, :laminar; flow = flow, model = model)
    return p, s
end

function _transition_laminar_partials(
    x::Real,
    u::Real,
    t::Real,
    d::Real;
    flow::BLFlowParams,
    model::BLModelParams,
)
    _, s0 = _transition_laminar_kinematics(x, u, t, d; flow = flow, model = model)

    function fd(value::Float64, update)
        step = 1.0e-6 * max(abs(value), 1.0e-4)
        plus = update(step)
        minus = update(-step)
        if !isnothing(plus) && !isnothing(minus)
            return ((plus.hk - minus.hk) / (2.0 * step), (plus.rt - minus.rt) / (2.0 * step))
        elseif !isnothing(plus)
            return ((plus.hk - s0.hk) / step, (plus.rt - s0.rt) / step)
        elseif !isnothing(minus)
            return ((s0.hk - minus.hk) / step, (s0.rt - minus.rt) / step)
        end
        return (NaN, NaN)
    end

    function eval_shift(du::Float64, dtv::Float64, dd::Float64)
        uf = max(float(u) + du, 1.0e-7)
        tf = max(float(t) + dtv, 1.0e-9)
        df = max(float(d) + dd, 1.0e-9)
        _, s = _transition_laminar_kinematics(x, uf, tf, df; flow = flow, model = model)
        return s
    end

    hk_t, rt_t = fd(float(t), h -> eval_shift(0.0, h, 0.0))
    hk_d, _ = fd(float(d), h -> eval_shift(0.0, 0.0, h))
    hk_u, rt_u = fd(float(u), h -> eval_shift(h, 0.0, 0.0))

    return (sec = s0, hk_t = hk_t, hk_d = hk_d, hk_u = hk_u, rt_t = rt_t, rt_u = rt_u)
end

function _trchek2_core_state(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ampl2::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
    xiforc::Float64 = Inf,
)
    x1 = max(p1.x, 1.0e-9)
    x2 = max(p2.x, x1 + 1.0e-6)
    dx = x2 - x1
    u1 = max(p1.u, 1.0e-7)
    u2 = max(p2.u, 1.0e-7)
    t1 = max(p1.t, 1.0e-9)
    t2 = max(p2.t, 1.0e-9)
    d1 = max(p1.d, 1.0e-9)
    d2 = max(p2.d, 1.0e-9)
    a1 = max(p1.ampl, 0.0)
    a2 = float(ampl2)

    _, sec1 = _transition_laminar_kinematics(x1, u1, t1, d1; flow = flow, model = model)
    _, sec2 = _transition_laminar_kinematics(x2, u2, t2, d2; flow = flow, model = model)

    if a2 <= ncrit
        amplt = a2
        amplt_a2 = 1.0
        sfa = 1.0
        sfa_a1 = 0.0
        sfa_a2 = 0.0
    else
        amplt = ncrit
        amplt_a2 = 0.0
        den = max(a2 - a1, 1.0e-12)
        sfa = (amplt - a1) / den
        sfa_a1 = (sfa - 1.0) / den
        sfa_a2 = -sfa / den
    end

    if isfinite(xiforc) && xiforc < x2
        sfx = (xiforc - x1) / dx
        sfx_x1 = (sfx - 1.0) / dx
        sfx_x2 = -sfx / dx
        sfx_xf = 1.0 / dx
    else
        sfx = 1.0
        sfx_x1 = 0.0
        sfx_x2 = 0.0
        sfx_xf = 0.0
    end

    if sfa < sfx
        wf2 = sfa
        wf2_a1 = sfa_a1
        wf2_a2 = sfa_a2
        wf2_x1 = 0.0
        wf2_x2 = 0.0
        wf2_xf = 0.0
    else
        wf2 = sfx
        wf2_a1 = 0.0
        wf2_a2 = 0.0
        wf2_x1 = sfx_x1
        wf2_x2 = sfx_x2
        wf2_xf = sfx_xf
    end
    wf1 = 1.0 - wf2
    wf1_a1 = -wf2_a1
    wf1_a2 = -wf2_a2
    wf1_x1 = -wf2_x1
    wf1_x2 = -wf2_x2
    wf1_xf = -wf2_xf

    xt = x1 * wf1 + x2 * wf2
    tt = t1 * wf1 + t2 * wf2
    dt = d1 * wf1 + d2 * wf2
    ut = u1 * wf1 + u2 * wf2

    _, sec_t = _transition_laminar_kinematics(xt, ut, tt, dt; flow = flow, model = model)
    pt = BLPrimaryState(xt, ut, tt, dt; s = 0.0, ampl = amplt, dw = 0.0)

    ax =
        axset(
            sec1.hk,
            t1,
            sec1.rt,
            a1,
            sec_t.hk,
            tt,
            sec_t.rt,
            amplt,
            ncrit;
            idampv = idampv,
        )
    res = a2 - a1 - ax * dx

    trfree = a2 >= ncrit
    trforc = isfinite(xiforc) && xiforc > x1 && xiforc <= x2
    if trfree && trforc
        trforc = xiforc < xt
        trfree = xiforc >= xt
    end

    return (
        ampl2 = a2,
        amplt = amplt,
        ax = ax,
        res = res,
        xt = xt,
        pt = pt,
        sec1 = sec1,
        sec2 = sec2,
        trfree = trfree,
        trforc = trforc,
        transition = trfree || trforc,
        forced_x = xiforc,
        wf1 = wf1,
        wf2 = wf2,
        wf1_a1 = wf1_a1,
        wf1_a2 = wf1_a2,
        wf1_x1 = wf1_x1,
        wf1_x2 = wf1_x2,
        wf1_xf = wf1_xf,
        wf2_a1 = wf2_a1,
        wf2_a2 = wf2_a2,
        wf2_x1 = wf2_x1,
        wf2_x2 = wf2_x2,
        wf2_xf = wf2_xf,
    )
end

function _trchek2_iteration_state(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ampl2::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
    xiforc::Float64 = Inf,
)
    base = _trchek2_core_state(
        p1,
        p2,
        ampl2,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
        xiforc = xiforc,
    )

    h = 1.0e-6 * max(abs(float(ampl2)), 1.0)
    plus =
        try
            _trchek2_core_state(
                p1,
                p2,
                float(ampl2) + h,
                ncrit;
                flow = flow,
                model = model,
                idampv = idampv,
                xiforc = xiforc,
            )
        catch
            nothing
        end
    minus =
        try
            _trchek2_core_state(
                p1,
                p2,
                float(ampl2) - h,
                ncrit;
                flow = flow,
                model = model,
                idampv = idampv,
                xiforc = xiforc,
            )
        catch
            nothing
        end

    res_a2 =
        if !isnothing(plus) && !isnothing(minus)
            (plus.res - minus.res) / (2.0 * h)
        elseif !isnothing(plus)
            (plus.res - base.res) / h
        elseif !isnothing(minus)
            (base.res - minus.res) / h
        else
            NaN
        end
    xt_a2 =
        if !isnothing(plus) && !isnothing(minus)
            (plus.xt - minus.xt) / (2.0 * h)
        elseif !isnothing(plus)
            (plus.xt - base.xt) / h
        elseif !isnothing(minus)
            (base.xt - minus.xt) / h
        else
            NaN
        end

    return merge(base, (res_a2 = res_a2, xt_a2 = xt_a2))
end

function _trchek2_interval(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
    xiforc::Float64 = Inf,
    tol::Float64 = 5.0e-5,
    max_iter::Int = 30,
)
    x1 = max(p1.x, 1.0e-9)
    x2 = max(p2.x, x1 + 1.0e-6)
    dx = x2 - x1

    _, sec1 = _transition_laminar_kinematics(x1, p1.u, p1.t, p1.d; flow = flow, model = model)
    _, sec2 = _transition_laminar_kinematics(x2, p2.u, p2.t, p2.d; flow = flow, model = model)
    a2_seed = max(p2.ampl, ncrit)
    ax0 =
        axset(
            sec1.hk,
            max(p1.t, 1.0e-9),
            sec1.rt,
            max(p1.ampl, 0.0),
            sec2.hk,
            max(p2.t, 1.0e-9),
            sec2.rt,
            a2_seed,
            ncrit;
            idampv = idampv,
        )[1]
    ampl2 = max(p1.ampl, 0.0) + ax0 * dx
    converged = false
    state = _trchek2_iteration_state(
        p1,
        p2,
        ampl2,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
        xiforc = xiforc,
    )

    for _ in 1:max_iter
        state = _trchek2_iteration_state(
            p1,
            p2,
            ampl2,
            ncrit;
            flow = flow,
            model = model,
            idampv = idampv,
            xiforc = xiforc,
        )
        if !isfinite(state.ax) || state.ax <= 0.0
            converged = true
            break
        end

        if !isfinite(state.res) || !isfinite(state.res_a2) || abs(state.res_a2) <= 1.0e-12
            break
        end

        da2 = -state.res / state.res_a2
        rlx = 1.0
        dxt = state.xt_a2 * da2
        if isfinite(dxt) && rlx * abs(dxt / dx) > 0.05
            rlx = 0.05 * abs(dx / dxt)
        end
        if isfinite(da2) && rlx * abs(da2) > 1.0
            rlx = min(rlx, abs(1.0 / da2))
        end
        if abs(da2) < tol
            converged = true
            break
        end

        anext = ampl2 + rlx * da2
        if (ampl2 > ncrit && anext < ncrit) || (ampl2 < ncrit && anext > ncrit)
            ampl2 = ncrit
        else
            ampl2 = anext
        end
    end

    state = _trchek2_iteration_state(
        p1,
        p2,
        ampl2,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
        xiforc = xiforc,
    )

    pt =
        if state.trforc && isfinite(xiforc)
            ut, tt, dt = _transition_interp_state(p1, p2, xiforc)
            BLPrimaryState(xiforc, ut, tt, dt; s = 0.0, ampl = ncrit, dw = 0.0)
        else
            state.pt
        end

    return (
        ampl2 = state.ampl2,
        xt = state.trforc && isfinite(xiforc) ? xiforc : state.xt,
        pt = pt,
        transition = state.transition,
        trfree = state.trfree,
        trforc = state.trforc,
        ok = converged || abs(state.res) <= max(5.0 * tol, 1.0e-4),
        residual = state.res,
    )
end

function _transition_growth_residual(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    sec1::BLSecondaryState,
    xt::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
)
    ut, tt, dt = _transition_interp_state(p1, p2, xt)
    pt = BLPrimaryState(xt, ut, tt, dt; s = 0.0, ampl = ncrit, dw = 0.0)
    st = blsecondary_state(pt, :laminar; flow = flow, model = model)
    ax = axset(
        sec1.hk,
        p1.t,
        sec1.rt,
        p1.ampl,
        st.hk,
        pt.t,
        st.rt,
        ncrit,
        ncrit;
        idampv = idampv,
    )
    return p1.ampl + ax * (pt.x - p1.x) - ncrit
end

function _solve_transition_location_interp(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
    xt_guess::Float64 = NaN,
    tol::Float64 = 1.0e-4,
    max_iter::Int = 18,
)
    x1 = p1.x
    x2 = max(p2.x, x1 + 1.0e-6)
    dx = x2 - x1
    epsx = max(1.0e-8, 1.0e-4 * dx)
    xl = x1 + epsx
    xr = x2 - epsx
    if !(xl < xr)
        return x1 + 0.5 * dx, false
    end

    sec1 = blsecondary_state(p1, :laminar; flow = flow, model = model)
    fres(x) = _transition_growth_residual(
        p1,
        p2,
        sec1,
        x,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
    )

    fl = fres(xl)
    fr = fres(xr)
    if isfinite(fl) && isfinite(fr) && fl * fr <= 0.0
        for _ in 1:max_iter
            xm = 0.5 * (xl + xr)
            fm = fres(xm)
            if !isfinite(fm)
                break
            end
            if abs(fm) <= tol || (xr - xl) <= 1.0e-6 * dx
                return xm, true
            end
            if fl * fm <= 0.0
                xr = xm
                fr = fm
            else
                xl = xm
                fl = fm
            end
        end
        xt = 0.5 * (xl + xr)
        ft = fres(xt)
        return xt, isfinite(ft) && abs(ft) <= 5.0 * tol
    end

    xt =
        if isfinite(xt_guess)
            clamp(xt_guess, xl, xr)
        else
            x1 + 0.5 * dx
        end

    for _ in 1:max_iter
        f0 = fres(xt)
        if !isfinite(f0)
            return x1 + 0.5 * dx, false
        end
        abs(f0) <= tol && return xt, true

        h = max(1.0e-6 * dx, 1.0e-8)
        xm = clamp(xt - h, xl, xr)
        xp = clamp(xt + h, xl, xr)
        fm = fres(xm)
        fp = fres(xp)

        xtn = 0.5 * (xl + xr)
        if isfinite(fm) && isfinite(fp) && abs(xp - xm) > 1.0e-12
            dfdx = (fp - fm) / (xp - xm)
            if isfinite(dfdx) && abs(dfdx) > 1.0e-12
                step = clamp(-f0 / dfdx, -0.4 * dx, 0.4 * dx)
                xtn = clamp(xt + step, xl, xr)
            end
        end

        fn = fres(xtn)
        if !isfinite(fn) || abs(fn) > abs(f0)
            xtn = clamp(0.5 * (xt + xtn), xl, xr)
            fn = fres(xtn)
        end
        if !isfinite(fn)
            return xt, false
        end

        xt = xtn
        abs(fn) <= tol && return xt, true
    end

    fend = fres(xt)
    return xt, isfinite(fend) && abs(fend) <= 5.0 * tol
end

function _transition_laminar_endpoint_hint(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
)
    if p2.s <= 1.0e-12
        return BLPrimaryState(
            p2.x,
            p2.u,
            p2.t,
            p2.d;
            s = 0.0,
            ampl = max(p2.ampl, p1.ampl, 0.0),
            dw = p2.dw,
        )
    end

    z_init =
        try
            _interval_seed_unknowns(:laminar, p2, p2.x, p2.u; flow = flow, model = model)
        catch
            nothing
        end
    p2_lam =
        try
            _solve_interval(
                :laminar,
                p1,
                p2.x,
                p2.u;
                flow = flow,
                model = model,
                acrit = ncrit,
                tol = 1.0e-6,
                max_iter = 20,
                z_init = z_init,
                dw = p2.dw,
            )[1]
        catch
            BLPrimaryState(
                p2.x,
                p2.u,
                p2.t,
                p2.d;
                s = 0.0,
                ampl = max(ncrit, p1.ampl, 0.0),
                dw = p2.dw,
            )
        end
    return BLPrimaryState(
        p2_lam.x,
        p2_lam.u,
        p2_lam.t,
        p2_lam.d;
        s = 0.0,
        ampl = max(p2_lam.ampl, p1.ampl, 0.0),
        dw = p2_lam.dw,
    )
end

function _eval_transition_interval(
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    idampv::Int = 0,
    hupwt::Float64 = 1.0,
    xt_guess::Float64 = NaN,
)
    p2_lam = _transition_laminar_endpoint_hint(p1, p2, ncrit; flow = flow, model = model)
    tr = _trchek2_interval(
        p1,
        p2_lam,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
    )
    xt = tr.xt
    xt_ok = tr.ok && tr.transition

    if !isfinite(xt) || !tr.transition
        x1 = p1.x
        x2 = max(p2_lam.x, x1 + 1.0e-6)
        epsx = max(1.0e-8, 1.0e-4 * (x2 - x1))
        if isfinite(xt_guess)
            xt = clamp(xt_guess, x1 + epsx, x2 - epsx)
            ut, tt, dt = _transition_interp_state(p1, p2_lam, xt)
            p_tr_l = BLPrimaryState(xt, ut, tt, dt; s = 0.0, ampl = ncrit, dw = 0.0)
            s1_l = blsecondary_state(p1, :laminar; flow = flow, model = model)
            s_tr_l = blsecondary_state(p_tr_l, :laminar; flow = flow, model = model)
            r_l = bldif_residuals(
                :laminar,
                p1,
                p_tr_l,
                s1_l,
                s_tr_l;
                model = model,
                acrit = ncrit,
                idampv = idampv,
                hupwt = hupwt,
            )

            sseed = _transition_ctau_seed()
            p_tr_t = BLPrimaryState(p_tr_l.x, p_tr_l.u, p_tr_l.t, p_tr_l.d; s = sseed, ampl = 0.0, dw = 0.0)
            s_tr_t = blsecondary_state(p_tr_t, :turbulent; flow = flow, model = model)
            s2_t = blsecondary_state(p2, :turbulent; flow = flow, model = model)
            r_t = bldif_residuals(
                :turbulent,
                p_tr_t,
                p2,
                s_tr_t,
                s2_t;
                model = model,
                acrit = ncrit,
                idampv = idampv,
                hupwt = hupwt,
            )
            rez = [r_t[1], r_l[2] + r_t[2], r_l[3] + r_t[3]]
            return rez, p2, s2_t, xt, false
        end

        bad = [1.0e3, 1.0e3, 1.0e3]
        s2 = blsecondary_state(p2, :turbulent; flow = flow, model = model)
        return bad, p2, s2, NaN, false
    end

    p_tr_l = tr.pt
    s1_l = blsecondary_state(p1, :laminar; flow = flow, model = model)
    s_tr_l = blsecondary_state(p_tr_l, :laminar; flow = flow, model = model)
    r_l = bldif_residuals(
        :laminar,
        p1,
        p_tr_l,
        s1_l,
        s_tr_l;
        model = model,
        acrit = ncrit,
        idampv = idampv,
        hupwt = hupwt,
    )

    sseed = _transition_ctau_seed()
    p_tr_t = BLPrimaryState(p_tr_l.x, p_tr_l.u, p_tr_l.t, p_tr_l.d; s = sseed, ampl = 0.0, dw = 0.0)
    s_tr_t = blsecondary_state(p_tr_t, :turbulent; flow = flow, model = model)
    s2_t = blsecondary_state(p2, :turbulent; flow = flow, model = model)
    r_t = bldif_residuals(
        :turbulent,
        p_tr_t,
        p2,
        s_tr_t,
        s2_t;
        model = model,
        acrit = ncrit,
        idampv = idampv,
        hupwt = hupwt,
    )

    rez = [r_t[1], r_l[2] + r_t[2], r_l[3] + r_t[3]]
    return rez, p2, s2_t, xt, xt_ok
end

function _solve_transition_interval_coupled(
    p1::BLPrimaryState,
    p2_lam_hint::BLPrimaryState,
    x2::Real,
    u2::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    idampv::Int = 0,
    hupwt::Float64 = 1.0,
)
    x1 = p1.x
    x2f = max(float(x2), x1 + 1.0e-6)
    u2f = max(float(u2), 1.0e-4)
    epsx = max(1.0e-8, 1.0e-4 * (x2f - x1))

    sseed = _transition_ctau_seed()
    z = [max(sseed, 1.0e-12), max(p2_lam_hint.t, 1.0e-9), max(p2_lam_hint.d, 1.0e-9)]
    best_z = copy(z)
    best_norm = Inf

    tr0 = _trchek2_interval(
        p1,
        p2_lam_hint,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
    )
    frac = (ncrit - p1.ampl) / max(p2_lam_hint.ampl - p1.ampl, 1.0e-8)
    xt_guess =
        if tr0.transition && isfinite(tr0.xt)
            clamp(tr0.xt, x1 + epsx, x2f - epsx)
        else
            clamp(x1 + frac * (x2f - x1), x1 + epsx, x2f - epsx)
        end
    best_xt = xt_guess

    for _ in 1:max_iter
        p2 = BLPrimaryState(x2f, u2f, z[2], z[3]; s = max(z[1], 1.0e-12), ampl = 0.0, dw = 0.0)
        sys = _blsys_interval_system(
            :transition,
            p1,
            p2;
            flow = flow,
            model = model,
            acrit = ncrit,
            idampv = idampv,
            hupwt = hupwt,
            xt_guess = xt_guess,
        )
        r = sys.residual
        s2 = sys.s2
        xt = sys.transition_x
        xt_ok = sys.transition_ok
        rnorm = maximum(abs, r)
        if xt_ok && isfinite(rnorm) && rnorm < best_norm
            best_norm = rnorm
            best_z .= z
            best_xt = xt
        end
        if xt_ok && isfinite(rnorm) && rnorm <= tol
            return p2, s2, xt, true, rnorm
        end

        J = copy(sys.vs2[1:3, 1:3])

        if any(!isfinite, J)
            break
        end

        dz =
            try
                J \ sys.vsrez[1:3]
            catch
                break
            end

        accepted = false
        lambda = 1.0
        for _ in 1:8
            ztrial = z .+ lambda .* dz
            ztrial[1] = max(ztrial[1], 1.0e-12)
            ztrial[2] = max(ztrial[2], 1.0e-9)
            ztrial[3] = max(ztrial[3], 1.0e-9)

            p2trial = BLPrimaryState(
                x2f,
                u2f,
                ztrial[2],
                ztrial[3];
                s = ztrial[1],
                ampl = 0.0,
                dw = 0.0,
            )
            rtrial, _, _, xt_trial, xt_ok_trial = _eval_transition_interval(
                p1,
                p2trial,
                ncrit;
                flow = flow,
                model = model,
                idampv = idampv,
                hupwt = hupwt,
                xt_guess = xt,
            )
            ntrial = maximum(abs, rtrial)
            if xt_ok_trial && isfinite(ntrial) && ntrial < rnorm
                z .= ztrial
                xt_guess = xt_trial
                accepted = true
                break
            end
            lambda *= 0.5
        end

        accepted || break
    end

    pbest = BLPrimaryState(x2f, u2f, best_z[2], best_z[3]; s = max(best_z[1], 1.0e-12), ampl = 0.0, dw = 0.0)
    rbest, _, sbest, xtbest, xtok = _eval_transition_interval(
        p1,
        pbest,
        ncrit;
        flow = flow,
        model = model,
        idampv = idampv,
        hupwt = hupwt,
        xt_guess = best_xt,
    )
    rnorm = maximum(abs, rbest)
    ok = xtok && isfinite(rnorm) && rnorm <= max(5.0 * tol, 1.0e-4)
    return pbest, sbest, xtbest, ok, rnorm
end

function _transition_interval_step_split(
    p1::BLPrimaryState,
    x2::Real,
    u2::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
)
    x1 = p1.x
    x2f = max(float(x2), x1 + 1.0e-6)
    u2f = max(float(u2), 1.0e-4)

    # First solve full interval as laminar to confirm transition crossing.
    p2_l, s2_l, ok_l, _ = _solve_interval(
        :laminar,
        p1,
        x2f,
        u2f;
        flow = flow,
        model = model,
        acrit = ncrit,
        tol = tol,
        max_iter = max_iter,
    )
    p2_l.ampl < ncrit && return p2_l, s2_l, NaN, ok_l

    tr = _trchek2_interval(
        p1,
        p2_l,
        ncrit;
        flow = flow,
        model = model,
    )
    (!tr.transition || !isfinite(tr.xt)) && return p2_l, s2_l, NaN, ok_l

    p_tr = tr.pt
    s_tr_lam = blsecondary_state(p_tr, :laminar; flow = flow, model = model)
    sseed = _transition_ctau_seed()
    p_tr_t = BLPrimaryState(p_tr.x, p_tr.u, p_tr.t, p_tr.d; s = sseed, ampl = 0.0, dw = 0.0)

    p2_t, s2_t, ok_2, _ = _solve_interval(
        :turbulent,
        p_tr_t,
        x2f,
        u2f;
        flow = flow,
        model = model,
        acrit = ncrit,
        tol = tol,
        max_iter = max_iter,
    )
    p2_t = BLPrimaryState(p2_t.x, p2_t.u, p2_t.t, p2_t.d; s = max(p2_t.s, 1.0e-12), ampl = 0.0, dw = p2_t.dw)
    s2_t = blsecondary_state(p2_t, :turbulent; flow = flow, model = model)

    return p2_t, s2_t, tr.xt, tr.ok && ok_l && ok_2
end

function _transition_interval_step(
    p1::BLPrimaryState,
    x2::Real,
    u2::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    p2_l_hint::Union{Nothing,BLPrimaryState} = nothing,
)
    x1 = p1.x
    x2f = max(float(x2), x1 + 1.0e-6)
    u2f = max(float(u2), 1.0e-4)

    p2_l, s2_l, ok_l =
        if isnothing(p2_l_hint)
            p2_lc, s2_lc, ok_lc, _ = _solve_interval(
                :laminar,
                p1,
                x2f,
                u2f;
                flow = flow,
                model = model,
                acrit = ncrit,
                tol = tol,
                max_iter = max_iter,
            )
            (p2_lc, s2_lc, ok_lc)
        else
            p2_lc = p2_l_hint
            s2_lc = blsecondary_state(p2_lc, :laminar; flow = flow, model = model)
            (p2_lc, s2_lc, true)
        end

    p2_l.ampl < ncrit && return p2_l, s2_l, NaN, ok_l

    p2_c, s2_c, xt_c, ok_c, _ = _solve_transition_interval_coupled(
        p1,
        p2_l,
        x2f,
        u2f,
        ncrit;
        flow = flow,
        model = model,
        tol = tol,
        max_iter = max_iter,
    )

    if ok_c &&
       isfinite(xt_c) &&
       xt_c > p1.x &&
       xt_c < x2f &&
       all(isfinite, (p2_c.s, p2_c.t, p2_c.d))
        p2_c = BLPrimaryState(p2_c.x, p2_c.u, p2_c.t, p2_c.d; s = max(p2_c.s, 1.0e-12), ampl = 0.0, dw = p2_c.dw)
        s2_c = blsecondary_state(p2_c, :turbulent; flow = flow, model = model)
        return p2_c, s2_c, xt_c, true
    end

    return _transition_interval_step_split(
        p1,
        x2f,
        u2f,
        ncrit;
        flow = flow,
        model = model,
        tol = tol,
        max_iter = max_iter,
    )
end

function _seeded_transition_xsi(surface::SurfaceMarchResult, stations::SurfaceStations, i::Integer)
    xtr = surface.transition_x
    isfinite(xtr) || return NaN
    1 <= i < length(stations.x) || throw(BoundsError(stations.x, i))

    x1 = stations.x[i]
    x2 = stations.x[i + 1]
    return clamp(xtr, min(x1, x2), max(x1, x2))
end

function _seeded_transition_interval_step(
    p1::BLPrimaryState,
    x2::Real,
    u2::Real,
    xt::Real,
    ncrit::Float64;
    flow::BLFlowParams,
    model::BLModelParams,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
)
    x1 = p1.x
    x2f = max(float(x2), x1 + 1.0e-6)
    u2f = max(float(u2), 1.0e-4)
    xtf = clamp(float(xt), x1 + 1.0e-6, x2f - 1.0e-6)
    utf = _interp_u_linear(x1, p1.u, x2f, u2f, xtf)

    p_tr_l, _, ok_l, _ = _solve_interval(
        :laminar,
        p1,
        xtf,
        utf;
        flow = flow,
        model = model,
        acrit = ncrit,
        tol = tol,
        max_iter = max_iter,
    )
    p_tr_l = BLPrimaryState(xtf, utf, p_tr_l.t, p_tr_l.d; s = 0.0, ampl = max(p_tr_l.ampl, 0.0), dw = p_tr_l.dw)

    sseed = _transition_ctau_seed()
    p_tr_t = BLPrimaryState(xtf, utf, p_tr_l.t, p_tr_l.d; s = sseed, ampl = 0.0, dw = p_tr_l.dw)

    p2_t, s2_t, ok_t, _ = _solve_interval(
        :turbulent,
        p_tr_t,
        x2f,
        u2f;
        flow = flow,
        model = model,
        acrit = ncrit,
        tol = tol,
        max_iter = max_iter,
    )
    p2_t = BLPrimaryState(p2_t.x, p2_t.u, p2_t.t, p2_t.d; s = max(p2_t.s, 1.0e-12), ampl = 0.0, dw = p2_t.dw)
    s2_t = blsecondary_state(p2_t, :turbulent; flow = flow, model = model)
    return p2_t, s2_t, xtf, ok_l && ok_t
end

function _initial_primary(x1::Real, ue1::Real, reynolds::Real)
    x = max(float(x1), 1.0e-5)
    ue = max(float(ue1), 0.02)
    re = max(float(reynolds), 1.0e3)

    # Match XFoil MRCHUE's similarity-station initialization exactly.
    bule = 1.0
    ucon = ue / x^bule
    tsq = 0.45 / max(ucon * (5.0 * bule + 1.0) * re, 1.0e-12) * x^(1.0 - bule)
    t = max(sqrt(max(tsq, 1.0e-14)), 1.0e-7)
    d = max(2.2 * t, 2.0e-7)
    return BLPrimaryState(x, ue, t, d; s = 0.0, ampl = 0.0, dw = 0.0)
end

function _machsq_from_ue(ue::Real; flow::BLFlowParams)
    uef = max(float(ue), 1.0e-7)
    denom = flow.gm1bl * (1.0 - 0.5 * uef^2 * flow.hstinv)
    abs(denom) < 1.0e-12 && (denom = sign(denom + 1.0e-12) * 1.0e-12)
    return uef^2 * flow.hstinv / denom
end

function _hkin_h_derivative(msq::Real)
    return 1.0 / (1.0 + 0.113 * float(msq))
end

function _dslim_dstr(
    dstr::Real,
    th::Real,
    ue::Real;
    flow::BLFlowParams,
    hklim::Float64,
)
    thf = max(float(th), 1.0e-12)
    dstrf = max(float(dstr), 1.0e-12)
    msq = _machsq_from_ue(ue; flow = flow)
    hk = hkin(dstrf / thf, msq)
    hk_h = max(_hkin_h_derivative(msq), 1.0e-12)
    dh = max(0.0, hklim - hk) / hk_h
    return dstrf + dh * thf
end

function _mrchue_hmax(mode::Symbol)
    mode in (:laminar, :similarity) && return 3.8
    return 2.5
end

function _mrchue_hklim(mode::Symbol)
    mode == :wake && return 1.00005
    return 1.02
end

function _mrchue_htarg(
    mode::Symbol,
    p1::BLPrimaryState,
    s1::BLSecondaryState,
    p2::BLPrimaryState;
    transition_x::Float64 = NaN,
)
    dx = max(p2.x - p1.x, 0.0)
    t1 = max(p1.t, 1.0e-12)
    hk1 = s1.hk

    if mode in (:laminar, :similarity)
        return max(hk1 + 0.03 * dx / t1, _mrchue_hmax(mode))
    elseif mode == :transition && isfinite(transition_x)
        xt = clamp(transition_x, p1.x, p2.x)
        return max(hk1 + (0.03 * (xt - p1.x) - 0.15 * (p2.x - xt)) / t1, 2.5)
    elseif mode == :wake
        const_term = 0.03 * dx / t1
        hk2 = hk1
        for _ in 1:3
            numer = hk2 + const_term * (hk2 - 1.0)^3 - hk1
            denom = 1.0 + 3.0 * const_term * (hk2 - 1.0)^2
            hk2 -= numer / max(denom, 1.0e-12)
        end
        return max(hk2, 1.01)
    end

    return max(hk1 - 0.15 * dx / t1, _mrchue_hmax(mode))
end

function _interval_hk_derivatives(
    p::BLPrimaryState,
    mode::Symbol;
    flow::BLFlowParams,
    model::BLModelParams,
)
    closure_mode = _mode_closure(mode)
    base = blsecondary_state(p, closure_mode; flow = flow, model = model)
    vals = Float64[]
    for col in 2:4
        step = _interval_fd_step(_interval_column_value(p, closure_mode, col), col)
        plus =
            try
                blsecondary_state(
                    _interval_column_assign(p, closure_mode, col, _interval_column_value(p, closure_mode, col) + step),
                    closure_mode;
                    flow = flow,
                    model = model,
                ).hk
            catch
                NaN
            end
        minus =
            try
                blsecondary_state(
                    _interval_column_assign(p, closure_mode, col, _interval_column_value(p, closure_mode, col) - step),
                    closure_mode;
                    flow = flow,
                    model = model,
                ).hk
            catch
                NaN
            end
        deriv =
            if isfinite(plus) && isfinite(minus)
                (plus - minus) / (2.0 * step)
            elseif isfinite(plus)
                (plus - base.hk) / step
            elseif isfinite(minus)
                (base.hk - minus) / step
            else
                NaN
            end
        push!(vals, deriv)
    end
    return (t = vals[1], d = vals[2], u = vals[3])
end

function _mrchue_delta_max(
    mode::Symbol,
    delta::AbstractVector{<:Real},
    p::BLPrimaryState;
    inverse::Bool,
)
    dmax = max(abs(delta[2] / max(p.t, 1.0e-12)), abs(delta[3] / max(p.d, 1.0e-12)))
    if inverse
        dmax = max(dmax, abs(delta[4] / max(p.u, 1.0e-8)))
    end
    if mode in (:laminar, :similarity)
        dmax = max(dmax, abs(delta[1] / 10.0))
    else
        dmax = max(dmax, abs(delta[1] / max(p.s, 1.0e-12)))
    end
    return dmax
end

function _mrchdu_delta_max(
    mode::Symbol,
    delta::AbstractVector{<:Real},
    p::BLPrimaryState,
)
    dmax = max(
        abs(delta[2] / max(p.t, 1.0e-12)),
        abs(delta[3] / max(p.d, 1.0e-12)),
        abs(delta[4] / max(p.u, 1.0e-8)),
    )
    if mode in (:laminar, :similarity)
        dmax = max(dmax, abs(delta[1] / 10.0))
    else
        dmax = max(dmax, abs(delta[1] / (10.0 * max(p.s, 1.0e-12))))
    end
    return dmax
end

function _apply_interval_delta(
    mode::Symbol,
    x2::Float64,
    p::BLPrimaryState,
    delta::AbstractVector{<:Real},
    rlx::Float64;
    flow::BLFlowParams,
    dw::Real = p.dw,
)
    z = _pack_unknowns(mode, p)
    ztrial = z .+ rlx .* delta[1:3]
    _clamp_interval_unknowns!(mode, ztrial)
    utrial = max(p.u + rlx * delta[4], 1.0e-7)
    ptrial = _unpack_primary(mode, x2, utrial, ztrial; dw = dw)
    dlim = _dslim_dstr(ptrial.d, ptrial.t, ptrial.u; flow = flow, hklim = _mrchue_hklim(mode))
    ptrial = _interval_column_assign(ptrial, _interval_unknown_mode(mode, 2), 3, dlim)
    return ptrial
end

function _mode_closure(mode::Symbol)
    if mode == :similarity
        return :laminar
    elseif mode == :transition
        return :turbulent
    elseif mode in (:laminar, :turbulent, :wake)
        return mode
    else
        throw(ArgumentError("invalid BL mode: $mode"))
    end
end

function _pack_unknowns(mode::Symbol, p::BLPrimaryState)
    if mode in (:laminar, :similarity)
        return [p.ampl, p.t, p.d]
    else
        return [p.s, p.t, p.d]
    end
end

function _unpack_primary(
    mode::Symbol,
    x::Real,
    u::Real,
    z::AbstractVector{<:Real};
    dw::Real = 0.0,
)
    z1 = float(z[1])
    t = max(float(z[2]), 1.0e-9)
    d = max(float(z[3]), 1.0e-9)
    dwf = mode == :wake ? max(float(dw), 0.0) : 0.0

    if mode in (:laminar, :similarity)
        ampl = max(z1, 0.0)
        return BLPrimaryState(x, u, t, d; s = 0.0, ampl = ampl, dw = 0.0)
    else
        s = max(z1, 1.0e-12)
        return BLPrimaryState(x, u, t, d; s = s, ampl = 0.0, dw = dwf)
    end
end

function _interval_unknown_mode(mode::Symbol, side::Integer)
    side in (1, 2) || throw(ArgumentError("interval side must be 1 or 2"))
    if mode == :transition
        return side == 1 ? :laminar : :turbulent
    elseif mode == :similarity
        return :similarity
    elseif mode in (:laminar, :turbulent, :wake)
        return mode
    end
    throw(ArgumentError("invalid interval mode: $mode"))
end

function _normalize_interval_primary(p::BLPrimaryState, mode::Symbol)
    x = max(p.x, 1.0e-9)
    u = max(p.u, 1.0e-7)
    t = max(p.t, 1.0e-9)
    d = max(p.d, 1.0e-9)
    if mode in (:laminar, :similarity)
        return BLPrimaryState(x, u, t, d; s = 0.0, ampl = max(p.ampl, 0.0), dw = p.dw)
    elseif mode in (:turbulent, :wake)
        return BLPrimaryState(x, u, t, d; s = max(p.s, 1.0e-12), ampl = 0.0, dw = p.dw)
    end
    throw(ArgumentError("invalid interval state mode: $mode"))
end

function _normalize_interval_pair(mode::Symbol, p1::BLPrimaryState, p2::BLPrimaryState)
    q1 = _normalize_interval_primary(p1, _interval_unknown_mode(mode, 1))
    q2 = _normalize_interval_primary(p2, _interval_unknown_mode(mode, 2))
    x1 = q1.x
    x2 = max(q2.x, x1 + 1.0e-8)
    q1 = BLPrimaryState(x1, q1.u, q1.t, q1.d; s = q1.s, ampl = q1.ampl, dw = q1.dw)
    q2 = BLPrimaryState(x2, q2.u, q2.t, q2.d; s = q2.s, ampl = q2.ampl, dw = q2.dw)
    return q1, q2
end

function _interval_column_value(p::BLPrimaryState, mode::Symbol, col::Integer)
    col in 1:5 || throw(ArgumentError("interval column must be between 1 and 5"))
    if col == 1
        return mode in (:laminar, :similarity) ? p.ampl : p.s
    elseif col == 2
        return p.t
    elseif col == 3
        return p.d
    elseif col == 4
        return p.u
    end
    return p.x
end

function _interval_column_assign(p::BLPrimaryState, mode::Symbol, col::Integer, value::Real)
    col in 1:5 || throw(ArgumentError("interval column must be between 1 and 5"))
    v = float(value)
    if col == 1
        if mode in (:laminar, :similarity)
            return BLPrimaryState(p.x, p.u, p.t, p.d; s = 0.0, ampl = max(v, 0.0), dw = p.dw)
        else
            return BLPrimaryState(
                p.x,
                p.u,
                p.t,
                p.d;
                s = max(v, 1.0e-12),
                ampl = 0.0,
                dw = p.dw,
            )
        end
    elseif col == 2
        return BLPrimaryState(
            p.x,
            p.u,
            max(v, 1.0e-9),
            p.d;
            s = p.s,
            ampl = p.ampl,
            dw = p.dw,
        )
    elseif col == 3
        return BLPrimaryState(
            p.x,
            p.u,
            p.t,
            max(v, 1.0e-9);
            s = p.s,
            ampl = p.ampl,
            dw = p.dw,
        )
    elseif col == 4
        return BLPrimaryState(
            p.x,
            max(v, 1.0e-7),
            p.t,
            p.d;
            s = p.s,
            ampl = p.ampl,
            dw = p.dw,
        )
    end
    return BLPrimaryState(
        max(v, 1.0e-9),
        p.u,
        p.t,
        p.d;
        s = p.s,
        ampl = p.ampl,
        dw = p.dw,
    )
end

function _interval_fd_step(value::Real, col::Integer)
    scale =
        if col == 1
            max(abs(float(value)), 1.0)
        elseif col in (2, 3)
            max(abs(float(value)), 1.0e-4)
        elseif col == 4
            max(abs(float(value)), 1.0e-2)
        else
            max(abs(float(value)), 1.0e-3)
        end
    return 1.0e-6 * scale
end

function _clamp_interval_unknowns!(mode::Symbol, z::Vector{Float64})
    length(z) == 3 || throw(ArgumentError("interval state must have length 3"))
    if mode in (:laminar, :similarity)
        z[1] = max(z[1], 0.0)
    else
        z[1] = max(z[1], 1.0e-12)
    end
    z[2] = max(z[2], 1.0e-9)
    z[3] = max(z[3], 1.0e-9)
    return z
end

function _seed_primary_for_mode(
    p::BLPrimaryState,
    mode::Symbol,
    x::Real,
    u::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    seeded = _surface_primary_with_station(p, mode, x, u; ueps = ueps)
    if mode in (:laminar, :similarity)
        return BLPrimaryState(
            seeded.x,
            seeded.u,
            seeded.t,
            seeded.d;
            s = 0.0,
            ampl = max(seeded.ampl, 0.0),
            dw = seeded.dw,
        )
    elseif mode in (:turbulent, :transition)
        sseed = seeded.s
        if sseed <= 1.0e-12
            sseed = _transition_ctau_seed()
        end
        return BLPrimaryState(
            seeded.x,
            seeded.u,
            seeded.t,
            seeded.d;
            s = max(sseed, 1.0e-12),
            ampl = 0.0,
            dw = seeded.dw,
        )
    elseif mode == :wake
        sseed = seeded.s
        if sseed <= 1.0e-12
            sseed = _transition_shear_seed(seeded.u, seeded.t, seeded.d; flow = flow, model = model)
        end
        return BLPrimaryState(
            seeded.x,
            seeded.u,
            seeded.t,
            seeded.d;
            s = max(sseed, 1.0e-12),
            ampl = 0.0,
            dw = seeded.dw,
        )
    end
    throw(ArgumentError("invalid BL mode: $mode"))
end

function _interval_seed_unknowns(
    mode::Symbol,
    p::BLPrimaryState,
    x::Real,
    u::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    seeded = _seed_primary_for_mode(p, mode, x, u; flow = flow, model = model, ueps = ueps)
    return _pack_unknowns(mode, seeded)
end

function _surface_seed_unknowns(
    mode::Symbol,
    p1::BLPrimaryState,
    seed::Union{Nothing,SurfaceMarchResult},
    idx::Integer,
    x::Real,
    u::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    isnothing(seed) && return nothing
    1 <= idx <= length(seed.primary) || throw(BoundsError(seed.primary, idx))

    pseed = seed.primary[idx]
    if mode != :turbulent || seed.mode[idx] == :turbulent
        return _interval_seed_unknowns(mode, pseed, x, u; flow = flow, model = model, ueps = ueps)
    end

    seeded = _surface_primary_with_station(pseed, :turbulent, x, u; ueps = ueps)
    pseed_t = BLPrimaryState(
        seeded.x,
        seeded.u,
        seeded.t,
        seeded.d;
        s = max(p1.s, 1.0e-12),
        ampl = 0.0,
        dw = seeded.dw,
    )
    return _pack_unknowns(:turbulent, pseed_t)
end

function _interval_initial_guess(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    dw::Float64 = 0.0,
)
    x2f = float(x2)
    u2f = max(float(u2), 1.0e-4)
    dx = max(x2f - p1.x, 1.0e-6)

    if mode == :similarity
        bule = 1.0
        ucon = u2f / max(x2f^bule, 1.0e-12)
        tsq = 0.45 / max(ucon * (5.0 * bule + 1.0) * flow.reybl, 1.0e-12) * x2f^(1.0 - bule)
        t2 = max(sqrt(max(tsq, 1.0e-14)), 1.0e-8)
        d2 = max(2.2 * t2, 2.0e-8)
        p2 = BLPrimaryState(x2f, u2f, t2, d2; s = 0.0, ampl = 0.0, dw = 0.0)
        return _pack_unknowns(mode, p2)
    elseif mode == :laminar
        sec1 = blsecondary_state(p1, :laminar; flow = flow, model = model)
        ax =
            axset(
                sec1.hk,
                p1.t,
                sec1.rt,
                p1.ampl,
                sec1.hk,
                p1.t,
                sec1.rt,
                p1.ampl,
                acrit,
            )
        ampl2 = max(p1.ampl + ax * dx, p1.ampl)
        p2 = BLPrimaryState(x2f, u2f, p1.t, p1.d; s = 0.0, ampl = ampl2, dw = 0.0)
        return _pack_unknowns(mode, p2)
    else
        sseed = _transition_shear_seed(p1.u, p1.t, p1.d; flow = flow, model = model)
        s2 = max(p1.s, sseed)
        p2 = BLPrimaryState(x2f, u2f, p1.t, p1.d; s = s2, ampl = 0.0, dw = dw)
        return _pack_unknowns(mode, p2)
    end
end

function _eval_interval(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real,
    z::AbstractVector{<:Real};
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    dw::Float64 = 0.0,
)
    p2 = _unpack_primary(mode, x2, u2, z; dw = dw)
    p1_eval = p1
    if mode == :similarity
        p1_eval = BLPrimaryState(p2.x, p2.u, p2.t, p2.d; s = 0.0, ampl = 0.0, dw = 0.0)
    end

    # When _USE_BLDIF_ANALYTIC[] is true (set during VISCAL Newton remarching),
    # use _bldif_analytic for residuals — the SAME function SETBL uses.
    # This eliminates divergence between bldif_residuals (different clamping)
    # and _bldif_analytic (XFoil-matching clamping) that causes ~0.5% Ue error.
    local r_vec
    if _USE_BLDIF_ANALYTIC[]
        r_vec = try
            _, _, vsrez, _, _ = _bldif_analytic(
                mode, p1_eval, p2;
                flow = flow, model = model,
                acrit = acrit, idampv = idampv,
                precision = Float64,
            )
            # vsrez = -residual (XFoil convention); negate for +residual
            collect(-vsrez)
        catch
            nothing
        end
    end

    if !_USE_BLDIF_ANALYTIC[] || isnothing(r_vec)
        s1 = blsecondary_state(p1_eval, mode == :similarity ? :laminar : _mode_closure(mode); flow = flow, model = model)
        s2 = blsecondary_state(p2, mode == :similarity ? :laminar : _mode_closure(mode); flow = flow, model = model)
        r_vec = collect(bldif_residuals(mode, p1_eval, p2, s1, s2;
            model = model, acrit = acrit, idampv = idampv,
            bule = bule, hupwt = hupwt))
    end

    s1 = blsecondary_state(p1_eval, mode == :similarity ? :laminar : _mode_closure(mode); flow = flow, model = model)
    s2 = blsecondary_state(p2, mode == :similarity ? :laminar : _mode_closure(mode); flow = flow, model = model)

    return r_vec, p2, s1, s2
end

function _blsys_interval_eval(
    mode::Symbol,
    p1::BLPrimaryState,
    p2::BLPrimaryState;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    xt_guess::Float64 = NaN,
)
    q1, q2 = _normalize_interval_pair(mode, p1, p2)
    if mode == :transition
        rez, p2_eval, s2, xt, xt_ok = _eval_transition_interval(
            q1,
            q2,
            acrit;
            flow = flow,
            model = model,
            idampv = idampv,
            hupwt = hupwt,
            xt_guess = xt_guess,
        )
        s1 = blsecondary_state(q1, :laminar; flow = flow, model = model)
        return collect(rez), q1, p2_eval, s1, s2, xt, xt_ok
    end

    rez, p2_eval, s1, s2 = _eval_interval(
        mode,
        q1,
        q2.x,
        q2.u,
        _pack_unknowns(mode, q2);
        flow = flow,
        model = model,
        acrit = acrit,
        idampv = idampv,
        bule = bule,
        hupwt = hupwt,
    )
    return rez, q1, p2_eval, s1, s2, NaN, false
end

function _blsys_interval_column(
    side::Integer,
    col::Integer,
    mode::Symbol,
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    residual::AbstractVector{<:Real};
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    xt_guess::Float64 = NaN,
)
    smode = _interval_unknown_mode(mode, side)
    pbase = side == 1 ? p1 : p2
    v0 = _interval_column_value(pbase, smode, col)
    step = _interval_fd_step(v0, col)

    function _eval_shift(delta::Float64)
        pshift = _interval_column_assign(pbase, smode, col, v0 + delta)
        rez, _, _, _, _, _, _ =
            if side == 1
                _blsys_interval_eval(
                    mode,
                    pshift,
                    p2;
                    flow = flow,
                    model = model,
                    acrit = acrit,
                    idampv = idampv,
                    bule = bule,
                    hupwt = hupwt,
                    xt_guess = xt_guess,
                )
            else
                _blsys_interval_eval(
                    mode,
                    p1,
                    pshift;
                    flow = flow,
                    model = model,
                    acrit = acrit,
                    idampv = idampv,
                    bule = bule,
                    hupwt = hupwt,
                    xt_guess = xt_guess,
                )
            end
        return rez
    end

    plus =
        try
            _eval_shift(step)
        catch
            nothing
        end
    minus =
        try
            _eval_shift(-step)
        catch
            nothing
        end

    if !isnothing(plus) && !isnothing(minus)
        return (plus .- minus) ./ (2.0 * step)
    elseif !isnothing(plus)
        return (plus .- residual) ./ step
    elseif !isnothing(minus)
        return (residual .- minus) ./ step
    end
    return fill(NaN, length(residual))
end

function _blsys_interval_system(
    mode::Symbol,
    p1::BLPrimaryState,
    p2::BLPrimaryState;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    xt_guess::Float64 = NaN,
)
    rez, q1, q2, s1, s2, xt, xt_ok = _blsys_interval_eval(
        mode,
        p1,
        p2;
        flow = flow,
        model = model,
        acrit = acrit,
        idampv = idampv,
        bule = bule,
        hupwt = hupwt,
        xt_guess = xt_guess,
    )

    vs1 = zeros(Float64, 4, 5)
    vs2 = zeros(Float64, 4, 5)
    for col in 1:5
        vs1[1:3, col] = _blsys_interval_column(
            1,
            col,
            mode,
            q1,
            q2,
            rez;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
            xt_guess = xt_guess,
        )
        vs2[1:3, col] = _blsys_interval_column(
            2,
            col,
            mode,
            q1,
            q2,
            rez;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
            xt_guess = xt_guess,
        )
    end

    vsrez = zeros(Float64, 4)
    vsrez[1:3] .= .-rez
    vsm = zeros(Float64, 4)
    vsr = zeros(Float64, 4)
    vsx = zeros(Float64, 4)

    return BLIntervalSystem(
        mode,
        rez,
        vs1,
        vs2,
        vsrez,
        vsm,
        vsr,
        vsx,
        q1,
        q2,
        s1,
        s2,
        xt,
        xt_ok,
    )
end

function _solve_interval(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    z_init::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dw::Float64 = 0.0,
)
    z =
        if isnothing(z_init)
            _interval_initial_guess(
                mode,
                p1,
                x2,
                u2;
                flow = flow,
                model = model,
                acrit = acrit,
                dw = dw,
            )
        else
            _clamp_interval_unknowns!(mode, [float(v) for v in z_init])
        end
    best_z = copy(z)
    best_norm = Inf

    for _ in 1:max_iter
        p2_guess = _unpack_primary(mode, x2, u2, z; dw = dw)
        sys = _blsys_interval_system(
            mode,
            p1,
            p2_guess;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
        )
        r = sys.residual
        p2 = sys.p2
        s2 = sys.s2
        rnorm = maximum(abs, r)
        if isfinite(rnorm) && rnorm < best_norm
            best_norm = rnorm
            best_z .= z
        end
        if isfinite(rnorm) && rnorm <= tol
            return p2, s2, true, rnorm
        end

        J = copy(sys.vs2[1:3, 1:3])

        if any(!isfinite, J)
            break
        end

        dz =
            try
                J \ sys.vsrez[1:3]
            catch
                break
            end

        accepted = false
        lambda = 1.0
        for _ in 1:8
            ztrial = z .+ lambda .* dz
            if mode in (:laminar, :similarity)
                ztrial[1] = max(ztrial[1], 0.0)
            else
                ztrial[1] = max(ztrial[1], 1.0e-12)
            end
            ztrial[2] = max(ztrial[2], 1.0e-9)
            ztrial[3] = max(ztrial[3], 1.0e-9)

            rtrial, _, _, _ = _eval_interval(
                mode,
                p1,
                x2,
                u2,
                ztrial;
                flow = flow,
                model = model,
                acrit = acrit,
                idampv = idampv,
                bule = bule,
                hupwt = hupwt,
                dw = dw,
            )
            ntrial = maximum(abs, rtrial)
            if isfinite(ntrial) && ntrial < rnorm
                z .= ztrial
                accepted = true
                break
            end
            lambda *= 0.5
        end

        accepted || break
    end

    r, p2, _, s2 = _eval_interval(
        mode,
        p1,
        x2,
        u2,
        best_z;
        flow = flow,
        model = model,
        acrit = acrit,
        idampv = idampv,
        bule = bule,
        hupwt = hupwt,
        dw = dw,
    )
    return p2, s2, false, maximum(abs, r)
end

function _solve_interval_mrchue(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    z_init::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dw::Float64 = 0.0,
)
    mode in (:laminar, :turbulent, :wake) ||
        throw(ArgumentError("MRCHUE interval solve only supports laminar, turbulent, or wake modes"))

    x2f = max(float(x2), p1.x + 1.0e-6)
    uref = max(float(u2), 1.0e-4)
    pcur =
        if isnothing(z_init)
            _unpack_primary(
                mode,
                x2f,
                uref,
                _interval_initial_guess(
                    mode,
                    p1,
                    x2f,
                    uref;
                    flow = flow,
                    model = model,
                    acrit = acrit,
                    dw = dw,
                );
                dw = dw,
            )
        else
            _unpack_primary(
                mode,
                x2f,
                uref,
                _clamp_interval_unknowns!(mode, [float(v) for v in z_init]);
                dw = dw,
            )
        end
    pbest = pcur
    sbest = blsecondary_state(pcur, _mode_closure(mode); flow = flow, model = model)
    best_metric = Inf
    direct = true
    htarg = NaN
    deps = max(float(tol), 1.0e-5)

    for _ in 1:max_iter
        sys = _blsys_interval_system(
            mode,
            p1,
            pcur;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
        )
        pnow = sys.p2
        snow = sys.s2
        if !all(isfinite, sys.vs2) || any(!isfinite, sys.vsrez)
            break
        end

        J = zeros(Float64, 4, 4)
        J[1:3, 1:3] .= sys.vs2[1:3, 1:3]
        J[1:3, 4] .= sys.vs2[1:3, 4]
        rhs = zeros(Float64, 4)
        rhs[1:3] .= sys.vsrez[1:3]

        if direct
            J[4, 4] = 1.0
            rhs[4] = 0.0
        else
            hk = _interval_hk_derivatives(pnow, mode; flow = flow, model = model)
            J[4, 2] = hk.t
            J[4, 3] = hk.d
            J[4, 4] = hk.u
            rhs[4] = htarg - snow.hk
        end

        any(!isfinite, J) && break

        delta =
            try
                J \ rhs
            catch
                break
            end
        any(!isfinite, delta) && break

        dmax = _mrchue_delta_max(mode, delta, pnow; inverse = !direct)
        if isfinite(dmax) && dmax < best_metric
            pbest = pnow
            sbest = snow
            best_metric = dmax
        end

        rlx = dmax > 0.3 ? 0.3 / dmax : 1.0
        rlx = clamp(rlx, 1.0e-6, 1.0)

        if direct
            ptest = _apply_interval_delta(mode, x2f, pnow, delta, rlx; flow = flow, dw = dw)
            stest = blsecondary_state(ptest, _mode_closure(mode); flow = flow, model = model)
            if stest.hk >= _mrchue_hmax(mode)
                htarg = _mrchue_htarg(mode, p1, sys.s1, ptest)
                direct = false
                continue
            end
            pcur = ptest
        else
            pcur = _apply_interval_delta(mode, x2f, pnow, delta, rlx; flow = flow, dw = dw)
        end

        if dmax <= deps
            scur = blsecondary_state(pcur, _mode_closure(mode); flow = flow, model = model)
            return pcur, scur, true, dmax
        end
    end

    return pbest, sbest, false, best_metric
end

function _solve_interval_mrchdu(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    z_init::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dw::Float64 = 0.0,
)
    mode in (:laminar, :turbulent) ||
        throw(ArgumentError("MRCHDU interval solve only supports laminar or turbulent modes"))

    x2f = max(float(x2), p1.x + 1.0e-6)
    uref = max(float(u2), 1.0e-4)
    pcur =
        if isnothing(z_init)
            _unpack_primary(
                mode,
                x2f,
                uref,
                _interval_initial_guess(
                    mode,
                    p1,
                    x2f,
                    uref;
                    flow = flow,
                    model = model,
                    acrit = acrit,
                    dw = dw,
                );
                dw = dw,
            )
        else
            _unpack_primary(
                mode,
                x2f,
                uref,
                _clamp_interval_unknowns!(mode, [float(v) for v in z_init]);
                dw = dw,
            )
        end

    pbest = pcur
    sbest = blsecondary_state(pcur, _mode_closure(mode); flow = flow, model = model)
    best_metric = Inf
    hkref = sbest.hk
    ueref_cur = max(pcur.u, 1.0e-8)
    sens = 0.0
    senswt = 1000.0
    deps = max(float(tol), 5.0e-6)

    for iter in 1:max_iter
        sys = _blsys_interval_system(
            mode,
            p1,
            pcur;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
        )
        pnow = sys.p2
        snow = sys.s2
        if !all(isfinite, sys.vs2) || any(!isfinite, sys.vsrez)
            break
        end

        if iter == 1
            ueref_cur = max(pnow.u, 1.0e-8)
            hkref = snow.hk
        end

        hk = _interval_hk_derivatives(pnow, mode; flow = flow, model = model)
        any(!isfinite, (hk.t, hk.d, hk.u, hkref, ueref_cur)) && break

        vtmp = zeros(Float64, 4, 4)
        vtmp[1:3, 1:3] .= sys.vs2[1:3, 1:3]
        vtmp[1:3, 4] .= sys.vs2[1:3, 4]
        vtmp[4, 2] = hk.t
        vtmp[4, 3] = hk.d
        vtmp[4, 4] = hk.u
        vz = [0.0, 0.0, 0.0, 1.0]
        duedhk =
            try
                (vtmp \ vz)[4]
            catch
                NaN
            end
        isfinite(duedhk) || break

        sennew = senswt * duedhk * hkref / max(ueref_cur, 1.0e-8)
        if iter <= 5
            sens = sennew
        elseif iter <= 15
            sens = 0.5 * (sens + sennew)
        end

        J = zeros(Float64, 4, 4)
        J[1:3, 1:3] .= sys.vs2[1:3, 1:3]
        J[1:3, 4] .= sys.vs2[1:3, 4]
        J[4, 2] = hk.t * hkref
        J[4, 3] = hk.d * hkref
        J[4, 4] = hk.u * hkref + sens / max(ueref_cur, 1.0e-8)
        rhs = zeros(Float64, 4)
        rhs[1:3] .= sys.vsrez[1:3]
        rhs[4] =
            -(hkref^2) * (snow.hk / max(hkref, 1.0e-8) - 1.0) -
            sens * (pnow.u / max(ueref_cur, 1.0e-8) - 1.0)

        any(!isfinite, J) && break
        any(!isfinite, rhs) && break

        delta =
            try
                J \ rhs
            catch
                break
            end
        any(!isfinite, delta) && break

        dmax = _mrchdu_delta_max(mode, delta, pnow)
        if isfinite(dmax) && dmax < best_metric
            pbest = pnow
            sbest = snow
            best_metric = dmax
        end

        rlx = dmax > 0.3 ? 0.3 / dmax : 1.0
        rlx = clamp(rlx, 1.0e-6, 1.0)

        pcur = _apply_interval_delta(mode, x2f, pnow, delta, rlx; flow = flow, dw = dw)
        if mode in (:turbulent, :wake)
            pcur = BLPrimaryState(
                pcur.x,
                pcur.u,
                pcur.t,
                pcur.d;
                s = clamp(pcur.s, 1.0e-7, 0.30),
                ampl = 0.0,
                dw = pcur.dw,
            )
        end

        if dmax <= deps
            scur = blsecondary_state(pcur, _mode_closure(mode); flow = flow, model = model)
            return pcur, scur, true, dmax
        end
    end

    return pbest, sbest, false, best_metric
end

function _solve_surface_interval(
    mode::Symbol,
    p1::BLPrimaryState,
    x2::Real,
    u2::Real;
    mrchue_local::Bool,
    mrchdu_local::Bool,
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Float64,
    idampv::Int = 0,
    bule::Float64 = 1.0,
    hupwt::Float64 = 1.0,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    z_init::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dw::Float64 = 0.0,
)
    if mrchdu_local && mode in (:laminar, :turbulent)
        return _solve_interval_mrchdu(
            mode,
            p1,
            x2,
            u2;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
            tol = tol,
            max_iter = max_iter,
            z_init = z_init,
            dw = dw,
        )
    end
    if mrchue_local && mode in (:laminar, :turbulent, :wake)
        return _solve_interval_mrchue(
            mode,
            p1,
            x2,
            u2;
            flow = flow,
            model = model,
            acrit = acrit,
            idampv = idampv,
            bule = bule,
            hupwt = hupwt,
            tol = tol,
            max_iter = max_iter,
            z_init = z_init,
            dw = dw,
        )
    end
    return _solve_interval(
        mode,
        p1,
        x2,
        u2;
        flow = flow,
        model = model,
        acrit = acrit,
        idampv = idampv,
        bule = bule,
        hupwt = hupwt,
        tol = tol,
        max_iter = max_iter,
        z_init = z_init,
        dw = dw,
    )
end

function _march_surface(
    stations::SurfaceStations;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    seed::Union{Nothing,SurfaceMarchResult} = nothing,
    mrchue_local::Bool = true,
    mrchdu_local::Bool = false,
    preserve_seed_transition::Bool = true,
    solve_similarity_station::Bool = false,
)
    n = length(stations.x)
    primary = Vector{BLPrimaryState}(undef, n)
    secondary = Vector{BLSecondaryState}(undef, n)
    mode = Vector{Symbol}(undef, n)
    conv = falses(n - 1)

    if !isnothing(seed)
        seed.side == stations.side || throw(ArgumentError("seed side mismatch"))
        length(seed.x) == n ||
            throw(ArgumentError("seed station length mismatch"))
        length(seed.primary) == n == length(seed.secondary) == length(seed.mode) ||
            throw(ArgumentError("seed surface state length mismatch"))
    end

    p1 =
        if isnothing(seed)
            _initial_primary(stations.x[1], stations.ue[1], flow.reybl)
        else
            _seed_primary_for_mode(
                seed.primary[1],
                :laminar,
                stations.x[1],
                stations.ue[1];
                flow = flow,
                model = model,
            )
        end
    if solve_similarity_station
        p1, s1, _, _ = _solve_interval(
            :similarity,
            p1,
            stations.x[1],
            stations.ue[1];
            flow = flow,
            model = model,
            acrit = ncrit,
            bule = 1.0,
            tol = tol,
            max_iter = max_iter,
            z_init =
                isnothing(seed) ? nothing :
                _interval_seed_unknowns(
                    :similarity,
                    seed.primary[1],
                    stations.x[1],
                    stations.ue[1];
                    flow = flow,
                    model = model,
                ),
        )
        p1 = BLPrimaryState(p1.x, p1.u, p1.t, p1.d; s = 0.0, ampl = 0.0, dw = p1.dw)
        s1 = blsecondary_state(p1, :laminar; flow = flow, model = model)
    else
        s1 = blsecondary_state(p1, :laminar; flow = flow, model = model)
    end
    mcur = :laminar

    primary[1] = p1
    secondary[1] = s1
    mode[1] = mcur

    xtr = isnothing(seed) ? NaN : seed.transition_x
    ncrit_trip = ncrit
    for i in 1:(n - 1)
        x2 = max(stations.x[i + 1], p1.x + 1.0e-6)
        u2 = stations.ue[i + 1]
        seed_next = isnothing(seed) ? nothing : seed.primary[i + 1]
        seeded_xtr = isnothing(seed) ? NaN : _seeded_transition_xsi(seed, stations, i)
        seeded_cross =
            isfinite(xtr) &&
            min(stations.x[i], stations.x[i + 1]) - 1.0e-10 <= xtr <= max(stations.x[i], stations.x[i + 1]) + 1.0e-10

        if !solve_similarity_station && mcur == :laminar && i == 1
            p2, s2, ok, _ = _solve_interval(
                :similarity,
                p1,
                x2,
                u2;
                flow = flow,
                model = model,
                acrit = ncrit,
                bule = 1.0,
                tol = tol,
                max_iter = max_iter,
                z_init =
                    isnothing(seed_next) ? nothing :
                    _interval_seed_unknowns(
                        :similarity,
                        seed_next,
                        x2,
                        u2;
                        flow = flow,
                        model = model,
                    ),
            )
            p2 = BLPrimaryState(p2.x, p2.u, p2.t, p2.d; s = 0.0, ampl = 0.0, dw = p2.dw)
            s2 = blsecondary_state(p2, :laminar; flow = flow, model = model)
            conv[i] = ok
            primary[i + 1] = p2
            secondary[i + 1] = s2
            mode[i + 1] = mcur
            p1 = p2
            continue
        end

        if mcur == :laminar && seeded_cross && preserve_seed_transition
            p2, s2, xtr_loc, ok_tr = _seeded_transition_interval_step(
                p1,
                x2,
                u2,
                seeded_xtr,
                ncrit_trip;
                flow = flow,
                model = model,
                tol = tol,
                max_iter = max_iter,
            )
            xtr =
                if isnothing(seed) || !preserve_seed_transition
                    xtr_loc
                else
                    seed.transition_x
                end
            mcur = :turbulent
            conv[i] = ok_tr
        elseif mcur == :laminar && (isnan(xtr) || xtr >= 1.0 || !preserve_seed_transition)
            p2_l, s2_l, ok_l, _ = _solve_surface_interval(
                :laminar,
                p1,
                x2,
                u2;
                mrchue_local = mrchue_local,
                mrchdu_local = mrchdu_local,
                flow = flow,
                model = model,
                acrit = ncrit,
                tol = tol,
                max_iter = max_iter,
                z_init =
                    isnothing(seed_next) ? nothing :
                    _interval_seed_unknowns(
                        :laminar,
                        seed_next,
                        x2,
                        u2;
                        flow = flow,
                        model = model,
                    ),
            )
            if p2_l.ampl >= ncrit_trip
                p2, s2, xtr_loc, ok_tr = _transition_interval_step(
                    p1,
                    x2,
                    p2_l.u,
                    ncrit_trip;
                    flow = flow,
                    model = model,
                    tol = tol,
                    max_iter = max_iter,
                    p2_l_hint = p2_l,
                )
                s1 = stations.x[i]
                s2m = stations.x[i + 1]
                t = clamp((xtr_loc - s1) / max(s2m - s1, 1.0e-12), 0.0, 1.0)
                xtr = (1.0 - t) * stations.x[i] + t * stations.x[i + 1]
                mcur = :turbulent
                conv[i] = ok_tr
            else
                p2, s2 = p2_l, s2_l
                conv[i] = ok_l
            end
        elseif mcur == :turbulent
            p2, s2, ok, _ = _solve_surface_interval(
                mcur,
                p1,
                x2,
                u2;
                mrchue_local = mrchue_local,
                mrchdu_local = mrchdu_local,
                flow = flow,
                model = model,
                acrit = ncrit,
                tol = tol,
                max_iter = max_iter,
                z_init =
                    _surface_seed_unknowns(
                        mcur,
                        p1,
                        seed,
                        i + 1,
                        x2,
                        u2;
                        flow = flow,
                        model = model,
                    ),
            )
            conv[i] = ok
            p2 =
                BLPrimaryState(
                    p2.x,
                    p2.u,
                    p2.t,
                    p2.d;
                    s = max(p2.s, 1.0e-12),
                    ampl = 0.0,
                    dw = p2.dw,
                )
            s2 = blsecondary_state(p2, :turbulent; flow = flow, model = model)
        else
            p2, s2, ok, _ = _solve_surface_interval(
                mcur,
                p1,
                x2,
                u2;
                mrchue_local = mrchue_local,
                mrchdu_local = mrchdu_local,
                flow = flow,
                model = model,
                acrit = ncrit,
                tol = tol,
                max_iter = max_iter,
            )
            conv[i] = ok
        end

        primary[i + 1] = p2
        secondary[i + 1] = s2
        mode[i + 1] = mcur

        p1 = p2
    end

    return SurfaceMarchResult(
        stations.side,
        stations.x,
        stations.xc,
        [p.u for p in primary],
        mode,
        primary,
        secondary,
        xtr,
        solve_similarity_station,
        conv,
    )
end

function _surface_station_unknown_mode(surface::SurfaceMarchResult, i::Integer)
    1 <= i <= length(surface.mode) || throw(BoundsError(surface.mode, i))
    return surface.mode[i] in (:turbulent, :wake) ? :turbulent : :laminar
end

@inline function _surface_transition_aware_ctau(surface::SurfaceMarchResult, i::Integer)
    1 <= i <= length(surface.primary) || throw(BoundsError(surface.primary, i))
    p = surface.primary[i]
    mode = surface.mode[i]
    if mode in (:turbulent, :wake)
        return p.s
    elseif mode == :transition && isfinite(float(p.s)) && float(p.s) > 0.0
        return p.s
    end
    return p.ampl
end

function _surface_interval_mode(surface::SurfaceMarchResult, i::Integer)
    n = length(surface.x)
    2 <= i <= n || throw(BoundsError(surface.x, i))

    if i == 2 && !surface.solved_similarity_station
        return :similarity
    end

    xtr = surface.transition_x
    if isfinite(xtr) &&
       surface.mode[i - 1] in (:laminar, :similarity) &&
       surface.mode[i] in (:turbulent, :wake)
        xlo = min(surface.x[i - 1], surface.x[i])
        xhi = max(surface.x[i - 1], surface.x[i])
        if xlo - 1.0e-10 <= xtr <= xhi + 1.0e-10
            return :transition
        end
    end

    if surface.mode[i - 1] == :wake || surface.mode[i] == :wake
        return :wake
    elseif surface.mode[i - 1] == :turbulent || surface.mode[i] == :turbulent
        return :turbulent
    end
    return :laminar
end

function _surface_dump_x(surface::SurfaceMarchResult, i::Integer)
    n = length(surface.x)
    1 <= i <= n || throw(BoundsError(surface.x, i))
    if surface.solved_similarity_station
        return i < n ? surface.xc[i + 1] : surface.x[end]
    end
    return surface.xc[i]
end

function _surface_transition_xsi_guess(surface::SurfaceMarchResult, i::Integer)
    xtr = surface.transition_x
    isfinite(xtr) || return NaN

    x1 = surface.x[i - 1]
    x2 = surface.x[i]
    return clamp(xtr, min(x1, x2), max(x1, x2))
end

function _surface_interval_systems(
    surface::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
)
    n = length(surface.x)
    systems = Vector{BLIntervalSystem}(undef, max(n - 1, 0))
    for i in 2:n
        mode = _surface_interval_mode(surface, i)
        systems[i - 1] = _blsys_interval_system(
            mode,
            surface.primary[i - 1],
            surface.primary[i];
            flow = flow,
            model = model,
            acrit = ncrit,
            xt_guess = mode == :transition ? _surface_transition_xsi_guess(surface, i) : NaN,
        )
    end
    return systems
end

function _surface_newton_station(
    p::BLPrimaryState,
    mode::Symbol,
    x::Real,
    u::Real,
    dz::AbstractVector{<:Real},
)
    length(dz) == 3 || throw(ArgumentError("surface station delta must have length 3"))
    xf = max(float(x), 1.0e-9)
    uf = max(float(u), 1.0e-7)
    tf = max(p.t + float(dz[2]), 1.0e-9)
    df = max(p.d + float(dz[3]), 1.0e-9)
    if mode == :laminar
        return BLPrimaryState(
            xf,
            uf,
            tf,
            df;
            s = 0.0,
            ampl = max(p.ampl + float(dz[1]), 0.0),
            dw = p.dw,
        )
    end
    return BLPrimaryState(
        xf,
        uf,
        tf,
        df;
        s = max(p.s + float(dz[1]), 1.0e-12),
        ampl = 0.0,
        dw = p.dw,
    )
end

function _surface_newton_update_metrics(
    surface::SurfaceMarchResult,
    delta::AbstractVector{<:Real},
)
    n = length(surface.x)
    length(delta) == 3 * max(n - 1, 0) || throw(ArgumentError("surface update vector length mismatch"))

    dhi = 1.5
    dlo = -0.5
    rlx = 1.0
    rms = 0.0
    rmx = 0.0
    imax = 2
    vmax = :ampl

    for i in 2:n
        off = 3 * (i - 2)
        mode = _surface_station_unknown_mode(surface, i)
        p = surface.primary[i]
        dn1 =
            if mode == :laminar
                delta[off + 1] / 10.0
            else
                delta[off + 1] / max(p.s, 1.0e-12)
            end
        dn2 = delta[off + 2] / max(p.t, 1.0e-12)
        dn3 = delta[off + 3] / max(p.d, 1.0e-12)

        rms += dn1^2 + dn2^2 + dn3^2
        for (dn, tag) in ((dn1, :ampl), (dn2, :theta), (dn3, :dstar))
            if abs(dn) > abs(rmx)
                rmx = dn
                imax = i
                vmax = tag
            end
            if rlx * dn > dhi
                rlx = dhi / dn
            elseif rlx * dn < dlo
                rlx = dlo / dn
            end
        end
    end

    ndof = 3 * max(n - 1, 1)
    rms = sqrt(rms / ndof)
    return (rlx = clamp(rlx, 0.0, 1.0), rms = rms, rmx = rmx, imax = imax, vmax = vmax)
end

function _surface_newton_step(
    stations::SurfaceStations,
    surface::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
)
    n = length(surface.x)
    n == length(stations.x) == length(stations.ue) == length(stations.xc) ||
        throw(ArgumentError("surface/station length mismatch"))
    n >= 2 || throw(ArgumentError("surface must contain at least two stations"))

    systems = _surface_interval_systems(surface; flow = flow, model = model, ncrit = ncrit)
    ndof = 3 * (n - 1)
    A = zeros(Float64, ndof, ndof)
    b = zeros(Float64, ndof)
    residual = 0.0

    for i in 2:n
        sys = systems[i - 1]
        row = (3 * (i - 2) + 1):(3 * (i - 1))
        b[row] .= sys.vsrez[1:3]
        residual = max(residual, maximum(abs, sys.residual))

        if i > 2
            col_prev = (3 * (i - 3) + 1):(3 * (i - 2))
            A[row, col_prev] .+= sys.vs1[1:3, 1:3]
        end
        col_cur = (3 * (i - 2) + 1):(3 * (i - 1))
        A[row, col_cur] .+= sys.vs2[1:3, 1:3]
    end

    delta =
        try
            A \ b
        catch err
            return (
                ok = false,
                reason = sprint(showerror, err),
                surface = surface,
                systems = systems,
                residual = residual,
                rlx = 0.0,
                rms = Inf,
                rmx = Inf,
                imax = 0,
                vmax = :none,
            )
        end

    if any(!isfinite, delta)
        return (
            ok = false,
            reason = "non-finite surface Newton delta",
            surface = surface,
            systems = systems,
            residual = residual,
            rlx = 0.0,
            rms = Inf,
            rmx = Inf,
            imax = 0,
            vmax = :none,
        )
    end

    metrics = _surface_newton_update_metrics(surface, delta)
    rlx = metrics.rlx

    primary = copy(surface.primary)
    secondary = copy(surface.secondary)
    mode = copy(surface.mode)
    xtr = surface.transition_x

    for i in 2:n
        off = 3 * (i - 2)
        stmode = _surface_station_unknown_mode(surface, i)
        pnew = _surface_newton_station(
            surface.primary[i],
            stmode,
            stations.x[i],
            stations.ue[i],
            rlx .* delta[(off + 1):(off + 3)],
        )
        primary[i] = pnew
        secondary[i] = blsecondary_state(
            pnew,
            _mode_closure(mode[i]);
            flow = flow,
            model = model,
        )
    end

    secondary[1] = blsecondary_state(primary[1], _mode_closure(mode[1]); flow = flow, model = model)

    for i in 2:n
        sys = systems[i - 1]
        if sys.mode == :transition && sys.transition_ok && isfinite(sys.transition_x)
            x1 = stations.x[i - 1]
            x2 = stations.x[i]
            xtr = clamp(sys.transition_x, min(x1, x2), max(x1, x2))
            break
        end
    end

    updated = SurfaceMarchResult(
        surface.side,
        copy(stations.x),
        copy(stations.xc),
        copy(stations.ue),
        mode,
        primary,
        secondary,
        xtr,
        surface.solved_similarity_station,
        copy(surface.converged_intervals),
    )

    return (
        ok = true,
        reason = "",
        surface = updated,
        systems = systems,
        residual = residual,
        rlx = rlx,
        rms = metrics.rms,
        rmx = metrics.rmx,
        imax = metrics.imax,
        vmax = metrics.vmax,
    )
end

function _surface_newton_solve(
    stations::SurfaceStations,
    surface::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 4,
)
    current = surface
    history = NamedTuple[]
    status = :max_iter
    failure_message = ""

    for iter in 1:max_iter
        step = _surface_newton_step(
            stations,
            current;
            flow = flow,
            model = model,
            ncrit = ncrit,
        )
        push!(history, (
            iteration = iter,
            residual = step.residual,
            rlx = step.rlx,
            rms = step.rms,
            rmx = step.rmx,
            imax = step.imax,
            vmax = step.vmax,
            ok = step.ok,
        ))
        if !step.ok
            status = :step_failure
            failure_message = step.reason
            break
        end
        current = step.surface
        if step.residual <= tol
            status = :converged
            break
        end
    end

    residual = isempty(history) ? Inf : history[end].residual
    converged = status == :converged
    return (
        surface = current,
        history = history,
        iterations = length(history),
        residual = residual,
        converged = converged,
        status = status,
        failure_message = failure_message,
    )
end

function _similarity_station_residual(p::BLPrimaryState; flow::BLFlowParams)
    x = max(p.x, 1.0e-9)
    u = max(p.u, 1.0e-7)
    re = max(flow.reybl, 1.0e3)
    t_eq = max(0.664 * sqrt(x / (re * u)), 1.0e-7)
    d_eq = max(2.2 * t_eq, 2.0e-7)
    return [max(p.ampl, 0.0), p.t - t_eq, p.d - d_eq]
end

function _coupled_surface_station_index(side::Symbol, i::Integer, nupper::Integer)
    side == :upper && return i
    side == :lower && return nupper + i
    throw(ArgumentError("side must be :upper or :lower"))
end

function _coupled_surface_station_block(side::Symbol, i::Integer, nupper::Integer)
    g = _coupled_surface_station_index(side, i, nupper)
    return (3 * (g - 1) + 1):(3 * g)
end

function _coupled_surface_interval_row(side::Symbol, i::Integer, nupper::Integer)
    base =
        if side == :upper
            3 * (i - 1) + 1
        else
            3 * nupper + 3 * (i - 1) + 1
        end
    return base:(base + 2)
end

function _surface_station_indices(topo::BLTopology, side::Symbol)
    st = _bl_side(topo, side)
    return st.ipan[2:end], st.vti[2:end]
end

function _surface_coupling_matrix(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
)
    up_idx, up_vti = _surface_station_indices(topo, :upper)
    lo_idx, lo_vti = _surface_station_indices(topo, :lower)

    idx = vcat(up_idx, lo_idx)
    vti = vcat(up_vti, lo_vti)
    nt = length(idx)
    K = zeros(Float64, nt, nt)
    uinv = zeros(Float64, nt)
    for i in 1:nt
        qbase = inv.qnode[idx[i]]
        uinv[i] = max(vti[i] * qbase, 1.0e-7)
        for j in 1:nt
            K[i, j] = -vti[i] * vti[j] * float(dij[idx[i], idx[j]])
        end
    end
    return K, uinv
end

function _coupled_surface_mass_sensitivities(
    p::BLPrimaryState,
    ue_m::AbstractVector{<:Real},
    self::Integer,
)
    u = max(p.u, 1.0e-7)
    d = max(p.d, 1.0e-9)
    d_m = collect((-(d / u)) .* float.(ue_m))
    d_m[self] += 1.0 / u
    return d_m
end

function _coupled_surface_xi_sensitivity(topo::BLTopology, side::Symbol)
    if side == :upper
        return topo.stagnation.sst_go, -topo.stagnation.sst_gp
    elseif side == :lower
        return -topo.stagnation.sst_go, topo.stagnation.sst_gp
    end
    throw(ArgumentError("side must be :upper or :lower"))
end

function _assemble_similarity_station_mass_row!(
    a::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    row,
    col,
    sys::BLIntervalSystem,
    p::BLPrimaryState,
    ue_m::AbstractVector{<:Real},
    d_m::AbstractVector{<:Real},
    due::Real;
    dxi_m::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dxi::Real = 0.0,
)
    vs1 = sys.vs1[1:3, :]
    vs2 = sys.vs2[1:3, :]
    vss = vs1 .+ vs2

    a[row, col[1]] .+= vss[:, 1]
    a[row, col[2]] .+= vss[:, 2]

    xisens =
        if isnothing(dxi_m)
            zeros(Float64, 3)
        else
            sys.vs1[1:3, 5] .+ sys.vs2[1:3, 5] .+ sys.vsx[1:3]
        end

    for eq in 1:3
        a[row[eq], 3:3:size(a, 2)] .+= vss[eq, 3] .* d_m .+ vss[eq, 4] .* ue_m
        if !isnothing(dxi_m)
            a[row[eq], 3:3:size(a, 2)] .+= xisens[eq] .* dxi_m
        end
    end

    dds = -(p.d / max(p.u, 1.0e-7)) * float(due)
    b[row] .= sys.vsrez[1:3] .+ vss[:, 4] .* float(due) .+ vss[:, 3] .* dds
    if !isnothing(dxi_m)
        b[row] .+= xisens .* float(dxi)
    end
    return nothing
end

function _approx_similarity_station_mass_row(
    p::BLPrimaryState,
    ue_m::AbstractVector{<:Real},
    self::Integer,
    due::Real;
    flow::BLFlowParams,
)
    x = max(p.x, 1.0e-9)
    u = max(p.u, 1.0e-7)
    re = max(flow.reybl, 1.0e3)
    t_eq = max(0.664 * sqrt(x / (re * u)), 1.0e-7)
    rez = _similarity_station_residual(p; flow = flow)

    jac_state = zeros(Float64, 3, 2)
    jac_state[1, 1] = 1.0
    jac_state[2, 2] = 1.0

    du = float.(ue_m)
    dd = _coupled_surface_mass_sensitivities(p, ue_m, self)
    jac_mass = zeros(Float64, 3, length(ue_m))
    jac_mass[2, :] .= (0.5 * t_eq / u) .* du
    jac_mass[3, :] .= dd .+ (1.1 * t_eq / u) .* du

    dstar_due = -(p.d / u) * float(due)
    rhs = .-rez
    rhs[2] += (0.5 * t_eq / u) * float(due)
    rhs[3] += dstar_due + (1.1 * t_eq / u) * float(due)

    return rez, jac_state, jac_mass, rhs
end

function _coupled_surface_update_metrics(
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    delta::AbstractVector{<:Real},
    due::AbstractVector{<:Real},
)
    nu = length(upper.x)
    nl = length(lower.x)
    ntotal = nu + nl
    length(delta) == 3 * ntotal || throw(ArgumentError("coupled update vector length mismatch"))
    length(due) == ntotal || throw(ArgumentError("coupled Ue delta length mismatch"))

    dhi = 1.5
    dlo = -0.5
    rlx = 1.0
    rms = 0.0
    rmx = 0.0
    loc = (side = :upper, station = 1, variable = :ampl)

    for (side, surface, base_station) in ((:upper, upper, 0), (:lower, lower, nu))
        for i in 1:length(surface.x)
            g = base_station + i
            off = 3 * (g - 1)
            mode = _surface_station_unknown_mode(surface, i)
            p = surface.primary[i]
            dz1 = float(delta[off + 1])
            dz2 = float(delta[off + 2])
            dm = float(delta[off + 3])
            dui = float(due[g])
            dd = (dm - p.d * dui) / max(p.u, 1.0e-7)
            dn1 =
                if mode == :laminar
                    dz1 / 10.0
                else
                    dz1 / max(p.s, 1.0e-12)
                end
            dn2 = dz2 / max(p.t, 1.0e-12)
            dn3 = dd / max(p.d, 1.0e-12)
            dn4 = abs(dui) / 0.25

            rms += dn1^2 + dn2^2 + dn3^2 + dn4^2
            for (dn, tag) in ((dn1, :ampl), (dn2, :theta), (dn3, :dstar), (dn4, :ue))
                if abs(dn) > abs(rmx)
                    rmx = tag == :ue ? dui : dn
                    loc = (side = side, station = i, variable = tag)
                end
                if rlx * dn > dhi
                    rlx = dhi / dn
                elseif rlx * dn < dlo
                    rlx = dlo / dn
                end
            end
        end
    end

    ndof = 4 * max(ntotal, 1)
    rms = sqrt(rms / ndof)
    return (rlx = clamp(rlx, 0.0, 1.0), rms = rms, rmx = rmx, loc = loc)
end

function _coupled_surface_updated_station(
    p::BLPrimaryState,
    mode::Symbol,
    x::Real,
    dctau::Real,
    dtheta::Real,
    dmass::Real,
    due::Real,
    rlx::Real;
    flow::BLFlowParams,
)
    uf0 = max(p.u, 1.0e-7)
    ddstr = (float(dmass) - p.d * float(due)) / uf0
    uf = max(p.u + float(rlx) * float(due), 1.0e-7)
    tf = max(p.t + float(rlx) * float(dtheta), 1.0e-9)
    df = max(p.d + float(rlx) * ddstr, 1.0e-9)

    if mode == :laminar
        pnew = BLPrimaryState(
            x,
            uf,
            tf,
            df;
            s = 0.0,
            ampl = max(p.ampl + float(rlx) * float(dctau), 0.0),
            dw = p.dw,
        )
    else
        snew = max(p.s + float(rlx) * float(dctau), 1.0e-12)
        snew = min(snew, 0.25)
        pnew = BLPrimaryState(
            x,
            uf,
            tf,
            df;
            s = snew,
            ampl = 0.0,
            dw = p.dw,
        )
    end

    hklim = mode == :wake ? 1.00005 : 1.02
    dsw = max(pnew.d - pnew.dw, 1.0e-9)
    dsw = _dslim_dstr(dsw, pnew.t, pnew.u; flow = flow, hklim = hklim)
    return BLPrimaryState(
        x,
        pnew.u,
        pnew.t,
        dsw + pnew.dw;
        s = pnew.s,
        ampl = pnew.ampl,
        dw = pnew.dw,
    )
end

function _coupled_surface_apply_update(
    surface::SurfaceMarchResult,
    stations::SurfaceStations,
    side::Symbol,
    nupper::Integer,
    delta::AbstractVector{<:Real},
    due::AbstractVector{<:Real};
    rlx::Float64,
    flow::BLFlowParams,
    model::BLModelParams,
)
    n = length(surface.x)
    n == length(stations.x) == length(stations.xc) == length(stations.ue) ||
        throw(ArgumentError("surface/station length mismatch"))

    primary = copy(surface.primary)
    secondary = copy(surface.secondary)
    for i in 1:n
        col = _coupled_surface_station_block(side, i, nupper)
        g = _coupled_surface_station_index(side, i, nupper)
        stmode = _surface_station_unknown_mode(surface, i)
        pnew = _coupled_surface_updated_station(
            surface.primary[i],
            stmode,
            stations.x[i],
            delta[col[1]],
            delta[col[2]],
            delta[col[3]],
            due[g],
            rlx;
            flow = flow,
        )
        primary[i] = pnew
        secondary[i] = blsecondary_state(
            pnew,
            _mode_closure(surface.mode[i]);
            flow = flow,
            model = model,
        )
    end

    return SurfaceMarchResult(
        surface.side,
        copy(stations.x),
        copy(stations.xc),
        [p.u for p in primary],
        copy(surface.mode),
        primary,
        secondary,
        surface.transition_x,
        surface.solved_similarity_station,
        copy(surface.converged_intervals),
    )
end

function _coupled_surface_transition_x(surface::SurfaceMarchResult, stations::SurfaceStations, systems)
    xtr = surface.transition_x
    for i in 2:length(surface.x)
        sys = systems[i - 1]
        if sys.mode == :transition && sys.transition_ok && isfinite(sys.transition_x)
            x1 = stations.x[i - 1]
            x2 = stations.x[i]
            xtr = clamp(sys.transition_x, min(x1, x2), max(x1, x2))
            break
        end
    end
    return xtr
end

function _update_surface_ue(
    surface::SurfaceMarchResult,
    ue_new::AbstractVector{<:Real};
    flow::BLFlowParams,
    model::BLModelParams,
)
    n = length(surface.x)
    length(ue_new) == n || throw(ArgumentError("ue_new length mismatch"))
    primary = copy(surface.primary)
    secondary = copy(surface.secondary)
    ue = copy(surface.ue)
    for i in 1:n
        p = surface.primary[i]
        u_new = max(float(ue_new[i]), 1.0e-7)
        primary[i] = BLPrimaryState(p.x, u_new, p.t, p.d; s = p.s, ampl = p.ampl, dw = p.dw)
        m = surface.mode[i] == :laminar || surface.mode[i] == :similarity ? :laminar : :turbulent
        secondary[i] = blsecondary_state(primary[i], m; flow = flow, model = model)
        ue[i] = u_new
    end
    return SurfaceMarchResult(
        surface.side,
        copy(surface.x),
        copy(surface.xc),
        ue,
        copy(surface.mode),
        primary,
        secondary,
        surface.transition_x,
        surface.solved_similarity_station,
        copy(surface.converged_intervals),
    )
end

function _surface_reclassify_transition(
    surface::SurfaceMarchResult,
    xtr::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    n = length(surface.x)
    n == length(surface.mode) == length(surface.primary) == length(surface.secondary) ||
        throw(ArgumentError("surface state length mismatch"))
    n == length(surface.xc) == length(surface.ue) || throw(ArgumentError("surface station length mismatch"))
    n == 0 && return surface

    xtrf = float(xtr)
    mode = fill(:laminar, n)
    itran = n + 1
    if isfinite(xtrf) && n >= 2
        for i in 2:n
            xlo = min(surface.x[i - 1], surface.x[i]) - 1.0e-10
            xhi = max(surface.x[i - 1], surface.x[i]) + 1.0e-10
            if xlo <= xtrf <= xhi
                itran = i
                break
            end
        end
        if itran == n + 1 && xtrf < min(surface.x[1], surface.x[min(2, n)])
            itran = 2
        end
        if 2 <= itran <= n
            mode[itran:end] .= :turbulent
        end
    end

    primary = copy(surface.primary)
    secondary = copy(surface.secondary)
    for i in 1:n
        pnew =
            if surface.mode[i] == mode[i]
                _surface_primary_with_station(surface.primary[i], mode[i], surface.x[i], surface.ue[i]; ueps = ueps)
            elseif mode[i] == :turbulent
                seeded = _surface_primary_with_station(
                    surface.primary[i],
                    :turbulent,
                    surface.x[i],
                    surface.ue[i];
                    ueps = ueps,
                )
                sseed =
                    if i > 1 && mode[i - 1] == :turbulent
                        primary[i - 1].s
                    else
                        _transition_ctau_seed()
                    end
                BLPrimaryState(
                    seeded.x,
                    seeded.u,
                    seeded.t,
                    seeded.d;
                    s = max(sseed, 1.0e-12),
                    ampl = 0.0,
                    dw = seeded.dw,
                )
            else
                _seed_primary_for_mode(
                    surface.primary[i],
                    mode[i],
                    surface.x[i],
                    surface.ue[i];
                    flow = flow,
                    model = model,
                    ueps = ueps,
                )
            end
        primary[i] = pnew
        secondary[i] = blsecondary_state(pnew, _mode_closure(mode[i]); flow = flow, model = model)
    end

    return SurfaceMarchResult(
        surface.side,
        copy(surface.x),
        copy(surface.xc),
        [p.u for p in primary],
        mode,
        primary,
        secondary,
        xtrf,
        surface.solved_similarity_station,
        copy(surface.converged_intervals),
    )
end

function _surface_transition_xc(surface::SurfaceMarchResult)
    n = length(surface.x)
    n == length(surface.xc) || throw(ArgumentError("surface transition coordinate length mismatch"))
    n == 0 && return float(surface.transition_x)
    any(m -> m in (:turbulent, :wake), surface.mode) || return 1.0

    xtr = float(surface.transition_x)
    if isfinite(xtr)
        return _transition_xc_from_stations(surface.x, surface.xc, xtr)
    end

    for i in 2:n
        prev = surface.mode[i - 1]
        cur = surface.mode[i]
        if prev in (:laminar, :similarity) && cur in (:turbulent, :wake)
            return float(surface.xc[i - 1])
        end
    end

    first_turbulent = findfirst(m -> m in (:turbulent, :wake), surface.mode)
    return isnothing(first_turbulent) ? 1.0 : float(surface.xc[Int(first_turbulent)])
end

@inline function _prefer_public_transition_xc(
    transition_xc::Real,
    surface::SurfaceMarchResult,
)
    return isfinite(float(transition_xc)) ? float(transition_xc) : _surface_transition_xc(surface)
end

function _transition_xc_from_stations(
    x::AbstractVector{<:Real},
    xc::AbstractVector{<:Real},
    transition_x::Real,
)
    n = length(x)
    n == length(xc) || throw(ArgumentError("station transition coordinate length mismatch"))
    n == 0 && return float(transition_x)

    xtr = float(transition_x)
    !isfinite(xtr) && return xtr
    xtr <= x[1] && return float(xc[1])
    xtr >= x[end] && return float(xc[end])

    for i in 1:(n - 1)
        x1 = float(x[i])
        x2 = float(x[i + 1])
        if x1 - 1.0e-12 <= xtr <= x2 + 1.0e-12
            t = clamp((xtr - x1) / max(x2 - x1, 1.0e-12), 0.0, 1.0)
            c1 = float(xc[i])
            c2 = float(xc[i + 1])
            return (1.0 - t) * c1 + t * c2
        end
    end

    return float(xc[end])
end

function _coupled_wake_station_index(i::Integer, nsurface::Integer)
    return nsurface + i
end

function _coupled_wake_station_block(i::Integer, nsurface::Integer)
    g = _coupled_wake_station_index(i, nsurface)
    return (3 * (g - 1) + 1):(3 * g)
end

function _coupled_wake_interval_row(i::Integer, nsurface::Integer)
    base = 3 * (nsurface + i - 1) + 1
    return base:(base + 2)
end

function _surface_wake_coupling_matrix(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    wake_dij::AbstractMatrix{<:Real},
)
    kss, uinv_s = _surface_coupling_matrix(inv, topo, dij)
    up_idx, up_vti = _surface_station_indices(topo, :upper)
    lo_idx, lo_vti = _surface_station_indices(topo, :lower)
    idxsurf = vcat(up_idx, lo_idx)
    vtisurf = vcat(up_vti, lo_vti)
    nsurface = length(idxsurf)
    nw = size(wake_dij, 2)
    ksw = zeros(Float64, nsurface, nw)
    return kss, ksw, uinv_s, idxsurf, vtisurf
end

function _wake_row_coupling_matrix(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    wake_st::WakeStations,
    wake_dij::AbstractMatrix{<:Real},
    idxsurf::AbstractVector{<:Integer},
    vtisurf::AbstractVector{<:Real},
)
    nsurface = length(idxsurf)
    nw = length(wake_st.x)
    size(wake_dij, 1) == length(inv.qnode) || throw(ArgumentError("wake influence row mismatch"))
    size(wake_dij, 2) == nw || throw(ArgumentError("wake influence column mismatch"))

    kws = zeros(Float64, nw, nsurface)
    kww = zeros(Float64, nw, nw)
    uinv_w = copy(wake_st.ue)
    cache = _wake_geometry_cache(inv)
    wake_ap =
        hasproperty(wake_st, :apanel) && length(wake_st.apanel) == nw ?
        wake_st.apanel : _wake_apanel(cache, wake_st.xw, wake_st.yw)
    te_idx = topo.lower.ipan[end]

    for i in 1:nw
        if i == 1
            for j in 1:nsurface
                kws[i, j] = float(vtisurf[j]) * float(dij[te_idx, idxsurf[j]])
            end
            kww[i, :] .= .-float.(wake_dij[te_idx, :])
            continue
        end

        psi_x, psi_y = _wake_streamfunction_gradient(
            cache,
            length(cache.x) + i,
            wake_st.xw[i],
            wake_st.yw[i],
        )
        gnorm = max(hypot(psi_x, psi_y), 1.0e-12)
        nxi = -psi_x / gnorm
        nyi = -psi_y / gnorm

        _, dqdg_air, _, dqdm_air = _psilin_source_vortex_derivatives(
            0,
            wake_st.xw[i],
            wake_st.yw[i],
            nxi,
            nyi,
            cache.x,
            cache.y,
            cache.s,
            cache.apanel;
            scs = cache.te.scs,
            sds = cache.te.sds,
        )
        _, dqdm_wake = _pswlin_source_derivatives(
            i,
            wake_st.xw[i],
            wake_st.yw[i],
            nxi,
            nyi,
            wake_st.xw,
            wake_st.yw,
            wake_ap;
            eval_on_wake = true,
        )

        di_surface = copy(dqdm_air)
        di_wake = copy(dqdm_wake)
        for j in eachindex(di_surface)
            acc = di_surface[j]
            for k in eachindex(dqdg_air)
                acc += dqdg_air[k] * float(dij[k, j])
            end
            di_surface[j] = acc
        end
        for j in eachindex(di_wake)
            acc = di_wake[j]
            for k in eachindex(dqdg_air)
                acc += dqdg_air[k] * float(wake_dij[k, j])
            end
            di_wake[j] = acc
        end

        for j in 1:nsurface
            kws[i, j] = float(vtisurf[j]) * di_surface[idxsurf[j]]
        end
        kww[i, :] .= .-di_wake
    end

    return kws, kww, uinv_w
end

function _global_coupling_matrix(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    wake_st::WakeStations,
    wake_dij::AbstractMatrix{<:Real},
)
    kss, ksw, uinv_s, idxsurf, vtisurf = _surface_wake_coupling_matrix(inv, topo, dij, wake_dij)
    kws, kww, uinv_w = _wake_row_coupling_matrix(inv, topo, dij, wake_st, wake_dij, idxsurf, vtisurf)

    nsurface = size(kss, 1)
    nw = size(kww, 1)
    k = zeros(Float64, nsurface + nw, nsurface + nw)
    k[1:nsurface, 1:nsurface] .= kss
    k[1:nsurface, (nsurface + 1):(nsurface + nw)] .= ksw
    k[(nsurface + 1):(nsurface + nw), 1:nsurface] .= kws
    k[(nsurface + 1):(nsurface + nw), (nsurface + 1):(nsurface + nw)] .= kww

    return k, vcat(uinv_s, uinv_w)
end

function _wake_interval_systems(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    wake::WakeMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
)
    nw = length(wake.x)
    systems = Vector{BLIntervalSystem}(undef, nw)
    systems[1] = _tesys_interval_system(inv, upper, lower, wake.primary[1]; flow = flow, model = model)
    for i in 2:nw
        systems[i] = _blsys_interval_system(
            :wake,
            wake.primary[i - 1],
            wake.primary[i];
            flow = flow,
            model = model,
            acrit = 0.0,
        )
    end
    return systems
end

function _assemble_interval_mass_row!(
    a::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    row,
    col_prev,
    col_cur,
    sys::BLIntervalSystem,
    pprev::BLPrimaryState,
    pcur::BLPrimaryState,
    ue1_m::AbstractVector{<:Real},
    ue2_m::AbstractVector{<:Real},
    d1_m::AbstractVector{<:Real},
    d2_m::AbstractVector{<:Real},
    due1::Real,
    due2::Real;
    dxi_m::Union{Nothing,AbstractVector{<:Real}} = nothing,
    dxi::Real = 0.0,
)
    a[row, col_prev[1]] .+= sys.vs1[1:3, 1]
    a[row, col_prev[2]] .+= sys.vs1[1:3, 2]
    a[row, col_cur[1]] .+= sys.vs2[1:3, 1]
    a[row, col_cur[2]] .+= sys.vs2[1:3, 2]

    xisens =
        if isnothing(dxi_m)
            zeros(Float64, 3)
        else
            sys.vs1[1:3, 5] .+ sys.vs2[1:3, 5] .+ sys.vsx[1:3]
        end

    for eq in 1:3
        a[row[eq], 3:3:size(a, 2)] .+=
            sys.vs1[eq, 3] .* d1_m .+
            sys.vs1[eq, 4] .* ue1_m .+
            sys.vs2[eq, 3] .* d2_m .+
            sys.vs2[eq, 4] .* ue2_m
        if !isnothing(dxi_m)
            a[row[eq], 3:3:size(a, 2)] .+= xisens[eq] .* dxi_m
        end
    end

    dds1 = -(pprev.d / max(pprev.u, 1.0e-7)) * float(due1)
    dds2 = -(pcur.d / max(pcur.u, 1.0e-7)) * float(due2)
    b[row] .=
        sys.vsrez[1:3] .+
        sys.vs1[1:3, 4] .* float(due1) .+
        sys.vs1[1:3, 3] .* dds1 .+
        sys.vs2[1:3, 4] .* float(due2) .+
        sys.vs2[1:3, 3] .* dds2
    if !isnothing(dxi_m)
        b[row] .+= xisens .* float(dxi)
    end
    return nothing
end

function _tesys_rows!(
    a::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    row,
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    wake::WakeMarchResult,
    k::AbstractMatrix{<:Real},
    forced_due::AbstractVector{<:Real},
)
    nu = length(upper.x)
    nl = length(lower.x)
    nsurface = nu + nl
    ndof = size(a, 2)
    g_u = _coupled_surface_station_index(:upper, nu, nu)
    g_l = _coupled_surface_station_index(:lower, nl, nu)
    g_w = _coupled_wake_station_index(1, nsurface)

    p_u = upper.primary[end]
    p_l = lower.primary[end]
    p_w = wake.primary[1]
    ct_u = _surface_station_ctau(upper, nu)
    ct_l = _surface_station_ctau(lower, nl)
    tte = max(p_u.t + p_l.t, 1.0e-12)
    cte = (ct_u * p_u.t + ct_l * p_l.t) / tte
    dte = p_u.d + p_l.d + _te_gap_geometry(inv).ante
    rez = [cte - p_w.s, tte - p_w.t, dte - p_w.d]

    col_u = _coupled_surface_station_block(:upper, nu, nu)
    col_l = _coupled_surface_station_block(:lower, nl, nu)
    col_w = _coupled_wake_station_block(1, nsurface)

    a[row[1], col_u[1]] += p_u.t / tte
    a[row[1], col_l[1]] += p_l.t / tte
    a[row[1], col_w[1]] += -1.0
    a[row[1], col_u[2]] += (ct_u - cte) / tte
    a[row[1], col_l[2]] += (ct_l - cte) / tte

    a[row[2], col_u[2]] += 1.0
    a[row[2], col_l[2]] += 1.0
    a[row[2], col_w[2]] += -1.0

    ue_u_m = Float64.(k[g_u, :])
    ue_l_m = Float64.(k[g_l, :])
    ue_w_m = Float64.(k[g_w, :])
    d_u_m = _coupled_surface_mass_sensitivities(p_u, ue_u_m, g_u)
    d_l_m = _coupled_surface_mass_sensitivities(p_l, ue_l_m, g_l)
    d_w_m = _coupled_surface_mass_sensitivities(p_w, ue_w_m, g_w)
    a[row[3], 3:3:ndof] .+= d_u_m .+ d_l_m .- d_w_m

    b[row] .= .-rez
    dds1 = -(p_u.d / max(p_u.u, 1.0e-7)) * float(forced_due[g_u]) - (p_l.d / max(p_l.u, 1.0e-7)) * float(forced_due[g_l])
    b[row[3]] += dds1
    return maximum(abs, rez)
end

function _xfoil_blsolv_single_rhs(
    a::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    nu::Integer,
    nl::Integer,
    nw::Integer,
    sspan::Real;
    vaccel::Real = 0.01,
    precision::Type{T} = Float32,
) where {T<:AbstractFloat}
    nsys = nu + nl + nw
    ndof = 3 * nsys
    size(a, 1) == ndof == size(a, 2) ||
        throw(ArgumentError("global coupled system matrix size mismatch"))
    length(b) == ndof || throw(ArgumentError("global coupled system rhs size mismatch"))

    va = zeros(T, 3, 2, nsys)
    vb = zeros(T, 3, 2, nsys)
    vm = zeros(T, 3, nsys, nsys)
    vz = zeros(T, 3, 2)
    vdel = zeros(T, 3, nsys)

    for iv in 1:nsys
        row = (3 * (iv - 1) + 1):(3 * iv)
        col_cur = (3 * (iv - 1) + 1):(3 * iv)
        va[:, 1, iv] .= T.(a[row, col_cur[1]])
        va[:, 2, iv] .= T.(a[row, col_cur[2]])
        for j in 1:nsys
            vm[:, j, iv] .= T.(a[row, 3 * j])
        end
        if iv > 1
            col_prev = (3 * (iv - 2) + 1):(3 * (iv - 1))
            vb[:, 1, iv] .= T.(a[row, col_prev[1]])
            vb[:, 2, iv] .= T.(a[row, col_prev[2]])
        end
        vdel[:, iv] .= T.(b[row])
    end

    ivte1 = nu
    ivz = nu + nl + 1
    if nw > 0 && 1 <= ivte1 <= nsys && 1 <= ivz <= nsys
        row = (3 * (ivz - 1) + 1):(3 * ivz)
        col_u = (3 * (ivte1 - 1) + 1):(3 * ivte1)
        vz[:, 1] .= T.(a[row, col_u[1]])
        vz[:, 2] .= T.(a[row, col_u[2]])
    end

    sspan_t = max(T(abs(float(sspan))), T(1.0e-7))
    vacc1 = T(vaccel)
    vacc2 = vacc1 * T(2.0) / sspan_t
    vacc3 = vacc2

    for iv in 1:nsys
        ivp = iv + 1

        pivot11 = va[1, 1, iv]
        abs(pivot11) > eps(T) || throw(ArgumentError("singular BL solver pivot VA(1,1,$iv)"))
        pivot = inv(pivot11)
        va[1, 2, iv] *= pivot
        for l in iv:nsys
            vm[1, l, iv] *= pivot
        end
        vdel[1, iv] *= pivot

        for k in 2:3
            vtmp = va[k, 1, iv]
            va[k, 2, iv] -= vtmp * va[1, 2, iv]
            for l in iv:nsys
                vm[k, l, iv] -= vtmp * vm[1, l, iv]
            end
            vdel[k, iv] -= vtmp * vdel[1, iv]
        end

        pivot22 = va[2, 2, iv]
        abs(pivot22) > eps(T) || throw(ArgumentError("singular BL solver pivot VA(2,2,$iv)"))
        pivot = inv(pivot22)
        for l in iv:nsys
            vm[2, l, iv] *= pivot
        end
        vdel[2, iv] *= pivot

        vtmp = va[3, 2, iv]
        for l in iv:nsys
            vm[3, l, iv] -= vtmp * vm[2, l, iv]
        end
        vdel[3, iv] -= vtmp * vdel[2, iv]

        pivot33 = vm[3, iv, iv]
        abs(pivot33) > eps(T) || throw(ArgumentError("singular BL solver pivot VM(3,$iv,$iv)"))
        pivot = inv(pivot33)
        for l in ivp:nsys
            vm[3, l, iv] *= pivot
        end
        vdel[3, iv] *= pivot

        vtmp1 = vm[1, iv, iv]
        vtmp2 = vm[2, iv, iv]
        for l in ivp:nsys
            vm[1, l, iv] -= vtmp1 * vm[3, l, iv]
            vm[2, l, iv] -= vtmp2 * vm[3, l, iv]
        end
        vdel[1, iv] -= vtmp1 * vdel[3, iv]
        vdel[2, iv] -= vtmp2 * vdel[3, iv]

        vtmp = va[1, 2, iv]
        for l in ivp:nsys
            vm[1, l, iv] -= vtmp * vm[2, l, iv]
        end
        vdel[1, iv] -= vtmp * vdel[2, iv]

        iv == nsys && continue

        for k in 1:3
            vtmp1 = vb[k, 1, ivp]
            vtmp2 = vb[k, 2, ivp]
            vtmp3 = vm[k, iv, ivp]
            for l in ivp:nsys
                vm[k, l, ivp] -=
                    vtmp1 * vm[1, l, iv] + vtmp2 * vm[2, l, iv] + vtmp3 * vm[3, l, iv]
            end
            vdel[k, ivp] -= vtmp1 * vdel[1, iv] + vtmp2 * vdel[2, iv] + vtmp3 * vdel[3, iv]
        end

        if iv == ivte1 && ivz <= nsys
            for k in 1:3
                vtmp1 = vz[k, 1]
                vtmp2 = vz[k, 2]
                for l in ivp:nsys
                    vm[k, l, ivz] -= vtmp1 * vm[1, l, iv] + vtmp2 * vm[2, l, iv]
                end
                vdel[k, ivz] -= vtmp1 * vdel[1, iv] + vtmp2 * vdel[2, iv]
            end
        end

        ivp == nsys && continue

        for kv in (iv + 2):nsys
            vtmp1 = vm[1, iv, kv]
            vtmp2 = vm[2, iv, kv]
            vtmp3 = vm[3, iv, kv]

            if abs(vtmp1) > vacc1
                for l in ivp:nsys
                    vm[1, l, kv] -= vtmp1 * vm[3, l, iv]
                end
                vdel[1, kv] -= vtmp1 * vdel[3, iv]
            end

            if abs(vtmp2) > vacc2
                for l in ivp:nsys
                    vm[2, l, kv] -= vtmp2 * vm[3, l, iv]
                end
                vdel[2, kv] -= vtmp2 * vdel[3, iv]
            end

            if abs(vtmp3) > vacc3
                for l in ivp:nsys
                    vm[3, l, kv] -= vtmp3 * vm[3, l, iv]
                end
                vdel[3, kv] -= vtmp3 * vdel[3, iv]
            end
        end
    end

    for iv in nsys:-1:2
        vtmp = vdel[3, iv]
        for kv in (iv - 1):-1:1
            vdel[1, kv] -= vm[1, iv, kv] * vtmp
            vdel[2, kv] -= vm[2, iv, kv] * vtmp
            vdel[3, kv] -= vm[3, iv, kv] * vtmp
        end
    end

    delta = zeros(Float64, ndof)
    for iv in 1:nsys
        row = (3 * (iv - 1) + 1):(3 * iv)
        delta[row] .= Float64.(vdel[:, iv])
    end
    return delta
end

function _coupled_global_update_metrics(
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    wake::WakeMarchResult,
    delta::AbstractVector{<:Real},
    due::AbstractVector{<:Real},
)
    nu = length(upper.x)
    nl = length(lower.x)
    nw = length(wake.x)
    ntotal = nu + nl + nw
    length(delta) == 3 * ntotal || throw(ArgumentError("global update vector length mismatch"))
    length(due) == ntotal || throw(ArgumentError("global Ue delta length mismatch"))

    dhi = 1.5
    dlo = -0.5
    rlx = 1.0
    rms = 0.0
    rmx = 0.0
    loc = (side = :upper, station = 1, variable = :ampl)

    for (side, primary, modefun, base_station) in (
        (:upper, upper.primary, i -> _surface_station_unknown_mode(upper, i), 0),
        (:lower, lower.primary, i -> _surface_station_unknown_mode(lower, i), nu),
        (:wake, wake.primary, _ -> :wake, nu + nl),
    )
        for i in eachindex(primary)
            g = base_station + i
            off = 3 * (g - 1)
            p = primary[i]
            mode = modefun(i)
            dz1 = float(delta[off + 1])
            dz2 = float(delta[off + 2])
            dm = float(delta[off + 3])
            dui = float(due[g])
            dd = (dm - p.d * dui) / max(p.u, 1.0e-7)
            dn1 =
                mode == :laminar ? dz1 / 10.0 : dz1 / max(p.s, 1.0e-12)
            dn2 = dz2 / max(p.t, 1.0e-12)
            dn3 = dd / max(p.d, 1.0e-12)
            dn4 = abs(dui) / 0.25

            rms += dn1^2 + dn2^2 + dn3^2 + dn4^2
            for (dn, tag) in ((dn1, :ampl), (dn2, :theta), (dn3, :dstar), (dn4, :ue))
                if abs(dn) > abs(rmx)
                    rmx = tag == :ue ? dui : dn
                    loc = (side = side, station = i, variable = tag)
                end
                if rlx * dn > dhi
                    rlx = dhi / dn
                elseif rlx * dn < dlo
                    rlx = dlo / dn
                end
            end
        end
    end

    rms = sqrt(rms / (4 * max(ntotal, 1)))
    return (rlx = clamp(rlx, 0.0, 1.0), rms = rms, rmx = rmx, loc = loc)
end

function _coupled_wake_apply_update(
    wake::WakeMarchResult,
    nsurface::Integer,
    delta::AbstractVector{<:Real},
    due::AbstractVector{<:Real};
    rlx::Float64,
    flow::BLFlowParams,
    model::BLModelParams,
)
    n = length(wake.x)
    primary = copy(wake.primary)
    secondary = copy(wake.secondary)
    for i in 1:n
        col = _coupled_wake_station_block(i, nsurface)
        g = _coupled_wake_station_index(i, nsurface)
        pnew = _coupled_surface_updated_station(
            wake.primary[i],
            :wake,
            wake.x[i],
            delta[col[1]],
            delta[col[2]],
            delta[col[3]],
            due[g],
            rlx;
            flow = flow,
        )
        primary[i] = pnew
        secondary[i] = blsecondary_state(pnew, :wake; flow = flow, model = model)
    end
    return WakeMarchResult(
        copy(wake.x),
        [p.u for p in primary],
        copy(wake.dw),
        copy(wake.xw),
        copy(wake.yw),
        copy(wake.apanel),
        primary,
        secondary,
        copy(wake.converged_intervals),
    )
end

function _coupled_global_newton_step(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper_st::SurfaceStations,
    lower_st::SurfaceStations,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    wake_st::WakeStations,
    wake::WakeMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    linear_solver::Symbol = :xfoil,
)
    nu = length(upper.x)
    nl = length(lower.x)
    nw = length(wake.x)
    nsurface = nu + nl
    ntotal = nsurface + nw
    ndof = 3 * ntotal

    wake_dij = _surface_wake_source_influence(inv, wake_st)
    k, uinv = _global_coupling_matrix(inv, topo, dij, wake_st, wake_dij)

    a = zeros(Float64, ndof, ndof)
    b = zeros(Float64, ndof)
    residual = 0.0

    upper_sys = _surface_interval_systems(upper; flow = flow, model = model, ncrit = ncrit)
    lower_sys = _surface_interval_systems(lower; flow = flow, model = model, ncrit = ncrit)
    wake_sys = _wake_interval_systems(inv, upper, lower, wake; flow = flow, model = model)

    primaries = vcat(upper.primary, lower.primary, wake.primary)
    mass = [p.d * p.u for p in primaries]
    ue_current = [p.u for p in primaries]
    ue_coupled = uinv .+ k * mass
    ue_mismatch = ue_coupled .- ue_current
    forced_due = .-ue_mismatch

    ule_upper = 1
    ule_lower = nu + 1
    ule_m = Float64.(k[ule_upper, :])
    lle_m = Float64.(k[ule_lower, :])

    row_u = _coupled_surface_interval_row(:upper, 1, nu)
    col_u = _coupled_surface_station_block(:upper, 1, nu)
    if upper.solved_similarity_station
        ue_u1_m = Float64.(k[ule_upper, :])
        d_u1_m = _coupled_surface_mass_sensitivities(upper.primary[1], ue_u1_m, ule_upper)
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :upper)
        dxi_u_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi_u = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        sys_u1 = _blsys_interval_system(
            :similarity,
            upper.primary[1],
            upper.primary[1];
            flow = flow,
            model = model,
            acrit = ncrit,
        )
        _assemble_similarity_station_mass_row!(
            a,
            b,
            row_u,
            col_u,
            sys_u1,
            upper.primary[1],
            ue_u1_m,
            d_u1_m,
            forced_due[ule_upper];
            dxi_m = dxi_u_m,
            dxi = dxi_u,
        )
        residual = max(residual, maximum(abs, sys_u1.residual), abs(ue_mismatch[ule_upper]))
    else
        rez_u, jac_u, jacm_u, rhs_u = _approx_similarity_station_mass_row(
            upper.primary[1],
            k[ule_upper, :],
            ule_upper,
            forced_due[ule_upper];
            flow = flow,
        )
        a[row_u, col_u[1]] .+= jac_u[:, 1]
        a[row_u, col_u[2]] .+= jac_u[:, 2]
        a[row_u, 3:3:ndof] .+= jacm_u
        b[row_u] .= rhs_u
        residual = max(residual, maximum(abs, rez_u), abs(ue_mismatch[ule_upper]))
    end

    row_l = _coupled_surface_interval_row(:lower, 1, nu)
    col_l = _coupled_surface_station_block(:lower, 1, nu)
    if lower.solved_similarity_station
        ue_l1_m = Float64.(k[ule_lower, :])
        d_l1_m = _coupled_surface_mass_sensitivities(lower.primary[1], ue_l1_m, ule_lower)
        xi_l1, xi_l2 = _coupled_surface_xi_sensitivity(topo, :lower)
        dxi_l_m = xi_l1 .* ule_m .+ xi_l2 .* lle_m
        dxi_l = xi_l1 * forced_due[ule_upper] + xi_l2 * forced_due[ule_lower]
        sys_l1 = _blsys_interval_system(
            :similarity,
            lower.primary[1],
            lower.primary[1];
            flow = flow,
            model = model,
            acrit = ncrit,
        )
        _assemble_similarity_station_mass_row!(
            a,
            b,
            row_l,
            col_l,
            sys_l1,
            lower.primary[1],
            ue_l1_m,
            d_l1_m,
            forced_due[ule_lower];
            dxi_m = dxi_l_m,
            dxi = dxi_l,
        )
        residual = max(residual, maximum(abs, sys_l1.residual), abs(ue_mismatch[ule_lower]))
    else
        rez_l, jac_l, jacm_l, rhs_l = _approx_similarity_station_mass_row(
            lower.primary[1],
            k[ule_lower, :],
            ule_lower,
            forced_due[ule_lower];
            flow = flow,
        )
        a[row_l, col_l[1]] .+= jac_l[:, 1]
        a[row_l, col_l[2]] .+= jac_l[:, 2]
        a[row_l, 3:3:ndof] .+= jacm_l
        b[row_l] .= rhs_l
        residual = max(residual, maximum(abs, rez_l), abs(ue_mismatch[ule_lower]))
    end

    for i in 2:nu
        sys = upper_sys[i - 1]
        row = _coupled_surface_interval_row(:upper, i, nu)
        gprev = _coupled_surface_station_index(:upper, i - 1, nu)
        gcur = _coupled_surface_station_index(:upper, i, nu)
        pprev = upper.primary[i - 1]
        pcur = upper.primary[i]
        ue1_m = Float64.(k[gprev, :])
        ue2_m = Float64.(k[gcur, :])
        d1_m = _coupled_surface_mass_sensitivities(pprev, ue1_m, gprev)
        d2_m = _coupled_surface_mass_sensitivities(pcur, ue2_m, gcur)
        due1 = forced_due[gprev]
        due2 = forced_due[gcur]
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :upper)
        dxi_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        _assemble_interval_mass_row!(
            a,
            b,
            row,
            _coupled_surface_station_block(:upper, i - 1, nu),
            _coupled_surface_station_block(:upper, i, nu),
            sys,
            pprev,
            pcur,
            ue1_m,
            ue2_m,
            d1_m,
            d2_m,
            due1,
            due2;
            dxi_m = dxi_m,
            dxi = dxi,
        )
        residual = max(residual, maximum(abs, sys.residual), abs(due1), abs(due2))
    end

    for i in 2:nl
        sys = lower_sys[i - 1]
        row = _coupled_surface_interval_row(:lower, i, nu)
        gprev = _coupled_surface_station_index(:lower, i - 1, nu)
        gcur = _coupled_surface_station_index(:lower, i, nu)
        pprev = lower.primary[i - 1]
        pcur = lower.primary[i]
        ue1_m = Float64.(k[gprev, :])
        ue2_m = Float64.(k[gcur, :])
        d1_m = _coupled_surface_mass_sensitivities(pprev, ue1_m, gprev)
        d2_m = _coupled_surface_mass_sensitivities(pcur, ue2_m, gcur)
        due1 = forced_due[gprev]
        due2 = forced_due[gcur]
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :lower)
        dxi_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        _assemble_interval_mass_row!(
            a,
            b,
            row,
            _coupled_surface_station_block(:lower, i - 1, nu),
            _coupled_surface_station_block(:lower, i, nu),
            sys,
            pprev,
            pcur,
            ue1_m,
            ue2_m,
            d1_m,
            d2_m,
            due1,
            due2;
            dxi_m = dxi_m,
            dxi = dxi,
        )
        residual = max(residual, maximum(abs, sys.residual), abs(due1), abs(due2))
    end

    row_w1 = _coupled_wake_interval_row(1, nsurface)
    residual = max(residual, _tesys_rows!(a, b, row_w1, inv, upper, lower, wake, k, forced_due))
    residual = max(residual, abs(ue_mismatch[_coupled_wake_station_index(1, nsurface)]))

    for i in 2:nw
        sys = wake_sys[i]
        row = _coupled_wake_interval_row(i, nsurface)
        gprev = _coupled_wake_station_index(i - 1, nsurface)
        gcur = _coupled_wake_station_index(i, nsurface)
        pprev = wake.primary[i - 1]
        pcur = wake.primary[i]
        ue1_m = Float64.(k[gprev, :])
        ue2_m = Float64.(k[gcur, :])
        d1_m = _coupled_surface_mass_sensitivities(pprev, ue1_m, gprev)
        d2_m = _coupled_surface_mass_sensitivities(pcur, ue2_m, gcur)
        due1 = forced_due[gprev]
        due2 = forced_due[gcur]
        _assemble_interval_mass_row!(
            a,
            b,
            row,
            _coupled_wake_station_block(i - 1, nsurface),
            _coupled_wake_station_block(i, nsurface),
            sys,
            pprev,
            pcur,
            ue1_m,
            ue2_m,
            d1_m,
            d2_m,
            due1,
            due2,
        )
        residual = max(residual, maximum(abs, sys.residual), abs(due1), abs(due2))
    end

    delta =
        try
            if linear_solver == :xfoil
                _xfoil_blsolv_single_rhs(
                    a,
                    b,
                    nu,
                    nl,
                    nw,
                    topo.s[end] - topo.s[1];
                    vaccel = 0.01,
                    precision = Float32,
                )
            elseif linear_solver == :dense
                a \ b
            else
                throw(ArgumentError("unknown coupled global Newton linear solver: $linear_solver"))
            end
        catch err
            return (
                ok = false,
                reason = sprint(showerror, err),
                upper = upper,
                lower = lower,
                wake = wake,
                residual = residual,
                rlx = 0.0,
                rms = Inf,
                rmx = Inf,
                max_location = (side = :none, station = 0, variable = :none),
            )
        end

    if any(!isfinite, delta)
        return (
            ok = false,
            reason = "non-finite global coupled Newton delta",
            upper = upper,
            lower = lower,
            wake = wake,
            residual = residual,
            rlx = 0.0,
            rms = Inf,
            rmx = Inf,
            max_location = (side = :none, station = 0, variable = :none),
        )
    end

    dmass = delta[3:3:ndof]
    due = ue_mismatch .+ k * dmass
    metrics = _coupled_global_update_metrics(upper, lower, wake, delta, due)
    rlx = metrics.rlx

    upper_new = _coupled_surface_apply_update(
        upper,
        upper_st,
        :upper,
        nu,
        delta,
        due;
        rlx = rlx,
        flow = flow,
        model = model,
    )
    lower_new = _coupled_surface_apply_update(
        lower,
        lower_st,
        :lower,
        nu,
        delta,
        due;
        rlx = rlx,
        flow = flow,
        model = model,
    )
    wake_new = _coupled_wake_apply_update(
        wake,
        nsurface,
        delta,
        due;
        rlx = rlx,
        flow = flow,
        model = model,
    )

    upper_sys_new = _surface_interval_systems(upper_new; flow = flow, model = model, ncrit = ncrit)
    lower_sys_new = _surface_interval_systems(lower_new; flow = flow, model = model, ncrit = ncrit)
    upper_xtr = _coupled_surface_transition_x(upper_new, upper_st, upper_sys_new)
    lower_xtr = _coupled_surface_transition_x(lower_new, lower_st, lower_sys_new)
    upper_new = _surface_reclassify_transition(
        upper_new,
        upper_xtr;
        flow = flow,
        model = model,
    )
    lower_new = _surface_reclassify_transition(
        lower_new,
        lower_xtr;
        flow = flow,
        model = model,
    )

    return (
        ok = true,
        reason = "",
        upper = upper_new,
        lower = lower_new,
        wake = wake_new,
        residual = residual,
        rlx = rlx,
        rms = metrics.rms,
        rmx = metrics.rmx,
        max_location = metrics.loc,
        delta = delta,
        due = due,
        ue_mismatch = ue_mismatch,
    )
end

function _coupled_global_newton_solve(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper_st::SurfaceStations,
    lower_st::SurfaceStations,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    wake_st::WakeStations,
    wake::WakeMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-3,
    max_iter::Int = 4,
)
    ucur = upper
    lcur = lower
    wcur = wake
    ubest = upper
    lbest = lower
    wbest = wake
    history = NamedTuple[]
    status = :max_iter
    failure_message = ""
    best_residual = Inf
    best_iteration = 0

    for iter in 1:max_iter
        step = _coupled_global_newton_step(
            inv,
            topo,
            dij,
            upper_st,
            lower_st,
            ucur,
            lcur,
            wake_st,
            wcur;
            flow = flow,
            model = model,
            ncrit = ncrit,
        )
        push!(history, (
            iteration = iter,
            residual = step.residual,
            rlx = step.rlx,
            rms = step.rms,
            rmx = step.rmx,
            max_location = step.max_location,
            ok = step.ok,
        ))
        if !step.ok
            status = :step_failure
            failure_message = step.reason
            break
        end
        ucur = step.upper
        lcur = step.lower
        wcur = step.wake
        if isfinite(step.residual) && step.residual < best_residual
            best_residual = step.residual
            best_iteration = iter
            ubest = step.upper
            lbest = step.lower
            wbest = step.wake
        end
        if step.residual <= tol
            status = :converged
            break
        end
    end

    residual = best_iteration == 0 ? (isempty(history) ? Inf : history[end].residual) : best_residual
    return (
        upper = best_iteration == 0 ? ucur : ubest,
        lower = best_iteration == 0 ? lcur : lbest,
        wake = best_iteration == 0 ? wcur : wbest,
        history = history,
        iterations = length(history),
        residual = residual,
        converged = status == :converged,
        status = status,
        failure_message = failure_message,
        best_iteration = best_iteration,
        best_residual = best_residual,
    )
end

function _coupled_surface_newton_step(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper_st::SurfaceStations,
    lower_st::SurfaceStations,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    wake = nothing,
    wake_dij::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    rlx_cap::Float64 = Inf,
)
    nu = length(upper.x)
    nl = length(lower.x)
    nu == length(upper_st.x) || throw(ArgumentError("upper station length mismatch"))
    nl == length(lower_st.x) || throw(ArgumentError("lower station length mismatch"))

    ntotal = nu + nl
    ndof = 3 * ntotal
    A = zeros(Float64, ndof, ndof)
    b = zeros(Float64, ndof)
    residual = 0.0

    upper_sys = _surface_interval_systems(upper; flow = flow, model = model, ncrit = ncrit)
    lower_sys = _surface_interval_systems(lower; flow = flow, model = model, ncrit = ncrit)
    K, uinv = _surface_coupling_matrix(inv, topo, dij)
    primaries = vcat(upper.primary, lower.primary)
    mass = [p.d * p.u for p in primaries]
    ue_current = [p.u for p in primaries]
    ue_coupled = uinv .+ K * mass

    # Add wake mass-defect contribution to surface Ue if wake info is provided.
    # wake_bias[i] = sum_k -vti_surface[i] * (-1) * dij[panel_i, N+k] * wake_mass[k]
    if !isnothing(wake) && !isnothing(wake_dij)
        up_idx, up_vti = _surface_station_indices(topo, :upper)
        lo_idx, lo_vti = _surface_station_indices(topo, :lower)
        s_idx = vcat(up_idx, lo_idx)
        s_vti = vcat(up_vti, lo_vti)
        wake_mass = [max(p.d, 0.0) * max(p.u, 1e-7) for p in wake.primary]
        nw = length(wake_mass)
        for i in 1:length(s_idx)
            bias = 0.0
            for k in 1:nw
                # wake vti = -1, wake panel = N+k, dij cols N+1:N+NW for wake
                if s_idx[i] >= 1 && s_idx[i] <= size(wake_dij, 1) && k <= size(wake_dij, 2)
                    bias += s_vti[i] * wake_dij[s_idx[i], k] * wake_mass[k]
                end
            end
            ue_coupled[i] += bias
        end
    end

    ue_mismatch = ue_coupled .- ue_current
    forced_due = .-ue_mismatch
    ule_upper = 1
    ule_lower = nu + 1
    ule_m = Float64.(K[ule_upper, :])
    lle_m = Float64.(K[ule_lower, :])

    row_u = _coupled_surface_interval_row(:upper, 1, nu)
    col_u = _coupled_surface_station_block(:upper, 1, nu)
    if upper.solved_similarity_station
        ue_u1_m = Float64.(K[ule_upper, :])
        d_u1_m = _coupled_surface_mass_sensitivities(upper.primary[1], ue_u1_m, ule_upper)
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :upper)
        dxi_u_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi_u = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        sys_u1 = _blsys_interval_system(
            :similarity,
            upper.primary[1],
            upper.primary[1];
            flow = flow,
            model = model,
            acrit = ncrit,
        )
        _assemble_similarity_station_mass_row!(
            A,
            b,
            row_u,
            col_u,
            sys_u1,
            upper.primary[1],
            ue_u1_m,
            d_u1_m,
            forced_due[ule_upper];
            dxi_m = dxi_u_m,
            dxi = dxi_u,
        )
        residual = max(residual, maximum(abs, sys_u1.residual), abs(ue_mismatch[ule_upper]))
    else
        rez_u, jac_u, jacm_u, rhs_u = _approx_similarity_station_mass_row(
            upper.primary[1],
            K[ule_upper, :],
            ule_upper,
            forced_due[ule_upper];
            flow = flow,
        )
        A[row_u, col_u[1]] .+= jac_u[:, 1]
        A[row_u, col_u[2]] .+= jac_u[:, 2]
        A[row_u, 3:3:ndof] .+= jacm_u
        b[row_u] .= rhs_u
        residual = max(residual, maximum(abs, rez_u), abs(ue_mismatch[ule_upper]))
    end

    row_l = _coupled_surface_interval_row(:lower, 1, nu)
    col_l = _coupled_surface_station_block(:lower, 1, nu)
    if lower.solved_similarity_station
        ue_l1_m = Float64.(K[ule_lower, :])
        d_l1_m = _coupled_surface_mass_sensitivities(lower.primary[1], ue_l1_m, ule_lower)
        xi_l1, xi_l2 = _coupled_surface_xi_sensitivity(topo, :lower)
        dxi_l_m = xi_l1 .* ule_m .+ xi_l2 .* lle_m
        dxi_l = xi_l1 * forced_due[ule_upper] + xi_l2 * forced_due[ule_lower]
        sys_l1 = _blsys_interval_system(
            :similarity,
            lower.primary[1],
            lower.primary[1];
            flow = flow,
            model = model,
            acrit = ncrit,
        )
        _assemble_similarity_station_mass_row!(
            A,
            b,
            row_l,
            col_l,
            sys_l1,
            lower.primary[1],
            ue_l1_m,
            d_l1_m,
            forced_due[ule_lower];
            dxi_m = dxi_l_m,
            dxi = dxi_l,
        )
        residual = max(residual, maximum(abs, sys_l1.residual), abs(ue_mismatch[ule_lower]))
    else
        rez_l, jac_l, jacm_l, rhs_l = _approx_similarity_station_mass_row(
            lower.primary[1],
            K[ule_lower, :],
            ule_lower,
            forced_due[ule_lower];
            flow = flow,
        )
        A[row_l, col_l[1]] .+= jac_l[:, 1]
        A[row_l, col_l[2]] .+= jac_l[:, 2]
        A[row_l, 3:3:ndof] .+= jacm_l
        b[row_l] .= rhs_l
        residual = max(residual, maximum(abs, rez_l), abs(ue_mismatch[ule_lower]))
    end

    for i in 2:nu
        sys = upper_sys[i - 1]
        row = _coupled_surface_interval_row(:upper, i, nu)
        gprev = _coupled_surface_station_index(:upper, i - 1, nu)
        gcur = _coupled_surface_station_index(:upper, i, nu)
        pprev = upper.primary[i - 1]
        pcur = upper.primary[i]
        ue1_m = Float64.(K[gprev, :])
        ue2_m = Float64.(K[gcur, :])
        d1_m = _coupled_surface_mass_sensitivities(pprev, ue1_m, gprev)
        d2_m = _coupled_surface_mass_sensitivities(pcur, ue2_m, gcur)
        due1 = forced_due[gprev]
        due2 = forced_due[gcur]
        dds1 = -(pprev.d / max(pprev.u, 1.0e-7)) * due1
        dds2 = -(pcur.d / max(pcur.u, 1.0e-7)) * due2
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :upper)
        dxi_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        xisens = sys.vs1[1:3, 5] .+ sys.vs2[1:3, 5] .+ sys.vsx[1:3]

        col_cur = _coupled_surface_station_block(:upper, i, nu)
        col_prev = _coupled_surface_station_block(:upper, i - 1, nu)
        A[row, col_prev[1]] .+= sys.vs1[1:3, 1]
        A[row, col_prev[2]] .+= sys.vs1[1:3, 2]
        A[row, col_cur[1]] .+= sys.vs2[1:3, 1]
        A[row, col_cur[2]] .+= sys.vs2[1:3, 2]
        for eq in 1:3
            A[row[eq], 3:3:ndof] .+=
                sys.vs1[eq, 3] .* d1_m .+
                sys.vs1[eq, 4] .* ue1_m .+
                sys.vs2[eq, 3] .* d2_m .+
                sys.vs2[eq, 4] .* ue2_m .+
                xisens[eq] .* dxi_m
        end
        b[row] .=
            sys.vsrez[1:3] .+
            sys.vs1[1:3, 4] .* due1 .+
            sys.vs1[1:3, 3] .* dds1 .+
            sys.vs2[1:3, 4] .* due2 .+
            sys.vs2[1:3, 3] .* dds2 .+
            xisens .* dxi
        residual = max(residual, maximum(abs, sys.residual), abs(due1), abs(due2))
    end

    for i in 2:nl
        sys = lower_sys[i - 1]
        row = _coupled_surface_interval_row(:lower, i, nu)
        gprev = _coupled_surface_station_index(:lower, i - 1, nu)
        gcur = _coupled_surface_station_index(:lower, i, nu)
        pprev = lower.primary[i - 1]
        pcur = lower.primary[i]
        ue1_m = Float64.(K[gprev, :])
        ue2_m = Float64.(K[gcur, :])
        d1_m = _coupled_surface_mass_sensitivities(pprev, ue1_m, gprev)
        d2_m = _coupled_surface_mass_sensitivities(pcur, ue2_m, gcur)
        due1 = forced_due[gprev]
        due2 = forced_due[gcur]
        dds1 = -(pprev.d / max(pprev.u, 1.0e-7)) * due1
        dds2 = -(pcur.d / max(pcur.u, 1.0e-7)) * due2
        xi_u1, xi_u2 = _coupled_surface_xi_sensitivity(topo, :lower)
        dxi_m = xi_u1 .* ule_m .+ xi_u2 .* lle_m
        dxi = xi_u1 * forced_due[ule_upper] + xi_u2 * forced_due[ule_lower]
        xisens = sys.vs1[1:3, 5] .+ sys.vs2[1:3, 5] .+ sys.vsx[1:3]

        col_cur = _coupled_surface_station_block(:lower, i, nu)
        col_prev = _coupled_surface_station_block(:lower, i - 1, nu)
        A[row, col_prev[1]] .+= sys.vs1[1:3, 1]
        A[row, col_prev[2]] .+= sys.vs1[1:3, 2]
        A[row, col_cur[1]] .+= sys.vs2[1:3, 1]
        A[row, col_cur[2]] .+= sys.vs2[1:3, 2]
        for eq in 1:3
            A[row[eq], 3:3:ndof] .+=
                sys.vs1[eq, 3] .* d1_m .+
                sys.vs1[eq, 4] .* ue1_m .+
                sys.vs2[eq, 3] .* d2_m .+
                sys.vs2[eq, 4] .* ue2_m .+
                xisens[eq] .* dxi_m
        end
        b[row] .=
            sys.vsrez[1:3] .+
            sys.vs1[1:3, 4] .* due1 .+
            sys.vs1[1:3, 3] .* dds1 .+
            sys.vs2[1:3, 4] .* due2 .+
            sys.vs2[1:3, 3] .* dds2 .+
            xisens .* dxi
        residual = max(residual, maximum(abs, sys.residual), abs(due1), abs(due2))
    end

    delta =
        try
            A \ b
        catch err
            return (
                ok = false,
                reason = sprint(showerror, err),
                upper = upper,
                lower = lower,
                upper_systems = upper_sys,
                lower_systems = lower_sys,
                residual = residual,
                rlx = 0.0,
                rms = Inf,
                rmx = Inf,
                max_location = (side = :none, station = 0, variable = :none),
            )
        end

    if any(!isfinite, delta)
        return (
            ok = false,
            reason = "non-finite coupled Newton delta",
            upper = upper,
            lower = lower,
            upper_systems = upper_sys,
            lower_systems = lower_sys,
            residual = residual,
            rlx = 0.0,
            rms = Inf,
            rmx = Inf,
            max_location = (side = :none, station = 0, variable = :none),
        )
    end

    dmass = delta[3:3:ndof]
    due = ue_mismatch .+ K * dmass
    metrics = _coupled_surface_update_metrics(upper, lower, delta, due)
    rlx = min(metrics.rlx, rlx_cap)
    upper_new = _coupled_surface_apply_update(
        upper,
        upper_st,
        :upper,
        nu,
        delta,
        due;
        rlx = rlx,
        flow = flow,
        model = model,
    )
    lower_new = _coupled_surface_apply_update(
        lower,
        lower_st,
        :lower,
        nu,
        delta,
        due;
        rlx = rlx,
        flow = flow,
        model = model,
    )
    upper_sys_new = _surface_interval_systems(upper_new; flow = flow, model = model, ncrit = ncrit)
    lower_sys_new = _surface_interval_systems(lower_new; flow = flow, model = model, ncrit = ncrit)
    upper_xtr = _coupled_surface_transition_x(upper_new, upper_st, upper_sys_new)
    lower_xtr = _coupled_surface_transition_x(lower_new, lower_st, lower_sys_new)
    upper_new = _surface_reclassify_transition(
        upper_new,
        upper_xtr;
        flow = flow,
        model = model,
    )
    lower_new = _surface_reclassify_transition(
        lower_new,
        lower_xtr;
        flow = flow,
        model = model,
    )

    return (
        ok = true,
        reason = "",
        upper = upper_new,
        lower = lower_new,
        upper_systems = upper_sys,
        lower_systems = lower_sys,
        residual = residual,
        rlx = rlx,
        rms = metrics.rms,
        rmx = metrics.rmx,
        max_location = metrics.loc,
    )
end

function _coupled_surface_newton_solve(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper_st::SurfaceStations,
    lower_st::SurfaceStations,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-3,
    max_iter::Int = 4,
    wake = nothing,
    wake_dij::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    rlx_cap::Float64 = Inf,
)
    ucur = upper
    lcur = lower
    ubest = upper
    lbest = lower
    history = NamedTuple[]
    status = :max_iter
    failure_message = ""
    best_residual = Inf
    best_iteration = 0

    for iter in 1:max_iter
        # MRCHDU re-march: re-establish BL consistency with current Ue
        if iter > 1
            ucur_st = _surface_stations(ucur)
            lcur_st = _surface_stations(lcur)
            ucur = _march_surface(
                ucur_st;
                flow = flow, model = model, ncrit = ncrit,
                tol = 1.0e-6, max_iter = 20,
                seed = ucur,
                mrchue_local = false,
                mrchdu_local = true,
                preserve_seed_transition = false,
                solve_similarity_station = true,
            )
            lcur = _march_surface(
                lcur_st;
                flow = flow, model = model, ncrit = ncrit,
                tol = 1.0e-6, max_iter = 20,
                seed = lcur,
                mrchue_local = false,
                mrchdu_local = true,
                preserve_seed_transition = false,
                solve_similarity_station = true,
            )
        end

        step = _coupled_surface_newton_step(
            inv,
            topo,
            dij,
            upper_st,
            lower_st,
            ucur,
            lcur;
            flow = flow,
            model = model,
            ncrit = ncrit,
            wake = wake,
            wake_dij = wake_dij,
            rlx_cap = rlx_cap,
        )
        push!(history, (
            iteration = iter,
            residual = step.residual,
            rlx = step.rlx,
            rms = step.rms,
            rmx = step.rmx,
            max_location = step.max_location,
            ok = step.ok,
        ))
        if !step.ok
            status = :step_failure
            failure_message = step.reason
            break
        end
        ucur = step.upper
        lcur = step.lower
        if isfinite(step.residual) && step.residual < best_residual
            best_residual = step.residual
            best_iteration = iter
            ubest = step.upper
            lbest = step.lower
        end
        if step.residual <= tol
            status = :converged
            break
        end
    end

    residual = best_iteration == 0 ? (isempty(history) ? Inf : history[end].residual) : best_residual
    converged = status == :converged
    return (
        upper = best_iteration == 0 ? ucur : ubest,
        lower = best_iteration == 0 ? lcur : lbest,
        history = history,
        iterations = length(history),
        residual = residual,
        converged = converged,
        status = status,
        failure_message = failure_message,
        best_iteration = best_iteration,
        best_residual = best_residual,
    )
end

function _mrchue_surface_step(
    inv::CaseResult,
    topo::BLTopology,
    dij::AbstractMatrix{<:Real},
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    wake_seed = nothing,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    ueps::Float64 = 1.0e-7,
    wake_coupled::Bool = false,
    newton_max_iter::Int = 1,
    newton_tol::Float64 = 1.0e-3,
)
    upper_st_cur = _surface_stations(upper)
    lower_st_cur = _surface_stations(lower)
    upper_cur = _march_surface(
        upper_st_cur;
        flow = flow,
        model = model,
        ncrit = ncrit,
        tol = tol,
        max_iter = max_iter,
        seed = upper,
        mrchue_local = false,
        mrchdu_local = true,
        preserve_seed_transition = false,
        solve_similarity_station = true,
    )
    lower_cur = _march_surface(
        lower_st_cur;
        flow = flow,
        model = model,
        ncrit = ncrit,
        tol = tol,
        max_iter = max_iter,
        seed = lower,
        mrchue_local = false,
        mrchdu_local = true,
        preserve_seed_transition = false,
        solve_similarity_station = true,
    )
    wake = nothing
    if wake_coupled
        wake =
            try
                _march_wake(
                    inv,
                    upper_cur,
                    lower_cur;
                    flow = flow,
                    model = model,
                    seed = wake_seed,
                    tol = tol,
                    max_iter = max_iter,
                    ueps = ueps,
                    mrchue_local = true,
                )
            catch
                nothing
            end
    end

    newton_status = :skipped
    newton_residual = Inf
    newton_refined = false
    newton_iterations = 0
    newton_iter_cap = max(Int(newton_max_iter), 1)

    function _evaluate_surface_state(upper_q::SurfaceMarchResult, lower_q::SurfaceMarchResult, wake_seed_local = nothing)
        qvis_local = _surface_qnode_from_surfaces(inv, topo, upper_q, lower_q; ueps = ueps)
        topo_local = _xfoil_bl_topology(inv; q = qvis_local)

        upper_st_local = _surface_stations(inv, :upper; ue_panel = qvis_local, topology = topo_local, uemin = ueps)
        lower_st_local = _surface_stations(inv, :lower; ue_panel = qvis_local, topology = topo_local, uemin = ueps)
        upper_new_local, lower_new_local = _xfoil_stmove_results(
            topo,
            topo_local,
            upper_q,
            lower_q,
            upper_st_local,
            lower_st_local;
            flow = flow,
            model = model,
            ueps = ueps,
        )
        qsurf_local = _surface_qnode_from_surfaces(inv, topo_local, upper_new_local, lower_new_local; ueps = ueps)

        wake_new_local = nothing
        wake_dij_new_local = nothing
        if wake_coupled
            wake_new_local =
                try
                _march_wake(
                    inv,
                    upper_new_local,
                    lower_new_local;
                    flow = flow,
                    model = model,
                    seed = wake_seed_local,
                    tol = tol,
                    max_iter = max_iter,
                    ueps = ueps,
                    mrchue_local = true,
                    )
                catch
                    nothing
                end
            wake_dij_new_local =
                if isnothing(wake_new_local)
                    nothing
                else
                    try
                        _surface_wake_source_influence(inv, wake_new_local)
                    catch
                        nothing
                    end
                end
            if isnothing(wake_dij_new_local)
                wake_new_local = nothing
            end
        end

        qvis_next_local = _ueset_surface_qvis(
            inv,
            topo_local,
            dij,
            upper_new_local,
            lower_new_local;
            wake = wake_new_local,
            wake_dij = wake_dij_new_local,
            ueps = ueps,
            include_wake = wake_coupled && !isnothing(wake_new_local) && !isnothing(wake_dij_new_local),
        )
        residual_local = _max_relative_change(qvis_next_local, qsurf_local; floor = ueps)
        return (
            qvis = qvis_local,
            topology = topo_local,
            upper = upper_new_local,
            lower = lower_new_local,
            wake = wake_new_local,
            qvis_next = qvis_next_local,
            qsurf = qsurf_local,
            residual = residual_local,
        )
    end

    state = _evaluate_surface_state(upper_cur, lower_cur, wake)
    base_residual = state.residual
    candidate_residual = Inf
    newton =
        try
            if wake_coupled && !isnothing(wake)
                wake_st = _wake_stations(inv, lower_cur.x[end]; ueps = ueps)
                _coupled_global_newton_solve(
                    inv,
                    topo,
                    dij,
                    upper_st_cur,
                    lower_st_cur,
                    upper_cur,
                    lower_cur,
                    wake_st,
                    wake;
                    flow = flow,
                    model = model,
                    ncrit = ncrit,
                    tol = float(newton_tol),
                    max_iter = newton_iter_cap,
                )
            else
                _coupled_surface_newton_solve(
                    inv,
                    topo,
                    dij,
                    upper_st_cur,
                    lower_st_cur,
                    upper_cur,
                    lower_cur;
                    flow = flow,
                    model = model,
                    ncrit = ncrit,
                    tol = float(newton_tol),
                    max_iter = newton_iter_cap,
                    wake = wake,
                )
            end
        catch
            nothing
        end
    if !isnothing(newton)
        newton_status = newton.status
        newton_residual = newton.residual
        newton_iterations = newton.iterations
        if newton.best_iteration > 0
            wake_seed_candidate = hasproperty(newton, :wake) ? newton.wake : nothing
            candidate = _evaluate_surface_state(newton.upper, newton.lower, wake_seed_candidate)
            candidate_residual = candidate.residual
            if newton.converged && isfinite(candidate.residual) && candidate.residual <= state.residual
                state = candidate
                newton_refined = true
            end
        end
    end

    return (
        topology = state.topology,
        qvis = state.qvis,
        qvis_next = state.qvis_next,
        qsurf = state.qsurf,
        upper = state.upper,
        lower = state.lower,
        wake = state.wake,
        residual = state.residual,
        stagnation_shift = state.topology.stagnation.ist - topo.stagnation.ist,
        base_residual = base_residual,
        candidate_residual = candidate_residual,
        newton_status = newton_status,
        newton_residual = newton_residual,
        newton_refined = newton_refined,
        newton_iterations = newton_iterations,
    )
end

function _mrchue_surface_initialize(
    inv::CaseResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
)
    topo = _xfoil_bl_topology(inv)
    upper = _march_surface(
        _surface_stations(inv, :upper; topology = topo);
        flow = flow,
        model = model,
        ncrit = ncrit,
        tol = tol,
        max_iter = max_iter,
        solve_similarity_station = true,
    )
    lower = _march_surface(
        _surface_stations(inv, :lower; topology = topo);
        flow = flow,
        model = model,
        ncrit = ncrit,
        tol = tol,
        max_iter = max_iter,
        solve_similarity_station = true,
    )
    return (topology = topo, upper = upper, lower = lower)
end

function _mrchue_surface_iteration(
    iter::Integer,
    step,
)
    return (
        iteration = Int(iter),
        residual = step.residual,
        stagnation_shift = step.stagnation_shift,
        upper_stations = length(step.upper.x),
        lower_stations = length(step.lower.x),
        top_xtr = _surface_transition_xc(step.upper),
        bot_xtr = _surface_transition_xc(step.lower),
        base_residual = step.base_residual,
        candidate_residual = step.candidate_residual,
        newton_status = step.newton_status,
        newton_residual = step.newton_residual,
        newton_refined = step.newton_refined,
        newton_iterations = step.newton_iterations,
    )
end

function _mrchue_surface_solve(
    inv::CaseResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    max_coupling_iter::Int = 8,
    coupling_tol::Float64 = 1.0e-3,
    ueps::Float64 = 1.0e-7,
    wake_coupled::Bool = false,
    newton_max_iter::Int = 1,
    newton_tol::Float64 = 1.0e-3,
)
    init = _mrchue_surface_initialize(
        inv;
        flow = flow,
        model = model,
        ncrit = ncrit,
        tol = tol,
        max_iter = max_iter,
    )
    dij = _surface_source_influence(inv)
    topo = init.topology
    upper = init.upper
    lower = init.lower
    wake = nothing
    qvis = copy(inv.qnode)
    qvis_next = copy(inv.qnode)
    topo_best = topo
    upper_best = upper
    lower_best = lower
    wake_best = wake
    qvis_best = copy(qvis)
    qvis_next_best = copy(qvis_next)
    history = NamedTuple[]
    converged = false
    best_iter = 0
    best_residual = Inf
    status = :ok
    failure_message = ""

    for iter in 1:max_coupling_iter
        step =
            try
                _mrchue_surface_step(
                    inv,
                    topo,
                    dij,
                    upper,
                    lower;
                    flow = flow,
                    model = model,
                    ncrit = ncrit,
                    tol = tol,
                    max_iter = max_iter,
                    ueps = ueps,
                    wake_coupled = wake_coupled,
                    wake_seed = wake,
                    newton_max_iter = newton_max_iter,
                    newton_tol = newton_tol,
                )
            catch err
                status = :step_failure
                failure_message = sprint(showerror, err)
                break
            end
        push!(history, _mrchue_surface_iteration(iter, step))

        if isfinite(step.residual) && step.residual < best_residual
            best_residual = step.residual
            best_iter = iter
            topo_best = step.topology
            upper_best = step.upper
            lower_best = step.lower
            wake_best = step.wake
            qvis_best = copy(step.qvis)
            qvis_next_best = copy(step.qvis_next)
        end

        topo = step.topology
        upper = step.upper
        lower = step.lower
        wake = step.wake
        qvis = step.qvis
        qvis_next = step.qvis_next

        if step.stagnation_shift == 0 && step.residual <= coupling_tol
            converged = true
            break
        end
    end

    if converged
        status = :converged
    elseif status == :ok && !isempty(history)
        status = :max_iter
    elseif status == :ok
        status = :no_steps
    end

    residual =
        if isempty(history)
            Inf
        else
            history[end].residual
        end
    iterations = length(history)
    return (
        topology = best_iter == 0 ? topo : topo_best,
        upper = best_iter == 0 ? upper : upper_best,
        lower = best_iter == 0 ? lower : lower_best,
        wake = best_iter == 0 ? wake : wake_best,
        qvis = best_iter == 0 ? qvis : qvis_best,
        qvis_next = best_iter == 0 ? qvis_next : qvis_next_best,
        history = history,
        iterations = iterations,
        converged = converged,
        residual = residual,
        best_iteration = best_iter,
        best_residual = best_residual,
        status = status,
        failure_message = failure_message,
    )
end

function _surface_station_tau(p::BLPrimaryState, s::BLSecondaryState; flow::BLFlowParams)
    kin = blkin_state(p.u, p.t, p.d, flow)
    return 0.5 * kin.r2 * kin.u2^2 * s.cf
end

function _surface_station_ctau(surface::SurfaceMarchResult, i::Integer)
    1 <= i <= length(surface.primary) || throw(BoundsError(surface.primary, i))
    return _surface_station_unknown_mode(surface, i) == :laminar ? surface.primary[i].ampl : surface.primary[i].s
end

function _te_gap_geometry(inv::CaseResult)
    T = _XFOIL_GGCALC_PRECISION
    x = T.(inv.xnode)
    y = T.(inv.ynode)
    n = length(x)
    n == length(y) || throw(ArgumentError("CaseResult node geometry mismatch"))
    n >= 3 || throw(ArgumentError("need at least 3 nodes for TE gap geometry"))

    s = _scalc(x, y)
    xp = _segspl(x, s)
    yp = _segspl(y, s)
    sle = _lefind(x, xp, y, yp, s)

    xle = _seval(sle, x, xp, s)
    yle = _seval(sle, y, yp, s)
    xte = T(0.5) * (x[1] + x[end])
    yte = T(0.5) * (y[1] + y[end])

    dxte = x[1] - x[end]
    dyte = y[1] - y[end]
    dxs = T(0.5) * (-xp[1] + xp[end])
    dys = T(0.5) * (-yp[1] + yp[end])

    ante = dxs * dyte - dys * dxte
    aste = dxs * dxte + dys * dyte
    dste = sqrt(dxte * dxte + dyte * dyte)
    chord = sqrt((xte - xle)^2 + (yte - yle)^2)
    sharp = dste < T(1.0e-4) * max(chord, T(1.0e-12))
    scs = sharp ? one(T) : ante / max(dste, T(1.0e-12))
    sds = sharp ? zero(T) : aste / max(dste, T(1.0e-12))

    return (
        dxte = float(dxte),
        dyte = float(dyte),
        dxs = float(dxs),
        dys = float(dys),
        ante = float(ante),
        aste = float(aste),
        dste = float(dste),
        chord = float(chord),
        sharp = sharp,
        scs = float(scs),
        sds = float(sds),
    )
end

function _te_wake_start_state(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
)
    length(upper.primary) >= 1 || throw(ArgumentError("upper surface must contain at least one station"))
    length(lower.primary) >= 1 || throw(ArgumentError("lower surface must contain at least one station"))

    gap = _te_gap_geometry(inv)
    thu = upper.primary[end].t
    thl = lower.primary[end].t
    tte = thu + thl
    dte = upper.primary[end].d + lower.primary[end].d + gap.ante
    ctu = _surface_station_ctau(upper, length(upper.primary))
    ctl = _surface_station_ctau(lower, length(lower.primary))
    cte = (ctu * thu + ctl * thl) / max(tte, 1.0e-12)
    uete = 0.5 * (upper.ue[end] + lower.ue[end])

    return (
        cte = float(cte),
        tte = float(tte),
        dte = float(dte),
        ante = gap.ante,
        uete = float(uete),
    )
end

function _tesys_interval_system(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult,
    pwake::BLPrimaryState;
    flow::BLFlowParams,
    model::BLModelParams,
)
    te = _te_wake_start_state(inv, upper, lower)
    q2 = _normalize_interval_primary(pwake, :wake)
    s2 = blsecondary_state(q2, :wake; flow = flow, model = model)
    p1 = BLPrimaryState(q2.x, te.uete, te.tte, te.dte; s = te.cte, ampl = 0.0, dw = te.ante)
    s1 = blsecondary_state(p1, :wake; flow = flow, model = model)

    vs1 = zeros(Float64, 4, 5)
    vs2 = zeros(Float64, 4, 5)
    vsrez = zeros(Float64, 4)
    vsm = zeros(Float64, 4)
    vsr = zeros(Float64, 4)
    vsx = zeros(Float64, 4)

    rez = zeros(Float64, 3)

    vs1[1, 1] = -1.0
    vs2[1, 1] = 1.0
    rez[1] = te.cte - q2.s

    vs1[2, 2] = -1.0
    vs2[2, 2] = 1.0
    rez[2] = te.tte - q2.t

    vs1[3, 3] = -1.0
    vs2[3, 3] = 1.0
    rez[3] = te.dte - q2.d
    vsrez[1:3] .= .-rez

    return BLIntervalSystem(
        :wake,
        rez,
        vs1,
        vs2,
        vsrez,
        vsm,
        vsr,
        vsx,
        p1,
        q2,
        s1,
        s2,
        NaN,
        false,
    )
end

function _wake_geometry_cache(inv::CaseResult)
    T = _XFOIL_GGCALC_PRECISION
    x = T.(inv.xnode)
    y = T.(inv.ynode)
    s = _scalc(x, y)
    nx, ny, xp, yp = _ncalc(x, y, s)
    te = _tecalc(x, y, xp, yp)
    apanel = _apcalc(x, y, nx, ny; sharp = te.sharp)
    alpha = T(deg2rad(inv.alpha_deg))
    gap = _te_gap_geometry(inv)
    return (
        x = x,
        y = y,
        s = s,
        xp = xp,
        yp = yp,
        te = te,
        gap = gap,
        apanel = apanel,
        q = T.(inv.qnode),
        sa = sin(alpha),
        ca = cos(alpha),
        chord = T(gap.chord),
    )
end

function _psilin_current_normal_derivative(
    io::Int,
    xi::Real,
    yi::Real,
    nxi::Real,
    nyi::Real,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    apanel::AbstractVector{<:Real},
    q::AbstractVector{<:Real};
    scs::Real,
    sds::Real,
    sa::Real,
    ca::Real,
    qinf::Real = 1.0,
)
    n = length(x)
    n == length(y) == length(s) == length(apanel) == length(q) ||
        throw(ArgumentError("psilin current length mismatch"))

    T = _XFOIL_GGCALC_PRECISION
    pi_t = T(pi)
    qopi = T(0.25) / pi_t
    hopi = T(0.5) / pi_t
    seps = max(T(s[n] - s[1]), T(1.0e-12)) * T(1.0e-5)

    psi_ni = zero(T)
    te_skip = false

    x1 = zero(T)
    x2 = zero(T)
    yy = zero(T)
    g1 = zero(T)
    g2 = zero(T)
    t1 = zero(T)
    t2 = zero(T)
    x1i = zero(T)
    x2i = zero(T)
    yyi = zero(T)
    apan = zero(T)
    jo_te = n
    jp_te = 1

    for jo in 1:n
        jp = jo == n ? 1 : jo + 1
        jo_te = jo
        jp_te = jp

        if jo == n
            d2 = (T(x[jo]) - T(x[jp]))^2 + (T(y[jo]) - T(y[jp]))^2
            if d2 < seps^2
                te_skip = true
                break
            end
        end

        dso = sqrt((T(x[jo]) - T(x[jp]))^2 + (T(y[jo]) - T(y[jp]))^2)
        dso == zero(T) && continue
        dsio = inv(dso)
        apan = T(apanel[jo])

        rx1 = T(xi) - T(x[jo])
        ry1 = T(yi) - T(y[jo])
        rx2 = T(xi) - T(x[jp])
        ry2 = T(yi) - T(y[jp])

        sx = (T(x[jp]) - T(x[jo])) * dsio
        sy = (T(y[jp]) - T(y[jo])) * dsio

        x1 = sx * rx1 + sy * ry1
        x2 = sx * rx2 + sy * ry2
        yy = sx * ry1 - sy * rx1

        rs1 = rx1 * rx1 + ry1 * ry1
        rs2 = rx2 * rx2 + ry2 * ry2
        # Match XFoil PSILIN exactly:
        #   surface rows (1..N) use SGN = 1
        #   wake/off-surface rows use SGN = SIGN(1,YY)
        sgn = if 1 <= io <= n
            one(T)
        else
            copysign(one(T), yy)
        end

        if io != jo && rs1 > zero(T)
            g1 = _xfoil_log(rs1)
            t1 = _xfoil_atan2(sgn * x1, sgn * yy) + (T(0.5) - T(0.5) * sgn) * pi_t
        else
            g1 = zero(T)
            t1 = zero(T)
        end

        if io != jp && rs2 > zero(T)
            g2 = _xfoil_log(rs2)
            t2 = _xfoil_atan2(sgn * x2, sgn * yy) + (T(0.5) - T(0.5) * sgn) * pi_t
        else
            g2 = zero(T)
            t2 = zero(T)
        end

        x1i = sx * T(nxi) + sy * T(nyi)
        x2i = x1i
        yyi = sx * T(nyi) - sy * T(nxi)

        if jo == n
            break
        end

        dxinv = inv(x1 - x2)
        psis = T(0.5) * x1 * g1 - T(0.5) * x2 * g2 + x2 - x1 + yy * (t1 - t2)
        psid =
            ((x1 + x2) * psis + T(0.5) * (rs2 * g2 - rs1 * g1 + x1^2 - x2^2)) * dxinv

        psx1 = T(0.5) * g1
        psx2 = -T(0.5) * g2
        psyy = t1 - t2
        pdx1 = ((x1 + x2) * psx1 + psis - x1 * g1 - psid) * dxinv
        pdx2 = ((x1 + x2) * psx2 + psis + x2 * g2 + psid) * dxinv
        pdyy = ((x1 + x2) * psyy - yy * (g1 - g2)) * dxinv

        psni = psx1 * x1i + psx2 * x2i + psyy * yyi
        pdni = pdx1 * x1i + pdx2 * x2i + pdyy * yyi
        gsum = T(q[jp]) + T(q[jo])
        gdif = T(q[jp]) - T(q[jo])
        psi_ni += qopi * (gsum * psni + gdif * pdni)
    end

    if !te_skip
        psig = T(0.5) * yy * (g1 - g2) + x2 * (t2 - apan) - x1 * (t1 - apan)
        pgam = T(0.5) * x1 * g1 - T(0.5) * x2 * g2 + x2 - x1 + yy * (t1 - t2)

        psigx1 = -(t1 - apan)
        psigx2 = t2 - apan
        psigyy = T(0.5) * (g1 - g2)
        pgamx1 = T(0.5) * g1
        pgamx2 = -T(0.5) * g2
        pgamyy = t1 - t2

        psigni = psigx1 * x1i + psigx2 * x2i + psigyy * yyi
        pgamni = pgamx1 * x1i + pgamx2 * x2i + pgamyy * yyi

        sigte = T(0.5) * T(scs) * (T(q[jp_te]) - T(q[jo_te]))
        gamte = -T(0.5) * T(sds) * (T(q[jp_te]) - T(q[jo_te]))
        psi_ni += hopi * (psigni * sigte + pgamni * gamte)
    end

    psi_ni += T(qinf) * (T(ca) * T(nyi) - T(sa) * T(nxi))
    return psi_ni
end

function _wake_streamfunction_gradient(cache, io::Int, xi::Real, yi::Real)
    psi_x = _psilin_current_normal_derivative(
        io,
        xi,
        yi,
        1.0,
        0.0,
        cache.x,
        cache.y,
        cache.s,
        cache.apanel,
        cache.q;
        scs = cache.te.scs,
        sds = cache.te.sds,
        sa = cache.sa,
        ca = cache.ca,
    )
    psi_y = _psilin_current_normal_derivative(
        io,
        xi,
        yi,
        0.0,
        1.0,
        cache.x,
        cache.y,
        cache.s,
        cache.apanel,
        cache.q;
        scs = cache.te.scs,
        sds = cache.te.sds,
        sa = cache.sa,
        ca = cache.ca,
    )
    return psi_x, psi_y
end

function _wake_streamfunction_gradient(cache, xi::Real, yi::Real)
    return _wake_streamfunction_gradient(cache, 0, xi, yi)
end

function _wake_streamfunction_normal_derivative(cache, io::Int, xi::Real, yi::Real, nxi::Real, nyi::Real)
    return _psilin_current_normal_derivative(
        io,
        xi,
        yi,
        nxi,
        nyi,
        cache.x,
        cache.y,
        cache.s,
        cache.apanel,
        cache.q;
        scs = cache.te.scs,
        sds = cache.te.sds,
        sa = cache.sa,
        ca = cache.ca,
    )
end

function _wake_streamfunction_normal_derivative(cache, xi::Real, yi::Real, nxi::Real, nyi::Real)
    return _wake_streamfunction_normal_derivative(cache, 0, xi, yi, nxi, nyi)
end

function _wake_streamfunction_gradient_old(cache, xi::Real, yi::Real)
    T = _XFOIL_GGCALC_PRECISION
    _, dqdg_x = _psilin_vortex_derivatives(
        0,
        T(xi),
        T(yi),
        one(T),
        zero(T),
        cache.x,
        cache.y,
        cache.s,
        cache.apanel;
        scs = cache.te.scs,
        sds = cache.te.sds,
    )
    _, dqdg_y = _psilin_vortex_derivatives(
        0,
        T(xi),
        T(yi),
        zero(T),
        one(T),
        cache.x,
        cache.y,
        cache.s,
        cache.apanel;
        scs = cache.te.scs,
        sds = cache.te.sds,
    )
    psi_x = -cache.sa
    psi_y = cache.ca
    for i in eachindex(dqdg_x, cache.q)
        psi_x += dqdg_x[i] * cache.q[i]
        psi_y += dqdg_y[i] * cache.q[i]
    end
    return psi_x, psi_y
end

function _wake_streamfunction_normal_derivative_old(cache, xi::Real, yi::Real, nxi::Real, nyi::Real)
    T = _XFOIL_GGCALC_PRECISION
    _, dqdg = _psilin_vortex_derivatives(
        0,
        T(xi),
        T(yi),
        T(nxi),
        T(nyi),
        cache.x,
        cache.y,
        cache.s,
        cache.apanel;
        scs = cache.te.scs,
        sds = cache.te.sds,
    )
    psi_n = -T(nxi) * cache.sa + T(nyi) * cache.ca
    for i in eachindex(dqdg, cache.q)
        psi_n += dqdg[i] * cache.q[i]
    end
    return psi_n
end

function _wake_gap_profile(cache, sw::AbstractVector{<:Real})
    nw = length(sw)
    wgap = zeros(Float64, nw)
    cache.gap.sharp && return wgap

    xp1 = cache.xp[1]
    yp1 = cache.yp[1]
    xpn = cache.xp[end]
    ypn = cache.yp[end]
    denom = sqrt((xp1^2 + yp1^2) * (xpn^2 + ypn^2))
    denom <= 1.0e-12 && return wgap

    telrat = 2.50
    ante = max(cache.gap.ante, 0.0)
    ante <= 1.0e-12 && return wgap

    crosp = (xp1 * ypn - yp1 * xpn) / denom
    crosp = clamp(crosp, -1.0 + 1.0e-12, 1.0 - 1.0e-12)
    dwdxte = crosp / sqrt(max(1.0 - crosp^2, 1.0e-12))
    dwdxte = clamp(dwdxte, -3.0 / telrat, 3.0 / telrat)
    aa = 3.0 + telrat * dwdxte
    bb = -2.0 - telrat * dwdxte

    for i in eachindex(sw)
        zn = 1.0 - float(sw[i]) / (telrat * ante)
        if zn >= 0.0
            wgap[i] = ante * (aa + bb * zn) * zn^2
        end
    end
    return wgap
end

function _wake_station_trace(
    inv::CaseResult;
    waklen::Float64 = 1.0,
)
    T = _XFOIL_GGCALC_PRECISION
    cache = _wake_geometry_cache(inv)
    n = length(cache.x)
    nw = div(n, 8) + 2
    ds1 = T(0.5) * (cache.s[2] - cache.s[1] + cache.s[end] - cache.s[end - 1])
    smax = T(waklen) * cache.chord
    ratio = _setexp_ratio(ds1, smax, nw)
    sw = _setexp_array(ds1, ratio, nw)
    if ratio != one(T)
        wake_ratio_tol = T(6.0) * eps(smax)
        if cache.gap.sharp || sw[end] < smax - wake_ratio_tol
            ratio = nextfloat(ratio)
            sw = _setexp_array(ds1, ratio, nw)
        else
            # The executable occasionally lands one REAL below the direct Julia
            # iterate when the discrete SETEXP residual is still positive after
            # rounding. Mirror that one-ulp correction before tracing the wake.
            ratio_prev = prevfloat(ratio)
            res = _setexp_residual(ratio, ds1, smax, nw)
            res_prev = _setexp_residual(ratio_prev, ds1, smax, nw)
            if res > zero(T) && abs(res_prev) < abs(res)
                ratio = ratio_prev
                sw = _setexp_array(ds1, ratio, nw)
            end
        end
    end

    xte = T(0.5) * (cache.x[1] + cache.x[end])
    yte = T(0.5) * (cache.y[1] + cache.y[end])
    sx = T(0.5) * (cache.yp[end] - cache.yp[1])
    sy = T(0.5) * (cache.xp[1] - cache.xp[end])
    smod = max(sqrt(sx * sx + sy * sy), T(1.0e-12))

    xw = zeros(T, nw)
    yw = zeros(T, nw)
    nxw = zeros(T, nw)
    nyw = zeros(T, nw)
    qinv = zeros(T, nw)
    apw = zeros(T, nw)
    psi_x_trace = zeros(T, nw)
    psi_y_trace = zeros(T, nw)
    gnorm_trace = zeros(T, nw)
    ds_trace = zeros(T, nw)

    nxw[1] = sx / smod
    nyw[1] = sy / smod
    xw[1] = xte - T(1.0e-4) * nyw[1]
    yw[1] = yte + T(1.0e-4) * nxw[1]
    qinv[1] = T(inv.qnode[end])

    psi_x_trace[1], psi_y_trace[1] = _wake_streamfunction_gradient(cache, n + 1, xw[1], yw[1])
    gnorm_trace[1] = max(sqrt(psi_x_trace[1] * psi_x_trace[1] + psi_y_trace[1] * psi_y_trace[1]), T(1.0e-12))
    apw[1] = _xfoil_atan2(psi_y_trace[1], psi_x_trace[1])
    if nw >= 2
        nxw[2] = -psi_x_trace[1] / gnorm_trace[1]
        nyw[2] = -psi_y_trace[1] / gnorm_trace[1]
    end

    for i in 2:nw
        ds = sw[i] - sw[i - 1]
        ds_trace[i] = ds
        xw[i] = xw[i - 1] - ds * nyw[i]
        yw[i] = yw[i - 1] + ds * nxw[i]
        qinv[i] = _wake_streamfunction_normal_derivative(cache, n + i, xw[i], yw[i], nxw[i], nyw[i])

        if i < nw
            psi_x_trace[i], psi_y_trace[i] = _wake_streamfunction_gradient(cache, n + i, xw[i], yw[i])
            gnorm_trace[i] = max(
                sqrt(psi_x_trace[i] * psi_x_trace[i] + psi_y_trace[i] * psi_y_trace[i]),
                T(1.0e-12),
            )
            apw[i] = _xfoil_atan2(psi_y_trace[i], psi_x_trace[i])
            nxw[i + 1] = -psi_x_trace[i] / gnorm_trace[i]
            nyw[i + 1] = -psi_y_trace[i] / gnorm_trace[i]
        end
    end

    return (
        cache = cache,
        sw = sw,
        xw = xw,
        yw = yw,
        nxw = nxw,
        nyw = nyw,
        qinv = qinv,
        apw = apw,
        psi_x = psi_x_trace,
        psi_y = psi_y_trace,
        gnorm = gnorm_trace,
        ds = ds_trace,
    )
end

function _wake_stations(
    inv::CaseResult,
    te_xi::Real;
    waklen::Float64 = 1.0,
    ueps::Float64 = 1.0e-7,
)
    T = _XFOIL_GGCALC_PRECISION
    trace = _wake_station_trace(inv; waklen = waklen)
    cache = trace.cache
    sw = trace.sw
    xw = trace.xw
    yw = trace.yw
    qinv = trace.qinv
    apw = trace.apw
    wake_x = Vector{Float64}(undef, length(xw))
    wake_x[1] = float(te_xi)
    for i in 2:length(xw)
        wake_x[i] = wake_x[i - 1] + hypot(float(xw[i] - xw[i - 1]), float(yw[i] - yw[i - 1]))
    end
    wake_xi = wake_x .- wake_x[1]
    return WakeStations(
        wake_x,
        Float64.(max.(-qinv, T(ueps))),
        _wake_gap_profile(cache, wake_xi),
        Float64.(xw),
        Float64.(yw),
        Float64.(apw),
    )
end

function _wake_primary_with_gap(
    p::BLPrimaryState,
    x::Real,
    u::Real,
    dw::Real;
    flow::BLFlowParams,
    model::BLModelParams,
    ueps::Float64 = 1.0e-7,
)
    seeded = _seed_primary_for_mode(p, :wake, x, u; flow = flow, model = model, ueps = ueps)
    dsw = max(seeded.d - seeded.dw, _mrchue_hklim(:wake) * seeded.t)
    return BLPrimaryState(
        seeded.x,
        seeded.u,
        seeded.t,
        dsw + max(float(dw), 0.0);
        s = max(seeded.s, 1.0e-12),
        ampl = 0.0,
        dw = max(float(dw), 0.0),
    )
end

function _march_wake(
    inv::CaseResult,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    seed::Union{Nothing,WakeMarchResult} = nothing,
    tol::Float64 = 1.0e-6,
    max_iter::Int = 20,
    waklen::Float64 = 1.0,
    ueps::Float64 = 1.0e-7,
    mrchue_local::Bool = false,
)
    stations = _wake_stations(inv, lower.x[end]; waklen = waklen, ueps = ueps)
    n = length(stations.x)
    n >= 1 || throw(ArgumentError("wake must contain at least one station"))

    primary = Vector{BLPrimaryState}(undef, n)
    secondary = Vector{BLSecondaryState}(undef, n)
    conv = falses(max(n - 1, 0))

    te = _te_wake_start_state(inv, upper, lower)
    dsw_te = max(te.dte - te.ante, _mrchue_hklim(:wake) * te.tte)
    seeded = !isnothing(seed) && length(seed.primary) == n
    if seeded
        p1_seed = seed.primary[1]
        t1 = max(p1_seed.t, 1.0e-9)
        dsw1 = max(p1_seed.d - p1_seed.dw, _mrchue_hklim(:wake) * t1)
        p1 = BLPrimaryState(
            stations.x[1],
            max(p1_seed.u, ueps),
            t1,
            dsw1 + stations.dw[1];
            s = max(p1_seed.s, 1.0e-12),
            ampl = 0.0,
            dw = stations.dw[1],
        )
    else
        p1 = BLPrimaryState(
            stations.x[1],
            stations.ue[1],
            te.tte,
            dsw_te + stations.dw[1];
            s = max(te.cte, 1.0e-12),
            ampl = 0.0,
            dw = stations.dw[1],
        )
    end
    s1 = blsecondary_state(p1, :wake; flow = flow, model = model)
    primary[1] = p1
    secondary[1] = s1

    for i in 1:(n - 1)
        x2 = max(stations.x[i + 1], p1.x + 1.0e-6)
        if seeded
            p2_seed = seed.primary[i + 1]
            u2 = max(p2_seed.u, ueps)
            pseed = _wake_primary_with_gap(
                p2_seed,
                x2,
                u2,
                stations.dw[i + 1];
                flow = flow,
                model = model,
                ueps = ueps,
            )
        else
            u2 = stations.ue[i + 1]
            pseed = _wake_primary_with_gap(
                p1,
                x2,
                u2,
                stations.dw[i + 1];
                flow = flow,
                model = model,
                ueps = ueps,
            )
        end
        p2, s2, ok, _ = _solve_surface_interval(
            :wake,
            p1,
            x2,
            u2;
            mrchue_local = mrchue_local,
            mrchdu_local = false,
            flow = flow,
            model = model,
            acrit = 0.0,
            tol = tol,
            max_iter = max_iter,
            z_init = _interval_seed_unknowns(:wake, pseed, x2, u2; flow = flow, model = model, ueps = ueps),
            dw = stations.dw[i + 1],
        )
        p2 = BLPrimaryState(
            p2.x,
            p2.u,
            p2.t,
            max(p2.d - p2.dw, _mrchue_hklim(:wake) * p2.t) + stations.dw[i + 1];
            s = max(p2.s, 1.0e-12),
            ampl = 0.0,
            dw = stations.dw[i + 1],
        )
        s2 = blsecondary_state(p2, :wake; flow = flow, model = model)
        primary[i + 1] = p2
        secondary[i + 1] = s2
        conv[i] = ok
        p1 = p2
    end

    return WakeMarchResult(
        copy(stations.x),
        [p.u for p in primary],
        copy(stations.dw),
        copy(stations.xw),
        copy(stations.yw),
        copy(stations.apanel),
        primary,
        secondary,
        conv,
    )
end

function _surface_skin_friction_drag(
    inv::CaseResult,
    surface::SurfaceMarchResult,
    alpha_deg::Real;
    flow::BLFlowParams,
    topology::Union{Nothing,BLTopology} = nothing,
)
    n = length(surface.x)
    n < 2 && return 0.0

    topo = isnothing(topology) ? _xfoil_bl_topology(inv) : topology
    st = _bl_side(topo, surface.side)
    n == length(st.ipan) - 1 || throw(ArgumentError("surface/topology length mismatch"))

    sa = sin(deg2rad(float(alpha_deg)))
    ca = cos(deg2rad(float(alpha_deg)))
    qinf = max(flow.qinfbl, 1.0e-12)
    cdf = 0.0

    tau = [_surface_station_tau(surface.primary[i], surface.secondary[i]; flow = flow) for i in 1:n]
    for ibl in 3:n
        i = st.ipan[ibl]
        im = st.ipan[ibl - 1]
        dx = (inv.xnode[i] - inv.xnode[im]) * ca + (inv.ynode[i] - inv.ynode[im]) * sa
        cdf += 0.5 * (tau[ibl] + tau[ibl - 1]) * dx * 2.0 / qinf^2
    end
    return max(cdf, 0.0)
end

function _wake_pressure_drag(upper::SurfaceMarchResult, lower::SurfaceMarchResult)
    # Squire-Young form used by XFoil CDCALC, evaluated from trailing-edge BL state.
    thwake = max(lower.primary[end].t, 1.0e-12)
    shwake = max(lower.primary[end].d / thwake, 1.01)
    uewake = max(lower.ue[end], 1.0e-8)
    cdp = 2.0 * thwake * uewake^(0.5 * (5.0 + shwake))
    return max(cdp, 0.0)
end

function _wake_pressure_drag(
    wake::WakeMarchResult;
    flow::BLFlowParams,
    minf::Real = 0.0,
)
    thwake = max(wake.primary[end].t, 1.0e-12)
    shwake = max(wake.primary[end].d / thwake, 1.01)
    qinf = max(flow.qinfbl, 1.0e-12)
    urat = wake.ue[end] / qinf
    beta = sqrt(max(1.0 - float(minf)^2, 1.0e-12))
    tklam = float(minf)^2 / (1.0 + beta)^2
    uewake = wake.ue[end] * (1.0 - tklam) / max(1.0 - tklam * urat^2, 1.0e-12)
    cdp = 2.0 * thwake * (uewake / qinf)^(0.5 * (5.0 + shwake))
    return max(cdp, 0.0)
end

function solve_viscous_case(
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    legacy::Bool = false,
)
    if !legacy
        return _as_viscous_case_result(
            solve_coupled_case_viscal(
                airfoil,
                alpha_deg;
                reynolds = reynolds,
                ncrit = ncrit,
                repanel = repanel,
                model = model,
                flow = flow,
                tol = tol,
                max_iter = max_iter,
                viscal_max_iter = viscal_max_iter,
                viscal_tol = viscal_tol,
                xfoil_load_normalize_digits = xfoil_load_normalize_digits,
                transition_recheck = transition_recheck,
                restart_on_nonconverged = restart_on_nonconverged,
            ),
        )
    end

    re = float(reynolds)
    nc = float(ncrit)
    inv = solve_inviscid_case(
        airfoil,
        alpha_deg;
        repanel = repanel,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
    )
    topo = _xfoil_bl_topology(inv)

    flowp = isnothing(flow) ? _xfoil_bl_flow_params(re; model = model) : flow
    upper_st = _surface_stations(inv, :upper)
    lower_st = _surface_stations(inv, :lower)

    upper = _march_surface(
        upper_st;
        flow = flowp,
        model = model,
        ncrit = nc,
        tol = float(tol),
        max_iter = Int(max_iter),
        solve_similarity_station = true,
    )
    lower = _march_surface(
        lower_st;
        flow = flowp,
        model = model,
        ncrit = nc,
        tol = float(tol),
        max_iter = Int(max_iter),
        solve_similarity_station = true,
    )

    cdf =
        _surface_skin_friction_drag(inv, upper, alpha_deg; flow = flowp) +
        _surface_skin_friction_drag(inv, lower, alpha_deg; flow = flowp)
    wake = _march_wake(
        inv,
        upper,
        lower;
        flow = flowp,
        model = model,
        tol = float(tol),
        max_iter = Int(max_iter),
        mrchue_local = false,
    )
    cd_raw = _wake_pressure_drag(wake; flow = flowp)
    ue_panel = _panel_ue_from_surfaces(inv, upper, lower)
    cp_panel = @. 1.0 - ue_panel^2
    cl, cm = _force_coefficients_from_cp(inv, alpha_deg, cp_panel)
    cd = cd_raw
    top_xtr = _surface_transition_xc(upper)
    bot_xtr = _surface_transition_xc(lower)
    interval_converged =
        all(upper.converged_intervals) &&
        all(lower.converged_intervals) &&
        all(wake.converged_intervals)
    provenance = _solver_provenance(
        backend = :legacy_surface_march,
        selection = :final_state,
        converged = interval_converged,
        status = interval_converged ? :converged : :interval_nonconverged,
    )

    return ViscousCaseResult(
        float(alpha_deg),
        re,
        nc,
        cl,
        cd,
        cm,
        top_xtr,
        bot_xtr,
        upper,
        lower,
        inv,
        provenance,
    )
end

function _mass_defect_ue(
    dij::Matrix{Float64},
    topo::BLTopology,
    uinv_upper::Vector{Float64},
    uinv_lower::Vector{Float64},
    mass_upper::Vector{Float64},
    mass_lower::Vector{Float64};
    uemin::Float64 = 1.0e-4,
)
    n_u = length(uinv_upper)
    n_l = length(uinv_lower)
    ue_upper = copy(uinv_upper)
    ue_lower = copy(uinv_lower)

    for k in 1:n_u
        i_node = topo.upper.ipan[k + 1]
        vti_k = topo.upper.vti[k + 1]
        dmue = 0.0
        for j in 1:n_u
            j_node = topo.upper.ipan[j + 1]
            vti_j = topo.upper.vti[j + 1]
            dmue += (-vti_k * vti_j * dij[i_node, j_node]) * mass_upper[j]
        end
        for j in 1:n_l
            j_node = topo.lower.ipan[j + 1]
            vti_j = topo.lower.vti[j + 1]
            dmue += (-vti_k * vti_j * dij[i_node, j_node]) * mass_lower[j]
        end
        ue_upper[k] = max(uinv_upper[k] + dmue, uemin)
    end

    for k in 1:n_l
        i_node = topo.lower.ipan[k + 1]
        vti_k = topo.lower.vti[k + 1]
        dmue = 0.0
        for j in 1:n_u
            j_node = topo.upper.ipan[j + 1]
            vti_j = topo.upper.vti[j + 1]
            dmue += (-vti_k * vti_j * dij[i_node, j_node]) * mass_upper[j]
        end
        for j in 1:n_l
            j_node = topo.lower.ipan[j + 1]
            vti_j = topo.lower.vti[j + 1]
            dmue += (-vti_k * vti_j * dij[i_node, j_node]) * mass_lower[j]
        end
        ue_lower[k] = max(uinv_lower[k] + dmue, uemin)
    end

    return ue_upper, ue_lower
end

function solve_coupled_case(
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    coupling_gain::Real = 4.0,
    coupling_grad_gain::Real = 0.8,
    coupling_smooth::Real = 0.2,
    coupling_relax::Real = 0.35,
    coupling_tol::Real = 1.0e-3,
    max_coupling_iter::Integer = 12,
    coupling_disp_scale::Real = 0.2,
    coupling_disp_clip::Real = 0.03,
    coupling_disp_relax::Real = 0.15,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    continuation_seed::Union{Nothing,CoupledCaseResult} = nothing,
    continuation_retry::Bool = true,
    legacy_nonconverged_retry::Bool = true,
    legacy::Bool = false,
)
    if !legacy
        exact_res = solve_coupled_case_viscal(
            airfoil,
            alpha_deg;
            reynolds = reynolds,
            ncrit = ncrit,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            transition_recheck = transition_recheck,
            restart_on_nonconverged = restart_on_nonconverged,
        )

        exact_score = _coupled_result_sanity_score(exact_res)
        if exact_res.provenance.converged
            return exact_res
        end

        legacy_res = solve_coupled_case(
            airfoil,
            alpha_deg;
            reynolds = reynolds,
            ncrit = ncrit,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            coupling_gain = coupling_gain,
            coupling_grad_gain = coupling_grad_gain,
            coupling_smooth = coupling_smooth,
            coupling_relax = coupling_relax,
            coupling_tol = coupling_tol,
            max_coupling_iter = max_coupling_iter,
            coupling_disp_scale = coupling_disp_scale,
            coupling_disp_clip = coupling_disp_clip,
            coupling_disp_relax = coupling_disp_relax,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            legacy = true,
        )
        legacy_score = _coupled_result_sanity_score(legacy_res)
        legacy_selected = legacy_res
        legacy_selected_score = legacy_score

        if legacy_nonconverged_retry &&
           !legacy_res.provenance.converged &&
           !exact_res.provenance.converged
            legacy_selected = _retry_nonconverged_legacy_coupled_case(
                legacy_res,
                airfoil,
                alpha_deg;
                reynolds = reynolds,
                ncrit = ncrit,
                repanel = repanel,
                model = model,
                flow = flow,
                tol = tol,
                max_iter = max_iter,
                coupling_tol = coupling_tol,
                max_coupling_iter = max_coupling_iter,
                viscal_max_iter = viscal_max_iter,
                viscal_tol = viscal_tol,
                xfoil_load_normalize_digits = xfoil_load_normalize_digits,
                continuation_retry = continuation_retry,
            )
            legacy_selected_score = _coupled_result_sanity_score(legacy_selected)
        end

        choose_legacy = false
        if exact_score > 0.0 && legacy_selected_score + 1.0e-12 < exact_score
            choose_legacy = true
        elseif exact_score > 0.0 &&
               _is_early_stalled_nonconverged_result(exact_res.provenance) &&
               legacy_selected_score <= exact_score + 1.0e-12 &&
               _is_meaningfully_better_residual(legacy_selected.provenance, exact_res.provenance)
            choose_legacy = true
        elseif !exact_res.provenance.converged &&
               legacy_selected_score <= exact_score + 1.0e-12 &&
               isfinite(exact_res.provenance.selected_residual) &&
               exact_res.provenance.selected_residual >= 1.0 &&
               _is_meaningfully_better_residual(
                   legacy_selected.provenance,
                   exact_res.provenance;
                   ratio = 0.1,
               )
            choose_legacy = true
        end

        if choose_legacy
            used_retry = legacy_selected.provenance.backend == :legacy_coupled_hardcase_retry
            fallback_provenance = _solver_provenance(
                backend = used_retry ? legacy_selected.provenance.backend : :legacy_coupled_fallback,
                selection = used_retry ? legacy_selected.provenance.selection : :legacy_nonconverged_fallback,
                converged = legacy_selected.provenance.converged,
                status = legacy_selected.provenance.status,
                failure_message =
                    "exact_nonconverged_sanity=$(round(exact_score; digits = 3)) " *
                    "legacy_sanity=$(round(legacy_score; digits = 3)) " *
                    "exact_residual=$(round(exact_res.provenance.selected_residual; digits = 6)) " *
                    "legacy_residual=$(round(legacy_res.provenance.selected_residual; digits = 6))" *
                    (
                        isempty(legacy_selected.provenance.failure_message) ?
                        "" : " " * legacy_selected.provenance.failure_message
                    ),
                iterations = legacy_selected.provenance.iterations,
                selected_iteration = legacy_selected.provenance.selected_iteration,
                selected_residual = legacy_selected.provenance.selected_residual,
            )
            return _with_provenance(legacy_selected, fallback_provenance)
        end

        return exact_res
    end

    re = float(reynolds)
    nc = float(ncrit)
    inv = solve_inviscid_case(
        airfoil,
        alpha_deg;
        repanel = repanel,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
    )

    flowp = isnothing(flow) ? _xfoil_bl_flow_params(re; model = model) : flow

    topo_n = _xfoil_bl_topology(inv)
    upper_inv = _surface_stations(inv, :upper; topology = topo_n)
    lower_inv = _surface_stations(inv, :lower; topology = topo_n)
    seed_case = _validated_continuation_seed(continuation_seed, upper_inv, lower_inv)
    seed_upper =
        isnothing(seed_case) ? nothing :
        _continuation_seed_surface(seed_case.upper, upper_inv; flow = flowp, model = model)
    seed_lower =
        isnothing(seed_case) ? nothing :
        _continuation_seed_surface(seed_case.lower, lower_inv; flow = flowp, model = model)
    iterative_surface_seeding = !isnothing(seed_case)
    ue_u =
        isnothing(seed_upper) ?
        copy(upper_inv.ue) :
        _continuation_seed_ue(seed_upper, upper_inv)
    ue_l =
        isnothing(seed_lower) ?
        copy(lower_inv.ue) :
        _continuation_seed_ue(seed_lower, lower_inv)
    inv_ue_u = copy(upper_inv.ue)
    inv_ue_l = copy(lower_inv.ue)

    upper = _march_surface(
        SurfaceStations(:upper, upper_inv.x, upper_inv.xc, ue_u);
        flow = flowp,
        model = model,
        ncrit = nc,
        tol = float(tol),
        max_iter = Int(max_iter),
        seed = seed_upper,
        preserve_seed_transition = !isnothing(seed_upper),
        solve_similarity_station = true,
    )
    lower = _march_surface(
        SurfaceStations(:lower, lower_inv.x, lower_inv.xc, ue_l);
        flow = flowp,
        model = model,
        ncrit = nc,
        tol = float(tol),
        max_iter = Int(max_iter),
        seed = seed_lower,
        preserve_seed_transition = !isnothing(seed_lower),
        solve_similarity_station = true,
    )

    cres = Inf
    nit = 1
    relax = clamp(float(coupling_relax), 0.01, 1.0)
    gain = max(float(coupling_gain), 0.0)
    grad_gain = max(float(coupling_grad_gain), 0.0)
    smooth = clamp(float(coupling_smooth), 0.0, 0.8)
    disp_scale = float(coupling_disp_scale)
    disp_clip = float(coupling_disp_clip)
    disp_relax = clamp(float(coupling_disp_relax), 0.0, 1.0)
    for it in 1:Int(max_coupling_iter)
        nit = it
        target_u =
            _coupled_ue_target(
                SurfaceStations(:upper, upper_inv.x, upper_inv.xc, inv_ue_u),
                upper;
                gain = gain,
                grad_gain = grad_gain,
                smooth = smooth,
            )
        target_l =
            _coupled_ue_target(
                SurfaceStations(:lower, lower_inv.x, lower_inv.xc, inv_ue_l),
                lower;
                gain = gain,
                grad_gain = grad_gain,
                smooth = smooth,
            )

        ue_u_new = @. (1.0 - relax) * ue_u + relax * target_u
        ue_l_new = @. (1.0 - relax) * ue_l + relax * target_l
        cres = max(
            maximum(abs.((ue_u_new .- ue_u) ./ max.(ue_u, 1.0e-6))),
            maximum(abs.((ue_l_new .- ue_l) ./ max.(ue_l, 1.0e-6))),
        )
        ue_u .= ue_u_new
        ue_l .= ue_l_new

        upper = _march_surface(
            SurfaceStations(:upper, upper_inv.x, upper_inv.xc, ue_u);
            flow = flowp,
            model = model,
            ncrit = nc,
            tol = float(tol),
            max_iter = Int(max_iter),
            seed = iterative_surface_seeding ? upper : nothing,
            preserve_seed_transition = iterative_surface_seeding,
            solve_similarity_station = true,
        )
        lower = _march_surface(
            SurfaceStations(:lower, lower_inv.x, lower_inv.xc, ue_l);
            flow = flowp,
            model = model,
            ncrit = nc,
            tol = float(tol),
            max_iter = Int(max_iter),
            seed = iterative_surface_seeding ? lower : nothing,
            preserve_seed_transition = iterative_surface_seeding,
            solve_similarity_station = true,
        )

        # Displacement-body inviscid refresh (quasi-BLSYS step).
        if disp_scale > 0.0
            try
                d_panel = _panel_dstar_from_surfaces(inv, upper, lower; topology = topo_n)
                af_disp = _displaced_airfoil(
                    airfoil,
                    d_panel;
                    scale = disp_scale,
                    max_frac = disp_clip,
                )
                inv_disp = solve_inviscid_case(af_disp, alpha_deg)
                ue_u_disp = _resampled_inviscid_ue(upper_inv, inv_disp, :upper)
                ue_l_disp = _resampled_inviscid_ue(lower_inv, inv_disp, :lower)

                ub_u = max.(upper_inv.ue, 1.0e-6)
                ub_l = max.(lower_inv.ue, 1.0e-6)
                ru = ue_u_disp ./ ub_u
                rl = ue_l_disp ./ ub_l
                sane = all(isfinite, ue_u_disp) &&
                       all(isfinite, ue_l_disp) &&
                       maximum(abs.(ru .- 1.0)) <= 0.35 &&
                       maximum(abs.(rl .- 1.0)) <= 0.35
                if sane
                    ue_u_guard = clamp.(ue_u_disp, 0.70 .* ub_u, 1.30 .* ub_u)
                    ue_l_guard = clamp.(ue_l_disp, 0.70 .* ub_l, 1.30 .* ub_l)
                    @. inv_ue_u = (1.0 - disp_relax) * inv_ue_u + disp_relax * ue_u_guard
                    @. inv_ue_l = (1.0 - disp_relax) * inv_ue_l + disp_relax * ue_l_guard
                else
                    inv_ue_u .= upper_inv.ue
                    inv_ue_l .= lower_inv.ue
                end
            catch
                inv_ue_u .= upper_inv.ue
                inv_ue_l .= lower_inv.ue
            end
        end

        cres <= float(coupling_tol) && break
    end

    ue_panel = _panel_ue_from_surfaces(inv, upper, lower; topology = topo_n)
    cp_panel = @. 1.0 - ue_panel^2
    cl, cm = _force_coefficients_from_cp(inv, alpha_deg, cp_panel)

    cdf =
        _surface_skin_friction_drag(inv, upper, alpha_deg; flow = flowp, topology = topo_n) +
        _surface_skin_friction_drag(inv, lower, alpha_deg; flow = flowp, topology = topo_n)
    wake = _march_wake(
        inv,
        upper,
        lower;
        flow = flowp,
        model = model,
        tol = float(tol),
        max_iter = Int(max_iter),
        mrchue_local = false,
    )
    cd_raw = _wake_pressure_drag(wake; flow = flowp)
    cd = cd_raw
    top_xtr = _surface_transition_xc(upper)
    bot_xtr = _surface_transition_xc(lower)
    legacy_converged = cres <= float(coupling_tol)
    provenance = _solver_provenance(
        backend = :legacy_quasi_coupled,
        selection = :final_state,
        converged = legacy_converged,
        status = legacy_converged ? :converged : :max_iter,
        iterations = nit,
        selected_iteration = nit,
        selected_residual = cres,
    )

    return CoupledCaseResult(
        float(alpha_deg),
        re,
        nc,
        cl,
        cd,
        cm,
        top_xtr,
        bot_xtr,
        nit,
        cres,
        upper,
        lower,
        inv,
        provenance,
    )
end

"""
    solve_coupled_polar(airfoil, alphas; kwargs...) -> Vector{CoupledCaseResult}

Solve a viscous polar with alpha continuation using the public stabilized
coupled solver path. The sweep anchors near `sweep_anchor` (default: 0 deg),
marches to higher alpha, then to lower alpha, reusing the last physically
plausible converged public result as the continuation seed.
"""
function solve_coupled_polar(
    airfoil::Airfoil,
    alphas::AbstractVector{<:Real};
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    coupling_gain::Real = 4.0,
    coupling_grad_gain::Real = 0.8,
    coupling_smooth::Real = 0.2,
    coupling_relax::Real = 0.35,
    coupling_tol::Real = 1.0e-3,
    max_coupling_iter::Integer = 12,
    coupling_disp_scale::Real = 0.2,
    coupling_disp_clip::Real = 0.03,
    coupling_disp_relax::Real = 0.15,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    continuation_retry::Bool = true,
    legacy::Bool = false,
    sweep_anchor::Union{Nothing,Real} = nothing,
    continuation_step_limit::Real = 2.0,
)
    alpha_list = sort!(unique(Float64.(alphas)))
    isempty(alpha_list) && return CoupledCaseResult[]

    anchor = isnothing(sweep_anchor) ? 0.0 : float(sweep_anchor)
    anchor_idx = argmin(abs.(alpha_list .- anchor))
    results = Dict{Float64,CoupledCaseResult}()

    function solve_one(alpha::Float64, seed_case::Union{Nothing,CoupledCaseResult})
        return solve_coupled_case(
            airfoil,
            alpha;
            reynolds = reynolds,
            ncrit = ncrit,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            coupling_gain = coupling_gain,
            coupling_grad_gain = coupling_grad_gain,
            coupling_smooth = coupling_smooth,
            coupling_relax = coupling_relax,
            coupling_tol = coupling_tol,
            max_coupling_iter = max_coupling_iter,
            coupling_disp_scale = coupling_disp_scale,
            coupling_disp_clip = coupling_disp_clip,
            coupling_disp_relax = coupling_disp_relax,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            transition_recheck = transition_recheck,
            restart_on_nonconverged = restart_on_nonconverged,
            continuation_seed = seed_case,
            continuation_retry = continuation_retry,
            legacy = legacy,
        )
    end

    function march_indices!(indices, seed_case::Union{Nothing,CoupledCaseResult})
        current_seed = seed_case
        current_seed_alpha = isnothing(seed_case) ? NaN : float(seed_case.alpha_deg)

        for idx in indices
            alpha = alpha_list[idx]
            use_seed =
                !isnothing(current_seed) &&
                isfinite(current_seed_alpha) &&
                abs(alpha - current_seed_alpha) <= float(continuation_step_limit) + 1.0e-12 ?
                current_seed : nothing

            result = solve_one(alpha, use_seed)
            results[alpha] = result

            if _plausible_converged_retry_candidate(result)
                current_seed = result
                current_seed_alpha = alpha
            end
        end
    end

    anchor_alpha = alpha_list[anchor_idx]
    anchor_result = solve_one(anchor_alpha, nothing)
    results[anchor_alpha] = anchor_result
    anchor_seed = _plausible_converged_retry_candidate(anchor_result) ? anchor_result : nothing

    if anchor_idx < length(alpha_list)
        march_indices!((anchor_idx + 1):length(alpha_list), anchor_seed)
    end
    if anchor_idx > 1
        march_indices!((anchor_idx - 1):-1:1, anchor_seed)
    end

    return [results[alpha] for alpha in alpha_list]
end


# ============================================================================
# XFoil-exact VISCAL Newton loop
# ============================================================================
# This implements XFoil's coupled viscous-inviscid Newton iteration:
#   SETBL (build system) → BLSOLV (solve) → UPDATE (apply deltas)
# The system couples BL equations at every station through the mass-defect
# influence matrix DIJ, exactly as in XFoil's xbl.f / xsolve.f.

"""
Mutable BL state arrays in XFoil layout.
Index convention: ibl=1 is stagnation, ibl=2..iblte are surface, ibl>iblte are wake.
IS=1 is upper surface, IS=2 is lower surface.
"""
mutable struct ViscalBLState
    nbl::Vector{Int}           # [2] number of BL stations per side
    iblte::Vector{Int}         # [2] TE station index per side
    itran::Vector{Int}         # [2] transition station per side
    uedg::Matrix{Float32}     # (max_ibl, 2) edge velocity — Float32 matches Fortran REAL
    thet::Matrix{Float32}     # (max_ibl, 2) momentum thickness
    dstr::Matrix{Float32}     # (max_ibl, 2) displacement thickness
    ctau::Matrix{Float32}     # (max_ibl, 2) shear lag / amplification
    mass::Matrix{Float32}     # (max_ibl, 2) mass defect = dstr * uedg
    xssi::Matrix{Float32}     # (max_ibl, 2) arc length coordinate
    uinv::Matrix{Float32}     # (max_ibl, 2) inviscid edge velocity
    vti::Matrix{Float32}      # (max_ibl, 2) tangential velocity sign
    ipan::Matrix{Int}         # (max_ibl, 2) panel index
    isys::Matrix{Int}         # (max_ibl, 2) system line number
    nsys::Int                 # total system size
    cl::Float64               # current viscous lift coefficient (Fortran global CL)
    wgap::Vector{Float32}     # wake gap displacement
end

@inline function _viscal_state_capacity(inv::CaseResult, nw::Integer = 0)
    # STMOVE can shift the stagnation split enough that one side temporarily
    # needs essentially the full surface panel count, plus any wake stations.
    return length(inv.qnode) + max(Int(nw), 0) + 1
end

"""
Initialize ViscalBLState from march results and topology.

When `wake_trace` is provided, wake stations are added to side 2 (lower)
following XFoil's IBLPAN convention: wake stations run from
`iblte[2]+1` to `nbl[2]`, with `ipan` pointing to panel indices `N+1:N+NW`.
"""
function _init_viscal_state(
    inv::CaseResult,
    topo::BLTopology,
    upper::SurfaceMarchResult,
    lower::SurfaceMarchResult;
    wake_trace = nothing,
    wake_march::Union{Nothing,WakeMarchResult} = nothing,
)
    nu = length(upper.primary)
    nl = length(lower.primary)
    n = length(inv.xnode)
    nw = isnothing(wake_trace) ? 0 : length(wake_trace.xw)
    max_ibl = _viscal_state_capacity(inv, nw)

    z2(sz) = zeros(Float32, sz, 2)
    zi2(sz) = zeros(Int, sz, 2)

    iblte1 = nu + 1
    iblte2 = nl + 1
    nbl1 = iblte1         # upper: surface only; upper wake slots are plotting/equalization only
    nbl2 = iblte2 + nw    # lower: surface + wake

    state = ViscalBLState(
        [nbl1, nbl2],
        [iblte1, iblte2],
        [nu + 2, nl + 2],   # itran default: beyond last station (all laminar)
        z2(max_ibl), z2(max_ibl), z2(max_ibl),
        z2(max_ibl), z2(max_ibl),
        z2(max_ibl), z2(max_ibl), z2(max_ibl),
        zi2(max_ibl), zi2(max_ibl),
        0,
        Float64(Float32(inv.cl)),
        Float32[],
    )

    # Fill upper surface (IS=1)
    for k in 1:nu
        ibl = k + 1
        p = upper.primary[k]
        m = upper.mode[k]
        state.xssi[ibl, 1] = p.x
        state.uedg[ibl, 1] = p.u
        state.thet[ibl, 1] = max(p.t, 1e-10)
        state.dstr[ibl, 1] = max(p.d, 1e-10)
        state.ctau[ibl, 1] = m in (:laminar, :similarity) ? p.ampl : p.s
        state.mass[ibl, 1] = state.dstr[ibl, 1] * state.uedg[ibl, 1]
        if ibl <= length(topo.upper.vti)
            state.vti[ibl, 1] = topo.upper.vti[ibl]
            state.ipan[ibl, 1] = topo.upper.ipan[ibl]
        end
        state.uinv[ibl, 1] = state.uedg[ibl, 1]  # initial: uinv = uedg
    end

    # Fill lower surface (IS=2)
    for k in 1:nl
        ibl = k + 1
        p = lower.primary[k]
        m = lower.mode[k]
        state.xssi[ibl, 2] = p.x
        state.uedg[ibl, 2] = p.u
        state.thet[ibl, 2] = max(p.t, 1e-10)
        state.dstr[ibl, 2] = max(p.d, 1e-10)
        state.ctau[ibl, 2] = m in (:laminar, :similarity) ? p.ampl : p.s
        state.mass[ibl, 2] = state.dstr[ibl, 2] * state.uedg[ibl, 2]
        if ibl <= length(topo.lower.vti)
            state.vti[ibl, 2] = topo.lower.vti[ibl]
            state.ipan[ibl, 2] = topo.lower.ipan[ibl]
        end
        state.uinv[ibl, 2] = state.uedg[ibl, 2]
    end

    # Fill wake stations on side 2 (IS=2), IBL = iblte[2]+1 .. nbl[2]
    if nw > 0 && !isnothing(wake_trace)
        # TE arc-length coordinate (last surface station on lower side)
        xi_te = state.xssi[iblte2, 2]
        wake_xi = Vector{Float64}(undef, nw)
        wake_xi[1] = 0.0
        for iw in 2:nw
            wake_xi[iw] = wake_xi[iw - 1] + hypot(
                float(wake_trace.xw[iw] - wake_trace.xw[iw - 1]),
                float(wake_trace.yw[iw] - wake_trace.yw[iw - 1]),
            )
        end

        wgap_vals = Float32.(wake_trace.cache !== nothing ?
            _wake_gap_profile(wake_trace.cache, wake_xi) :
            zeros(nw))
        state.wgap = wgap_vals

        # Compute initial wake BL from TESYS: TTE, DTE from upper+lower TE
        tte_init = state.thet[iblte1, 1] + state.thet[iblte2, 2]
        dte_init = state.dstr[iblte1, 1] + state.dstr[iblte2, 2]
        cte_init = (state.ctau[iblte1, 1] * state.thet[iblte1, 1] +
                    state.ctau[iblte2, 2] * state.thet[iblte2, 2]) / max(tte_init, 1e-10)

        for iw in 1:nw
            ibl = iblte2 + iw
            state.ipan[ibl, 2] = n + iw            # panel index N+iw
            state.vti[ibl, 2] = -1.0               # wake always -1
            state.xssi[ibl, 2] = xi_te + wake_xi[iw]

            # Use wake march result if available; otherwise seed from TE
            if !isnothing(wake_march) && iw <= length(wake_march.primary)
                pw = wake_march.primary[iw]
                state.uedg[ibl, 2] = max(pw.u, 1e-6)
                state.thet[ibl, 2] = max(pw.t, 1e-10)
                state.dstr[ibl, 2] = max(pw.d, 1e-10)
                state.ctau[ibl, 2] = pw.s   # wake is always turbulent
            else
                # Simple seed: copy TE combined values with mild decay
                state.uedg[ibl, 2] = max(abs(Float64(wake_trace.qinv[iw])), 1e-6)
                state.thet[ibl, 2] = max(tte_init, 1e-10)
                state.dstr[ibl, 2] = max(dte_init + wgap_vals[iw], 1e-10)
                state.ctau[ibl, 2] = max(cte_init, 1e-6)
            end
            state.mass[ibl, 2] = state.dstr[ibl, 2] * state.uedg[ibl, 2]
            state.uinv[ibl, 2] = max(abs(Float64(wake_trace.qinv[iw])), 1e-6)
        end

        # Upper side wake: copy from lower wake (Fortran equalization)
        # VTI, ipan, and all BL state must match lower wake
        for iw in 1:nw
            ibl1 = iblte1 + iw
            ibl2 = iblte2 + iw
            if ibl1 <= max_ibl
                state.ipan[ibl1, 1] = state.ipan[ibl2, 2]
                state.vti[ibl1, 1] = 1.0f0
                state.xssi[ibl1, 1] = state.xssi[iblte1, 1] +
                                      (state.xssi[ibl2, 2] - state.xssi[iblte2, 2])
                state.uedg[ibl1, 1] = state.uedg[ibl2, 2]
                state.thet[ibl1, 1] = state.thet[ibl2, 2]
                state.dstr[ibl1, 1] = state.dstr[ibl2, 2]
                state.ctau[ibl1, 1] = state.ctau[ibl2, 2]
                state.mass[ibl1, 1] = state.mass[ibl2, 2]
                state.uinv[ibl1, 1] = state.uinv[ibl2, 2]
            end
        end
    end

    # Set transition indices
    for (is_side, march) in ((1, upper), (2, lower))
        itran_found = false
        for k in 1:length(march.mode)
            if march.mode[k] in (:turbulent, :transition)
                state.itran[is_side] = k + 1  # IBL index
                itran_found = true
                break
            end
        end
        if !itran_found
            state.itran[is_side] = state.iblte[is_side] + 1  # all laminar
        end
    end

    # Set system line numbering (IBLSYS equivalent)
    # Fortran loops: DO IS=1,2; DO IBL=2,NBL(IS); IV=IV+1; ISYS(IBL,IS)=IV
    # Upper: IBL 2..nbl[1] (surface + wake)
    # Lower: IBL 2..nbl[2] (surface + wake)
    iv = 0
    for is in 1:2
        for ibl in 2:state.nbl[is]
            iv += 1
            state.isys[ibl, is] = iv
        end
    end
    state.nsys = iv

    return state
end

"""
Compute inviscid edge velocities from panel solution (UICALC equivalent).
Sets state.uinv from the panel tangential velocities.
"""
function _viscal_uicalc!(state::ViscalBLState, inv::CaseResult)
    for is in 1:2
        for ibl in 2:state.nbl[is]
            i = state.ipan[ibl, is]
            if i >= 1 && i <= length(inv.qnode)
                state.uinv[ibl, is] = state.vti[ibl, is] * inv.qnode[i]
            end
        end
    end
end

"""
    _viscal_retransition!(state, is; flow, ncrit)

Lightweight transition re-check for side `is`.  Walks the n-factor
amplification through the CURRENT BL state (thet, dstr, uedg) and
updates state.itran[is] and state.ctau (amplification values) if the
n-factor exceeds ncrit at a new station.

Updates itran AND ctau for consistency: laminar stations get their
computed amplification, and the first turbulent station gets an
initial Ctau seed (0.03).  This prevents the Newton discontinuity
that occurs when itran changes without ctau update.
"""
function _viscal_retransition!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
)
    iblte = state.iblte[is]
    itrold = state.itran[is]

    # Walk surface stations, accumulating amplification
    ami = 0.0
    prev_hk = 2.0; prev_t = 1e-6; prev_rt = 100.0; prev_x = 0.0
    # Store amplification at each station for ctau update
    ampl_at = zeros(iblte)

    itran_new = iblte + 2  # default: no transition

    for ibl in 2:iblte
        x = Float64(state.xssi[ibl, is])
        u = Float64(state.uedg[ibl, is])
        t = Float64(state.thet[ibl, is])
        d = Float64(state.dstr[ibl, is])

        kin = blkin_state(u, t, d, flow)
        hk = max(kin.hk2, 1.05)
        rt = max(kin.rt2, 1.0)

        if ibl >= 3
            dx = x - prev_x
            if dx > 0
                ax = axset(prev_hk, prev_t, prev_rt, ami,
                    hk, t, rt, ami, ncrit; idampv=0)
                ami = max(ami + ax * dx, 0.0)
            end
        elseif ibl == 2
            # Seed from stored amplification if station is laminar
            if ibl < itrold
                ami = max(Float64(state.ctau[ibl, is]), 0.0)
            end
        end
        ampl_at[ibl] = ami

        if ami >= ncrit && itran_new > iblte
            itran_new = ibl
        end

        prev_hk = hk; prev_t = t; prev_rt = rt; prev_x = x
    end

    # Only update if transition moved (allow movement in either direction)
    if itran_new != itrold && itran_new < iblte
        state.itran[is] = itran_new

        # Update ctau for affected stations to maintain consistency
        for ibl in 2:iblte
            if ibl < itran_new
                # Laminar: ctau = amplification
                state.ctau[ibl, is] = ampl_at[ibl]
            elseif ibl == itran_new
                # Transition station: seed turbulent Ctau
                if ibl >= itrold  # was turbulent, keep existing ctau
                    # already has a turbulent ctau, keep it
                else
                    # was laminar, initialize turbulent ctau
                    state.ctau[ibl, is] = 0.03
                end
            elseif ibl > itran_new && ibl < itrold
                # Newly turbulent (was laminar): seed Ctau
                state.ctau[ibl, is] = 0.03
            end
            # Stations that were already turbulent (ibl >= itrold) keep their ctau
        end
    end

    return nothing
end

"""
Compute mass-consistent edge velocities (UESET equivalent).
Returns usav = UINV + DIJ*MASS (what Ue should be from mass defect coupling).
"""
function _viscal_ueset(state::ViscalBLState, dij::Matrix{Float64})
    FT = Float32
    usav = zeros(FT, size(state.uedg))
    for is in 1:2
        for ibl in 2:state.nbl[is]
            i = state.ipan[ibl, is]
            dui = zero(FT)
            for js in 1:2
                for jbl in 2:state.nbl[js]
                    j = state.ipan[jbl, js]
                    if i >= 1 && j >= 1 && i <= size(dij, 1) && j <= size(dij, 2)
                        ue_m = -state.vti[ibl, is] * state.vti[jbl, js] * FT(dij[i, j])
                        dui += ue_m * state.mass[jbl, js]
                    end
                end
            end
            usav[ibl, is] = state.uinv[ibl, is] + dui
        end
    end
    return Float64.(usav)
end

"""
Get BLPrimaryState for a given station from the ViscalBLState.
"""
function _viscal_get_primary(state::ViscalBLState, ibl::Int, is::Int)
    is_lam = ibl < state.itran[is]
    dswaki = 0.0f0
    if ibl > state.iblte[is]
        iw = ibl - state.iblte[is]
        dswaki = iw <= length(state.wgap) ? state.wgap[iw] : 0.0f0
    end
    return BLPrimaryState(
        state.xssi[ibl, is],
        state.uedg[ibl, is],
        state.thet[ibl, is],
        state.dstr[ibl, is],
        is_lam ? 0.0f0 : state.ctau[ibl, is],  # s (turbulent)
        is_lam ? state.ctau[ibl, is] : 0.0f0,   # ampl (laminar)
        dswaki,
    )
end

"""
Get the BL mode for a given station.
"""
function _viscal_station_mode(state::ViscalBLState, ibl::Int, is::Int)
    if ibl == 2
        return :similarity
    elseif ibl > state.iblte[is]
        return :wake
    elseif ibl == state.itran[is]
        return :transition
    elseif ibl < state.itran[is]
        return :laminar
    else
        return :turbulent
    end
end

"""
BLSOLV: Block elimination solver for the coupled BL Newton system.
Port of XFoil's xsolve.f BLSOLV subroutine.

System structure per station IV:
  VA(k,j,IV) * [dS, dT]_IV + VB(k,j,IV) * [dS, dT]_{IV-1}
    + SUM_JV VM(k,JV,IV) * dM_JV = -VDEL(k,1,IV)

After solve, VDEL(k,1,IV) contains the Newton deltas:
  VDEL(1,1,IV) = dCtau/dAmpl
  VDEL(2,1,IV) = dTheta
  VDEL(3,1,IV) = dMass
"""
function _viscal_blsolv!(
    va::Array{T,3},    # (3, 2, nsys) diagonal blocks
    vb::Array{T,3},    # (3, 2, nsys) lower diagonal blocks
    vm::Array{T,3},    # (3, nsys, nsys) mass coupling
    vdel::Array{T,3},  # (3, 2, nsys) residuals and Re sensitivity
    vz::Matrix{T},     # (3, 2) TE coupling block
    nsys::Int,
    ivte1::Int,              # system line at upper TE
    ivz_target::Int;         # system line for lower wake entry (VZ target)
    sspan::Real = 2.0,    # S(N)-S(1) total panel arc length
) where {T<:AbstractFloat}
    vaccel = T(0.01)  # acceleration parameter (matches XFoil VACCEL)
    vacc1 = vaccel
    sspan_t = max(abs(T(sspan)), T(1e-7))
    vacc2 = vaccel * T(2) / sspan_t
    vacc3 = vacc2

    for iv in 1:nsys
        ivp = iv + 1

        # ====== Invert VA(IV) block ======

        # Normalize first row — no tolerance check, matching Fortran BLSOLV
        pivot = one(T) / va[1, 1, iv]
        va[1, 2, iv] *= pivot
        for l in iv:nsys
            vm[1, l, iv] *= pivot
        end
        vdel[1, 1, iv] *= pivot
        vdel[1, 2, iv] *= pivot

        # Eliminate lower first column in VA block
        for k in 2:3
            vtmp = va[k, 1, iv]
            va[k, 2, iv] -= vtmp * va[1, 2, iv]
            for l in iv:nsys
                vm[k, l, iv] -= vtmp * vm[1, l, iv]
            end
            vdel[k, 1, iv] -= vtmp * vdel[1, 1, iv]
            vdel[k, 2, iv] -= vtmp * vdel[1, 2, iv]
        end

        # Normalize second row — no tolerance check, matching Fortran BLSOLV
        pivot = one(T) / va[2, 2, iv]
        for l in iv:nsys
            vm[2, l, iv] *= pivot
        end
        vdel[2, 1, iv] *= pivot
        vdel[2, 2, iv] *= pivot

        # Eliminate lower second column (row 3 only)
        vtmp = va[3, 2, iv]
        for l in iv:nsys
            vm[3, l, iv] -= vtmp * vm[2, l, iv]
        end
        vdel[3, 1, iv] -= vtmp * vdel[2, 1, iv]
        vdel[3, 2, iv] -= vtmp * vdel[2, 2, iv]

        # Normalize third row by VM(3,IV,IV) — no tolerance check, matching Fortran
        pivot = one(T) / vm[3, iv, iv]
        for l in (iv + 1):nsys
            vm[3, l, iv] *= pivot
        end
        vdel[3, 1, iv] *= pivot
        vdel[3, 2, iv] *= pivot

        # Eliminate upper third column in VA block
        vtmp1 = vm[1, iv, iv]
        vtmp2 = vm[2, iv, iv]
        for l in (iv + 1):nsys
            vm[1, l, iv] -= vtmp1 * vm[3, l, iv]
            vm[2, l, iv] -= vtmp2 * vm[3, l, iv]
        end
        vdel[1, 1, iv] -= vtmp1 * vdel[3, 1, iv]
        vdel[2, 1, iv] -= vtmp2 * vdel[3, 1, iv]
        vdel[1, 2, iv] -= vtmp1 * vdel[3, 2, iv]
        vdel[2, 2, iv] -= vtmp2 * vdel[3, 2, iv]

        # Eliminate upper second column in VA block
        vtmp = va[1, 2, iv]
        for l in (iv + 1):nsys
            vm[1, l, iv] -= vtmp * vm[2, l, iv]
        end
        vdel[1, 1, iv] -= vtmp * vdel[2, 1, iv]
        vdel[1, 2, iv] -= vtmp * vdel[2, 2, iv]

        iv == nsys && continue

        # ====== Eliminate VB(IV+1) block ======
        for k in 1:3
            vtmp1 = vb[k, 1, ivp]
            vtmp2 = vb[k, 2, ivp]
            vtmp3 = vm[k, iv, ivp]
            for l in ivp:nsys
                corr = vtmp1 * vm[1, l, iv]
                corr += vtmp2 * vm[2, l, iv]
                corr += vtmp3 * vm[3, l, iv]
                vm[k, l, ivp] -= corr
            end
            corr1 = vtmp1 * vdel[1, 1, iv]
            corr1 += vtmp2 * vdel[2, 1, iv]
            corr1 += vtmp3 * vdel[3, 1, iv]
            vdel[k, 1, ivp] -= corr1
            corr2 = vtmp1 * vdel[1, 2, iv]
            corr2 += vtmp2 * vdel[2, 2, iv]
            corr2 += vtmp3 * vdel[3, 2, iv]
            vdel[k, 2, ivp] -= corr2
        end

        # ====== Eliminate VZ block (TE coupling) ======
        if iv == ivte1 && ivz_target >= 1 && ivz_target <= nsys
            for k in 1:3
                vtmp1 = vz[k, 1]
                vtmp2 = vz[k, 2]
                for l in ivp:nsys
                    corr = vtmp1 * vm[1, l, iv]
                    corr += vtmp2 * vm[2, l, iv]
                    vm[k, l, ivz_target] -= corr
                end
                corr1 = vtmp1 * vdel[1, 1, iv]
                corr1 += vtmp2 * vdel[2, 1, iv]
                vdel[k, 1, ivz_target] -= corr1
                corr2 = vtmp1 * vdel[1, 2, iv]
                corr2 += vtmp2 * vdel[2, 2, iv]
                vdel[k, 2, ivz_target] -= corr2
            end
        end

        ivp == nsys && continue

        # ====== Eliminate lower VM column ======
        for kv in (iv + 2):nsys
            vtmp1 = vm[1, iv, kv]
            vtmp2 = vm[2, iv, kv]
            vtmp3 = vm[3, iv, kv]

            if abs(vtmp1) > vacc1
                for l in ivp:nsys
                    vm[1, l, kv] -= vtmp1 * vm[3, l, iv]
                end
                vdel[1, 1, kv] -= vtmp1 * vdel[3, 1, iv]
                vdel[1, 2, kv] -= vtmp1 * vdel[3, 2, iv]
            end

            if abs(vtmp2) > vacc2
                for l in ivp:nsys
                    vm[2, l, kv] -= vtmp2 * vm[3, l, iv]
                end
                vdel[2, 1, kv] -= vtmp2 * vdel[3, 1, iv]
                vdel[2, 2, kv] -= vtmp2 * vdel[3, 2, iv]
            end

            if abs(vtmp3) > vacc3
                for l in ivp:nsys
                    vm[3, l, kv] -= vtmp3 * vm[3, l, iv]
                end
                vdel[3, 1, kv] -= vtmp3 * vdel[3, 1, iv]
                vdel[3, 2, kv] -= vtmp3 * vdel[3, 2, iv]
            end
        end
    end

    # Back-substitution
    for iv in nsys:-1:2
        vtmp = vdel[3, 1, iv]
        for kv in (iv - 1):-1:1
            vdel[1, 1, kv] -= vm[1, iv, kv] * vtmp
            vdel[2, 1, kv] -= vm[2, iv, kv] * vtmp
            vdel[3, 1, kv] -= vm[3, iv, kv] * vtmp
        end

        vtmp = vdel[3, 2, iv]
        for kv in (iv - 1):-1:1
            vdel[1, 2, kv] -= vm[1, iv, kv] * vtmp
            vdel[2, 2, kv] -= vm[2, iv, kv] * vtmp
            vdel[3, 2, kv] -= vm[3, iv, kv] * vtmp
        end
    end
end

"""
SETBL: Build the coupled BL Newton system.
Port of XFoil's xbl.f SETBL subroutine.
Returns (va, vb, vm, vdel, vz) arrays for BLSOLV.
"""
function _viscal_setbl(
    state::ViscalBLState,
    dij::Matrix{Float64},
    usav::Matrix{Float64};  # mass-consistent Ue from UESET
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    ante::Float64 = 0.0,    # TE base area (normal-projected gap)
    topo::Union{Nothing,BLTopology} = nothing,  # for stagnation sensitivities
    mrchdu_cache::Union{Nothing, NTuple{2,Dict}} = nothing,  # (side1_cache, side2_cache), each Dict{Int,NamedTuple}
)
    nsys = state.nsys
    # Allocate Newton system arrays in Float32 to match XFoil's gfortran
    # REAL (single precision).  The Float32 arithmetic in BLSOLV naturally
    # regularizes the ill-conditioned system from degenerate wake equations.
    # Newton system arrays: Float32 matches XFoil's REAL precision exactly.
    # The Float32 truncation acts as natural regularization that prevents
    # Newton from overshooting — critical for MRCHDU per-iteration stability.
    va = zeros(Float32, 3, 2, nsys)
    vb = zeros(Float32, 3, 2, nsys)
    vm = zeros(Float32, 3, nsys, nsys)
    vdel = zeros(Float32, 3, 2, nsys)
    vz = zeros(Float32, 3, 2)

    iblte1 = state.iblte[1]
    iblte2 = state.iblte[2]
    ile1 = state.ipan[2, 1]
    ile2 = state.ipan[2, 2]
    ite1 = state.ipan[iblte1, 1]
    ite2 = state.ipan[iblte2, 2]
    jvte1 = state.isys[iblte1, 1]
    jvte2 = state.isys[iblte2, 2]

    # Pre-compute LE and TE Ue mass sensitivities
    ule1_m = zeros(Float64, nsys)
    ule2_m = zeros(Float64, nsys)
    ute1_m = zeros(Float64, nsys)
    ute2_m = zeros(Float64, nsys)
    for js in 1:2
        for jbl in 2:state.nbl[js]
            j = state.ipan[jbl, js]
            jv = state.isys[jbl, js]
            if jv >= 1 && jv <= nsys && j >= 1 && j <= size(dij, 2)
                vtj = state.vti[jbl, js]
                if ile1 >= 1 && ile1 <= size(dij, 1)
                    ule1_m[jv] = -state.vti[2, 1] * vtj * dij[ile1, j]
                end
                if ile2 >= 1 && ile2 <= size(dij, 1)
                    ule2_m[jv] = -state.vti[2, 2] * vtj * dij[ile2, j]
                end
                if ite1 >= 1 && ite1 <= size(dij, 1)
                    ute1_m[jv] = -state.vti[iblte1, 1] * vtj * dij[ite1, j]
                end
                if ite2 >= 1 && ite2 <= size(dij, 1)
                    ute2_m[jv] = -state.vti[iblte2, 2] * vtj * dij[ite2, j]
                end
            end
        end
    end

    dule1 = state.uedg[2, 1] - usav[2, 1]
    dule2 = state.uedg[2, 2] - usav[2, 2]

    # Stagnation point sensitivities (SST_GO, SST_GP)
    # These control how the arc-length coordinate XI responds to changes in
    # the LE edge velocities.  XFoil computes them from the stagnation point
    # location relative to the neighbouring panel nodes.
    sst_go = isnothing(topo) ? 0.0 : topo.stagnation.sst_go
    sst_gp = isnothing(topo) ? 0.0 : topo.stagnation.sst_gp

    for is in 1:2
        # Initialize "1" station mass sensitivities to zero
        u1_m = zeros(Float64, nsys)
        d1_m = zeros(Float64, nsys)
        due1 = 0.0
        dds1 = 0.0

        # COM1 kinematic state — flows forward station-to-station,
        # matching XFoil's COM1(ICOM) = COM2(ICOM) at end of each station.
        kin1_carried = nothing  # will be set after first station

        # BLCom state carried from station to station (literal COM1=COM2 chain)
        _c1_carried = BLCom()
        _c1_blsys = BLCom()
        _c2_blsys = BLCom()

        for ibl in 2:state.nbl[is]
            iv = state.isys[ibl, is]
            i = state.ipan[ibl, is]
            mode = _viscal_station_mode(state, ibl, is)
            is_tesys = (ibl == state.iblte[is] + 1) && (ibl <= state.nbl[is])

            # Current station primary state
            p2 = _viscal_get_primary(state, ibl, is)

            # Compute BLKIN at current station (COM2) and save for COM1 chain
            kin2_current = try
                _blkin_full(Float64(p2.u), Float64(p2.t), Float64(p2.d - p2.dw), flow; precision=Float32)
            catch
                nothing
            end

            # Derivatives of DSI = D2 wrt mass
            uei = state.uedg[ibl, is]
            dsi = state.dstr[ibl, is]
            d2_m2 = 1.0 / max(uei, 1e-10)
            d2_u2 = -dsi / max(uei, 1e-10)

            # Compute U2_M and D2_M for all stations
            u2_m = zeros(Float64, nsys)
            d2_m = zeros(Float64, nsys)
            for js in 1:2
                for jbl in 2:state.nbl[js]
                    j = state.ipan[jbl, js]
                    jv = state.isys[jbl, js]
                    if jv >= 1 && jv <= nsys && i >= 1 && j >= 1 &&
                       i <= size(dij, 1) && j <= size(dij, 2)
                        u2_m[jv] = -state.vti[ibl, is] * state.vti[jbl, js] * dij[i, j]
                        d2_m[jv] = d2_u2 * u2_m[jv]
                    end
                end
            end
            d2_m[iv] += d2_m2

            # "forced" changes from UEDG-USAV mismatch
            due2 = state.uedg[ibl, is] - usav[ibl, is]
            dds2 = d2_u2 * due2

            if is_tesys
                # ---- TESYS: TE-to-wake junction system ----
                tte = state.thet[iblte1, 1] + state.thet[iblte2, 2]
                dte = state.dstr[iblte1, 1] + state.dstr[iblte2, 2] + ante
                cte = (state.ctau[iblte1, 1] * state.thet[iblte1, 1] +
                       state.ctau[iblte2, 2] * state.thet[iblte2, 2]) / max(tte, 1e-10)

                # Chain-rule derivatives
                tte_tte1 = 1.0
                tte_tte2 = 1.0
                dte_mte1 = 1.0 / max(state.uedg[iblte1, 1], 1e-10)
                dte_ute1 = -state.dstr[iblte1, 1] / max(state.uedg[iblte1, 1], 1e-10)
                dte_mte2 = 1.0 / max(state.uedg[iblte2, 2], 1e-10)
                dte_ute2 = -state.dstr[iblte2, 2] / max(state.uedg[iblte2, 2], 1e-10)
                cte_cte1 = state.thet[iblte1, 1] / max(tte, 1e-10)
                cte_cte2 = state.thet[iblte2, 2] / max(tte, 1e-10)
                cte_tte1 = (state.ctau[iblte1, 1] - cte) / max(tte, 1e-10)
                cte_tte2 = (state.ctau[iblte2, 2] - cte) / max(tte, 1e-10)

                # TESYS system
                dw2 = p2.dw
                vs1_tesys = zeros(Float64, 3, 5)
                vs2_tesys = zeros(Float64, 3, 5)
                vsrez_tesys = zeros(Float64, 3)
                vs1_tesys[1, 1] = -1.0
                vs2_tesys[1, 1] = 1.0
                vsrez_tesys[1] = cte - p2.s
                vs1_tesys[2, 2] = -1.0
                vs2_tesys[2, 2] = 1.0
                vsrez_tesys[2] = tte - p2.t
                vs1_tesys[3, 3] = -1.0
                vs2_tesys[3, 3] = 1.0
                vsrez_tesys[3] = dte - (p2.d - dw2) - dw2

                # Redefine D1_M for TESYS: D1=DTE depends on both TE masses
                for jv in 1:nsys
                    d1_m[jv] = dte_ute1 * ute1_m[jv] + dte_ute2 * ute2_m[jv]
                end
                if jvte1 >= 1 && jvte1 <= nsys
                    d1_m[jvte1] += dte_mte1
                end
                if jvte2 >= 1 && jvte2 <= nsys
                    d1_m[jvte2] += dte_mte2
                end

                # Redefine forced changes for TESYS
                due1 = 0.0
                dds1 = dte_ute1 * (state.uedg[iblte1, 1] - usav[iblte1, 1]) +
                       dte_ute2 * (state.uedg[iblte2, 2] - usav[iblte2, 2])

                # XI sensitivities (zero for wake)
                xi_ule1 = 0.0
                xi_ule2 = 0.0

                # Fill VA, VB, VM, VDEL for TESYS station
                for k in 1:3
                    va[k, 1, iv] = vs2_tesys[k, 1]
                    va[k, 2, iv] = vs2_tesys[k, 2]

                    # VM: mass coupling
                    for jv in 1:nsys
                        vm[k, jv, iv] = vs1_tesys[k, 3] * d1_m[jv] + vs1_tesys[k, 4] * u1_m[jv] +
                                        vs2_tesys[k, 3] * d2_m[jv] + vs2_tesys[k, 4] * u2_m[jv]
                    end

                    # VDEL: residual with forcing
                    vdel[k, 1, iv] = vsrez_tesys[k] +
                                     vs1_tesys[k, 4] * due1 + vs1_tesys[k, 3] * dds1 +
                                     vs2_tesys[k, 4] * due2 + vs2_tesys[k, 3] * dds2
                    vdel[k, 2, iv] = 0.0
                end

                # VZ: coupling to upper TE (applied in BLSOLV at IV=JVTE1 → IVZ row)
                for k in 1:3
                    vz[k, 1] = vs1_tesys[k, 1] * cte_cte1
                    vz[k, 2] = vs1_tesys[k, 1] * cte_tte1 + vs1_tesys[k, 2] * tte_tte1
                end

                # VB: coupling to lower TE (previous station on side 2)
                for k in 1:3
                    vb[k, 1, iv] = vs1_tesys[k, 1] * cte_cte2
                    vb[k, 2, iv] = vs1_tesys[k, 1] * cte_tte2 + vs1_tesys[k, 2] * tte_tte2
                end

            else
                # ---- Normal station ----
                # Get previous station state
                if ibl >= 3
                    p1 = _viscal_get_primary(state, ibl - 1, is)
                else
                    p1 = BLPrimaryState(
                        max(p2.x * 0.5, 1e-10),
                        p2.u, p2.t, p2.d, p2.s, p2.ampl, p2.dw,
                    )
                end

                # Compute interval Jacobian using literal BL system (xbl_blsys!)
                # matching the SAME equations used by xbl_mrchdu!
                local vs1_loc, vs2_loc, vsrez_loc
                try
                    _fl = BLFlowF32(XFT(0), XFT(flow.qinfbl), XFT(0), XFT(0),
                        XFT(flow.rstbl), XFT(flow.rstbl_ms), XFT(flow.hstinv), XFT(flow.hstinv_ms),
                        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
                        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
                        XFT(1), XFT(1e6), XFT(ncrit), Int32(0))
                    _pr = BLParF32()
                    # Station 1: use carried COM1 if available
                    if ibl >= 3 && _c1_carried.t > XFT(0)
                        copycom!(_c1_blsys, _c1_carried)
                    else
                        xbl_blprv!(_c1_blsys, XFT(p1.x), XFT(p1.ampl), XFT(p1.s),
                            XFT(p1.t), XFT(p1.d), XFT(p1.dw), XFT(p1.u), _fl)
                        xbl_blkin!(_c1_blsys, _fl)
                    end
                    # Station 2
                    xbl_blprv!(_c2_blsys, XFT(p2.x), XFT(p2.ampl), XFT(p2.s),
                        XFT(p2.t), XFT(p2.d), XFT(p2.dw), XFT(p2.u), _fl)
                    xbl_blkin!(_c2_blsys, _fl)

                    _vs1 = zeros(XFT,4,5); _vs2 = zeros(XFT,4,5); _vr = zeros(XFT,4)
                    _vm = zeros(XFT,4); _vsr = zeros(XFT,4); _vsx = zeros(XFT,4)
                    xbl_blsys!(_vs1, _vs2, _vr, _vm, _vsr, _vsx,
                        _c1_blsys, _c2_blsys,
                        mode == :similarity, mode == :transition,
                        mode == :turbulent, mode == :wake,
                        _fl, _pr)

                    vs1_loc = zeros(Float64,4,5); vs2_loc = zeros(Float64,4,5); vsrez_loc = zeros(Float64,4)
                    for k in 1:4, l in 1:5
                        vs1_loc[k,l] = Float64(_vs1[k,l]); vs2_loc[k,l] = Float64(_vs2[k,l])
                    end
                    for k in 1:4; vsrez_loc[k] = Float64(_vr[k]); end
                    # COM1=COM2 propagation
                    copycom!(_c1_carried, _c2_blsys)
                catch
                    vs1_loc = nothing
                end

                if isnothing(vs1_loc)
                    va[1, 1, iv] = 1.0; va[2, 2, iv] = 1.0; vm[3, iv, iv] = 1.0
                    u1_m .= u2_m; d1_m .= d2_m; due1 = due2; dds1 = dds2
                    continue
                end

                vs1 = vs1_loc
                vs2 = vs2_loc
                vsrez = vsrez_loc

                # XI sensitivities wrt LE Ue changes
                xi_ule1 = is == 1 ? sst_go : -sst_go
                xi_ule2 = is == 1 ? -sst_gp : sst_gp

                # Fill VA, VB, VM, VDEL for equations 1-3
                for k in 1:3
                    va[k, 1, iv] = vs2[k, 1]
                    va[k, 2, iv] = vs2[k, 2]

                    vb[k, 1, iv] = vs1[k, 1]
                    vb[k, 2, iv] = vs1[k, 2]

                    for jv in 1:nsys
                        vm[k, jv, iv] = vs1[k, 3] * d1_m[jv] + vs1[k, 4] * u1_m[jv] +
                                        vs2[k, 3] * d2_m[jv] + vs2[k, 4] * u2_m[jv] +
                                        (vs1[k, 5] + vs2[k, 5]) *
                                        (xi_ule1 * ule1_m[jv] + xi_ule2 * ule2_m[jv])
                    end

                    vdel[k, 1, iv] = vsrez[k] +
                                     vs1[k, 4] * due1 + vs1[k, 3] * dds1 +
                                     vs2[k, 4] * due2 + vs2[k, 3] * dds2 +
                                     (vs1[k, 5] + vs2[k, 5]) *
                                     (xi_ule1 * dule1 + xi_ule2 * dule2)
                    vdel[k, 2, iv] = 0.0
                end
            end

            # Shift "2" → "1" for next station (matching XFoil COM1=COM2)
            u1_m .= u2_m
            d1_m .= d2_m
            due1 = due2
            dds1 = dds2
            kin1_carried = kin2_current  # COM2 → COM1 for next station
        end
    end

    return va, vb, vm, vdel, vz
end

"""
UPDATE: Apply Newton deltas to BL variables with underrelaxation.
Port of XFoil's xbl.f UPDATE subroutine.
"""
function _viscal_update!(
    state::ViscalBLState,
    vdel::Array{Float64,3},
    dij::Matrix{Float64},
    inv::CaseResult,
    alpha_deg::Float64;
    flow::BLFlowParams,
)
    FT = Float32
    alfa = FT(alpha_deg) * FT(π) / FT(180.0)
    sa = sin(alfa)
    ca = cos(alfa)
    n = length(inv.xnode)
    xnode = FT.(inv.xnode)
    ynode = FT.(inv.ynode)
    qnode = FT.(inv.qnode)
    qinf = max(FT(flow.qinfbl), FT(1e-10))

    # Compute new Ue distribution and its AC sensitivity (Fortran UPDATE lines 1293-1320)
    unew = zeros(FT, size(state.uedg))
    u_ac = zeros(FT, size(state.uedg))
    for is in 1:2
        for ibl in 2:state.nbl[is]
            i = state.ipan[ibl, is]
            dui = zero(FT)
            dui_ac = zero(FT)
            for js in 1:2
                for jbl in 2:state.nbl[js]
                    j = state.ipan[jbl, js]
                    jv = state.isys[jbl, js]
                    if jv >= 1 && jv <= state.nsys && i >= 1 && j >= 1 &&
                       i <= size(dij, 1) && j <= size(dij, 2)
                        ue_m = -state.vti[ibl, is] * state.vti[jbl, js] * FT(dij[i, j])
                        dui    += ue_m * (state.mass[jbl, js] + FT(vdel[3, 1, jv]))
                        dui_ac += ue_m * (                   - FT(vdel[3, 2, jv]))
                    end
                end
            end
            # LALFA=true → UINV_AC = 0 (line 1311)
            unew[ibl, is] = state.uinv[ibl, is] + dui
            u_ac[ibl, is] = dui_ac
        end
    end

    # Set new Qtan from new Ue (Fortran lines 1323-1329) — surface panels only
    qnew = copy(qnode)  # default: inviscid
    q_ac = zeros(FT, n)
    for is in 1:2
        for ibl in 2:state.iblte[is]
            i = state.ipan[ibl, is]
            if i >= 1 && i <= n
                qnew[i] = state.vti[ibl, is] * unew[ibl, is]
                q_ac[i] = state.vti[ibl, is] * u_ac[ibl, is]
            end
        end
    end

    # Calculate new CL from panel velocities (Fortran lines 1342-1383)
    # Incompressible: Cp = 1 - (Q/QINF)^2, CL = ∮ Cp·dx
    clnew = zero(FT)
    cl_ac = zero(FT)
    cpg1 = FT(1.0) - (qnew[1] / qinf)^2
    cpg1_ac = -FT(2.0) * qnew[1] / qinf^2 * q_ac[1]
    for i in 1:n
        ip = i == n ? 1 : i + 1
        cpg2 = FT(1.0) - (qnew[ip] / qinf)^2
        cpg2_ac = -FT(2.0) * qnew[ip] / qinf^2 * q_ac[ip]

        dx = (xnode[ip] - xnode[i]) * ca + (ynode[ip] - ynode[i]) * sa
        ag = FT(0.5) * (cpg2 + cpg1)
        ag_ac = FT(0.5) * (cpg2_ac + cpg1_ac)

        clnew += dx * ag
        cl_ac += dx * ag_ac

        cpg1 = cpg2
        cpg1_ac = cpg2_ac
    end

    # Compute DAC (Fortran lines 1388-1396): LALFA=true → AC is CL
    # DAC = (CLNEW - CL) / (1 - CL_AC)
    # CL_MS terms vanish for incompressible (MINF=0)
    cl_old = FT(state.cl)
    denom_ac = FT(1.0) - cl_ac
    dac = (clnew - cl_old) / denom_ac

    # Underrelax DAC (DCLMAX/DCLMIN = ±0.5, Fortran lines 1285-1296)
    rlx = FT(1.0)
    dclmax = FT(0.5)
    dclmin = FT(-0.5)
    if rlx * dac > dclmax; rlx = dclmax / dac; end
    if rlx * dac < dclmin; rlx = dclmin / dac; end

    dhi = FT(1.5)
    dlo = FT(-0.5)
    rmsbl = zero(FT)

    # First pass: compute changes and determine relaxation (Fortran lines 1417-1486)
    for is in 1:2
        for ibl in 2:state.nbl[is]
            iv = state.isys[ibl, is]
            if iv < 1 || iv > state.nsys; continue; end

            # Apply DAC correction (Fortran lines 1424-1428)
            dctau = FT(vdel[1, 1, iv]) - dac * FT(vdel[1, 2, iv])
            dthet = FT(vdel[2, 1, iv]) - dac * FT(vdel[2, 2, iv])
            dmass = FT(vdel[3, 1, iv]) - dac * FT(vdel[3, 2, iv])
            duedg = unew[ibl, is] + dac * u_ac[ibl, is] - state.uedg[ibl, is]
            ddstr = (dmass - state.dstr[ibl, is] * duedg) / state.uedg[ibl, is]

            # Normalize changes (Fortran lines 1431-1435)
            if ibl < state.itran[is]
                dn1 = dctau / FT(10.0)
            else
                dn1 = dctau / state.ctau[ibl, is]
            end
            dn2 = dthet / state.thet[ibl, is]
            dn3 = ddstr / state.dstr[ibl, is]
            dn4 = abs(duedg) / FT(0.25)

            rmsbl += dn1^2 + dn2^2 + dn3^2 + dn4^2

            # Underrelax each variable (Fortran lines 1441-1483)
            for dn in (dn1, dn2, dn3, dn4)
                rdn = rlx * dn
                if rdn > dhi; rlx = dhi / dn; end
                if rdn < dlo; rlx = dlo / dn; end
            end
        end
    end

    # Set RMS (Fortran line 1489)
    ntotal = FT(state.nbl[1] + state.nbl[2])
    rmsbl = sqrt(rmsbl / max(FT(4.0) * ntotal, FT(1.0)))

    state.cl = Float64(cl_old + rlx * dac)

    # Second pass: apply underrelaxed changes (Fortran lines 1502-1552)
    hstinv = FT(flow.hstinv)
    gm1bl = FT(flow.gm1bl)
    for is in 1:2
        for ibl in 2:state.nbl[is]
            iv = state.isys[ibl, is]
            if iv < 1 || iv > state.nsys; continue; end

            dctau = FT(vdel[1, 1, iv]) - dac * FT(vdel[1, 2, iv])
            dthet = FT(vdel[2, 1, iv]) - dac * FT(vdel[2, 2, iv])
            dmass = FT(vdel[3, 1, iv]) - dac * FT(vdel[3, 2, iv])
            duedg = unew[ibl, is] + dac * u_ac[ibl, is] - state.uedg[ibl, is]
            ddstr = (dmass - state.dstr[ibl, is] * duedg) / state.uedg[ibl, is]

            state.ctau[ibl, is] += rlx * dctau
            state.thet[ibl, is] += rlx * dthet
            state.dstr[ibl, is] += rlx * ddstr
            state.uedg[ibl, is] += rlx * duedg

            # Wake gap
            dswaki = zero(FT)
            if ibl > state.iblte[is]
                iw = ibl - state.iblte[is]
                dswaki = iw <= length(state.wgap) ? state.wgap[iw] : zero(FT)
            end

            # Ctau clamp (Fortran line 1526)
            if ibl >= state.itran[is]
                state.ctau[ibl, is] = min(state.ctau[ibl, is], FT(0.25))
            end

            # DSLIM (Fortran lines 1528-1537)
            hklim = ibl <= state.iblte[is] ? FT(1.02) : FT(1.00005)
            ue = state.uedg[ibl, is]
            msq_local = ue^2 * hstinv / max(gm1bl * (FT(1.0) - FT(0.5) * ue^2 * hstinv), FT(1e-10))
            dsw = state.dstr[ibl, is] - dswaki
            h = dsw / max(state.thet[ibl, is], FT(1e-10))
            hk, hk_h, _ = hkin_jac(h, msq_local)
            dh = iszero(hk_h) ? zero(FT) : max(zero(FT), hklim - FT(hk)) / FT(hk_h)
            isfinite(dh) || (dh = zero(FT))
            state.dstr[ibl, is] = dsw + dh * state.thet[ibl, is] + dswaki

            # Nonlinear mass defect update (Fortran line 1540)
            state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
        end

        # No islands of negative Ue (Fortran lines 1545-1551)
        for ibl in 3:state.iblte[is]
            if state.uedg[ibl - 1, is] > FT(0.0) && state.uedg[ibl, is] <= FT(0.0)
                state.uedg[ibl, is] = state.uedg[ibl - 1, is]
                state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
            end
        end
    end

    # Equate upper wake to lower wake (Fortran lines 1556-1566)
    nwake = state.nbl[2] - state.iblte[2]
    for kbl in 1:nwake
        ibl1 = state.iblte[1] + kbl
        ibl2 = state.iblte[2] + kbl
        if ibl1 <= size(state.ctau, 1) && ibl2 <= size(state.ctau, 1)
            state.ctau[ibl1, 1] = state.ctau[ibl2, 2]
            state.thet[ibl1, 1] = state.thet[ibl2, 2]
            state.dstr[ibl1, 1] = state.dstr[ibl2, 2]
            state.uedg[ibl1, 1] = state.uedg[ibl2, 2]
            state.mass[ibl1, 1] = state.mass[ibl2, 2]
        end
    end

    return Float64(rmsbl), Float64(rlx)
end

"""
Refresh laminar amplification values on the converged primary state by replaying
XFoil's TRCHEK2 march over the final surface stations up to the current
transition index. This preserves the converged force solution while making the
reported laminar state more consistent with the final Ue/theta/dstar fields.
"""
function _viscal_refresh_laminar_ctau!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
)
    iblte = state.iblte[is]
    itran = clamp(state.itran[is], 2, iblte + 2)
    itran <= 3 && return nothing

    fl32 = BLFlowF32(
        XFT(0), XFT(flow.qinfbl),
        XFT(0), XFT(0),
        XFT(flow.rstbl), XFT(flow.rstbl_ms),
        XFT(flow.hstinv), XFT(flow.hstinv_ms),
        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
        XFT(1), XFT(1e6), XFT(ncrit), Int32(0),
    )

    function _build_laminar_com(ibl::Int, ampl::Float32)
        c = BLCom()
        xbl_blprv!(
            c,
            state.xssi[ibl, is],
            ampl,
            XFT(0.03),
            state.thet[ibl, is],
            state.dstr[ibl, is],
            XFT(0),
            state.uedg[ibl, is],
            fl32,
        )
        xbl_blkin!(c, fl32)
        return c
    end

    # The remaining exact-current-dump tail is concentrated in the final
    # transition-adjacent laminar neighborhood. Replay the upstream laminar
    # chain to get a better local seed, but only overwrite the last few
    # stations so the already-matched upstream coupled state stays intact.
    refresh_start = max(3, min(itran - 4, iblte))
    ampl_prev = state.ctau[2, is]
    refresh_end = min(itran - 1, iblte)
    for ibl in 3:refresh_end
        c1 = _build_laminar_com(ibl - 1, ampl_prev)
        c2 = _build_laminar_com(ibl, state.ctau[ibl, is])
        c2sav = BLCom()
        tr = xbl_trchek2!(c1, c2, c2sav, fl32)
        tr.tran && break
        ampl_prev = c2.ampl
        if ibl >= refresh_start
            state.ctau[ibl, is] = ampl_prev
        end
    end

    return nothing
end

"""
Recover the exact transition location (in XFoil's XSSI coordinate) from the
converged transition interval instead of snapping to the first turbulent
station. This mirrors XFoil's TRCHEK2 solve on the final interval state.
"""
function _viscal_transition_xsi(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
)
    iblte = state.iblte[is]
    itran = state.itran[is]

    itran > iblte && return NaN
    itran <= 2 && return Float64(state.xssi[clamp(itran, 2, iblte), is])

    fl = BLFlowF32(
        XFT(0), XFT(flow.qinfbl),
        XFT(0), XFT(0),
        XFT(flow.rstbl), XFT(flow.rstbl_ms),
        XFT(flow.hstinv), XFT(flow.hstinv_ms),
        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
        XFT(1), XFT(1e6), XFT(ncrit), Int32(0),
    )
    fl = xbl_with_xiforc(fl, XFT(state.xssi[iblte, is]))

    ibl1 = itran - 1
    ibl2 = itran
    dswak1 = ibl1 > iblte ? (ibl1 - iblte <= length(state.wgap) ? XFT(state.wgap[ibl1 - iblte]) : XFT(0)) : XFT(0)
    dswak2 = ibl2 > iblte ? (ibl2 - iblte <= length(state.wgap) ? XFT(state.wgap[ibl2 - iblte]) : XFT(0)) : XFT(0)

    ampl1 = ibl1 < itran ? XFT(state.ctau[ibl1, is]) : XFT(0)
    ampl2 = ibl2 < itran ? XFT(state.ctau[ibl2, is]) : XFT(0)
    cti1 = ibl1 >= itran ? max(XFT(state.ctau[ibl1, is]), XFT(1e-7)) : XFT(0.03)
    cti2 = ibl2 >= itran ? max(XFT(state.ctau[ibl2, is]), XFT(1e-7)) : XFT(0.03)

    c1 = BLCom()
    c2 = BLCom()
    c2sav = BLCom()

    try
        xbl_blprv!(
            c1,
            XFT(state.xssi[ibl1, is]),
            ampl1,
            cti1,
            XFT(state.thet[ibl1, is]),
            XFT(state.dstr[ibl1, is]),
            dswak1,
            XFT(state.uedg[ibl1, is]),
            fl,
        )
        xbl_blkin!(c1, fl)

        xbl_blprv!(
            c2,
            XFT(state.xssi[ibl2, is]),
            ampl2,
            cti2,
            XFT(state.thet[ibl2, is]),
            XFT(state.dstr[ibl2, is]),
            dswak2,
            XFT(state.uedg[ibl2, is]),
            fl,
        )
        xbl_blkin!(c2, fl)

        tr = xbl_trchek2!(c1, c2, c2sav, fl)
        return tr.tran ? Float64(tr.xt) : NaN
    catch
        return NaN
    end
end

function _viscal_surface_qvis(state::ViscalBLState, inv::CaseResult)
    qvis = Float64.(inv.qnode)
    for is in 1:2
        for ibl in 2:state.iblte[is]
            i = state.ipan[ibl, is]
            if 1 <= i <= length(qvis)
                qvis[i] = Float64(state.vti[ibl, is] * state.uedg[ibl, is])
            end
        end
    end
    return qvis
end

function _viscal_trace_sample(
    state::ViscalBLState,
    inv::CaseResult;
    flow::BLFlowParams,
    ncrit::Float64,
    residual::Real,
    global_iteration::Integer,
    segment::Symbol,
    segment_iteration::Integer,
    mrchdu_active::Bool,
    transition_recheck_active::Bool,
)
    qvis = _viscal_surface_qvis(state, inv)
    topo = _xfoil_bl_topology_f32(inv; q = qvis)
    upper_xtr = _viscal_transition_xsi(state, 1; flow = flow, ncrit = ncrit)
    lower_xtr = _viscal_transition_xsi(state, 2; flow = flow, ncrit = ncrit)

    cl, cm_c4, _ = _clcalc_from_surface_speed(
        inv.xnode,
        inv.ynode,
        qvis,
        inv.alpha_deg;
        xref = 0.25,
        yref = 0.0,
        qinf = Float64(flow.qinfbl),
        minf = 0.0,
    )
    cd = _viscal_pressure_drag(state; flow = flow)
    top_xtr = _viscal_transition_xc(
        state,
        inv,
        :upper;
        qvis = qvis,
        topology = topo,
        transition_x = upper_xtr,
    )
    bot_xtr = _viscal_transition_xc(
        state,
        inv,
        :lower;
        qvis = qvis,
        topology = topo,
        transition_x = lower_xtr,
    )

    return CoupledIterationSample(
        Int(global_iteration),
        segment,
        Int(segment_iteration),
        float(residual),
        cl,
        cd,
        cm_c4,
        top_xtr,
        bot_xtr,
        Int(state.itran[1]),
        Int(state.itran[2]),
        mrchdu_active,
        transition_recheck_active,
    )
end

function _viscal_pressure_drag(
    state::ViscalBLState;
    flow::BLFlowParams,
    minf::Real = 0.0,
)
    ibl = max(state.nbl[2], 2)
    thwake = max(Float64(state.thet[ibl, 2]), 1.0e-12)
    dswake = max(Float64(state.dstr[ibl, 2]), 1.01 * thwake)
    qinf = max(Float64(flow.qinfbl), 1.0e-12)
    uewake_raw = Float64(state.uedg[ibl, 2])
    beta = sqrt(max(1.0 - float(minf)^2, 1.0e-12))
    tklam = float(minf)^2 / (1.0 + beta)^2
    urat = uewake_raw / qinf
    uewake = uewake_raw * (1.0 - tklam) / max(1.0 - tklam * urat^2, 1.0e-12)
    shwake = dswake / thwake
    cdp = 2.0 * thwake * (uewake / qinf)^(0.5 * (5.0 + shwake))
    return max(cdp, 0.0)
end

function _viscal_transition_xc(
    state::ViscalBLState,
    inv::CaseResult,
    side::Symbol;
    qvis::Union{Nothing,AbstractVector{<:Real}} = nothing,
    topology::Union{Nothing,BLTopology} = nothing,
    transition_x::Union{Nothing,Real} = nothing,
    flow::Union{Nothing,BLFlowParams} = nothing,
    ncrit::Union{Nothing,Real} = nothing,
)
    side == :upper ? (is = 1; iblte = state.iblte[1]) :
    side == :lower ? (is = 2; iblte = state.iblte[2]) :
        throw(ArgumentError("side must be :upper or :lower"))

    if state.itran[is] >= iblte
        return 1.0
    end

    xtr =
        if isnothing(transition_x)
            isnothing(flow) && throw(ArgumentError("flow is required when transition_x is not provided"))
            isnothing(ncrit) && throw(ArgumentError("ncrit is required when transition_x is not provided"))
            _viscal_transition_xsi(state, is; flow = flow, ncrit = float(ncrit))
        else
            float(transition_x)
        end
    !isfinite(xtr) && return xtr

    qvis_local = isnothing(qvis) ? _viscal_surface_qvis(state, inv) : qvis
    topo_local = isnothing(topology) ? _xfoil_bl_topology_f32(inv; q = qvis_local) : topology
    stations = _surface_stations(inv, side; ue_panel = qvis_local, topology = topo_local)
    return _transition_xc_from_stations(stations.x, stations.xc, xtr)
end

"""
Convert ViscalBLState back to SurfaceMarchResult for output.
"""
function _viscal_to_surface(
    state::ViscalBLState,
    is::Int,
    inv::Union{Nothing,CaseResult} = nothing;
    flow::BLFlowParams,
    model::BLModelParams,
    transition_x::Union{Nothing,Real} = nothing,
)
    n = state.iblte[is] - 1  # surface stations only (excluding stagnation)
    x = Vector{Float64}(undef, n)
    xc = Vector{Float64}(undef, n)
    ue = Vector{Float64}(undef, n)
    modes = Vector{Symbol}(undef, n)
    primaries = Vector{BLPrimaryState}(undef, n)
    secondaries = Vector{BLSecondaryState}(undef, n)
    converged = trues(n)
    xtr = isnothing(transition_x) ? NaN : float(transition_x)

    has_xnode = !isnothing(inv) && length(inv.xnode) > 0

    for k in 1:n
        ibl = k + 1
        mode = _viscal_station_mode(state, ibl, is)
        modes[k] = mode
        p = _viscal_get_primary(state, ibl, is)
        primaries[k] = p
        x[k] = p.x
        # Fill xc from panel x/c coordinates
        ipan = state.ipan[ibl, is]
        if has_xnode && ipan >= 1 && ipan <= length(inv.xnode)
            xc[k] = float(inv.xnode[ipan])
        else
            xc[k] = 0.0
        end
        ue[k] = p.u

        ityp = mode == :laminar || mode == :similarity ? :laminar :
               mode == :wake ? :wake : :turbulent
        secondaries[k] = blsecondary_state(p, ityp; flow = flow, model = model)

        if !isfinite(xtr) && (mode == :transition || (mode == :turbulent && k > 1 && modes[k - 1] == :laminar))
            xtr = p.x
        end
    end

    side = is == 1 ? :upper : :lower
    return SurfaceMarchResult(
        side, x, xc, ue, modes, primaries, secondaries,
        xtr, true, converged,
    )
end

function _viscal_to_wake(
    state::ViscalBLState;
    flow::BLFlowParams,
    model::BLModelParams,
    wake_trace = nothing,
)
    nw = state.nbl[2] - state.iblte[2]
    nw <= 0 && return WakeMarchResult(
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        BLPrimaryState[],
        BLSecondaryState[],
        BitVector(),
    )

    x = Vector{Float64}(undef, nw)
    ue = Vector{Float64}(undef, nw)
    dw = Vector{Float64}(undef, nw)
    xw = Vector{Float64}(undef, nw)
    yw = Vector{Float64}(undef, nw)
    apanel = Vector{Float64}(undef, nw)
    primary = Vector{BLPrimaryState}(undef, nw)
    secondary = Vector{BLSecondaryState}(undef, nw)
    converged = trues(max(nw - 1, 0))

    for iw in 1:nw
        ibl = state.iblte[2] + iw
        p = _viscal_get_primary(state, ibl, 2)
        primary[iw] = p
        secondary[iw] = blsecondary_state(p, :wake; flow = flow, model = model)
        x[iw] = Float64(state.xssi[ibl, 2])
        ue[iw] = Float64(state.uedg[ibl, 2])
        dw[iw] = iw <= length(state.wgap) ? Float64(state.wgap[iw]) : Float64(p.dw)

        if !isnothing(wake_trace) && iw <= length(wake_trace.xw)
            xw[iw] = Float64(wake_trace.xw[iw])
            yw[iw] = Float64(wake_trace.yw[iw])
            apanel[iw] = Float64(wake_trace.apw[iw])
        else
            xw[iw] = 0.0
            yw[iw] = 0.0
            apanel[iw] = 0.0
        end
    end

    return WakeMarchResult(x, ue, dw, xw, yw, apanel, primary, secondary, converged)
end

"""
    _gauss4!(Z, R)

Solve 4×4 dense system Z*x = R in-place (R overwritten with solution).
Direct translation of XFoil's GAUSS subroutine with partial pivoting.
"""
function _gauss4!(Z::AbstractMatrix{T}, R::AbstractVector{T}) where T
    nn = 4
    @inbounds for np in 1:nn-1
        np1 = np + 1
        # partial pivoting
        nx = np
        for n in np1:nn
            if abs(Z[n, np]) > abs(Z[nx, np])
                nx = n
            end
        end
        pivot = one(T) / Z[nx, np]
        Z[nx, np] = Z[np, np]
        for l in np1:nn
            tmp = Z[nx, l] * pivot
            Z[nx, l] = Z[np, l]
            Z[np, l] = tmp
        end
        tmp = R[nx] * pivot
        R[nx] = R[np]
        R[np] = tmp
        # forward elimination
        for k in np1:nn
            ztmp = Z[k, np]
            for l in np1:nn
                Z[k, l] -= ztmp * Z[np, l]
            end
            R[k] -= ztmp * R[np]
        end
    end
    # last row
    R[nn] /= Z[nn, nn]
    # back substitution
    for np in nn-1:-1:1
        for k in np+1:nn
            R[np] -= Z[np, k] * R[k]
        end
    end
    return R
end

"""
    _viscal_mrchdu!(state, is; flow, model, ncrit, ante)

Line-by-line translation of XFoil's MRCHDU subroutine (xbl.f 886-1202).
Marches the BL on side `is` in mixed Ue-Hk mode using the current edge
velocity.  At each station a local 4×4 Newton system is solved for
(dCtau/dAmpl, dTheta, dDstr, dUe).  The 4th equation prescribes a
combination of Ue and Hk quasi-normal to the natural Ue-Hk characteristic,
which avoids the Goldstein singularity near separation.

This replaces the previous `_viscal_remarch!` which delegated to the
generic `_march_surface` function and didn't match XFoil's MRCHDU closely
enough, causing Newton-MRCHDU oscillation in the global VISCAL loop.
"""
function _viscal_mrchdu!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    ante::Float64 = 0.0,
    lock_ue::Bool = false,  # if true, prescribe Ue (MRCHUE mode) instead of Ue-Hk tracking
    max_local_iter::Int = 25,  # max Newton iterations per station (XFoil default: 25)
    preserve_itran::Bool = false,  # if true, keep existing itran instead of rediscovering
)
    FT = Float32   # match XFoil single precision
    deps = FT(5e-6)
    senswt = FT(1000)

    iblte = state.iblte[is]
    nbl   = state.nbl[is]

    # Cache for the converged BLSYS at each station — used by SETBL to avoid
    # recomputing the system (which would give subtly different results).
    # Indexed by ibl; positive keys hold (vs1, vs2, vsrez) BLSYS results,
    # negative keys hold kinematic state NamedTuples from _blkin_full.
    blsys_cache = Dict{Int, Any}()
    itrold = state.itran[is]

    tran  = false
    turb  = false
    if !preserve_itran
        state.itran[is] = iblte  # default: fully laminar; will be updated by TRCHEK2
    end
    # When preserve_itran=true, state.itran[is] keeps its existing value.
    # The tran/turb flags will be set naturally as the march proceeds:
    # the ityp logic at each station checks ibl >= state.itran[is].

    bule = FT(1)  # leading edge pressure gradient parameter

    # Previous-station primary state (COM1 equivalent) —
    # initialised from station IBL=2 at the first step.
    p1_x  = FT(0); p1_u  = FT(0); p1_t  = FT(0); p1_d  = FT(0)
    p1_s  = FT(0); p1_ampl = FT(0); p1_dw = FT(0)

    sens = FT(0)

    for ibl in 2:nbl
        simi = (ibl == 2)
        wake = (ibl > iblte)

        # ── initialise current station from stored arrays ─────────────
        xsi = FT(state.xssi[ibl, is])
        uei = FT(state.uedg[ibl, is])
        thi = FT(state.thet[ibl, is])
        dsi = FT(state.dstr[ibl, is])

        if ibl < itrold
            ami = FT(state.ctau[ibl, is])   # amplification
            cti = FT(0.03)
        else
            cti = FT(state.ctau[ibl, is])
            if cti <= FT(0); cti = FT(0.03); end
            ami = FT(0)
        end

        dswaki = FT(0)
        if wake
            iw = ibl - iblte
            dswaki = iw <= length(state.wgap) ? FT(state.wgap[iw]) : FT(0)
        end

        # clamp DSI so that HK stays above limit
        if ibl <= iblte
            dsi = max(dsi - dswaki, FT(1.02) * thi) + dswaki
        else
            dsi = max(dsi - dswaki, FT(1.00005) * thi) + dswaki
        end

        ueref = FT(0)
        hkref = FT(0)
        sennew = FT(0)
        converged_station = false
        last_vs1_a = zeros(FT, 3, 5)
        last_vs2_a = zeros(FT, 3, 5)
        last_vsrez_a = zeros(FT, 3)

        # ── local Newton iterations ───────────────────────────────────
        for itbl in 1:max_local_iter
            # Build BLPrimaryState for stations 1 and 2
            p2 = BLPrimaryState(xsi, uei, thi, dsi;
                s = (ibl >= state.itran[is]) ? cti : FT(0),
                ampl = (ibl < state.itran[is]) ? ami : FT(0),
                dw = dswaki)

            if simi
                p1 = p2  # similarity: station 1 = station 2
            else
                p1 = BLPrimaryState(p1_x, p1_u, p1_t, p1_d;
                    s = p1_s, ampl = p1_ampl, dw = p1_dw)
            end

            # Determine mode for BLDIF
            ityp = if simi
                :similarity
            elseif wake
                :wake
            elseif tran
                :transition
            elseif ibl >= state.itran[is]
                :turbulent
            else
                :laminar
            end

            # ── TRCHEK2: implicit amplification check for transition ────
            if !simi && !turb && ityp != :wake && !preserve_itran
                # Compute kinematic state at stations 1 and 2
                kin1 = p1_t > FT(0) ? blkin_state(p1_u, p1_t, p1_d, flow) : nothing
                kin2 = blkin_state(uei, thi, dsi - dswaki, flow)

                hk1 = !isnothing(kin1) ? max(kin1.hk2, FT(1.05)) : FT(2)
                t1  = max(p1_t, FT(1e-12))
                rt1 = !isnothing(kin1) ? max(kin1.rt2, FT(1)) : FT(100)
                a1  = p1_ampl
                hk2c = max(kin2.hk2, FT(1.05))
                t2  = max(thi, FT(1e-12))
                rt2c = max(kin2.rt2, FT(1e-12))
                x1  = p1_x
                x2  = xsi
                dx  = x2 - x1
                amcrit = FT(ncrit)

                # Initial guess (forward Euler)
                ax_init = FT(axset(Float64(hk1), Float64(t1), Float64(rt1), Float64(a1),
                    Float64(hk2c), Float64(t2), Float64(rt2c), Float64(ami),
                    Float64(ncrit); idampv=0))
                ami = max(a1 + ax_init * dx, FT(0))

                # Implicit Newton iteration (TRCHEK2 algorithm)
                daeps = FT(5e-5)
                for itam in 1:30
                    # Interpolation weight for transition point
                    if ami <= amcrit
                        amplt = ami
                        sfa = FT(1)
                    else
                        amplt = amcrit
                        denom_sfa = max(abs(ami - a1), FT(1e-12))
                        sfa = (amplt - a1) / denom_sfa
                        sfa = clamp(sfa, FT(0), FT(1))
                    end
                    wf1 = FT(1) - sfa
                    wf2 = sfa

                    # Interpolate to transition point
                    tt = t1 * wf1 + t2 * wf2
                    dt_interp = (p1_d) * wf1 + (dsi - dswaki) * wf2
                    ut = p1_u * wf1 + uei * wf2

                    # Kinematic state at transition point
                    kin_t = blkin_state(ut, tt, dt_interp, flow)
                    hkt = max(kin_t.hk2, FT(1.05))
                    rtt = max(kin_t.rt2, FT(1e-12))

                    # Amplification rate over X1..XT interval
                    ax_new = FT(axset(Float64(hk1), Float64(t1), Float64(rt1), Float64(a1),
                        Float64(hkt), Float64(tt), Float64(rtt), Float64(amplt),
                        Float64(ncrit); idampv=0))

                    if ax_new <= FT(0); break; end

                    # Residual: AMPL2 - AMPL1 - AX*(X2-X1) = 0
                    res = ami - a1 - ax_new * dx

                    # Approximate Jacobian (neglecting AX sensitivity to ami for stability)
                    res_a2 = FT(1)  # dRes/dAMPL2 ≈ 1
                    da2 = -res / res_a2

                    # Underrelax
                    rlx = FT(1)
                    if abs(da2) > FT(1); rlx = FT(1) / abs(da2); end

                    if abs(da2) < daeps; break; end

                    # Don't let ami cross amcrit in one step
                    if (ami > amcrit && ami + rlx * da2 < amcrit) ||
                       (ami < amcrit && ami + rlx * da2 > amcrit)
                        ami = amcrit
                    else
                        ami = ami + rlx * da2
                    end
                end

                ami = max(ami, FT(0))

                if ami >= amcrit
                    tran = true
                    state.itran[is] = ibl
                    ityp = :transition
                else
                    state.itran[is] = ibl + 2
                end
            end

            # ── TESYS or BLSYS: assemble 4×5 system ──────────────────
            vs2_44 = zeros(FT, 4, 4)
            vsrez4 = zeros(FT, 4)

            is_tesys = (ibl == iblte + 1) && (ibl <= nbl) && !simi

            if is_tesys
                # TE junction: prescribe continuity
                iblte1 = state.iblte[1]
                iblte2 = state.iblte[2]
                tte = FT(state.thet[iblte1, 1]) + FT(state.thet[iblte2, 2])
                dte = FT(state.dstr[iblte1, 1]) + FT(state.dstr[iblte2, 2]) + FT(ante)
                cte = (FT(state.ctau[iblte1, 1]) * FT(state.thet[iblte1, 1]) +
                       FT(state.ctau[iblte2, 2]) * FT(state.thet[iblte2, 2])) / max(tte, FT(1e-10))

                vsrez4[1] = cte - cti
                vsrez4[2] = tte - thi
                vsrez4[3] = dte - (dsi - dswaki) - dswaki
                vsrez4[4] = FT(0)
                vs2_44[1, 1] = FT(1)   # d(rez1)/d(cti)
                vs2_44[2, 2] = FT(1)   # d(rez2)/d(thi)
                vs2_44[3, 3] = FT(1)   # d(rez3)/d(dsi)
                vs2_44[4, 4] = FT(1)   # d(rez4)/d(uei) — prescribe Ue

                ueref = uei
            else
                # ── BLSYS: call _bldif_analytic or _trdif_analytic for rows 1-3
                vs1_a, vs2_a, vsrez_a, _, _ = try
                    if ityp == :transition
                        _trdif_analytic(
                            p1, p2;
                            flow = flow, model = model,
                            acrit = FT(ncrit), idampv = 0,
                            precision = FT,
                        )
                    else
                        _bldif_analytic(
                            ityp, p1, p2;
                            flow = flow, model = model,
                            acrit = FT(ncrit), idampv = 0,
                            precision = FT,
                        )
                    end
                catch
                    # Fallback: identity system (skip this station)
                    vs2_44[1,1] = FT(1); vs2_44[2,2] = FT(1)
                    vs2_44[3,3] = FT(1); vs2_44[4,4] = FT(1)
                    break
                end

                # Fill rows 1-3 of the 4×4 system and cache for SETBL
                for k in 1:3
                    vs2_44[k, 1] = vs2_a[k, 1]
                    vs2_44[k, 2] = vs2_a[k, 2]
                    vs2_44[k, 3] = vs2_a[k, 3]
                    vs2_44[k, 4] = vs2_a[k, 4]
                    vsrez4[k]    = vsrez_a[k]
                end
                last_vs1_a .= vs1_a
                last_vs2_a .= vs2_a
                last_vsrez_a .= vsrez_a
            end

            # ── First iteration: set baseline Ue, Hk ─────────────────
            if itbl == 1
                # Compute HK2 and U2_UEI from current state
                kin2 = blkin_state(uei, thi, dsi - dswaki, flow)
                hk2 = max(kin2.hk2, wake ? FT(1.00005) : FT(1.05))

                # Compressibility: U2 = UEI*(1-TKBL)/(1-TKBL*(UEI/QINFBL)^2)
                # For incompressible (TKBL≈0): U2≈UEI, U2_UEI≈1
                tkbl = FT(0)  # incompressible approximation (Minf≈0)
                u2 = uei
                u2_uei = FT(1)

                ueref = u2
                hkref = hk2

                # If station was turbulent but is now laminar, extrapolate Hk
                if ibl < state.itran[is] && ibl >= itrold && ibl > 2
                    uem = FT(state.uedg[ibl-1, is])
                    dsm = FT(state.dstr[ibl-1, is])
                    thm = FT(state.thet[ibl-1, is])
                    msq = uem^2 * flow.hstinv / max(flow.gm1bl * (FT(1) - FT(0.5) * uem^2 * flow.hstinv), FT(1e-12))
                    hkref = hkin(dsm / max(thm, FT(1e-12)), msq)
                end

                # If station was laminar, reinitialise Ctau if now turbulent
                if ibl < itrold
                    if tran
                        cti = FT(0.03)
                        state.ctau[ibl, is] = FT(0.03)
                    end
                    if turb && ibl > 2
                        cti = FT(state.ctau[ibl-1, is])
                        state.ctau[ibl, is] = cti
                    end
                end

                # Compute HK2 derivatives for 4th equation
                # HK2_T2 ≈ -HK2/T2 * HK2_H2,  HK2_D2 ≈ HK2_H2/T2
                hk2_h2 = FT(1) / (FT(1) + FT(0.113) * kin2.m2)   # ∂HK/∂H
                h2 = (dsi - dswaki) / max(thi, FT(1e-12))
                hk2_t2 = hk2_h2 * (-h2 / max(thi, FT(1e-12)))
                hk2_d2 = hk2_h2 / max(thi, FT(1e-12))
                hk2_u2 = FT(0)  # neglect Mach correction for incompressible
            else
                # Recompute HK2 derivatives at current state
                kin2 = blkin_state(uei, thi, dsi - dswaki, flow)
                hk2 = max(kin2.hk2, wake ? FT(1.00005) : FT(1.05))
                u2 = uei
                u2_uei = FT(1)
                hk2_h2 = FT(1) / (FT(1) + FT(0.113) * kin2.m2)
                h2 = (dsi - dswaki) / max(thi, FT(1e-12))
                hk2_t2 = hk2_h2 * (-h2 / max(thi, FT(1e-12)))
                hk2_d2 = hk2_h2 / max(thi, FT(1e-12))
                hk2_u2 = FT(0)
            end

            # ── 4th equation: Ue-Hk coupling or prescribed Ue ────────
            if !is_tesys
                if lock_ue || simi || (ibl == iblte + 1)
                    # Prescribe Ue (MRCHUE mode or similarity/first-wake)
                    # 4th equation: U2 = UEREF  →  solve for θ, δ*, Ctau only
                    vs2_44[4, 1] = FT(0)
                    vs2_44[4, 2] = FT(0)
                    vs2_44[4, 3] = FT(0)
                    vs2_44[4, 4] = u2_uei
                    vsrez4[4]    = ueref - u2
                else
                    # ── Ue-Hk characteristic tracking (MRCHDU mode) ───
                    vtmp = copy(vs2_44)
                    vztmp = copy(vsrez4)

                    vtmp[4, 1] = FT(0)
                    vtmp[4, 2] = hk2_t2
                    vtmp[4, 3] = hk2_d2
                    vtmp[4, 4] = hk2_u2 * u2_uei
                    vztmp[4]   = FT(1)

                    _gauss4!(vtmp, vztmp)

                    sennew = senswt * vztmp[4] * hkref / max(abs(ueref), FT(1e-10))
                    if itbl <= 5
                        sens = sennew
                    elseif itbl <= 15
                        sens = FT(0.5) * (sens + sennew)
                    end

                    vs2_44[4, 1] = FT(0)
                    vs2_44[4, 2] = hk2_t2 * hkref
                    vs2_44[4, 3] = hk2_d2 * hkref
                    vs2_44[4, 4] = (hk2_u2 * hkref + sens / max(abs(ueref), FT(1e-10))) * u2_uei
                    vsrez4[4]    = -(hkref^2) * (hk2 / max(hkref, FT(1e-10)) - FT(1)) -
                                    sens * (u2 / max(abs(ueref), FT(1e-10)) - FT(1))
                end
            end

            # ── Solve 4×4 Newton system ───────────────────────────────
            _gauss4!(vs2_44, vsrez4)

            # ── Underrelax ────────────────────────────────────────────
            dmax = max(
                abs(vsrez4[2] / max(thi, FT(1e-10))),
                abs(vsrez4[3] / max(dsi, FT(1e-10))),
                abs(vsrez4[4] / max(abs(uei), FT(1e-10))),
            )
            if ibl >= state.itran[is]
                dmax = max(dmax, abs(vsrez4[1] / (FT(10) * max(cti, FT(1e-6)))))
            end

            rlx = FT(1)
            if dmax > FT(0.3)
                rlx = FT(0.3) / dmax
            end

            # ── Apply update (with NaN guard) ─────────────────────────
            if any(!isfinite, vsrez4)
                break  # solver produced NaN — abandon this station
            end
            if ibl < state.itran[is]
                ami += rlx * vsrez4[1]
            else
                cti += rlx * vsrez4[1]
            end
            thi += rlx * vsrez4[2]
            dsi += rlx * vsrez4[3]
            uei += rlx * vsrez4[4]

            # ── Clamp Ctau ────────────────────────────────────────────
            if ibl >= state.itran[is]
                cti = clamp(cti, FT(1e-7), FT(0.30))
            end

            # ── DSLIM: prevent Hk going below limit ──────────────────
            hklim = ibl <= iblte ? FT(1.02) : FT(1.00005)
            msq_loc = uei^2 * flow.hstinv / max(flow.gm1bl * (FT(1) - FT(0.5) * uei^2 * flow.hstinv), FT(1e-12))
            dsw = dsi - dswaki
            h_loc = dsw / max(thi, FT(1e-12))
            hk_loc = hkin(h_loc, msq_loc)
            hk_h_loc = FT(1) / (FT(1) + FT(0.113) * msq_loc)
            dh = max(FT(0), hklim - hk_loc) / max(hk_h_loc, FT(1e-6))
            dsi = dsw + dh * thi + dswaki

            # ── Check convergence ─────────────────────────────────────
            if dmax <= deps
                converged_station = true
                break
            end
        end  # Newton iterations

        # ── Extrapolate if not converged (matching XFoil lines 1120-1138) ──
        # (Skipped for brevity — the unconverged solution is used as-is
        #  if DMAX < 0.1, which is the common case.)

        sens = sennew

        # ── Cache the converged BLSYS for this station ────────────────
        blsys_cache[ibl] = (vs1=copy(last_vs1_a), vs2=copy(last_vs2_a), vsrez=copy(last_vsrez_a))

        # ── Store converged/best values (with NaN guard) ─────────────
        if isfinite(thi) && isfinite(dsi) && isfinite(uei)
            if ibl < state.itran[is]
                state.ctau[ibl, is] = isfinite(ami) ? FT(ami) : state.ctau[ibl, is]
            else
                state.ctau[ibl, is] = isfinite(cti) ? FT(clamp(cti, FT(1e-7), FT(0.3))) : state.ctau[ibl, is]
            end
            state.thet[ibl, is] = FT(max(thi, FT(1e-10)))
            state.dstr[ibl, is] = FT(max(dsi, FT(1e-10)))
            if !lock_ue
                state.uedg[ibl, is] = FT(uei)
            end
            state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
        end
        # If NaN, keep the old values (don't modify state)

        # ── Save "2" → "1" for next station (COM2 → COM1) ────────────
        # Compute full kinematic state at converged values (matching XFoil's
        # CALL BLPRV; CALL BLKIN; COM1=COM2 at end of each MRCHDU station).
        # This kin1_saved will be passed to _bldif_analytic via kin1_pre to
        # ensure SETBL sees exactly the same station-1 state as MRCHDU.
        kin1_saved = try
            _blkin_full(uei, thi, dsi - dswaki, flow; precision=FT)
        catch
            nothing
        end
        # Store in cache for SETBL to use as kin1_pre at the NEXT station
        if !isnothing(kin1_saved)
            blsys_cache[-ibl] = kin1_saved  # negative key = kinematic state for COM1
        end

        p1_x    = xsi
        p1_u    = uei
        p1_t    = thi
        p1_d    = dsi - dswaki
        p1_s    = ibl >= state.itran[is] ? cti : FT(0)
        p1_ampl = ibl < state.itran[is] ? ami : FT(0)
        p1_dw   = dswaki

        # ── Turbulent intervals follow transition or TE ───────────────
        if tran || ibl == iblte
            turb = true
        end
        tran = false

    end  # station loop

    return blsys_cache
end

"""
MRCHDU-equivalent re-march: update BL state to be consistent with current UEDG.
XFoil calls MRCHDU at the start of each SETBL to re-establish BL state
after UPDATE has modified edge velocities via mass-defect coupling.
"""
function _viscal_remarch!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    tol::Float64 = 1e-6,
    max_iter::Int = 20,
    use_mrchdu::Bool = false,
    lock_transition::Bool = false,
    max_itran_move::Int = 0,      # 0 = unlimited; >0 = limit transition movement
)
    # Only remarch surface stations (up to iblte).  Wake is handled separately.
    nsurf = state.iblte[is] - 1  # surface stations excluding stagnation

    # Build SurfaceStations with current (coupling-adjusted) Ue
    x = [state.xssi[ibl, is] for ibl in 2:state.iblte[is]]
    xc = zeros(nsurf)
    ue = [state.uedg[ibl, is] for ibl in 2:state.iblte[is]]
    side = is == 1 ? :upper : :lower
    stations = SurfaceStations(side, x, xc, ue)

    # Build seed from current surface state (only surface stations)
    seed_primaries = Vector{BLPrimaryState}(undef, nsurf)
    seed_modes = Vector{Symbol}(undef, nsurf)
    seed_secondaries = Vector{BLSecondaryState}(undef, nsurf)
    seed_x = Vector{Float64}(undef, nsurf)
    seed_xc = zeros(nsurf)
    seed_ue = Vector{Float64}(undef, nsurf)
    seed_conv = trues(nsurf)
    seed_xtr = NaN
    for k in 1:nsurf
        ibl = k + 1
        mode = _viscal_station_mode(state, ibl, is)
        seed_modes[k] = mode
        p = _viscal_get_primary(state, ibl, is)
        seed_primaries[k] = p
        seed_x[k] = p.x
        seed_ue[k] = p.u
        ityp = mode == :laminar || mode == :similarity ? :laminar : :turbulent
        seed_secondaries[k] = blsecondary_state(p, ityp; flow = flow, model = model)
        if mode in (:transition, :turbulent) && !isfinite(seed_xtr)
            seed_xtr = p.x
        end
    end
    seed = SurfaceMarchResult(side, seed_x, seed_xc, seed_ue, seed_modes,
        seed_primaries, seed_secondaries, seed_xtr, true, seed_conv)

    # Re-march surface BL to be consistent with current UEDG.
    # For Newton coupling (use_mrchdu=true), use MRCHDU mode which allows Ue
    # to deviate via Hk-Ue coupling, avoiding the Goldstein singularity near
    # separation.  For Picard coupling (use_mrchdu=false), use MRCHUE mode
    # which prescribes Ue exactly.
    result = _march_surface(
        stations;
        flow = flow, model = model, ncrit = ncrit,
        tol = tol, max_iter = max_iter,
        seed = seed,
        mrchue_local = !use_mrchdu,
        mrchdu_local = use_mrchdu,
        preserve_seed_transition = false,
        solve_similarity_station = true,
    )

    # Update surface state arrays from march result
    for k in 1:nsurf
        ibl = k + 1
        p = result.primary[k]
        m = result.mode[k]
        state.thet[ibl, is] = max(p.t, 1e-10)
        state.dstr[ibl, is] = max(p.d, 1e-10)
        state.ctau[ibl, is] = m in (:laminar, :similarity) ? p.ampl : p.s
        state.uedg[ibl, is] = p.u
        state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
    end

    # Update transition index from remarch result (unless locked)
    if !lock_transition
        itran_new = state.iblte[is] + 1  # default: fully laminar
        for k in 1:nsurf
            if result.mode[k] in (:turbulent, :transition)
                itran_new = k + 1
                break
            end
        end
        state.itran[is] = itran_new
    end

    return result
end

"""
Remarch wake stations on side 2 after surface remarch.
Uses TESYS to seed first wake station from combined TE values,
then marches remaining wake stations with wake BL equations.
"""
function _viscal_remarch_wake!(
    state::ViscalBLState;
    flow::BLFlowParams,
    model::BLModelParams,
    tol::Float64 = 1e-6,
    max_iter::Int = 20,
)
    nwake = state.nbl[2] - state.iblte[2]
    nwake <= 0 && return

    iblte1 = state.iblte[1]
    iblte2 = state.iblte[2]
    ueps = 1.0e-7

    # TESYS: combined TE values
    tte = state.thet[iblte1, 1] + state.thet[iblte2, 2]
    dte = state.dstr[iblte1, 1] + state.dstr[iblte2, 2]
    cte = (state.ctau[iblte1, 1] * state.thet[iblte1, 1] +
           state.ctau[iblte2, 2] * state.thet[iblte2, 2]) / max(tte, 1e-10)
    cte_clamped = clamp(cte, 1e-7, 0.30)
    ibl1 = iblte2 + 1
    dw1 = ibl1 <= size(state.wgap, 1) && 1 <= length(state.wgap) ? state.wgap[1] : 0.0
    p1 = BLPrimaryState(
        state.xssi[ibl1, 2],
        max(state.uedg[ibl1, 2], ueps),
        max(tte, 1e-10),
        max(dte + dw1, _mrchue_hklim(:wake) * max(tte, 1e-10) + dw1);
        s = max(cte_clamped, 1e-12),
        ampl = 0.0,
        dw = dw1,
    )
    state.thet[ibl1, 2] = p1.t
    state.dstr[ibl1, 2] = p1.d
    state.ctau[ibl1, 2] = p1.s
    state.mass[ibl1, 2] = p1.d * p1.u

    # March subsequent wake stations
    for iw in 2:nwake
        ibl = iblte2 + iw
        dw2 = iw <= length(state.wgap) ? state.wgap[iw] : 0.0
        x2 = max(state.xssi[ibl, 2], p1.x + 1e-6)
        u2 = max(state.uedg[ibl, 2], ueps)

        pseed = _wake_primary_with_gap(
            p1, x2, u2, dw2;
            flow = flow, model = model, ueps = ueps,
        )
        p2, s2, ok, _ = _solve_surface_interval(
            :wake, p1, x2, u2;
            mrchue_local = false,
            mrchdu_local = true,
            flow = flow, model = model,
            acrit = 0.0,
            tol = tol, max_iter = max_iter,
            z_init = _interval_seed_unknowns(:wake, pseed, x2, u2;
                flow = flow, model = model, ueps = ueps),
            dw = dw2,
        )
        p2 = BLPrimaryState(
            p2.x, p2.u, p2.t,
            max(p2.d - p2.dw, _mrchue_hklim(:wake) * p2.t) + dw2;
            s = max(p2.s, 1e-12), ampl = 0.0, dw = dw2,
        )
        state.thet[ibl, 2] = max(p2.t, 1e-10)
        state.dstr[ibl, 2] = max(p2.d, 1e-10)
        state.ctau[ibl, 2] = p2.s
        state.uedg[ibl, 2] = p2.u
        state.mass[ibl, 2] = state.dstr[ibl, 2] * state.uedg[ibl, 2]
        p1 = p2
    end

    # Equalize upper wake pointers
    for iw in 1:nwake
        ibl1u = state.iblte[1] + iw
        ibl2w = iblte2 + iw
        if ibl1u <= size(state.ctau, 1)
            state.ctau[ibl1u, 1] = state.ctau[ibl2w, 2]
            state.thet[ibl1u, 1] = state.thet[ibl2w, 2]
            state.dstr[ibl1u, 1] = state.dstr[ibl2w, 2]
            state.uedg[ibl1u, 1] = state.uedg[ibl2w, 2]
            state.mass[ibl1u, 1] = state.mass[ibl2w, 2]
        end
    end
end

"""
QVFUE + GAMQV + STMOVE equivalent:
After UPDATE, convert UEDG → QVIS → GAM, find new stagnation point,
and recompute BL arc length array XSSI.  If the stagnation panel changes,
also recompute UINV.
"""
function _viscal_qvfue_gamqv_stmove!(
    state::ViscalBLState,
    inv::CaseResult,
    topo::BLTopology,
)
    n = length(inv.qnode)

    # QVFUE + GAMQV: compute viscous panel velocity qvis = VTI * UEDG
    qvis = Float32.(inv.qnode)  # start from inviscid as default
    for is in 1:2
        for ibl in 2:state.nbl[is]
            i = state.ipan[ibl, is]
            if i >= 1 && i <= n
                qvis[i] = state.vti[ibl, is] * state.uedg[ibl, is]
            end
        end
    end

    # STFIND: locate the current stagnation point from the updated qvis field.
    stag_new, s = try
        _xfoil_stfind_f32(inv, qvis)
    catch
        # If STFIND fails, still apply Ue clamping and mass update
        _viscal_stmove_clamp!(state)
        return
    end

    istold = state.iblte[1] - 1
    istnew = stag_new.ist

    nw = state.nbl[2] - state.iblte[2]
    old_iblte1 = state.iblte[1]
    old_iblte2 = state.iblte[2]
    xi_te_u_old = Float64(state.xssi[old_iblte1, 1])
    xi_te_l_old = Float64(state.xssi[old_iblte2, 2])
    wake_offsets = [Float64(state.xssi[old_iblte2 + iw, 2]) - xi_te_l_old for iw in 1:nw]
    wake_uinv = [Float32(state.uinv[old_iblte2 + iw, 2]) for iw in 1:nw]
    reuse_cached_surface_topology =
        istnew == istold &&
        _viscal_surface_topology_indices_valid(state, 1, n) &&
        _viscal_surface_topology_indices_valid(state, 2, n)

    if reuse_cached_surface_topology
        upper_side = BLSideTopology(
            :upper,
            Int.(state.ipan[1:old_iblte1, 1]),
            Float64.(state.vti[1:old_iblte1, 1]),
            zeros(Float64, old_iblte1),
        )
        lower_side = BLSideTopology(
            :lower,
            Int.(state.ipan[1:old_iblte2, 2]),
            Float64.(state.vti[1:old_iblte2, 2]),
            zeros(Float64, old_iblte2),
        )

        upper_xi = _xfoil_xicalc_f32(s, stag_new, upper_side)
        lower_xi = _xfoil_xicalc_f32(s, stag_new, lower_side)
        state.xssi[1:old_iblte1, 1] .= Float32.(upper_xi.xssi)
        state.xssi[1:old_iblte2, 2] .= Float32.(lower_xi.xssi)

        xi_te_u_new = Float64(state.xssi[old_iblte1, 1])
        xi_te_l_new = Float64(state.xssi[old_iblte2, 2])
        for iw in 1:nw
            ibl1 = old_iblte1 + iw
            ibl2 = old_iblte2 + iw
            offset = Float32(wake_offsets[iw])
            state.xssi[ibl2, 2] = Float32(xi_te_l_new) + offset
            state.uinv[ibl2, 2] = wake_uinv[iw]
            state.xssi[ibl1, 1] = Float32(xi_te_u_new) + offset
            state.uinv[ibl1, 1] = wake_uinv[iw]
        end

        if nw > 0 && !isempty(state.wgap)
            cache = _wake_geometry_cache(inv)
            wake_sw = [
                Float64(state.xssi[old_iblte2 + iw, 2]) - xi_te_l_new
                for iw in 1:nw
            ]
            state.wgap .= Float32.(_wake_gap_profile(cache, wake_sw))
        end

        _viscal_stmove_clamp!(state)
        return
    end

    # IBLPAN / UICALC / XICALC / IBLSYS: rebuild the current topology only
    # when the stagnation panel changes, matching XFoil STMOVE.
    topo_new = _xfoil_bl_topology_f32(inv; q = qvis)

    iblte1_new = length(topo_new.upper.ipan)
    iblte2_new = length(topo_new.lower.ipan)
    state.iblte[1] = iblte1_new
    state.iblte[2] = iblte2_new
    state.nbl[1] = iblte1_new
    state.nbl[2] = iblte2_new + nw

    state.ipan[1, 1] = 0
    state.ipan[1, 2] = 0
    state.vti[1, 1] = 0
    state.vti[1, 2] = 0
    state.xssi[1, 1] = 0
    state.xssi[1, 2] = 0

    for ibl in 2:iblte1_new
        state.ipan[ibl, 1] = topo_new.upper.ipan[ibl]
        state.vti[ibl, 1] = Float32(topo_new.upper.vti[ibl])
        state.xssi[ibl, 1] = Float32(topo_new.upper.xssi[ibl])
    end
    for ibl in 2:iblte2_new
        state.ipan[ibl, 2] = topo_new.lower.ipan[ibl]
        state.vti[ibl, 2] = Float32(topo_new.lower.vti[ibl])
        state.xssi[ibl, 2] = Float32(topo_new.lower.xssi[ibl])
    end

    xi_te_u_new = Float64(state.xssi[iblte1_new, 1])
    xi_te_l_new = Float64(state.xssi[iblte2_new, 2])
    for iw in 1:nw
        ipw = n + iw
        ibl1 = iblte1_new + iw
        ibl2 = iblte2_new + iw
        offset = Float32(wake_offsets[iw])

        state.ipan[ibl2, 2] = ipw
        state.vti[ibl2, 2] = -1.0f0
        state.xssi[ibl2, 2] = Float32(xi_te_l_new) + offset
        state.uinv[ibl2, 2] = wake_uinv[iw]

        state.ipan[ibl1, 1] = ipw
        state.vti[ibl1, 1] = 1.0f0
        state.xssi[ibl1, 1] = Float32(xi_te_u_new) + offset
        state.uinv[ibl1, 1] = wake_uinv[iw]
    end

    if nw > 0 && !isempty(state.wgap)
        cache = _wake_geometry_cache(inv)
        wake_sw = [
            Float64(state.xssi[iblte2_new + iw, 2]) - xi_te_l_new
            for iw in 1:nw
        ]
        state.wgap .= Float32.(_wake_gap_profile(cache, wake_sw))
    end

    for is in 1:2
        for ibl in 2:state.iblte[is]
            i = state.ipan[ibl, is]
            state.uinv[ibl, is] = state.vti[ibl, is] * inv.qnode[i]
        end
    end

    iv = 0
    for is in 1:2
        for ibl in 2:state.nbl[is]
            iv += 1
            state.isys[ibl, is] = iv
        end
    end
    state.nsys = iv

    if istnew != istold
        # Shift BL arrays if stagnation moved to a different panel
        # (Fortran STMOVE lines 2319-2390)
        idif = abs(istnew - istold)
        if idif >= 1 && idif < min(state.nbl[1], state.nbl[2]) - 2
            if istnew > istold
                # More points on top side (IS=1): shift top downstream, bottom upstream
                state.itran[1] = state.itran[1] + idif
                state.itran[2] = state.itran[2] - idif
                # Shift top side downstream
                for ibl in state.nbl[1]:-1:(idif+2)
                    for f in (state.ctau, state.thet, state.dstr, state.uedg)
                        f[ibl, 1] = f[ibl-idif, 1]
                    end
                end
                # Interpolate gap stations on top
                dudx = state.xssi[idif+2, 1] > 0 ?
                    state.uedg[idif+2, 1] / state.xssi[idif+2, 1] : Float32(0)
                for ibl in (idif+1):-1:2
                    state.ctau[ibl, 1] = state.ctau[idif+2, 1]
                    state.thet[ibl, 1] = state.thet[idif+2, 1]
                    state.dstr[ibl, 1] = state.dstr[idif+2, 1]
                    state.uedg[ibl, 1] = dudx * state.xssi[ibl, 1]
                end
                # Shift bottom side upstream
                for ibl in 2:state.nbl[2]
                    src = ibl + idif
                    for f in (state.ctau, state.thet, state.dstr, state.uedg)
                        f[ibl, 2] = f[src, 2]
                    end
                end
            else
                # More points on bottom side (IS=2): shift bottom downstream, top upstream
                state.itran[1] = state.itran[1] - idif
                state.itran[2] = state.itran[2] + idif
                # Shift bottom side downstream
                for ibl in state.nbl[2]:-1:(idif+2)
                    for f in (state.ctau, state.thet, state.dstr, state.uedg)
                        f[ibl, 2] = f[ibl-idif, 2]
                    end
                end
                # Interpolate gap stations on bottom
                dudx = state.xssi[idif+2, 2] > 0 ?
                    state.uedg[idif+2, 2] / state.xssi[idif+2, 2] : Float32(0)
                for ibl in (idif+1):-1:2
                    state.ctau[ibl, 2] = state.ctau[idif+2, 2]
                    state.thet[ibl, 2] = state.thet[idif+2, 2]
                    state.dstr[ibl, 2] = state.dstr[idif+2, 2]
                    state.uedg[ibl, 2] = dudx * state.xssi[ibl, 2]
                end
                # Shift top side upstream
                for ibl in 2:state.nbl[1]
                    src = ibl + idif
                    for f in (state.ctau, state.thet, state.dstr, state.uedg)
                        f[ibl, 1] = f[src, 1]
                    end
                end
            end
        end
    end

    # Fortran STMOVE lines 2392-2403: clamp ALL UEDG >= UEPS
    _viscal_stmove_clamp!(state)
end

@inline function _viscal_surface_topology_indices_valid(
    state::ViscalBLState,
    is::Int,
    n::Int,
)
    iblte = state.iblte[is]
    1 <= iblte <= size(state.ipan, 1) || return false
    state.ipan[1, is] == 0 || return false
    for ibl in 2:iblte
        i = state.ipan[ibl, is]
        1 <= i <= n || return false
    end
    return true
end

"""
Clamp all UEDG values to be >= UEPS (Fortran STMOVE lines 2392-2403, 2408-2412).
Prevents negative or zero Ue values that would cause Newton divergence.
Also updates MASS = DSTR * UEDG for consistency.
"""
function _viscal_stmove_clamp!(state::ViscalBLState)
    ueps = Float32(1.0e-7)
    for is in 1:2
        for ibl in 2:state.nbl[is]
            if state.uedg[ibl, is] <= ueps
                state.uedg[ibl, is] = ueps
            end
            # Always update mass (Fortran STMOVE lines 2408-2412)
            state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
        end
    end
end

"""
TRCHEK2-style transition recheck on the current ViscalBLState.
Walks through laminar stations on side `is`, computing implicit amplification
using axset (matching XFoil TRCHEK2), and updates state.itran[is] if
transition is detected earlier than what the march found.

This matches XFoil's architecture where TRCHEK2 is called inside MRCHDU's
per-station Newton loop and uses the coupled Ue to determine transition.
"""

"""
    _viscal_mrchdu_consistent!(state, is; flow, model, ncrit, ante)

Equation-consistent MRCHDU: marches the BL using _bldif_analytic (the SAME
equations as SETBL) with a 4th Ue-Hk tracking equation.  This ensures the
march and the Newton system see identical BL equations, preventing the
MRCHDU-Newton oscillation that occurs with the separate _viscal_mrchdu!.
"""
function _viscal_mrchdu_consistent!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    ante::Float64 = 0.0,
)
    iblte = state.iblte[is]
    nbl = state.nbl[is]
    itrold = state.itran[is]
    # Keep initial transition as baseline; TRCHEK2 can adjust within range
    # state.itran[is] = iblte  # DON'T reset — use initial march's transition

    deps = 5e-6
    tran = false
    turb = false
    # Allow itran to move but start from the initial march's value
    itran_current = state.itran[is]

    # Previous station primary state
    p1_prev = nothing

    for ibl in 2:nbl
        simi = (ibl == 2)
        wake = (ibl > iblte)

        xsi = Float64(state.xssi[ibl, is])
        uei = Float64(state.uedg[ibl, is])
        thi = Float64(state.thet[ibl, is])
        dsi = Float64(state.dstr[ibl, is])

        if ibl < itrold
            ami = Float64(state.ctau[ibl, is])
            cti = 0.03
        else
            cti = Float64(state.ctau[ibl, is])
            cti = cti <= 0.0 ? 0.03 : cti
            ami = 0.0
        end

        dswaki = 0.0
        if wake
            iw = ibl - iblte
            dswaki = iw <= length(state.wgap) ? Float64(state.wgap[iw]) : 0.0
        end

        # Clamp DSI
        if ibl <= iblte
            dsi = max(dsi - dswaki, 1.02 * thi) + dswaki
        else
            dsi = max(dsi - dswaki, 1.00005 * thi) + dswaki
        end

        ueref = 0.0
        hkref = 0.0

        for itbl in 1:25
            # Determine mode
            mode = if simi; :similarity
            elseif wake; :wake
            elseif tran; :transition
            elseif ibl >= state.itran[is]; :turbulent
            else; :laminar; end

            s_val = (ibl >= state.itran[is]) ? cti : 0.0
            ampl_val = (ibl < state.itran[is]) ? ami : 0.0

            p2 = BLPrimaryState(xsi, uei, thi, dsi;
                s=s_val, ampl=ampl_val, dw=dswaki)

            if isnothing(p1_prev)
                p1 = p2  # similarity: p1 = p2
            else
                p1 = p1_prev
            end

            # TRCHEK2: check transition at every sub-iteration (matching Fortran
            # MRCHDU line 975: called inside DO 100 ITBL loop).
            # Only check for stations that were laminar in the initial march
            # (ibl < itrold).  This allows transition to move LATER (downstream)
            # with coupled Ue but prevents false early transitions.
            if !simi && !turb && mode != :wake && !isnothing(p1_prev) && ibl < itrold
                kin2 = blkin_state(uei, thi, dsi - dswaki, flow)
                hk2c = max(kin2.hk2, 1.05)
                rt2c = max(kin2.rt2, 1e-12)
                kin1 = blkin_state(p1_prev.u, p1_prev.t,
                    p1_prev.d - p1_prev.dw, flow)
                hk1 = max(kin1.hk2, 1.05)
                t1_tr = max(p1_prev.t, 1e-12)
                rt1 = max(kin1.rt2, 1e-12)
                a1 = max(p1_prev.ampl, 0.0)
                dx = max(xsi - p1_prev.x, 1e-12)

                # Implicit TRCHEK2 Newton iteration (matching xblsys.f:277-420)
                ax_init = axset(hk1, t1_tr, rt1, a1, hk2c, thi, rt2c, ami,
                    ncrit; idampv=0)
                ami = max(a1 + ax_init * dx, 0.0)
                for itam in 1:30
                    if ami <= ncrit
                        amplt = ami; sfa = 1.0
                    else
                        amplt = ncrit
                        sfa = clamp((amplt - a1) / max(abs(ami - a1), 1e-12), 0.0, 1.0)
                    end
                    wf1 = 1.0 - sfa; wf2 = sfa
                    tt = t1_tr * wf1 + thi * wf2
                    dt_i = (p1_prev.d - p1_prev.dw) * wf1 + (dsi - dswaki) * wf2
                    ut = p1_prev.u * wf1 + uei * wf2
                    kin_t = blkin_state(ut, tt, dt_i, flow)
                    hkt = max(kin_t.hk2, 1.05)
                    rtt = max(kin_t.rt2, 1e-12)
                    ax_new = axset(hk1, t1_tr, rt1, a1, hkt, tt, rtt, amplt,
                        ncrit; idampv=0)
                    ax_new <= 0.0 && break
                    res = ami - a1 - ax_new * dx
                    da2 = -res
                    rlx_tr = abs(da2) > 1.0 ? 1.0 / abs(da2) : 1.0
                    abs(da2) < 5e-5 && break
                    if (ami > ncrit && ami + rlx_tr * da2 < ncrit) ||
                       (ami < ncrit && ami + rlx_tr * da2 > ncrit)
                        ami = ncrit
                    else
                        ami = ami + rlx_tr * da2
                    end
                end
                ami = max(ami, 0.0)
                if ami >= ncrit
                    tran = true
                    state.itran[is] = ibl
                    if cti <= 0.0; cti = 0.03; end
                    mode = :transition
                else
                    state.itran[is] = ibl + 2
                end
            end

            # Build 3×5 BL system using _bldif_analytic (SAME as SETBL)
            local vs1_a, vs2_a, vsrez_a
            try
                if mode == :transition
                    vs1_a, vs2_a, vsrez_a, _, _ = _trdif_analytic(
                        p1, p2; flow=flow, model=model, acrit=ncrit, idampv=0,
                        precision=Float64)
                else
                    vs1_a, vs2_a, vsrez_a, _, _ = _bldif_analytic(
                        mode, p1, p2; flow=flow, model=model, acrit=ncrit,
                        idampv=0, precision=Float64)
                end
            catch
                break  # Can't build system; skip station
            end

            # Build 4×4 system: rows 1-3 from BL equations, row 4 from Ue-Hk tracking
            J = zeros(4, 4)
            rhs = zeros(4)
            J[1:3, 1:4] .= vs2_a[1:3, 1:4]
            rhs[1:3] .= vsrez_a[1:3]

            # Get kinematic state for 4th equation
            kin = _blkin_full(uei, thi, dsi - dswaki, flow; precision=Float64)

            if itbl == 1
                ueref = max(abs(uei), 1e-10)
                hkref = max(kin.hk2, 1.05)
                # Reinitialize Ctau for mode change
                if ibl < state.itran[is] && ibl >= itrold
                    kin_prev = !isnothing(p1_prev) ?
                        _blkin_full(p1_prev.u, p1_prev.t, p1_prev.d - p1_prev.dw,
                            flow; precision=Float64) : nothing
                    if !isnothing(kin_prev)
                        hkref = max(kin_prev.hk2, 1.05)
                    end
                end
                if ibl < itrold
                    if tran; cti = 0.03; end
                    if turb && ibl >= 3; cti = max(Float64(state.ctau[ibl-1, is]), 0.0000001); end
                end
            end

            if simi || (ibl == iblte + 1) || wake
                # Prescribe Ue for similarity, first wake, and wake stations
                J[4, :] .= 0.0
                J[4, 4] = 1.0
                rhs[4] = ueref - uei
            else
                # Ue-Hk tracking for surface stations (Fortran MRCHDU lines 1036-1067)
                # Compute dUe/dHk sensitivity from the 4×4 BL system
                vtmp = copy(J)
                vztmp = copy(rhs)
                vtmp[4, :] .= 0.0
                vtmp[4, 2] = kin.hk2_t2
                vtmp[4, 3] = kin.hk2_d2
                vtmp[4, 4] = kin.hk2_u2
                vztmp[4] = 1.0
                local sennew = 0.0
                try
                    vtmp_solve = copy(vtmp); vztmp_solve = copy(vztmp)
                    _gauss4!(vtmp_solve, vztmp_solve)
                    sennew = 1000.0 * vztmp_solve[4] * hkref / max(ueref, 1e-10)
                catch; end

                # Set prescribed Ue-Hk combination (Fortran MRCHDU lines 1062-1067)
                J[4, :] .= 0.0
                J[4, 2] = kin.hk2_t2 * hkref
                J[4, 3] = kin.hk2_d2 * hkref
                J[4, 4] = kin.hk2_u2 * hkref + sennew / max(ueref, 1e-10)
                rhs[4] = -(hkref^2) * (kin.hk2 / hkref - 1.0) -
                          sennew * (uei / ueref - 1.0)
            end

            # Solve 4×4
            delta = try; J \ rhs; catch; break; end
            any(!isfinite, delta) && break

            # Underrelax
            dmax = max(abs(delta[2] / max(thi, 1e-10)),
                       abs(delta[3] / max(dsi, 1e-10)),
                       abs(delta[4] / max(abs(uei), 1e-10)))
            if ibl >= state.itran[is]
                dmax = max(dmax, abs(delta[1] / (10.0 * max(abs(cti), 1e-10))))
            end
            rlx = dmax > 0.3 ? 0.3 / dmax : 1.0

            # Update
            if ibl < state.itran[is]; ami += rlx * delta[1]; end
            if ibl >= state.itran[is]; cti += rlx * delta[1]; end
            thi += rlx * delta[2]
            dsi += rlx * delta[3]
            uei += rlx * delta[4]

            # Clamp
            if ibl >= state.itran[is]
                cti = clamp(cti, 0.0000001, 0.30)
            end
            hklim = ibl <= iblte ? 1.02 : 1.00005
            msq = uei^2 * flow.hstinv / max(flow.gm1bl * (1.0 - 0.5 * uei^2 * flow.hstinv), 1e-10)
            dsw = dsi - dswaki
            hk_val, hk_h, _ = hkin_jac(dsw / max(thi, 1e-10), msq)
            dh = max(0.0, hklim - hk_val) / max(hk_h, 1e-6)
            dsi = dsw + dh * thi + dswaki

            if dmax <= deps
                break
            end
        end

        # Store results
        if ibl < state.itran[is]; state.ctau[ibl, is] = Float32(ami); end
        if ibl >= state.itran[is]; state.ctau[ibl, is] = Float32(cti); end
        state.thet[ibl, is] = Float32(thi)
        state.dstr[ibl, is] = Float32(dsi)
        state.uedg[ibl, is] = Float32(uei)
        state.mass[ibl, is] = Float32(dsi * uei)

        # Set COM1 for next station
        s_val = (ibl >= state.itran[is]) ? cti : 0.0
        ampl_val = (ibl < state.itran[is]) ? ami : 0.0
        p1_prev = BLPrimaryState(xsi, uei, thi, dsi;
            s=s_val, ampl=ampl_val, dw=dswaki)

        # Transition logic
        if tran || ibl == iblte
            turb = true
        end
        tran = false
    end
end

"""
    _viscal_transition_recheck!(state, is; flow, ncrit)

Full-sweep transition recheck: walks ALL surface stations (ignoring current
itran) and re-discovers transition from the current Newton-corrected state.
Only modifies state.itran and state.ctau — does NOT touch uedg/thet/dstr.

This replaces the Fortran MRCHDU's per-station TRCHEK2 for transition
location update without the full march.
"""
function _viscal_transition_recheck!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
)
    iblte = state.iblte[is]
    amcrit = ncrit
    daeps = 5e-5

    old_itran = state.itran[is]
    new_itran = iblte  # default: fully laminar (transition at TE)

    ampl1 = 0.0
    hk1 = 2.0
    t1 = 1e-6
    rt1 = 100.0

    for ibl in 2:iblte
        ui = Float64(state.uedg[ibl, is])
        ti = max(Float64(state.thet[ibl, is]), 1e-12)
        di = max(Float64(state.dstr[ibl, is]), 1e-12)
        xi = max(Float64(state.xssi[ibl, is]), 1e-12)

        # Skip wake gap
        dswaki = 0.0
        if ibl > iblte
            iw = ibl - iblte
            dswaki = iw <= length(state.wgap) ? Float64(state.wgap[iw]) : 0.0
        end

        kin2 = blkin_state(ui, ti, di - dswaki, flow)
        hk2 = max(kin2.hk2, 1.05)
        rt2 = max(kin2.rt2, 1e-12)

        if ibl == 2
            hk1 = hk2; t1 = ti; rt1 = rt2
            # Use stored amplification if laminar, else 0
            ampl1 = ibl < old_itran ? max(Float64(state.ctau[ibl, is]), 0.0) : 0.0
            continue
        end

        x1 = max(Float64(state.xssi[ibl - 1, is]), 1e-12)
        dx = max(xi - x1, 1e-12)
        a1 = max(ampl1, 0.0)

        # Initial guess
        ax_init = axset(hk1, t1, rt1, a1, hk2, ti, rt2, a1, ncrit; idampv=0)
        ami = max(a1 + ax_init * dx, 0.0)

        # TRCHEK2 Newton iteration
        for itam in 1:30
            if ami <= amcrit
                amplt = ami; sfa = 1.0
            else
                amplt = amcrit
                sfa = clamp((amplt - a1) / max(abs(ami - a1), 1e-12), 0.0, 1.0)
            end
            wf1 = 1.0 - sfa; wf2 = sfa

            tt = t1 * wf1 + ti * wf2
            dt_i = Float64(state.dstr[ibl - 1, is]) * wf1 + di * wf2
            ut = Float64(state.uedg[ibl - 1, is]) * wf1 + ui * wf2

            kin_t = blkin_state(ut, tt, dt_i, flow)
            hkt = max(kin_t.hk2, 1.05)
            rtt = max(kin_t.rt2, 1e-12)

            ax_new = axset(hk1, t1, rt1, a1, hkt, tt, rtt, amplt, ncrit; idampv=0)
            ax_new <= 0.0 && break

            res = ami - a1 - ax_new * dx
            da2 = -res
            rlx = abs(da2) > 1.0 ? 1.0 / abs(da2) : 1.0
            abs(da2) < daeps && break

            if (ami > amcrit && ami + rlx * da2 < amcrit) ||
               (ami < amcrit && ami + rlx * da2 > amcrit)
                ami = amcrit
            else
                ami = ami + rlx * da2
            end
        end

        ami = max(ami, 0.0)

        if ami >= amcrit
            new_itran = ibl
            break
        end

        ampl1 = ami
        hk1 = hk2; t1 = ti; rt1 = rt2
    end

    # Only update if transition moved — re-initialize ctau at affected stations
    if new_itran != old_itran
        state.itran[is] = new_itran
        # Stations that switched from laminar to turbulent need ctau initialization
        if new_itran < old_itran
            for ibl in new_itran:min(old_itran, iblte)
                if ibl == new_itran
                    state.ctau[ibl, is] = 0.03
                elseif ibl > new_itran && ibl - 1 >= 2
                    state.ctau[ibl, is] = max(state.ctau[ibl - 1, is], 0.0000001)
                end
            end
        end
        # Stations that switched from turbulent to laminar need amplification values
        if new_itran > old_itran
            for ibl in old_itran:min(new_itran - 1, iblte)
                state.ctau[ibl, is] = max(ampl1 * Float64(ibl - 2) / max(new_itran - 2, 1), 0.0)
            end
        end
    end
end

function _viscal_trchek2!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
)
    iblte = state.iblte[is]
    amcrit = ncrit
    daeps = 5e-5

    # Walk through stations, computing implicit amplification
    ampl1 = 0.0  # amplification at previous station
    hk1 = 2.0    # kinematic shape parameter at previous station
    t1 = 1e-6    # momentum thickness at previous station
    rt1 = 100.0  # momentum thickness Reynolds number at previous station

    for ibl in 2:iblte
        # Get current station state
        ui = Float64(state.uedg[ibl, is])
        ti = max(Float64(state.thet[ibl, is]), 1e-12)
        di = max(Float64(state.dstr[ibl, is]), 1e-12)
        xi = max(Float64(state.xssi[ibl, is]), 1e-12)

        # Kinematic state at station 2
        kin2 = blkin_state(ui, ti, di, flow)
        hk2 = max(kin2.hk2, 1.05)
        rt2 = max(kin2.rt2, 1e-12)

        if ibl == 2
            # Initialize station 1 from station 2 (similarity)
            hk1 = hk2
            t1 = ti
            rt1 = rt2
            ampl1 = Float64(state.ctau[ibl, is])  # amplification stored in ctau for laminar
            continue
        end

        # Only check transition for laminar stations before current itran
        if ibl >= state.itran[is]
            break  # already turbulent; no need to recheck
        end

        x1 = max(Float64(state.xssi[ibl - 1, is]), 1e-12)
        dx = max(xi - x1, 1e-12)
        a1 = max(ampl1, 0.0)

        # Initial amplification guess (forward Euler)
        ax_init = axset(hk1, t1, rt1, a1, hk2, ti, rt2, a1, ncrit; idampv=0)
        ami = max(a1 + ax_init * dx, 0.0)

        # Implicit Newton iteration on amplification (TRCHEK2 algorithm)
        for itam in 1:30
            if ami <= amcrit
                amplt = ami
                sfa = 1.0
            else
                amplt = amcrit
                denom_sfa = max(abs(ami - a1), 1e-12)
                sfa = clamp((amplt - a1) / denom_sfa, 0.0, 1.0)
            end
            wf1 = 1.0 - sfa
            wf2 = sfa

            # Interpolate to transition point
            tt = t1 * wf1 + ti * wf2
            dt_interp = Float64(state.dstr[ibl - 1, is]) * wf1 + di * wf2
            ut = Float64(state.uedg[ibl - 1, is]) * wf1 + ui * wf2

            # Kinematic state at transition point
            kin_t = blkin_state(ut, tt, dt_interp, flow)
            hkt = max(kin_t.hk2, 1.05)
            rtt = max(kin_t.rt2, 1e-12)

            ax_new = axset(hk1, t1, rt1, a1, hkt, tt, rtt, amplt, ncrit; idampv=0)
            ax_new <= 0.0 && break

            res = ami - a1 - ax_new * dx

            res = ami - a1 - ax_new * dx
            da2 = -res  # dRes/dAMPL2 ≈ 1

            rlx = 1.0
            if abs(da2) > 1.0; rlx = 1.0 / abs(da2); end
            abs(da2) < daeps && break

            if (ami > amcrit && ami + rlx * da2 < amcrit) ||
               (ami < amcrit && ami + rlx * da2 > amcrit)
                ami = amcrit
            else
                ami = ami + rlx * da2
            end
        end

        ami = max(ami, 0.0)

        if ami >= amcrit
            # Transition detected at this station — update itran
            state.itran[is] = ibl
            # Update amplification/ctau for remaining laminar stations
            state.ctau[ibl, is] = ami
            return  # transition found, done
        end

        # Update amplification for this station
        state.ctau[ibl, is] = ami
        ampl1 = ami
        hk1 = hk2
        t1 = ti
        rt1 = rt2
    end
end

"""
Main VISCAL Newton solver.
Implements XFoil's coupled viscous-inviscid Newton iteration:
  MRCHDU → UESET → SETBL → BLSOLV → UPDATE → QVFUE/GAMQV/STMOVE
"""
function _viscal_solve(
    inv::CaseResult,
    topo::BLTopology,
    dij::Matrix{Float64},
    upper_march::SurfaceMarchResult,
    lower_march::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    max_iter::Int = 20,
    tol::Float64 = 1e-4,
    wake_trace = nothing,
    wake_march = nothing,
    transition_recheck::Bool = true,
    transition_recheck_upper_mode::Symbol = :literal,
    transition_recheck_lower_mode::Symbol = :literal,
    capture_trace::Bool = false,
    trace_segment::Symbol = :direct_exact,
    trace_offset::Int = 0,
    capture_separated_fallback::Bool = false,
    separated_fallback_kind::Symbol = :generic,
    separated_fallback_ref = nothing,
)
    alpha_deg = inv.alpha_deg

    # Initialize state from march results (with wake if provided)
    state = _init_viscal_state(inv, topo, upper_march, lower_march;
        wake_trace = wake_trace, wake_march = wake_march)
    best_state = deepcopy(state)
    trace_samples = CoupledIterationSample[]

    # Set inviscid velocities (UICALC) — includes wake panels if present
    _viscal_uicalc!(state, inv)

    # For wake panels beyond N, set UINV from wake_trace qinv
    if !isnothing(wake_trace)
        nw = length(wake_trace.xw)
        for iw in 1:nw
            ibl = state.iblte[2] + iw
            if ibl <= state.nbl[2]
                state.uinv[ibl, 2] = max(abs(Float64(wake_trace.qinv[iw])), 1e-7)
            end
        end
    end

    # Compute S(N)-S(1) for VACCEL scaling
    xn, yn = inv.xnode, inv.ynode
    sspan = 0.0
    for i in 2:length(xn)
        sspan += sqrt((xn[i] - xn[i-1])^2 + (yn[i] - yn[i-1])^2)
    end

    # TE base area (normal-projected gap) for TESYS DTE
    te_ante = Float64(_te_gap_geometry(inv).ante)

    rmsbl = Inf
    rlx = 1.0
    niter_done = 0
    best_iteration = 0
    best_residual = Inf
    separated_fallback_state = nothing
    separated_fallback_iteration = 0
    separated_fallback_residual = Inf
    separated_fallback_score = Inf

    # Build Float32 flow params for the literal BL system (xfoil_blsys.jl)
    _fl32 = BLFlowF32(
        XFT(0), XFT(flow.qinfbl),
        XFT(0), XFT(0),
        XFT(flow.rstbl), XFT(flow.rstbl_ms),
        XFT(flow.hstinv), XFT(flow.hstinv_ms),
        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
        XFT(1), XFT(1e6), XFT(ncrit), Int32(0))
    _par32 = BLParF32()
    _ante32 = XFT(te_ante)

    # Newton iterations: keep MRCHDU active only during the startup phase
    # while the coupled transition location is settling. Continuing to remarch
    # on every outer iteration can keep the system moving even after ITRAN has
    # stabilized, which hurts convergence on otherwise well-behaved cases.
    #
    # For the public transition-rechecked path, allow MRCHDU to continue past
    # the nominal startup window when the transition indices are still moving.
    # Freezing too early can lock the solver onto an unrealistically early
    # transition branch on some hard cases. The strict stagedump-parity path
    # keeps the old fixed startup window by disabling transition recheck.
    mrchdu_min_iters = min(max_iter, 20)
    mrchdu_max_iters = transition_recheck ? min(max_iter, 40) : mrchdu_min_iters
    mrchdu_frozen = false
    transition_stable_iters = 0
    prev_transition_signature = (Int(state.itran[1]), Int(state.itran[2]))
    for iter in 1:max_iter
        qvis_iter = copy(inv.qnode)
        for is in 1:2
            for ibl in 2:state.iblte[is]
                i = state.ipan[ibl, is]
                if i >= 1 && i <= length(qvis_iter)
                    qvis_iter[i] = state.vti[ibl, is] * state.uedg[ibl, is]
                end
            end
        end
        topo_iter = _xfoil_bl_topology_f32(inv; q = qvis_iter)

        itran_save = copy(state.itran)
        uedg_save = copy(state.uedg); thet_save = copy(state.thet)
        dstr_save = copy(state.dstr); ctau_save = copy(state.ctau)
        mass_save = copy(state.mass)

        do_mrchdu = !mrchdu_frozen && iter <= mrchdu_max_iters
        va, vb, vm, vdel, vz = try
            xbl_setbl!(state, dij, _fl32, _par32, _ante32;
                       topo=topo_iter, skip_mrchdu=!do_mrchdu)
        catch
            state.itran .= itran_save; state.uedg .= uedg_save; state.thet .= thet_save
            state.dstr .= dstr_save; state.ctau .= ctau_save; state.mass .= mass_save
            xbl_setbl!(state, dij, _fl32, _par32, _ante32;
                       topo=topo_iter, skip_mrchdu=true)
        end
        # Guard: if itran went past nbl (corrupt state), full restore.
        if any(state.itran[is] > state.nbl[is] for is in 1:2)
            state.itran .= itran_save; state.uedg .= uedg_save; state.thet .= thet_save
            state.dstr .= dstr_save; state.ctau .= ctau_save; state.mass .= mass_save
            va, vb, vm, vdel, vz = xbl_setbl!(state, dij, _fl32, _par32, _ante32;
                                               topo=topo_iter, skip_mrchdu=true)
        end
        ivte1 = state.isys[state.iblte[1], 1]
        nwake = state.nbl[2] - state.iblte[2]
        ivz_target = nwake > 0 ? state.isys[state.iblte[2] + 1, 2] : -1
        _viscal_blsolv!(va, vb, vm, vdel, vz, state.nsys, ivte1, ivz_target; sspan = sspan)
        vdel64 = Array{Float64,3}(undef, 3, 2, state.nsys)
        for iv in 1:state.nsys, k in 1:3
            v1 = Float64(vdel[k, 1, iv])
            v2 = Float64(vdel[k, 2, iv])
            vdel64[k, 1, iv] = isfinite(v1) ? v1 : 0.0
            vdel64[k, 2, iv] = isfinite(v2) ? v2 : 0.0
        end
        rmsbl, _ = _viscal_update!(state, vdel64, dij, inv, alpha_deg; flow = flow)
        _viscal_qvfue_gamqv_stmove!(state, inv, topo)

        transition_recheck_applied = transition_recheck && !mrchdu_frozen
        if transition_recheck
            # Keep the laminar/turbulent split synchronized only during the
            # startup phase while the transition indices are still settling.
            # Once MRCHDU startup is frozen, continuing to recheck transition
            # on every Newton step can re-perturb an otherwise consistent
            # branch and block final convergence.
            if transition_recheck_applied
                _viscal_apply_transition_recheck!(
                    state,
                    1;
                    flow = flow,
                    ncrit = ncrit,
                    mode = transition_recheck_upper_mode,
                )
                _viscal_apply_transition_recheck!(
                    state,
                    2;
                    flow = flow,
                    ncrit = ncrit,
                    mode = transition_recheck_lower_mode,
                )
            end

            transition_signature = (Int(state.itran[1]), Int(state.itran[2]))
            if transition_signature == prev_transition_signature
                transition_stable_iters += 1
            else
                transition_stable_iters = 0
            end
            if !mrchdu_frozen &&
               iter >= mrchdu_min_iters &&
               transition_stable_iters >= 3
                mrchdu_frozen = true
            end
            prev_transition_signature = transition_signature
        end

        capture_fallback_sample =
            capture_separated_fallback &&
            !transition_recheck_applied &&
            isfinite(rmsbl) &&
            rmsbl <= 0.25
        if capture_trace || capture_fallback_sample
            iteration_sample = _viscal_trace_sample(
                state,
                inv;
                flow = flow,
                ncrit = ncrit,
                residual = rmsbl,
                global_iteration = trace_offset + iter,
                segment = trace_segment,
                segment_iteration = iter,
                mrchdu_active = do_mrchdu,
                transition_recheck_active = transition_recheck_applied,
            )
            if capture_trace
                push!(trace_samples, iteration_sample)
            end
            if capture_fallback_sample
                fallback_score = _viscal_separated_fallback_score(
                    iteration_sample;
                    alpha_deg = alpha_deg,
                    branch_kind = separated_fallback_kind,
                )
                if fallback_score + 1.0e-12 < separated_fallback_score
                    separated_fallback_state = deepcopy(state)
                    separated_fallback_iteration = iter
                    separated_fallback_residual = float(rmsbl)
                    separated_fallback_score = fallback_score
                end
            end
        end

        niter_done = iter
        if isfinite(rmsbl) && rmsbl < best_residual
            best_residual = rmsbl
            best_iteration = iter
            best_state = deepcopy(state)
        end
        if rmsbl < tol; break; end
    end

    function build_viscal_result(
        result_state,
        result_converged::Bool,
        result_status::Symbol,
        result_selection::Symbol,
        result_selected_iteration::Integer,
        result_selected_residual::Real;
        result_trace = trace_samples,
    )
        qvis_final = copy(inv.qnode)
        for is in 1:2
            for ibl in 2:result_state.iblte[is]
                i = result_state.ipan[ibl, is]
                if i >= 1 && i <= length(qvis_final)
                    qvis_final[i] = result_state.vti[ibl, is] * result_state.uedg[ibl, is]
                end
            end
        end
        topo_final = _xfoil_bl_topology_f32(inv; q = qvis_final)
        upper_xtr = _viscal_transition_xsi(result_state, 1; flow = flow, ncrit = ncrit)
        lower_xtr = _viscal_transition_xsi(result_state, 2; flow = flow, ncrit = ncrit)

        cl_exact, cm_exact, _ = _clcalc_from_surface_speed(
            inv.xnode,
            inv.ynode,
            qvis_final,
            alpha_deg;
            xref = 0.25,
            yref = 0.0,
            qinf = Float64(flow.qinfbl),
            minf = 0.0,
        )
        cd_exact = _viscal_pressure_drag(result_state; flow = flow)
        top_xtr_exact = _viscal_transition_xc(
            result_state,
            inv,
            :upper;
            qvis = qvis_final,
            topology = topo_final,
            transition_x = upper_xtr,
        )
        bot_xtr_exact = _viscal_transition_xc(
            result_state,
            inv,
            :lower;
            qvis = qvis_final,
            topology = topo_final,
            transition_x = lower_xtr,
        )

        wake_final = _viscal_to_wake(result_state; flow = flow, model = model, wake_trace = wake_trace)

        # Convert state to SurfaceMarchResult (surface stations only).
        _viscal_refresh_laminar_ctau!(result_state, 1; flow = flow, ncrit = ncrit)
        _viscal_refresh_laminar_ctau!(result_state, 2; flow = flow, ncrit = ncrit)
        upper = _viscal_to_surface(
            result_state,
            1,
            inv;
            flow = flow,
            model = model,
            transition_x = upper_xtr,
        )
        lower = _viscal_to_surface(
            result_state,
            2,
            inv;
            flow = flow,
            model = model,
            transition_x = lower_xtr,
        )
        top_xtr_public = _prefer_public_transition_xc(top_xtr_exact, upper)
        bot_xtr_public = _prefer_public_transition_xc(bot_xtr_exact, lower)

        return (
            upper = upper,
            lower = lower,
            wake = wake_final,
            topology = topo_final,
            cl = cl_exact,
            cd = cd_exact,
            cm = cm_exact,
            top_xtr = top_xtr_public,
            bot_xtr = bot_xtr_public,
            iterations = niter_done,
            residual = float(result_selected_residual),
            converged = result_converged,
            status = result_status,
            selected_iteration = Int(result_selected_iteration),
            selection = result_selection,
            trace = result_trace,
        )
    end

    converged = rmsbl < tol
    selected_state = converged ? state : (best_iteration > 0 ? best_state : state)
    selected_iteration = converged ? niter_done : best_iteration
    selected_residual = converged ? rmsbl : (best_iteration > 0 ? best_residual : rmsbl)
    selection = converged ? :final_state : :best_available_nonconverged

    separated_fallback = isnothing(separated_fallback_state) ? nothing :
        build_viscal_result(
            separated_fallback_state,
            false,
            :max_iter,
            :separated_continuation_fallback,
            separated_fallback_iteration,
            separated_fallback_residual;
            result_trace = CoupledIterationSample[],
        )
    if !isnothing(separated_fallback_ref)
        separated_fallback_ref[] = separated_fallback
    end

    return build_viscal_result(
        selected_state,
        converged,
        converged ? :converged : :max_iter,
        selection,
        selected_iteration,
        selected_residual;
        result_trace = trace_samples,
    )
end

@inline function _viscal_apply_transition_recheck!(
    state::ViscalBLState,
    is::Int;
    flow::BLFlowParams,
    ncrit::Float64,
    mode::Symbol,
)
    if mode === :aggressive
        _viscal_transition_recheck!(state, is; flow = flow, ncrit = ncrit)
    elseif mode === :literal
        _viscal_trchek2!(state, is; flow = flow, ncrit = ncrit)
    else
        throw(ArgumentError("unsupported transition recheck mode $mode"))
    end
    return nothing
end

"""
    _viscal_picard_solve(inv, topo, dij, upper_march, lower_march; ...)

DIJ-coupled Picard iteration: replaces the heuristic blockage coupling with
proper XFoil-style UESET (Ue = UINV + Σ DIJ * MASS).  Uses the ViscalBLState
infrastructure for correct panel mapping but avoids the full Newton system
(BLSOLV), doing simple Picard iteration instead.

Each iteration:
1. UESET: compute mass-consistent Ue from DIJ coupling
2. Update UEDG with relaxation
3. Re-march surface BL (MRCHUE/MRCHDU) with new UEDG
4. Re-march wake BL
5. QVFUE + STMOVE: update stagnation and arc lengths
"""
function _viscal_picard_solve(
    inv::CaseResult,
    topo::BLTopology,
    dij::Matrix{Float64},
    upper_march::SurfaceMarchResult,
    lower_march::SurfaceMarchResult;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
    max_iter::Int = 30,
    tol::Float64 = 1e-4,
    relax::Float64 = 0.35,
    wake_trace = nothing,
)
    # Initialize state from march results (with wake if provided)
    state = _init_viscal_state(inv, topo, upper_march, lower_march;
        wake_trace = wake_trace)

    # Set inviscid velocities (UICALC)
    _viscal_uicalc!(state, inv)

    # For wake panels beyond N, set UINV from wake_trace qinv
    if !isnothing(wake_trace)
        nw = length(wake_trace.xw)
        for iw in 1:nw
            ibl = state.iblte[2] + iw
            if ibl <= state.nbl[2]
                state.uinv[ibl, 2] = max(abs(Float64(wake_trace.qinv[iw])), 1e-7)
            end
        end
    end

    nwake = state.nbl[2] - state.iblte[2]
    rmsbl = Inf
    niter_done = 0

    for iter in 1:max_iter
        # UESET: compute mass-consistent Ue from DIJ coupling
        usav = _viscal_ueset(state, dij)

        # Apply relaxed update with safeguards.
        # Ramp up relaxation gradually to avoid early divergence.
        rlx = iter <= 2 ? min(relax, 0.15) :
              iter <= 5 ? min(relax, 0.25) : relax

        for is in 1:2
            for ibl in 2:state.nbl[is]
                ue_old = state.uedg[ibl, is]
                ue_new = usav[ibl, is]

                # Protect stagnation-adjacent stations (ibl=2): mass coupling
                # produces huge corrections where Ue is small.  Blend toward
                # UINV instead of UESET for the first few surface stations.
                if ibl <= 3
                    ue_new = state.uinv[ibl, is]
                end

                # Clamp UESET result: don't let it go below a fraction of UINV
                ue_floor = max(abs(state.uinv[ibl, is]) * 0.3, 1e-5)
                ue_new = max(ue_new, ue_floor)

                due = ue_new - ue_old

                # Limit relative change per iteration (prevent runaway)
                max_rel_change = 0.3
                if abs(due) > max_rel_change * abs(ue_old)
                    due = copysign(max_rel_change * abs(ue_old), due)
                end

                state.uedg[ibl, is] = max(ue_old + rlx * due, ue_floor)
            end
        end

        # Re-march surfaces with updated UEDG
        try
            _viscal_remarch!(state, 1; flow = flow, model = model, ncrit = ncrit)
        catch; end
        try
            _viscal_remarch!(state, 2; flow = flow, model = model, ncrit = ncrit)
        catch; end

        # Re-march wake
        if nwake > 0
            try
                _viscal_remarch_wake!(state; flow = flow, model = model)
            catch; end
        end

        # Update mass defect arrays from new BL state
        for is in 1:2
            for ibl in 2:state.nbl[is]
                state.mass[ibl, is] = state.dstr[ibl, is] * state.uedg[ibl, is]
            end
        end

        # QVFUE + STMOVE: update stagnation point and arc lengths
        _viscal_qvfue_gamqv_stmove!(state, inv, topo)

        niter_done = iter

        # Convergence metric: measure RMS of ACTUAL applied change
        # (after relaxation + clamping + remarch).  Compare current UEDG
        # against the new UESET target to see how close we are to the
        # coupled fixed point.
        usav2 = _viscal_ueset(state, dij)
        rms_sum = 0.0
        ntot = 0
        for is in 1:2
            for ibl in 4:state.nbl[is]  # skip stagnation-adjacent
                rel = (usav2[ibl, is] - state.uedg[ibl, is]) / max(abs(state.uedg[ibl, is]), 0.01)
                rms_sum += rel^2
                ntot += 1
            end
        end
        rmsbl = sqrt(rms_sum / max(ntot, 1))

        # Check for NaN/Inf divergence
        if isnan(rmsbl) || isinf(rmsbl)
            break
        end

        rmsbl < tol && break
    end

    # Preserve transition from the exact remarched BL state instead of the
    # coarse first-turbulent-station fallback in _viscal_to_surface.
    upper_xtr = _viscal_transition_xsi(state, 1; flow = flow, ncrit = ncrit)
    lower_xtr = _viscal_transition_xsi(state, 2; flow = flow, ncrit = ncrit)
    upper = _viscal_to_surface(
        state,
        1,
        inv;
        flow = flow,
        model = model,
        transition_x = upper_xtr,
    )
    lower = _viscal_to_surface(
        state,
        2,
        inv;
        flow = flow,
        model = model,
        transition_x = lower_xtr,
    )

    return upper, lower, rmsbl, niter_done
end

"""
    _mrchue_initial_march(inv, topo; flow, model, ncrit)

Build a temporary ViscalBLState from the inviscid solution, run the literal
Fortran MRCHUE port (`xbl_mrchue!`) for both sides, and convert the result
back to `SurfaceMarchResult` objects.  This replaces the analytical
`_march_surface` for the initial BL establishment, eliminating the 1%
stagnation-Ue seed error that the analytical march introduces.
"""
function _mrchue_initial_march(
    inv::CaseResult,
    topo::BLTopology;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
)
    upper_topo = _bl_side(topo, :upper)
    lower_topo = _bl_side(topo, :lower)
    nu = length(upper_topo.ipan) - 1   # surface stations (excl stagnation IBL=1)
    nl = length(lower_topo.ipan) - 1

    # No wake in the initial MRCHUE march (wake is added later)
    iblte1 = nu + 1;  iblte2 = nl + 1
    nbl1 = iblte1;    nbl2 = iblte2
    max_ibl = _viscal_state_capacity(inv)

    z2(sz) = zeros(Float32, sz, 2)
    zi2(sz) = zeros(Int, sz, 2)

    state = ViscalBLState(
        [nbl1, nbl2],
        [iblte1, iblte2],
        [iblte1, iblte2],   # itran: default at TE (fully laminar)
        z2(max_ibl), z2(max_ibl), z2(max_ibl),
        z2(max_ibl), z2(max_ibl),
        z2(max_ibl), z2(max_ibl), z2(max_ibl),
        zi2(max_ibl), zi2(max_ibl),
        0,
        Float64(Float32(inv.cl)),
        Float32[],
    )

    # Fill upper surface (IS=1): IBL 2..iblte1
    for k in 1:nu
        ibl = k + 1
        state.xssi[ibl, 1] = Float32(upper_topo.xssi[ibl])
        uei = Float32(abs(upper_topo.vti[ibl] * inv.qnode[upper_topo.ipan[ibl]]))
        state.uedg[ibl, 1] = max(uei, Float32(1e-4))
        state.vti[ibl, 1] = Float32(upper_topo.vti[ibl])
        state.ipan[ibl, 1] = upper_topo.ipan[ibl]
        state.uinv[ibl, 1] = state.uedg[ibl, 1]
    end
    # Fill lower surface (IS=2): IBL 2..iblte2
    for k in 1:nl
        ibl = k + 1
        state.xssi[ibl, 2] = Float32(lower_topo.xssi[ibl])
        uei = Float32(abs(lower_topo.vti[ibl] * inv.qnode[lower_topo.ipan[ibl]]))
        state.uedg[ibl, 2] = max(uei, Float32(1e-4))
        state.vti[ibl, 2] = Float32(lower_topo.vti[ibl])
        state.ipan[ibl, 2] = lower_topo.ipan[ibl]
        state.uinv[ibl, 2] = state.uedg[ibl, 2]
    end

    # System line numbering (for _viscal_to_surface compatibility)
    iv = 0
    for is in 1:2, ibl in 2:state.nbl[is]
        iv += 1; state.isys[ibl, is] = iv
    end
    state.nsys = iv

    # Build Float32 flow parameters (same as in _viscal_solve)
    _fl32 = BLFlowF32(
        XFT(0), XFT(flow.qinfbl),
        XFT(0), XFT(0),
        XFT(flow.rstbl), XFT(flow.rstbl_ms),
        XFT(flow.hstinv), XFT(flow.hstinv_ms),
        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
        XFT(1), XFT(1e6), XFT(ncrit), Int32(0))
    _par32 = BLParF32()
    _ante32 = XFT(_te_gap_geometry(inv).ante)

    # Run MRCHUE for side 1 (upper), then side 2 (lower)
    # Side 1 must complete first because TESYS at wake junction needs both sides.
    # (No wake here, but TE station values are still stored for reference.)
    xbl_mrchue!(state, 1, _fl32, _par32, _ante32)
    xbl_mrchue!(state, 2, _fl32, _par32, _ante32)

    # Reconstruct transition from the raw startup BL state instead of using
    # the coarse first-turbulent-station fallback in _viscal_to_surface.
    # This preserves the physically implied transition point without copying
    # XFoil's occasionally stale startup XTR bookkeeping.
    upper_xtr = _viscal_transition_xsi(state, 1; flow = flow, ncrit = ncrit)
    lower_xtr = _viscal_transition_xsi(state, 2; flow = flow, ncrit = ncrit)

    # Convert to SurfaceMarchResult
    upper = _viscal_to_surface(
        state,
        1,
        inv;
        flow = flow,
        model = model,
        transition_x = upper_xtr,
    )
    lower = _viscal_to_surface(
        state,
        2,
        inv;
        flow = flow,
        model = model,
        transition_x = lower_xtr,
    )

    return upper, lower
end

"""
    _mrchue_initial_march_with_wake(inv, topo; flow, model, ncrit)

Initialize the viscous state the same way XFoil does when `LBLINI` is false:
first establish both surface boundary layers with `MRCHUE`, then march the
lower side again with the wake present so the startup wake is formed by the
same direct-mode equations instead of the generic analytical wake marcher.
"""
function _mrchue_initial_march_with_wake(
    inv::CaseResult,
    topo::BLTopology;
    flow::BLFlowParams,
    model::BLModelParams,
    ncrit::Float64,
)
    upper_surface, lower_surface = _mrchue_initial_march(
        inv, topo; flow = flow, model = model, ncrit = ncrit)

    wake_trace = _wake_station_trace(inv)
    wake_seed = _march_wake(
        inv,
        upper_surface,
        lower_surface;
        flow = flow,
        model = model,
        seed = nothing,
        tol = 1.0e-6,
        max_iter = 20,
        ueps = 1.0e-7,
        mrchue_local = true,
    )

    state = _init_viscal_state(
        inv,
        topo,
        upper_surface,
        lower_surface;
        wake_trace = wake_trace,
        wake_march = nothing,
    )

    _viscal_uicalc!(state, inv)

    for iw in 1:length(wake_trace.xw)
        ibl = state.iblte[2] + iw
        if ibl <= state.nbl[2]
            state.uinv[ibl, 2] = max(abs(Float64(wake_trace.qinv[iw])), 1e-7)
        end
    end

    _fl32 = BLFlowF32(
        XFT(0), XFT(flow.qinfbl),
        XFT(0), XFT(0),
        XFT(flow.rstbl), XFT(flow.rstbl_ms),
        XFT(flow.hstinv), XFT(flow.hstinv_ms),
        XFT(flow.reybl), XFT(flow.reybl_ms), XFT(flow.reybl_re),
        XFT(1.4), XFT(flow.gm1bl), XFT(flow.hvrat),
        XFT(1), XFT(1e6), XFT(ncrit), Int32(0))
    _par32 = BLParF32()
    _ante32 = XFT(_te_gap_geometry(inv).ante)

    # The wake belongs to the lower-side BL chain in XFoil's storage/layout,
    # so a direct lower-side march establishes the startup wake consistently
    # without perturbing the already-matched upper surface.
    thet_before = copy(state.thet[:, 2])
    dstr_before = copy(state.dstr[:, 2])
    ctau_before = copy(state.ctau[:, 2])
    uedg_before = copy(state.uedg[:, 2])
    xbl_mrchue!(state, 2, _fl32, _par32, _ante32)

    # Always recover the direct-marched wake seed.  We only trust it when the
    # lower-side direct remarch stays broadly consistent with the already-
    # established surface state; otherwise the remarched wake can be
    # inconsistent with the untouched surface seed and send the coupled solve
    # onto a disconnected startup branch.
    wake_direct = _viscal_to_wake(state; flow = flow, model = model, wake_trace = wake_trace)

    # Accept the full direct wake refinement only if it leaves the already-
    # established lower surface essentially unchanged.  When it perturbs the
    # surface too much, keep the original surface seed paired with the
    # consistent analytical wake seed.
    maxsurf = 0.0
    for ibl in 2:state.iblte[2]
        maxsurf = max(
            maxsurf,
            abs(Float64(state.thet[ibl, 2] - thet_before[ibl])),
            abs(Float64(state.dstr[ibl, 2] - dstr_before[ibl])),
            abs(Float64(state.ctau[ibl, 2] - ctau_before[ibl])),
            abs(Float64(state.uedg[ibl, 2] - uedg_before[ibl])),
        )
    end
    if maxsurf > 1.0e-5
        return upper_surface, lower_surface, wake_seed, wake_trace
    end

    upper_xtr = _viscal_transition_xsi(state, 1; flow = flow, ncrit = ncrit)
    lower_xtr = _viscal_transition_xsi(state, 2; flow = flow, ncrit = ncrit)
    upper = _viscal_to_surface(
        state,
        1,
        inv;
        flow = flow,
        model = model,
        transition_x = upper_xtr,
    )
    lower = _viscal_to_surface(
        state,
        2,
        inv;
        flow = flow,
        model = model,
        transition_x = lower_xtr,
    )

    return upper, lower, wake_direct, wake_trace
end

function _solve_coupled_case_viscal_impl(
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    continuation_seed::Union{Nothing,CoupledCaseResult} = nothing,
    capture_trace::Bool = false,
    allow_plateau_public_convergence::Bool = true,
    plateau_verify::Bool = true,
    converged_branch_verify::Bool = true,
)
    re = float(reynolds)
    nc = float(ncrit)
    trace_samples = CoupledIterationSample[]
    trace_offset = Ref(0)

    function record_viscal_trace!(candidate)
        if capture_trace
            append!(trace_samples, candidate.trace)
            trace_offset[] += length(candidate.trace)
        end
        return candidate
    end

    function replay_exact_branch(seed_result::CoupledCaseResult; trace::Bool)
        return _solve_coupled_case_viscal_impl(
            airfoil,
            alpha_deg;
            reynolds = re,
            ncrit = nc,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            viscal_max_iter = min(Int(viscal_max_iter), 40),
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            transition_recheck = transition_recheck,
            restart_on_nonconverged = restart_on_nonconverged,
            continuation_seed = seed_result,
            capture_trace = trace,
            allow_plateau_public_convergence = false,
            plateau_verify = false,
            converged_branch_verify = false,
        )
    end

    inv = solve_inviscid_case(
        airfoil,
        alpha_deg;
        repanel = repanel,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
    )

    flowp = isnothing(flow) ? _xfoil_bl_flow_params(re; model = model) : flow

    topo = _xfoil_bl_topology_f32(inv)

    # Initial BL/wake march using the literal Fortran MRCHUE port.
    # This replaces the analytical wake seed with the same direct-mode
    # lower-side+wake march used by XFoil's startup path.
    upper_init, lower_init, wake_init, wake_trace = _mrchue_initial_march_with_wake(
        inv, topo; flow = flowp, model = model, ncrit = nc)
    seed_case = _validated_continuation_seed(
        continuation_seed,
        SurfaceStations(:upper, upper_init.x, upper_init.xc, upper_init.ue),
        SurfaceStations(:lower, lower_init.x, lower_init.xc, lower_init.ue),
    )
    if isnothing(seed_case)
        startup_upper = upper_init
        startup_lower = lower_init
    else
        seed_upper, seed_lower = _continuation_seed_pair(
            seed_case,
            upper_init,
            lower_init;
            flow = flowp,
            model = model,
        )
        startup_upper = _march_surface(
            _surface_stations(seed_upper);
            flow = flowp,
            model = model,
            ncrit = nc,
            tol = float(tol),
            max_iter = Int(max_iter),
            seed = seed_upper,
            preserve_seed_transition = true,
            solve_similarity_station = true,
        )
        startup_lower = _march_surface(
            _surface_stations(seed_lower);
            flow = flowp,
            model = model,
            ncrit = nc,
            tol = float(tol),
            max_iter = Int(max_iter),
            seed = seed_lower,
            preserve_seed_transition = true,
            solve_similarity_station = true,
        )
    end
    inv_cl = float(inv.cl)
    te_gap_ante = _te_gap_geometry(inv).ante

    # Compute full (N+NW)x(N+NW) DIJ influence matrix
    dij = _qdcalc_full(inv, wake_trace)

    # Run the primary exact Newton solve on the direct MRCHUE/MRCHDU branch.
    # Transition recheck remains available as a rescue path after a clean exact
    # solve fails to converge, but we do not let it perturb an otherwise
    # convergent exact branch.
    fresh_start = nothing
    aggressive_startup = nothing
    viscal = record_viscal_trace!(_viscal_solve(
        inv, topo, dij, startup_upper, startup_lower;
        flow = flowp, model = model, ncrit = nc,
        max_iter = Int(viscal_max_iter), tol = Float64(viscal_tol),
        wake_trace = wake_trace, wake_march = wake_init,
        transition_recheck = false,
        capture_trace = capture_trace,
        trace_segment = isnothing(seed_case) ? :direct_exact : :direct_exact_seeded,
        trace_offset = trace_offset[],
    ))
    total_viscal_iterations = viscal.iterations
    restart_notes = String[]
    if !isnothing(seed_case)
        _append_restart_note!(restart_notes, "startup_seed=continuation_surface")
    end

    @inline function restart_candidate_seed_result(candidate_viscal)
        return CoupledCaseResult(
            float(alpha_deg),
            re,
            nc,
            candidate_viscal.cl,
            candidate_viscal.cd,
            candidate_viscal.cm,
            candidate_viscal.top_xtr,
            candidate_viscal.bot_xtr,
            0,
            float(candidate_viscal.residual),
            candidate_viscal.upper,
            candidate_viscal.lower,
            inv,
            _solver_provenance(
                backend = :exact_viscal,
                selection = candidate_viscal.selection,
                converged = candidate_viscal.converged,
                status = candidate_viscal.status,
                selected_iteration = candidate_viscal.selected_iteration,
                selected_residual = candidate_viscal.residual,
            ),
        )
    end

    @inline function restart_startup_family_label(candidate_viscal)
        candidate_result = restart_candidate_seed_result(candidate_viscal)

        if !isnothing(aggressive_startup) &&
           _startup_family_output_matches(
               candidate_result,
               restart_candidate_seed_result(aggressive_startup),
           )
            return :aggressive_startup
        end

        if !isnothing(fresh_start) &&
           _startup_family_output_matches(
               candidate_result,
               restart_candidate_seed_result(fresh_start),
           )
            return :literal_fresh_start
        end

        return nothing
    end

    @inline function startup_family_allows_restart_replacement(candidate_viscal, incumbent_viscal)
        protected_label = restart_startup_family_label(incumbent_viscal)
        isnothing(protected_label) && return true
        return _startup_family_output_matches(
            restart_candidate_seed_result(incumbent_viscal),
            restart_candidate_seed_result(candidate_viscal),
        )
    end

    if !isnothing(seed_case) && !_viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
        fresh_start = record_viscal_trace!(_viscal_solve(
            inv, topo, dij, upper_init, lower_init;
            flow = flowp, model = model, ncrit = nc,
            max_iter = Int(viscal_max_iter), tol = Float64(viscal_tol),
            wake_trace = wake_trace, wake_march = wake_init,
            transition_recheck = false,
            capture_trace = capture_trace,
            trace_segment = :direct_exact_fresh_start,
            trace_offset = trace_offset[],
        ))
        total_viscal_iterations += fresh_start.iterations

        if _prefer_viscal_candidate(fresh_start, viscal, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
            _append_restart_note!(
                restart_notes,
                "startup_seed=fallback_to_literal_mrchue " *
                "seed_residual=$(round(viscal.residual; digits = 6)) " *
                "fresh_residual=$(round(fresh_start.residual; digits = 6))",
            )
            viscal = fresh_start
        end
    end

    if transition_recheck &&
       (!viscal.converged ||
        _should_probe_aggressive_transition_startup_branch(viscal, inv.cl))
        aggressive_startup = record_viscal_trace!(_viscal_solve(
            inv, topo, dij, upper_init, lower_init;
            flow = flowp, model = model, ncrit = nc,
            max_iter = Int(viscal_max_iter), tol = Float64(viscal_tol),
            wake_trace = wake_trace, wake_march = wake_init,
            transition_recheck = true,
            transition_recheck_upper_mode = :aggressive,
            transition_recheck_lower_mode = :aggressive,
            capture_trace = capture_trace,
            trace_segment = :startup_transition_recheck_aggressive,
            trace_offset = trace_offset[],
        ))
        total_viscal_iterations += aggressive_startup.iterations

        aggressive_branch_preferred = _prefer_aggressive_transition_startup_branch(
            aggressive_startup,
            viscal,
            inv.cl,
        )
        aggressive_candidate_preferred =
            _prefer_viscal_candidate(aggressive_startup, viscal, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
        _append_restart_note!(
            restart_notes,
            _startup_transition_recheck_note(
                viscal,
                aggressive_startup,
                aggressive_branch_preferred,
                aggressive_candidate_preferred,
                te_gap_ante = te_gap_ante,
            ),
        )

        if aggressive_branch_preferred && aggressive_candidate_preferred
            viscal = aggressive_startup
        end
    end

    if restart_on_nonconverged &&
       transition_recheck &&
       !_viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
        initial_residual = viscal.residual
        exact_restart_passes = 0
        if !viscal.converged
            restart_upper = viscal.upper
            restart_lower = viscal.lower

            for pass in 1:3
                restart_selected = try
                    record_viscal_trace!(_viscal_solve(
                        inv, topo, dij, restart_upper, restart_lower;
                        flow = flowp, model = model, ncrit = nc,
                        max_iter = Int(viscal_max_iter), tol = Float64(viscal_tol),
                        wake_trace = wake_trace, wake_march = wake_init,
                        transition_recheck = transition_recheck,
                        capture_trace = capture_trace,
                        trace_segment = Symbol("restart_exact_$(pass)"),
                        trace_offset = trace_offset[],
                    ))
                catch
                    nothing
                end
                isnothing(restart_selected) && break

                total_viscal_iterations += restart_selected.iterations
                exact_restart_passes = pass

                if _prefer_viscal_candidate(restart_selected, viscal, inv_cl, alpha_deg; te_gap_ante = te_gap_ante) &&
                   startup_family_allows_restart_replacement(restart_selected, viscal)
                    _append_restart_note!(
                        restart_notes,
                        "restart_seed=exact_selected_surface " *
                        "passes=$(pass) " *
                        "initial_residual=$(round(initial_residual; digits = 6)) " *
                        "restart_residual=$(round(restart_selected.residual; digits = 6))",
                    )
                    viscal = restart_selected
                end

                if _viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
                    break
                end

                restart_upper = restart_selected.upper
                restart_lower = restart_selected.lower
            end
        end

        if !_viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
            literal_seed_residual = viscal.residual
            literal_restart_upper = upper_init
            literal_restart_lower = lower_init

            for pass in 1:2
                restart_literal = record_viscal_trace!(_viscal_solve(
                    inv, topo, dij, literal_restart_upper, literal_restart_lower;
                    flow = flowp, model = model, ncrit = nc,
                    max_iter = Int(viscal_max_iter), tol = Float64(viscal_tol),
                    wake_trace = wake_trace, wake_march = wake_init,
                    transition_recheck = transition_recheck,
                    transition_recheck_upper_mode = :literal,
                    transition_recheck_lower_mode = :literal,
                    capture_trace = capture_trace,
                    trace_segment = Symbol("restart_literal_$(pass)"),
                    trace_offset = trace_offset[],
                ))
                total_viscal_iterations += restart_literal.iterations

                if _prefer_viscal_candidate(restart_literal, viscal, inv_cl, alpha_deg; te_gap_ante = te_gap_ante) &&
                   startup_family_allows_restart_replacement(restart_literal, viscal)
                    _append_restart_note!(
                        restart_notes,
                        "restart_seed=exact_literal_initial_surface " *
                        "exact_passes=$(exact_restart_passes) " *
                        "passes=$(pass) " *
                        "initial_residual=$(round(initial_residual; digits = 6)) " *
                        "literal_seed_residual=$(round(literal_seed_residual; digits = 6)) " *
                        "restart_residual=$(round(restart_literal.residual; digits = 6))",
                    )
                    viscal = restart_literal
                end

                if _viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
                    break
                end

                literal_restart_upper = restart_literal.upper
                literal_restart_lower = restart_literal.lower
            end
        end

        if !viscal.converged
            continuation_restart = _viscal_alpha_continuation_rescue(
                airfoil,
                alpha_deg,
                viscal,
                model,
                flowp;
                reynolds = re,
                ncrit = nc,
                repanel = repanel,
                max_iter = max_iter,
                tol = tol,
                viscal_max_iter = viscal_max_iter,
                viscal_tol = viscal_tol,
                xfoil_load_normalize_digits = xfoil_load_normalize_digits,
                transition_recheck = transition_recheck,
                capture_trace = capture_trace,
                trace_offset = trace_offset[],
            )
            total_viscal_iterations += continuation_restart.iterations
            if capture_trace
                append!(trace_samples, continuation_restart.trace)
                trace_offset[] += length(continuation_restart.trace)
            end

            if _prefer_viscal_candidate(continuation_restart.candidate, viscal, inv_cl, alpha_deg; te_gap_ante = te_gap_ante) &&
               startup_family_allows_restart_replacement(continuation_restart.candidate, viscal)
                _append_restart_note!(
                    restart_notes,
                    continuation_restart.note *
                    " exact_passes=$(exact_restart_passes)",
                )
                viscal = continuation_restart.candidate
            end
        end
    end

    @inline function candidate_seed_result(candidate_viscal)
        return CoupledCaseResult(
            float(alpha_deg),
            re,
            nc,
            candidate_viscal.cl,
            candidate_viscal.cd,
            candidate_viscal.cm,
            candidate_viscal.top_xtr,
            candidate_viscal.bot_xtr,
            0,
            float(candidate_viscal.residual),
            candidate_viscal.upper,
            candidate_viscal.lower,
            inv,
            _solver_provenance(
                backend = :exact_viscal,
                selection = candidate_viscal.selection,
                converged = candidate_viscal.converged,
                status = candidate_viscal.status,
                selected_iteration = candidate_viscal.selected_iteration,
                selected_residual = candidate_viscal.residual,
            ),
        )
    end

    @inline function corroborated_branch_label(candidate_viscal)
        candidate_result = candidate_seed_result(candidate_viscal)

        if !isnothing(aggressive_startup) &&
           _startup_family_output_matches(
               candidate_result,
               candidate_seed_result(aggressive_startup),
           )
            return :aggressive_startup
        end

        if !isnothing(fresh_start) &&
           _startup_family_output_matches(
               candidate_result,
               candidate_seed_result(fresh_start),
           )
            return :literal_fresh_start
        end

        return nothing
    end

    @inline function startup_family_allows_replacement(candidate_viscal, incumbent_viscal)
        protected_label = corroborated_branch_label(incumbent_viscal)
        isnothing(protected_label) && return true
        return _startup_family_output_matches(
            candidate_seed_result(incumbent_viscal),
            candidate_seed_result(candidate_viscal),
        )
    end

    function direct_seed_exact_branch(
        seed_result::CoupledCaseResult;
        trace::Bool,
        trace_segment::Symbol,
    )
        seed_upper, seed_lower = _continuation_seed_pair(
            seed_result,
            upper_init,
            lower_init;
            flow = flowp,
            model = model,
        )
        return _viscal_solve(
            inv,
            topo,
            dij,
            seed_upper,
            seed_lower;
            flow = flowp,
            model = model,
            ncrit = nc,
            max_iter = max(Int(viscal_max_iter), 80),
            tol = Float64(viscal_tol),
            wake_trace = wake_trace,
            wake_march = wake_init,
            transition_recheck = transition_recheck,
            transition_recheck_upper_mode = :literal,
            transition_recheck_lower_mode = :literal,
            capture_trace = trace,
            trace_segment = trace_segment,
            trace_offset = trace ? trace_offset[] : 0,
        )
    end

    @inline function _meaningfully_prefer_viscal_candidate(
        candidate,
        incumbent,
        inv_cl::Real,
        alpha_deg::Real;
        te_gap_ante::Real = NaN,
        merit_tol::Real = 1.0e-6,
        residual_factor::Real = 0.5,
    )
        cand_ok = _viscal_candidate_acceptable(candidate, inv_cl, alpha_deg)
        inc_ok = _viscal_candidate_acceptable(incumbent, inv_cl, alpha_deg)
        cand_ok == inc_ok || return cand_ok

        cand_merit = _viscal_candidate_merit(candidate, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
        inc_merit = _viscal_candidate_merit(incumbent, inv_cl, alpha_deg; te_gap_ante = te_gap_ante)
        cand_merit + float(merit_tol) < inc_merit && return true
        inc_merit + float(merit_tol) < cand_merit && return false

        cand_res = float(candidate.residual)
        inc_res = float(incumbent.residual)
        isfinite(cand_res) && isfinite(inc_res) || return false
        return cand_res <= float(residual_factor) * inc_res
    end

    function verify_converged_branch(
        candidate_viscal;
        selected_via_continuation_audit::Bool = false,
        protected_branch_label::Union{Nothing,Symbol} = nothing,
    )
        candidate_note = ""
        candidate_verified = true
        candidate_acceptable = _viscal_candidate_acceptable(candidate_viscal, inv_cl, alpha_deg)
        if !(converged_branch_verify && candidate_viscal.converged && candidate_acceptable)
            return (
                viscal = candidate_viscal,
                acceptable = candidate_acceptable,
                verified = candidate_verified,
                note = candidate_note,
            )
        end

        provisional_result = candidate_seed_result(candidate_viscal)
        verification_result, _, verification_error = try
            result, trace = replay_exact_branch(provisional_result; trace = false)
            result, trace, nothing
        catch err
            nothing, CoupledIterationSample[], replace(sprint(showerror, err), '\n' => ' ')
        end
        if !isnothing(verification_error)
            candidate_note =
                "converged_verify=error " *
                "verify_error=$(verification_error)"
            return (
                viscal = candidate_viscal,
                acceptable = candidate_acceptable,
                verified = candidate_verified,
                note = candidate_note,
            )
        end

        verification_mismatch = _result_output_mismatch(provisional_result, verification_result)
        verification_consistent = _plateau_verification_consistent(provisional_result, verification_result)
        if verification_consistent
            candidate_note =
                "converged_verify=accepted " *
                "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6))"
            return (
                viscal = candidate_viscal,
                acceptable = candidate_acceptable,
                verified = candidate_verified,
                note = candidate_note,
            )
        end

        verification_result2, _, verification_result2_error = try
            result, trace = replay_exact_branch(verification_result; trace = false)
            result, trace, nothing
        catch err
            nothing, CoupledIterationSample[], replace(sprint(showerror, err), '\n' => ' ')
        end
        if !isnothing(verification_result2_error)
            candidate_note =
                "converged_verify=error " *
                "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                "stable_error=$(verification_result2_error)"
            return (
                viscal = candidate_viscal,
                acceptable = candidate_acceptable,
                verified = candidate_verified,
                note = candidate_note,
            )
        end

        verification_stable = _plateau_verification_consistent(verification_result, verification_result2)
        stability_mismatch = _result_output_mismatch(verification_result, verification_result2)
        verification_candidate = _result_replay_candidate(verification_result)
        verification_candidate_acceptable =
            _viscal_candidate_acceptable(verification_candidate, inv_cl, alpha_deg)
        verification_candidate2 = _result_replay_candidate(verification_result2)
        verification_candidate2_acceptable =
            _viscal_candidate_acceptable(verification_candidate2, inv_cl, alpha_deg)
        residual_protected_candidate =
            _viscal_low_residual_branch_protected(
                candidate_viscal,
                inv_cl,
                alpha_deg,
                viscal_tol,
            )
        orbit_kind = _replay_orbit_kind(
            provisional_result,
            verification_result,
            verification_result2,
        )
        verification_result3 = nothing
        verification_result3_error = nothing
        if orbit_kind == :unresolved && !verification_stable
            verification_result3, _, verification_result3_error = try
                result, trace = replay_exact_branch(verification_result2; trace = false)
                result, trace, nothing
            catch err
                nothing, CoupledIterationSample[], replace(sprint(showerror, err), '\n' => ' ')
            end
            if isnothing(verification_result3_error)
                orbit_kind = _replay_orbit_kind(
                    provisional_result,
                    verification_result,
                    verification_result2;
                    verification3 = verification_result3,
                )
            end
        end

        if orbit_kind == :fixed_point && verification_candidate_acceptable
            verification_meaningfully_preferred =
                _meaningfully_prefer_viscal_candidate(
                    verification_candidate,
                    candidate_viscal,
                    inv_cl,
                    alpha_deg;
                    te_gap_ante = te_gap_ante,
                )
            preserve_fixed_point =
                !isnothing(protected_branch_label) ||
                (selected_via_continuation_audit && !verification_meaningfully_preferred) ||
                !verification_meaningfully_preferred
            if preserve_fixed_point
                preserved_tag =
                    !isnothing(protected_branch_label) ?
                    "corroborated_preserved" :
                    (selected_via_continuation_audit ?
                        "continuation_preserved" :
                        "orbit_fixed_point_preserved")
                protected_note =
                    isnothing(protected_branch_label) ?
                    "" :
                    " protected_by=$(protected_branch_label)"
                candidate_note =
                    "converged_verify=$(preserved_tag) " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "orbit_kind=$(orbit_kind) " *
                    "verify_converged=$(verification_result.provenance.converged) " *
                    "stable_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                    "stable_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                    "stable_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                    "stable_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                    "stable_dxtr=$(round(stability_mismatch.dxtr; digits = 6)) " *
                    "verify_meaningful=$(verification_meaningfully_preferred)" *
                    protected_note
            else
                candidate_viscal = verification_candidate
                candidate_acceptable = true
                candidate_verified = true
                candidate_note =
                    "converged_verify=$(verification_result.provenance.converged ? "replaced" : "orbit_fixed_point") " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "orbit_kind=$(orbit_kind) " *
                    "verify_converged=$(verification_result.provenance.converged) " *
                    "stable_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                    "stable_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                    "stable_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                    "stable_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                    "stable_dxtr=$(round(stability_mismatch.dxtr; digits = 6))"
            end
        elseif orbit_kind == :contracting &&
               verification_candidate_acceptable &&
               verification_candidate2_acceptable &&
               _converged_replay_contracts(
                   provisional_result,
                   verification_result,
                   verification_result2,
               )
            candidate_viscal = verification_candidate
            candidate_acceptable = true
            candidate_verified = true
            candidate_note =
                "converged_verify=contracted_replay " *
                "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                "orbit_kind=$(orbit_kind) " *
                "stable_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                "stable_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                "stable_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                "stable_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                "stable_dxtr=$(round(stability_mismatch.dxtr; digits = 6))"
        elseif orbit_kind == :two_cycle
            cycle_mismatch = isnothing(verification_result3) ?
                (dcl = NaN, dcd = NaN, dcm = NaN, dxtr = NaN) :
                _result_output_mismatch(verification_result, verification_result3)
            verification_severe = _converged_verification_severe(provisional_result, verification_result)
            if verification_severe
                rescue_best = nothing
                rescue_followup = nothing
                rescue_followup_scale = Inf
                rescue_seed_label = ""
                rescue_seed_residual = NaN

                for (label, seed_result) in (
                    ("verify", verification_result),
                    ("next", verification_result2),
                    ("reference", provisional_result),
                )
                    direct_candidate = try
                        direct_seed_exact_branch(
                            seed_result;
                            trace = capture_trace,
                            trace_segment = Symbol("orbit_direct_$(label)"),
                        )
                    catch
                        nothing
                    end
                    isnothing(direct_candidate) && continue

                    total_viscal_iterations += direct_candidate.iterations
                    if capture_trace
                        append!(trace_samples, direct_candidate.trace)
                        trace_offset[] += length(direct_candidate.trace)
                    end

                    _viscal_candidate_acceptable(direct_candidate, inv_cl, alpha_deg) || continue
                    direct_candidate.converged || continue

                    direct_seed = candidate_seed_result(direct_candidate)
                    direct_followup_candidate = try
                        direct_seed_exact_branch(
                            direct_seed;
                            trace = capture_trace,
                            trace_segment = Symbol("orbit_direct_followup_$(label)"),
                        )
                    catch
                        nothing
                    end
                    isnothing(direct_followup_candidate) && continue

                    total_viscal_iterations += direct_followup_candidate.iterations
                    if capture_trace
                        append!(trace_samples, direct_followup_candidate.trace)
                        trace_offset[] += length(direct_followup_candidate.trace)
                    end

                    _viscal_candidate_acceptable(direct_followup_candidate, inv_cl, alpha_deg) || continue
                    direct_followup_candidate.converged || continue
                    _plateau_verification_consistent(direct_candidate, direct_followup_candidate) || continue
                    direct_followup_mismatch = _result_output_mismatch(direct_candidate, direct_followup_candidate)
                    direct_followup_scale = _result_mismatch_scale(direct_followup_mismatch)

                    if isnothing(rescue_best) ||
                       direct_followup_scale + 1.0e-12 < rescue_followup_scale ||
                       (
                           abs(direct_followup_scale - rescue_followup_scale) <= 1.0e-12 &&
                           _prefer_viscal_candidate(
                               direct_candidate,
                               rescue_best,
                               inv_cl,
                               alpha_deg;
                               te_gap_ante = te_gap_ante,
                           )
                       )
                        rescue_best = direct_candidate
                        rescue_followup = direct_followup_candidate
                        rescue_followup_scale = direct_followup_scale
                        rescue_seed_label = label
                        rescue_seed_residual = _result_selected_residual(seed_result)
                    end
                end

                if !isnothing(rescue_best)
                    rescue_mismatch = _result_output_mismatch(provisional_result, rescue_best)
                    rescue_followup_mismatch = _result_output_mismatch(rescue_best, rescue_followup)
                    candidate_viscal = rescue_best
                    candidate_acceptable = true
                    candidate_verified = true
                    candidate_note =
                        "converged_verify=orbit_direct_rescue " *
                        "orbit_kind=$(orbit_kind) " *
                        "rescue_seed=$(rescue_seed_label) " *
                        "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                        "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                        "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                        "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                        "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                        "rescue_seed_residual=$(round(rescue_seed_residual; digits = 6)) " *
                        "rescue_residual=$(round(rescue_best.residual; digits = 6)) " *
                        "rescue_dcl=$(round(rescue_mismatch.dcl; digits = 6)) " *
                        "rescue_dcd=$(round(rescue_mismatch.dcd; digits = 6)) " *
                        "rescue_dcm=$(round(rescue_mismatch.dcm; digits = 6)) " *
                        "rescue_dxtr=$(round(rescue_mismatch.dxtr; digits = 6)) " *
                        "stable_residual=$(round(rescue_followup.residual; digits = 6)) " *
                        "stable_dcl=$(round(rescue_followup_mismatch.dcl; digits = 6)) " *
                        "stable_dcd=$(round(rescue_followup_mismatch.dcd; digits = 6)) " *
                        "stable_dcm=$(round(rescue_followup_mismatch.dcm; digits = 6)) " *
                        "stable_dxtr=$(round(rescue_followup_mismatch.dxtr; digits = 6)) " *
                        "stable_scale=$(round(rescue_followup_scale; digits = 6))"
                elseif residual_protected_candidate
                    candidate_verified = true
                    candidate_note =
                        "converged_verify=low_residual_preserved " *
                        "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                        "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                        "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                        "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                        "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                        "orbit_kind=$(orbit_kind) " *
                        "candidate_residual=$(round(candidate_viscal.residual; digits = 6)) " *
                        "candidate_sanity=$(round(_viscal_candidate_sanity_score(candidate_viscal, inv_cl, alpha_deg); digits = 6)) " *
                        "next_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                        "next_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                        "next_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                        "next_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                        "next_dxtr=$(round(stability_mismatch.dxtr; digits = 6)) " *
                        "verify_severe=true"
                else
                    candidate_verified = false
                    candidate_note =
                        "converged_verify=orbit_two_cycle " *
                        "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                        "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                        "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                        "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                        "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                        "next_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                        "next_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                        "next_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                        "next_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                        "next_dxtr=$(round(stability_mismatch.dxtr; digits = 6)) " *
                        "cycle_converged=$(isnothing(verification_result3) ? false : verification_result3.provenance.converged) " *
                        "cycle_residual=$(round(isnothing(verification_result3) ? NaN : _result_selected_residual(verification_result3); digits = 6)) " *
                        "cycle_dcl=$(round(cycle_mismatch.dcl; digits = 6)) " *
                        "cycle_dcd=$(round(cycle_mismatch.dcd; digits = 6)) " *
                        "cycle_dcm=$(round(cycle_mismatch.dcm; digits = 6)) " *
                        "cycle_dxtr=$(round(cycle_mismatch.dxtr; digits = 6)) " *
                        "verify_severe=true"
                end
            else
                candidate_verified = true
                candidate_note =
                    "converged_verify=orbit_two_cycle_warning " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "next_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                    "next_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                    "next_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                    "next_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                    "next_dxtr=$(round(stability_mismatch.dxtr; digits = 6)) " *
                    "cycle_converged=$(isnothing(verification_result3) ? false : verification_result3.provenance.converged) " *
                    "cycle_residual=$(round(isnothing(verification_result3) ? NaN : _result_selected_residual(verification_result3); digits = 6)) " *
                    "cycle_dcl=$(round(cycle_mismatch.dcl; digits = 6)) " *
                    "cycle_dcd=$(round(cycle_mismatch.dcd; digits = 6)) " *
                    "cycle_dcm=$(round(cycle_mismatch.dcm; digits = 6)) " *
                    "cycle_dxtr=$(round(cycle_mismatch.dxtr; digits = 6)) " *
                    "verify_severe=false"
            end
        else
            verification_severe = _converged_verification_severe(provisional_result, verification_result)
            rescue_best = nothing
            rescue_followup = nothing
            rescue_followup_scale = Inf
            rescue_seed_label = ""
            rescue_seed_residual = NaN
            if verification_severe
                for (label, seed_result) in (
                    ("verify", verification_result),
                    ("next", verification_result2),
                    ("reference", provisional_result),
                )
                    direct_candidate = try
                        direct_seed_exact_branch(
                            seed_result;
                            trace = capture_trace,
                            trace_segment = Symbol("orbit_direct_rescue_$(label)"),
                        )
                    catch
                        nothing
                    end
                    isnothing(direct_candidate) && continue

                    total_viscal_iterations += direct_candidate.iterations
                    if capture_trace
                        append!(trace_samples, direct_candidate.trace)
                        trace_offset[] += length(direct_candidate.trace)
                    end

                    _viscal_candidate_acceptable(direct_candidate, inv_cl, alpha_deg) || continue
                    direct_candidate.converged || continue

                    direct_seed = candidate_seed_result(direct_candidate)
                    direct_followup_candidate = try
                        direct_seed_exact_branch(
                            direct_seed;
                            trace = capture_trace,
                            trace_segment = Symbol("orbit_direct_rescue_followup_$(label)"),
                        )
                    catch
                        nothing
                    end
                    isnothing(direct_followup_candidate) && continue

                    total_viscal_iterations += direct_followup_candidate.iterations
                    if capture_trace
                        append!(trace_samples, direct_followup_candidate.trace)
                        trace_offset[] += length(direct_followup_candidate.trace)
                    end

                    _viscal_candidate_acceptable(direct_followup_candidate, inv_cl, alpha_deg) || continue
                    direct_followup_candidate.converged || continue
                    _plateau_verification_consistent(direct_candidate, direct_followup_candidate) || continue
                    direct_followup_mismatch = _result_output_mismatch(direct_candidate, direct_followup_candidate)
                    direct_followup_scale = _result_mismatch_scale(direct_followup_mismatch)

                    if isnothing(rescue_best) ||
                       direct_followup_scale + 1.0e-12 < rescue_followup_scale ||
                       (
                           abs(direct_followup_scale - rescue_followup_scale) <= 1.0e-12 &&
                           _prefer_viscal_candidate(
                               direct_candidate,
                               rescue_best,
                               inv_cl,
                               alpha_deg;
                               te_gap_ante = te_gap_ante,
                           )
                       )
                        rescue_best = direct_candidate
                        rescue_followup = direct_followup_candidate
                        rescue_followup_scale = direct_followup_scale
                        rescue_seed_label = label
                        rescue_seed_residual = _result_selected_residual(seed_result)
                    end
                end
            end

            if !isnothing(rescue_best) &&
               _prefer_viscal_candidate(
                   rescue_best,
                   candidate_viscal,
                   inv_cl,
                   alpha_deg;
                   te_gap_ante = te_gap_ante,
                )
                rescue_mismatch = _result_output_mismatch(provisional_result, rescue_best)
                rescue_followup_mismatch = _result_output_mismatch(rescue_best, rescue_followup)
                candidate_viscal = rescue_best
                candidate_acceptable = true
                candidate_verified = true
                candidate_note =
                    "converged_verify=orbit_direct_rescue " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "orbit_kind=$(orbit_kind) " *
                    "verify_converged=$(verification_result.provenance.converged) " *
                    "verify_severe=$(verification_severe) " *
                    "verify_acceptable=$(verification_candidate_acceptable) " *
                    "rescue_seed=$(rescue_seed_label) " *
                    "rescue_seed_residual=$(round(rescue_seed_residual; digits = 6)) " *
                    "rescue_residual=$(round(rescue_best.residual; digits = 6)) " *
                    "rescue_dcl=$(round(rescue_mismatch.dcl; digits = 6)) " *
                    "rescue_dcd=$(round(rescue_mismatch.dcd; digits = 6)) " *
                    "rescue_dcm=$(round(rescue_mismatch.dcm; digits = 6)) " *
                    "rescue_dxtr=$(round(rescue_mismatch.dxtr; digits = 6)) " *
                    "stable_residual=$(round(rescue_followup.residual; digits = 6)) " *
                    "stable_dcl=$(round(rescue_followup_mismatch.dcl; digits = 6)) " *
                    "stable_dcd=$(round(rescue_followup_mismatch.dcd; digits = 6)) " *
                    "stable_dcm=$(round(rescue_followup_mismatch.dcm; digits = 6)) " *
                    "stable_dxtr=$(round(rescue_followup_mismatch.dxtr; digits = 6)) " *
                    "stable_scale=$(round(rescue_followup_scale; digits = 6))"
            elseif verification_severe && residual_protected_candidate
                candidate_verified = true
                candidate_note =
                    "converged_verify=low_residual_preserved " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "orbit_kind=$(orbit_kind) " *
                    "verify_converged=$(verification_result.provenance.converged) " *
                    "verify_acceptable=$(verification_candidate_acceptable) " *
                    "candidate_residual=$(round(candidate_viscal.residual; digits = 6)) " *
                    "candidate_sanity=$(round(_viscal_candidate_sanity_score(candidate_viscal, inv_cl, alpha_deg); digits = 6)) " *
                    "stable=$(verification_stable) " *
                    "stable_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                    "stable_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                    "stable_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                    "stable_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                    "stable_dxtr=$(round(stability_mismatch.dxtr; digits = 6))"
            else
                candidate_verified = !verification_severe
                candidate_note =
                    "converged_verify=$(!verification_severe ? "warning" : "rejected") " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "orbit_kind=$(orbit_kind) " *
                    "verify_converged=$(verification_result.provenance.converged) " *
                    "stable=$(verification_stable) " *
                    "stable_residual=$(round(_result_selected_residual(verification_result2); digits = 6)) " *
                    "stable_dcl=$(round(stability_mismatch.dcl; digits = 6)) " *
                    "stable_dcd=$(round(stability_mismatch.dcd; digits = 6)) " *
                    "stable_dcm=$(round(stability_mismatch.dcm; digits = 6)) " *
                    "stable_dxtr=$(round(stability_mismatch.dxtr; digits = 6)) " *
                    (!isnothing(verification_result3_error) ?
                        "orbit_error=$(verification_result3_error) " :
                        "") *
                    "verify_severe=$(verification_severe) " *
                    "verify_acceptable=$(verification_candidate_acceptable)"
            end
        end

        return (
            viscal = candidate_viscal,
            acceptable = candidate_acceptable,
            verified = candidate_verified,
            note = candidate_note,
        )
    end

    converged_verification_note = ""
    converged_branch_verified = true
    viscal_acceptable = _viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
    startup_protected_label = corroborated_branch_label(viscal)
    if !isnothing(startup_protected_label)
        _append_restart_note!(restart_notes, "startup_family=$(startup_protected_label)")
    end
    verification = verify_converged_branch(
        viscal;
        protected_branch_label = startup_protected_label,
    )
    viscal = verification.viscal
    viscal_acceptable = verification.acceptable
    converged_branch_verified = verification.verified
    converged_verification_note = verification.note
    if converged_branch_verify &&
       transition_recheck &&
       viscal.converged &&
       viscal_acceptable
        continuation_audit = _viscal_alpha_continuation_rescue(
            airfoil,
            alpha_deg,
            viscal,
            model,
            flowp;
            reynolds = re,
            ncrit = nc,
            repanel = repanel,
            max_iter = max_iter,
            tol = tol,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            transition_recheck = transition_recheck,
            capture_trace = capture_trace,
            trace_offset = trace_offset[],
        )
        total_viscal_iterations += continuation_audit.iterations
        if capture_trace
            append!(trace_samples, continuation_audit.trace)
            trace_offset[] += length(continuation_audit.trace)
        end
        protected_startup_label = corroborated_branch_label(viscal)
        continuation_candidate_preferred =
            continuation_audit.accepted &&
            _prefer_viscal_candidate(
                continuation_audit.candidate,
                viscal,
                inv_cl,
                alpha_deg;
                te_gap_ante = te_gap_ante,
            )
        continuation_high_lift_fallback_preferred =
            continuation_audit.accepted &&
            _viscal_separated_fallback_preferred(
                continuation_audit.candidate,
                viscal;
                alpha_deg = alpha_deg,
                branch_kind = _viscal_separated_fallback_probe_kind(viscal),
                allow_near_zero = false,
            )
        continuation_near_zero_fallback_preferred =
            continuation_audit.accepted &&
            !converged_branch_verified &&
            _viscal_separated_fallback_preferred(
                continuation_audit.candidate,
                viscal;
                alpha_deg = alpha_deg,
                branch_kind = _viscal_separated_fallback_probe_kind(viscal),
                allow_high_lift = false,
            )
        continuation_separated_fallback_preferred =
            continuation_high_lift_fallback_preferred ||
            continuation_near_zero_fallback_preferred
        continuation_preferred =
            (continuation_candidate_preferred || continuation_separated_fallback_preferred) &&
            isnothing(protected_startup_label)
        if !isempty(continuation_audit.note)
            _append_restart_note!(
                restart_notes,
                continuation_audit.note *
                " converged_audit=$(continuation_preferred ? "selected" : "rejected")" *
                (continuation_separated_fallback_preferred ?
                    " audit_reason=separated_fallback" :
                    "") *
                (continuation_candidate_preferred && !isnothing(protected_startup_label) ?
                    " protected_by=$(protected_startup_label)" :
                    ""),
            )
        end
        if continuation_preferred
            viscal = continuation_audit.candidate
            viscal_acceptable = _viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
            verification = verify_converged_branch(
                viscal;
                selected_via_continuation_audit = true,
                protected_branch_label = corroborated_branch_label(viscal),
            )
            viscal = verification.viscal
            viscal_acceptable = verification.acceptable
            converged_branch_verified = verification.verified
            converged_verification_note = verification.note
        end
    end

    upper = viscal.upper
    lower = viscal.lower
    cl = viscal.cl
    cm = viscal.cm
    cd = viscal.cd
    top_xtr = viscal.top_xtr
    bot_xtr = viscal.bot_xtr
    sanity_score = _viscal_candidate_sanity_score(viscal, inv_cl, alpha_deg)
    viscal_acceptable = _viscal_candidate_acceptable(viscal, inv_cl, alpha_deg)
    viscal_publicly_converged = viscal.converged && viscal_acceptable && converged_branch_verified
    plateau_converged = false
    plateau_verification_note = ""
    if allow_plateau_public_convergence &&
       !viscal_acceptable &&
       _viscal_candidate_plateau_converged(viscal, inv_cl, alpha_deg)
        provisional_provenance = _solver_provenance(
            backend = :exact_viscal,
            selection = viscal.selection,
            converged = false,
            status = viscal.status,
            iterations = total_viscal_iterations,
            selected_iteration = viscal.selected_iteration,
            selected_residual = viscal.residual,
        )
        provisional_result = CoupledCaseResult(
            float(alpha_deg),
            re,
            nc,
            cl,
            cd,
            cm,
            top_xtr,
            bot_xtr,
            0,
            float(viscal.residual),
            upper,
            lower,
            inv,
            provisional_provenance,
        )

        if plateau_verify
            verification_result, verification_trace, verification_error = try
                result, trace = replay_exact_branch(
                    provisional_result;
                    trace = true,
                )
                result, trace, nothing
            catch err
                nothing, CoupledIterationSample[], replace(sprint(showerror, err), '\n' => ' ')
            end
            if !isnothing(verification_error)
                plateau_converged = false
                plateau_verification_note =
                    "plateau_verify=error " *
                    "verify_error=$(verification_error)"
            else
                verification_mismatch = _result_output_mismatch(provisional_result, verification_result)
                verification_consistent = _plateau_verification_consistent(provisional_result, verification_result)
                verification_cycle = _plateau_trace_cycle_detected(verification_trace, provisional_result)
                plateau_converged = verification_consistent && !verification_cycle
                plateau_verification_note =
                    "plateau_verify=$(plateau_converged ? "accepted" : "rejected") " *
                    "verify_residual=$(round(_result_selected_residual(verification_result); digits = 6)) " *
                    "verify_dcl=$(round(verification_mismatch.dcl; digits = 6)) " *
                    "verify_dcd=$(round(verification_mismatch.dcd; digits = 6)) " *
                    "verify_dcm=$(round(verification_mismatch.dcm; digits = 6)) " *
                    "verify_dxtr=$(round(verification_mismatch.dxtr; digits = 6)) " *
                    "verify_cycle=$(verification_cycle)"
            end
        else
            plateau_converged = true
        end
    end
    provenance_status =
        viscal.converged && viscal_acceptable && !converged_branch_verified ? :unverified_branch :
        viscal.converged && !viscal_acceptable ? :implausible_state :
        (plateau_converged ? :converged : viscal.status)
    provenance_selection =
        viscal.converged && viscal_acceptable && !converged_branch_verified ? :rejected_unverified_branch :
        viscal.converged && !viscal_acceptable ? :rejected_implausible_branch :
        (plateau_converged ? :accepted_plateau_state : viscal.selection)
    provenance_converged = viscal_publicly_converged || plateau_converged
    provenance_message = _restart_notes_message(restart_notes)
    if !isempty(converged_verification_note)
        provenance_message =
            isempty(provenance_message) ?
            converged_verification_note :
            provenance_message * " " * converged_verification_note
    end
    if !isempty(plateau_verification_note)
        provenance_message =
            isempty(provenance_message) ? plateau_verification_note : provenance_message * " " * plateau_verification_note
    end
    if plateau_converged
        plateau_note =
            "public_convergence=plateau " *
            "plateau_residual=$(round(viscal.residual; digits = 6)) " *
            "sanity_score=$(round(sanity_score; digits = 3))"
        provenance_message =
            isempty(provenance_message) ? plateau_note : provenance_message * " " * plateau_note
    end
    if !viscal_acceptable
        sanity_note = "sanity_score=$(round(sanity_score; digits = 3))"
        provenance_message = isempty(provenance_message) ? sanity_note : provenance_message * " " * sanity_note
    end
    if !viscal_acceptable && sanity_score > 5.0
        cl = NaN
        cm = NaN
        cd = NaN
        top_xtr = NaN
        bot_xtr = NaN
        redact_note = "public_outputs=redacted_implausible"
        provenance_message = isempty(provenance_message) ? redact_note : provenance_message * " " * redact_note
    end
    provenance = _solver_provenance(
        backend = :exact_viscal,
        selection = provenance_selection,
        converged = provenance_converged,
        status = provenance_status,
        failure_message = provenance_message,
        iterations = total_viscal_iterations,
        selected_iteration = viscal.selected_iteration,
        selected_residual = viscal.residual,
    )

    result = CoupledCaseResult(
        float(alpha_deg), re, nc, cl, cd, cm,
        top_xtr, bot_xtr, total_viscal_iterations, viscal.residual,
        upper, lower, inv,
        provenance,
    )
    return result, trace_samples
end

"""
Coupled case solver using XFoil-exact VISCAL Newton loop.
"""
function solve_coupled_case_viscal(
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    continuation_seed::Union{Nothing,CoupledCaseResult} = nothing,
)
    result, _ = _solve_coupled_case_viscal_impl(
        airfoil,
        alpha_deg;
        reynolds = reynolds,
        ncrit = ncrit,
        repanel = repanel,
        model = model,
        flow = flow,
        tol = tol,
        max_iter = max_iter,
        viscal_max_iter = viscal_max_iter,
        viscal_tol = viscal_tol,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
        transition_recheck = transition_recheck,
        restart_on_nonconverged = restart_on_nonconverged,
        continuation_seed = continuation_seed,
        capture_trace = false,
    )
    return result
end

function solve_coupled_case_viscal_trace(
    airfoil::Airfoil,
    alpha_deg::Real;
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    continuation_seed::Union{Nothing,CoupledCaseResult} = nothing,
)
    result, samples = _solve_coupled_case_viscal_impl(
        airfoil,
        alpha_deg;
        reynolds = reynolds,
        ncrit = ncrit,
        repanel = repanel,
        model = model,
        flow = flow,
        tol = tol,
        max_iter = max_iter,
        viscal_max_iter = viscal_max_iter,
        viscal_tol = viscal_tol,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
        transition_recheck = transition_recheck,
        restart_on_nonconverged = restart_on_nonconverged,
        continuation_seed = continuation_seed,
        capture_trace = true,
    )
    return CoupledSolveTrace(
        airfoil.name,
        float(alpha_deg),
        float(reynolds),
        float(ncrit),
        result,
        samples,
    )
end

"""
    solve_coupled_polar_viscal(airfoil, alphas; kwargs...) -> Vector{CoupledCaseResult}

Solve a viscous polar with alpha continuation. The sweep anchors near
`sweep_anchor` (default: 0 deg), marches to higher alpha, then marches to
lower alpha while reusing the last physically plausible converged solution as
the continuation seed.
"""
function solve_coupled_polar_viscal(
    airfoil::Airfoil,
    alphas::AbstractVector{<:Real};
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    model::BLModelParams = BLModelParams(),
    flow::Union{Nothing,BLFlowParams} = nothing,
    tol::Real = 1.0e-6,
    max_iter::Integer = 20,
    viscal_max_iter::Integer = 80,
    viscal_tol::Real = 1.0e-4,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    transition_recheck::Bool = true,
    restart_on_nonconverged::Bool = true,
    sweep_anchor::Union{Nothing,Real} = nothing,
    continuation_step_limit::Real = 2.0,
)
    alpha_list = sort!(unique(Float64.(alphas)))
    isempty(alpha_list) && return CoupledCaseResult[]

    anchor = isnothing(sweep_anchor) ? 0.0 : float(sweep_anchor)
    anchor_idx = argmin(abs.(alpha_list .- anchor))
    results = Dict{Float64,CoupledCaseResult}()

    function solve_one(alpha::Float64, seed_case::Union{Nothing,CoupledCaseResult})
        return solve_coupled_case_viscal(
            airfoil,
            alpha;
            reynolds = reynolds,
            ncrit = ncrit,
            repanel = repanel,
            model = model,
            flow = flow,
            tol = tol,
            max_iter = max_iter,
            viscal_max_iter = viscal_max_iter,
            viscal_tol = viscal_tol,
            xfoil_load_normalize_digits = xfoil_load_normalize_digits,
            transition_recheck = transition_recheck,
            restart_on_nonconverged = restart_on_nonconverged,
            continuation_seed = seed_case,
        )
    end

    function march_indices!(indices, seed_case::Union{Nothing,CoupledCaseResult})
        current_seed = seed_case
        current_seed_alpha = isnothing(seed_case) ? NaN : float(seed_case.alpha_deg)

        for idx in indices
            alpha = alpha_list[idx]
            use_seed =
                !isnothing(current_seed) &&
                isfinite(current_seed_alpha) &&
                abs(alpha - current_seed_alpha) <= float(continuation_step_limit) + 1.0e-12 ?
                current_seed : nothing

            result = solve_one(alpha, use_seed)
            results[alpha] = result

            if _plausible_converged_retry_candidate(result)
                current_seed = result
                current_seed_alpha = alpha
            end
        end
    end

    anchor_alpha = alpha_list[anchor_idx]
    anchor_result = solve_one(anchor_alpha, nothing)
    results[anchor_alpha] = anchor_result
    anchor_seed = _plausible_converged_retry_candidate(anchor_result) ? anchor_result : nothing

    if anchor_idx < length(alpha_list)
        march_indices!((anchor_idx + 1):length(alpha_list), anchor_seed)
    end
    if anchor_idx > 1
        march_indices!((anchor_idx - 1):-1:1, anchor_seed)
    end

    return [results[alpha] for alpha in alpha_list]
end

"""
    _blkin_full(u, t, d, flow; precision) -> NamedTuple

Standalone BLKIN: computes kinematic state + all derivatives for one station.
Identical to the inner `_blkin` in `_bldif_analytic` but callable externally.
Used by `_viscal_mrchdu!` to compute COM state for passing to SETBL.
"""
function _blkin_full(u::Real, t::Real, d::Real, flow::BLFlowParams;
                     precision::Type{FT} = Float32) where {FT}
    eps = FT(1e-12)
    u2f = FT(u); t2f = max(FT(t), eps); d2f = max(FT(d), eps)

    denom_m = FT(flow.gm1bl) * (FT(1) - FT(0.5) * u2f^2 * FT(flow.hstinv))
    denom_m = abs(denom_m) < eps ? sign(denom_m + eps) * eps : denom_m
    m2    = u2f^2 * FT(flow.hstinv) / denom_m
    tr2   = FT(1) + FT(0.5) * FT(flow.gm1bl) * m2
    m2_u2 = FT(2) * m2 * tr2 / u2f
    m2_ms = u2f^2 * tr2 / denom_m * FT(flow.hstinv_ms)

    r2    = FT(flow.rstbl) * tr2^(FT(-1) / FT(flow.gm1bl))
    r2_u2 = -r2 / tr2 * FT(0.5) * m2_u2
    r2_ms = -r2 / tr2 * FT(0.5) * m2_ms + FT(flow.rstbl_ms) * tr2^(FT(-1) / FT(flow.gm1bl))

    h2    = d2f / t2f
    h2_d2 = FT(1) / t2f
    h2_t2 = -h2 / t2f

    herat  = FT(1) - FT(0.5) * u2f^2 * FT(flow.hstinv)
    herat  = max(herat, eps)
    he_u2  = -u2f * FT(flow.hstinv)
    he_ms  = -FT(0.5) * u2f^2 * FT(flow.hstinv_ms)

    v2    = sqrt(herat^3) * (FT(1) + FT(flow.hvrat)) / (herat + FT(flow.hvrat)) / FT(flow.reybl)
    v2    = max(v2, FT(1e-16))
    v2_he = v2 * (FT(1.5) / herat - FT(1) / (herat + FT(flow.hvrat)))
    v2_u2 = v2_he * he_u2
    v2_ms = -v2 / FT(flow.reybl) * FT(flow.reybl_ms) + v2_he * he_ms
    v2_re = -v2 / FT(flow.reybl) * FT(flow.reybl_re)

    # Analytical HKIN Jacobian (matching Fortran HKIN output)
    hk2, hk2_h2, hk2_m2 = hkin_jac(Float64(h2), Float64(m2))
    hk2_u2 = hk2_m2 * m2_u2
    hk2_t2 = hk2_h2 * h2_t2
    hk2_d2 = hk2_h2 * h2_d2
    hk2_ms = hk2_m2 * m2_ms

    rt2    = r2 * u2f * t2f / v2
    rt2_u2 = rt2 * (FT(1) / u2f + r2_u2 / r2 - v2_u2 / v2)
    rt2_t2 = rt2 / t2f
    rt2_ms = rt2 * (r2_ms / r2 - v2_ms / v2)
    rt2_re = rt2 * (-v2_re / v2)

    return (;
        u2 = u2f, t2 = t2f, d2 = d2f,
        m2, m2_u2, m2_ms,
        r2, r2_u2, r2_ms,
        h2, h2_d2, h2_t2,
        hk2, hk2_u2, hk2_t2, hk2_d2, hk2_ms,
        rt2, rt2_u2, rt2_t2, rt2_ms, rt2_re,
        v2, v2_u2, v2_ms, v2_re,
        herat,
    )
end

# ────────────────────────────────────────────────────────────────────────────
#  _bldif_analytic  –  BLKIN + BLVAR + BLMID + BLDIF with full analytic
#  Jacobians, direct port of XFoil Fortran.
#
#  Returns (vs1, vs2, vsrez, vsm, vsr)
#    vs1[3,5]   – sensitivities of 3 residuals wrt station-1 vars [S/A,T,D,U,X]
#    vs2[3,5]   – sensitivities of 3 residuals wrt station-2 vars [S/A,T,D,U,X]
#    vsrez[3]   – negative residuals  (VSREZ = -residual, XFoil sign convention)
#    vsm[3]     – Mach-number (M²) sensitivities
#    vsr[3]     – Reynolds-number sensitivities
# ────────────────────────────────────────────────────────────────────────────

function _bldif_analytic(
    ityp::Symbol,
    p1::BLPrimaryState,
    p2::BLPrimaryState;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Real = 9.0f0,
    idampv::Int    = 0,
    precision::Type{FT} = Float32,
    kin1_pre = nothing,  # pre-computed station-1 kinematic state from MRCHDU
) where {FT<:AbstractFloat}
    _ityp_symbol(ityp)  # validate

    _FT = FT  # local alias for precision type
    eps = _FT(1e-12)
    # Closure derivatives now use analytical Jacobians from viscous_closure.jl

    # Convert primary states to target precision
    p1 = BLPrimaryState(_FT(p1.x), _FT(p1.u), _FT(p1.t), _FT(p1.d);
        s=_FT(p1.s), ampl=_FT(p1.ampl), dw=_FT(p1.dw))
    p2 = BLPrimaryState(_FT(p2.x), _FT(p2.u), _FT(p2.t), _FT(p2.d);
        s=_FT(p2.s), ampl=_FT(p2.ampl), dw=_FT(p2.dw))

    SIMI = (ityp == :similarity)

    # ── Analytical Jacobians for closure functions ──────────────────────
    # Use exact analytical derivatives from viscous_closure.jl, matching
    # XFoil's BLVAR/BLKIN chain.  Arguments are promoted to Float64 for
    # full derivative accuracy.
    _hkin_jac(h, msq) = hkin_jac(Float64(h), Float64(msq))
    _hct_jac(hk, msq) = hct_jac(Float64(hk), Float64(msq))
    _hsl_jac(hk, rt, msq) = hsl_jac(Float64(hk), Float64(rt), Float64(msq))
    _hst_jac(hk, rt, msq) = hst_jac(Float64(hk), Float64(rt), Float64(msq))
    _cfl_jac(hk, rt, msq) = cfl_jac(Float64(hk), Float64(rt), Float64(msq))
    _cft_jac(hk, rt, msq) = cft_jac(Float64(hk), Float64(rt), Float64(msq);
        cffac=Float64(model.cffac), gamma=Float64(model.gamma))
    _dil_jac(hk, rt) = dil_jac(Float64(hk), Float64(rt))
    _dilw_jac(hk, rt) = dilw_jac(Float64(hk), Float64(rt))
    _dampl_jac(hk, th, rt) = (idampv == 0 ? dampl_jac : dampl2_jac)(Float64(hk), Float64(th), Float64(rt))

    # ── BLKIN for a single station ──────────────────────────────────────
    # Returns all kinematic quantities + their sensitivities wrt (U,T,D,MS,RE)
    function _blkin(u2, t2, d2)
        u2f = _FT(u2)
        t2f = max(_FT(t2), eps)
        d2f = max(_FT(d2), eps)

        # Mach²
        denom_m = _FT(flow.gm1bl) * (_FT(1) - _FT(0.5) * u2f^2 * _FT(flow.hstinv))
        denom_m = abs(denom_m) < eps ? sign(denom_m + eps) * eps : denom_m
        m2    = u2f^2 * _FT(flow.hstinv) / denom_m
        tr2   = _FT(1) + _FT(0.5) * _FT(flow.gm1bl) * m2
        m2_u2 = _FT(2) * m2 * tr2 / u2f
        m2_ms = u2f^2 * tr2 / denom_m * _FT(flow.hstinv_ms)

        # density
        r2    = _FT(flow.rstbl) * tr2^(_FT(-1) / _FT(flow.gm1bl))
        r2_u2 = -r2 / tr2 * _FT(0.5) * m2_u2
        r2_ms = -r2 / tr2 * _FT(0.5) * m2_ms + _FT(flow.rstbl_ms) * tr2^(_FT(-1) / _FT(flow.gm1bl))

        # shape parameter
        h2    = d2f / t2f
        h2_d2 = _FT(1) / t2f
        h2_t2 = -h2 / t2f

        # static/stagnation enthalpy ratio
        herat  = _FT(1) - _FT(0.5) * u2f^2 * _FT(flow.hstinv)
        herat  = max(herat, eps)
        he_u2  = -u2f * _FT(flow.hstinv)
        he_ms  = -_FT(0.5) * u2f^2 * _FT(flow.hstinv_ms)

        # molecular viscosity
        v2    = sqrt(herat^3) * (_FT(1) + _FT(flow.hvrat)) / (herat + _FT(flow.hvrat)) / _FT(flow.reybl)
        v2    = max(v2, _FT(1e-16))
        v2_he = v2 * (_FT(1.5) / herat - _FT(1) / (herat + _FT(flow.hvrat)))
        v2_u2 = v2_he * he_u2
        v2_ms = -v2 / _FT(flow.reybl) * _FT(flow.reybl_ms) + v2_he * he_ms
        v2_re = -v2 / _FT(flow.reybl) * _FT(flow.reybl_re)

        # kinematic shape parameter
        hk2, hk2_h2, hk2_m2 = _hkin_jac(h2, m2)
        hk2_u2 = hk2_m2 * m2_u2
        hk2_t2 = hk2_h2 * h2_t2
        hk2_d2 = hk2_h2 * h2_d2
        hk2_ms = hk2_m2 * m2_ms

        # momentum-thickness Reynolds number
        rt2    = r2 * u2f * t2f / v2
        rt2_u2 = rt2 * (_FT(1) / u2f + r2_u2 / r2 - v2_u2 / v2)
        rt2_t2 = rt2 / t2f
        rt2_ms = rt2 * (r2_ms / r2 - v2_ms / v2)
        rt2_re = rt2 * (-v2_re / v2)

        return (;
            u2 = u2f, t2 = t2f, d2 = d2f,
            m2, m2_u2, m2_ms,
            r2, r2_u2, r2_ms,
            h2, h2_d2, h2_t2,
            hk2, hk2_u2, hk2_t2, hk2_d2, hk2_ms,
            rt2, rt2_u2, rt2_t2, rt2_ms, rt2_re,
            v2, v2_u2, v2_ms, v2_re,
            herat,
        )
    end

    # ── BLVAR for a single station ──────────────────────────────────────
    # Returns closure quantities + full Jacobians wrt (U,T,D,S,MS,RE)
    function _blvar(kin, ityp_loc, s_val)
        hk2 = kin.hk2
        # Clamp hk
        hk_clamped = false
        if ityp_loc == :wake
            if hk2 < _FT(1.00005)
                hk2 = _FT(1.00005); hk_clamped = true
            end
        else
            if hk2 < _FT(1.05)
                hk2 = _FT(1.05); hk_clamped = true
            end
        end
        hk2_u2 = hk_clamped ? _FT(0) : kin.hk2_u2
        hk2_t2 = hk_clamped ? _FT(0) : kin.hk2_t2
        hk2_d2 = hk_clamped ? _FT(0) : kin.hk2_d2
        hk2_ms = hk_clamped ? _FT(0) : kin.hk2_ms

        # HC2 – density thickness
        hc2, hc2_hk2, hc2_m2 = _hct_jac(hk2, kin.m2)
        hc2_u2 = hc2_hk2 * hk2_u2 + hc2_m2 * kin.m2_u2
        hc2_t2 = hc2_hk2 * hk2_t2
        hc2_d2 = hc2_hk2 * hk2_d2
        hc2_ms = hc2_hk2 * hk2_ms + hc2_m2 * kin.m2_ms

        # HS2 – energy thickness shape parameter
        if ityp_loc == :laminar || ityp_loc == :similarity
            hs2, hs2_hk2, hs2_rt2, hs2_m2 = _hsl_jac(hk2, kin.rt2, kin.m2)
        else
            hs2, hs2_hk2, hs2_rt2, hs2_m2 = _hst_jac(hk2, kin.rt2, kin.m2)
        end
        hs2_u2 = hs2_hk2 * hk2_u2 + hs2_rt2 * kin.rt2_u2 + hs2_m2 * kin.m2_u2
        hs2_t2 = hs2_hk2 * hk2_t2 + hs2_rt2 * kin.rt2_t2
        hs2_d2 = hs2_hk2 * hk2_d2
        hs2_ms = hs2_hk2 * hk2_ms + hs2_rt2 * kin.rt2_ms + hs2_m2 * kin.m2_ms
        hs2_re = hs2_rt2 * kin.rt2_re

        # US2 – slip velocity
        h2 = max(kin.h2, eps)
        us2     = _FT(0.5) * hs2 * (_FT(1) - (hk2 - _FT(1)) / (_FT(model.gbcon) * h2))
        us2_hs2 = _FT(0.5)       * (_FT(1) - (hk2 - _FT(1)) / (_FT(model.gbcon) * h2))
        us2_hk2 = _FT(0.5) * hs2 * (-_FT(1) / (_FT(model.gbcon) * h2))
        us2_h2  = _FT(0.5) * hs2 * (hk2 - _FT(1)) / (_FT(model.gbcon) * h2^2)

        us2_u2 = us2_hs2 * hs2_u2 + us2_hk2 * hk2_u2
        us2_t2 = us2_hs2 * hs2_t2 + us2_hk2 * hk2_t2 + us2_h2 * kin.h2_t2
        us2_d2 = us2_hs2 * hs2_d2 + us2_hk2 * hk2_d2 + us2_h2 * kin.h2_d2
        us2_ms = us2_hs2 * hs2_ms + us2_hk2 * hk2_ms
        us2_re = us2_hs2 * hs2_re

        # Clamp US2
        if ityp_loc != :wake && us2 > _FT(0.95)
            us2 = _FT(0.98)
            us2_u2 = _FT(0); us2_t2 = _FT(0); us2_d2 = _FT(0); us2_ms = _FT(0); us2_re = _FT(0)
        end
        if ityp_loc == :wake && us2 > _FT(0.99995)
            us2 = _FT(0.99995)
            us2_u2 = _FT(0); us2_t2 = _FT(0); us2_d2 = _FT(0); us2_ms = _FT(0); us2_re = _FT(0)
        end

        # CQ2 – equilibrium shear stress
        gcc = _FT(0)
        hkc     = hk2 - _FT(1)
        hkc_hk2 = _FT(1)
        hkc_rt2 = _FT(0)
        rt2_safe = max(kin.rt2, eps)
        if ityp_loc == :turbulent
            gcc = _FT(model.gccon)
            hkc     = hk2 - _FT(1) - gcc / rt2_safe
            hkc_hk2 = _FT(1)
            hkc_rt2 = gcc / rt2_safe^2
            if hkc < _FT(0.01)
                hkc = _FT(0.01); hkc_hk2 = _FT(0); hkc_rt2 = _FT(0)
            end
        end
        hkb = hk2 - _FT(1)
        usb = max(_FT(1) - us2, _FT(1e-8))
        cqarg = _FT(model.ctcon) * hs2 * hkb * hkc^2 / (usb * h2 * hk2^2)
        if cqarg <= _FT(0)
            cq2 = eps  # avoid division by zero below
            cq2_hs2 = _FT(0); cq2_us2 = _FT(0); cq2_hk2 = _FT(0); cq2_rt2 = _FT(0); cq2_h2 = _FT(0)
        else
            cq2 = sqrt(cqarg)
            half_cq2 = _FT(0.5) / cq2
            cq2_hs2 = _FT(model.ctcon) * hkb * hkc^2 / (usb * h2 * hk2^2) * half_cq2
            cq2_us2 = _FT(model.ctcon) * hs2 * hkb * hkc^2 / (usb * h2 * hk2^2) / usb * half_cq2
            cq2_hk2 = (_FT(model.ctcon) * hs2 * hkc^2 / (usb * h2 * hk2^2) -
                        _FT(model.ctcon) * hs2 * hkb * hkc^2 / (usb * h2 * hk2^3) * _FT(2) +
                        _FT(model.ctcon) * hs2 * hkb * hkc / (usb * h2 * hk2^2) * _FT(2) * hkc_hk2) * half_cq2
            cq2_rt2 = _FT(model.ctcon) * hs2 * hkb * hkc / (usb * h2 * hk2^2) * _FT(2) * hkc_rt2 * half_cq2
            cq2_h2  = -_FT(model.ctcon) * hs2 * hkb * hkc^2 / (usb * h2 * hk2^2) / h2 * half_cq2
        end

        cq2_u2 = cq2_hs2 * hs2_u2 + cq2_us2 * us2_u2 + cq2_hk2 * hk2_u2 + cq2_rt2 * kin.rt2_u2
        cq2_t2 = cq2_hs2 * hs2_t2 + cq2_us2 * us2_t2 + cq2_hk2 * hk2_t2 + cq2_h2 * kin.h2_t2 + cq2_rt2 * kin.rt2_t2
        cq2_d2 = cq2_hs2 * hs2_d2 + cq2_us2 * us2_d2 + cq2_hk2 * hk2_d2 + cq2_h2 * kin.h2_d2
        cq2_ms = cq2_hs2 * hs2_ms + cq2_us2 * us2_ms + cq2_hk2 * hk2_ms + cq2_rt2 * kin.rt2_ms
        cq2_re = cq2_hs2 * hs2_re + cq2_us2 * us2_re + cq2_rt2 * kin.rt2_re

        # CF2 – skin friction
        if ityp_loc == :wake
            cf2 = _FT(0)
            cf2_u2 = _FT(0); cf2_t2 = _FT(0); cf2_d2 = _FT(0); cf2_ms = _FT(0); cf2_re = _FT(0)
            cf2_hk2 = _FT(0); cf2_rt2 = _FT(0); cf2_m2 = _FT(0)
        elseif ityp_loc == :laminar || ityp_loc == :similarity
            cf2, cf2_hk2, cf2_rt2, cf2_m2 = _cfl_jac(hk2, kin.rt2, kin.m2)
            cf2_u2 = cf2_hk2 * hk2_u2 + cf2_rt2 * kin.rt2_u2 + cf2_m2 * kin.m2_u2
            cf2_t2 = cf2_hk2 * hk2_t2 + cf2_rt2 * kin.rt2_t2
            cf2_d2 = cf2_hk2 * hk2_d2
            cf2_ms = cf2_hk2 * hk2_ms + cf2_rt2 * kin.rt2_ms + cf2_m2 * kin.m2_ms
            cf2_re = cf2_rt2 * kin.rt2_re
        else  # turbulent
            cf2t, cf2t_hk2, cf2t_rt2, cf2t_m2 = _cft_jac(hk2, kin.rt2, kin.m2)
            cf2l, cf2l_hk2, cf2l_rt2, cf2l_m2 = _cfl_jac(hk2, kin.rt2, kin.m2)
            if cf2l > cf2t
                cf2 = cf2l; cf2_hk2 = cf2l_hk2; cf2_rt2 = cf2l_rt2; cf2_m2 = cf2l_m2
            else
                cf2 = cf2t; cf2_hk2 = cf2t_hk2; cf2_rt2 = cf2t_rt2; cf2_m2 = cf2t_m2
            end
            cf2_u2 = cf2_hk2 * hk2_u2 + cf2_rt2 * kin.rt2_u2 + cf2_m2 * kin.m2_u2
            cf2_t2 = cf2_hk2 * hk2_t2 + cf2_rt2 * kin.rt2_t2
            cf2_d2 = cf2_hk2 * hk2_d2
            cf2_ms = cf2_hk2 * hk2_ms + cf2_rt2 * kin.rt2_ms + cf2_m2 * kin.m2_ms
            cf2_re = cf2_rt2 * kin.rt2_re
        end

        # DI2 – dissipation
        di2_s2 = _FT(0)
        if ityp_loc == :laminar || ityp_loc == :similarity
            di2, di2_hk2, di2_rt2 = _dil_jac(hk2, kin.rt2)
            di2_u2 = di2_hk2 * hk2_u2 + di2_rt2 * kin.rt2_u2
            di2_t2 = di2_hk2 * hk2_t2 + di2_rt2 * kin.rt2_t2
            di2_d2 = di2_hk2 * hk2_d2
            di2_ms = di2_hk2 * hk2_ms + di2_rt2 * kin.rt2_ms
            di2_re = di2_rt2 * kin.rt2_re
        elseif ityp_loc == :turbulent
            # turbulent wall contribution using cft
            cf2t_val, cf2t_hk2v, cf2t_rt2v, cf2t_m2v = _cft_jac(hk2, kin.rt2, kin.m2)
            cf2t_u2v = cf2t_hk2v * hk2_u2 + cf2t_rt2v * kin.rt2_u2 + cf2t_m2v * kin.m2_u2
            cf2t_t2v = cf2t_hk2v * hk2_t2 + cf2t_rt2v * kin.rt2_t2
            cf2t_d2v = cf2t_hk2v * hk2_d2
            cf2t_msv = cf2t_hk2v * hk2_ms + cf2t_rt2v * kin.rt2_ms + cf2t_m2v * kin.m2_ms
            cf2t_rev = cf2t_rt2v * kin.rt2_re

            hs2_safe = max(hs2, eps)
            di2       = (_FT(0.5) * cf2t_val * us2) * _FT(2) / hs2_safe
            di2_hs2_w = -(_FT(0.5) * cf2t_val * us2) * _FT(2) / hs2_safe^2
            di2_us2_w = (_FT(0.5) * cf2t_val) * _FT(2) / hs2_safe
            di2_cf2t  = (_FT(0.5) * us2) * _FT(2) / hs2_safe

            di2_s2 = _FT(0)
            di2_u2 = di2_hs2_w * hs2_u2 + di2_us2_w * us2_u2 + di2_cf2t * cf2t_u2v
            di2_t2 = di2_hs2_w * hs2_t2 + di2_us2_w * us2_t2 + di2_cf2t * cf2t_t2v
            di2_d2 = di2_hs2_w * hs2_d2 + di2_us2_w * us2_d2 + di2_cf2t * cf2t_d2v
            di2_ms = di2_hs2_w * hs2_ms + di2_us2_w * us2_ms + di2_cf2t * cf2t_msv
            di2_re = di2_hs2_w * hs2_re + di2_us2_w * us2_re + di2_cf2t * cf2t_rev

            # DFAC correction for low Hk
            grt_val = log(rt2_safe)
            hmin = _FT(1) + _FT(2.1) / max(grt_val, eps)
            hm_rt2 = -(_FT(2.1) / max(grt_val, eps)^2) / rt2_safe

            fl     = (hk2 - _FT(1)) / max(hmin - _FT(1), eps)
            fl_hk2 = _FT(1) / max(hmin - _FT(1), eps)
            fl_rt2 = (-fl / max(hmin - _FT(1), eps)) * hm_rt2

            tfl  = tanh(fl)
            dfac = _FT(0.5) + _FT(0.5) * tfl
            df_fl = _FT(0.5) * (_FT(1) - tfl^2)

            df_hk2 = df_fl * fl_hk2
            df_rt2 = df_fl * fl_rt2

            di2_s2 = di2_s2 * dfac
            di2_u2 = di2_u2 * dfac + di2 * (df_hk2 * hk2_u2 + df_rt2 * kin.rt2_u2)
            di2_t2 = di2_t2 * dfac + di2 * (df_hk2 * hk2_t2 + df_rt2 * kin.rt2_t2)
            di2_d2 = di2_d2 * dfac + di2 * (df_hk2 * hk2_d2)
            di2_ms = di2_ms * dfac + di2 * (df_hk2 * hk2_ms + df_rt2 * kin.rt2_ms)
            di2_re = di2_re * dfac + di2 * (df_rt2 * kin.rt2_re)
            di2    = di2 * dfac

        else  # wake – zero wall contribution
            di2    = _FT(0); di2_s2 = _FT(0); di2_u2 = _FT(0); di2_t2 = _FT(0)
            di2_d2 = _FT(0); di2_ms = _FT(0); di2_re = _FT(0)
        end

        # Turbulent outer layer contribution (ityp != laminar/similarity)
        if ityp_loc == :turbulent || ityp_loc == :wake
            hs2_safe = max(hs2, eps)
            s2f = _FT(s_val)

            dd     = s2f^2 * (_FT(0.995) - us2) * _FT(2) / hs2_safe
            dd_hs2 = -s2f^2 * (_FT(0.995) - us2) * _FT(2) / hs2_safe^2
            dd_us2 = -s2f^2 * _FT(2) / hs2_safe
            dd_s2  = s2f * _FT(2) * (_FT(0.995) - us2) * _FT(2) / hs2_safe

            di2    += dd
            di2_s2  = dd_s2
            di2_u2 += dd_hs2 * hs2_u2 + dd_us2 * us2_u2
            di2_t2 += dd_hs2 * hs2_t2 + dd_us2 * us2_t2
            di2_d2 += dd_hs2 * hs2_d2 + dd_us2 * us2_d2
            di2_ms += dd_hs2 * hs2_ms + dd_us2 * us2_ms
            di2_re += dd_hs2 * hs2_re + dd_us2 * us2_re

            # laminar stress contribution
            dd2     = _FT(0.15) * (_FT(0.995) - us2)^2 / rt2_safe * _FT(2) / hs2_safe
            dd2_us2 = -_FT(0.15) * (_FT(0.995) - us2) * _FT(2) / rt2_safe * _FT(2) / hs2_safe
            dd2_hs2 = -dd2 / hs2_safe
            dd2_rt2 = -dd2 / rt2_safe

            di2    += dd2
            di2_u2 += dd2_hs2 * hs2_u2 + dd2_us2 * us2_u2 + dd2_rt2 * kin.rt2_u2
            di2_t2 += dd2_hs2 * hs2_t2 + dd2_us2 * us2_t2 + dd2_rt2 * kin.rt2_t2
            di2_d2 += dd2_hs2 * hs2_d2 + dd2_us2 * us2_d2
            di2_ms += dd2_hs2 * hs2_ms + dd2_us2 * us2_ms + dd2_rt2 * kin.rt2_ms
            di2_re += dd2_hs2 * hs2_re + dd2_us2 * us2_re + dd2_rt2 * kin.rt2_re
        end

        # For turbulent: check if laminar DI is larger
        if ityp_loc == :turbulent
            di2l, di2l_hk2, di2l_rt2 = _dil_jac(hk2, kin.rt2)
            if di2l > di2
                di2    = di2l
                di2_s2 = _FT(0)
                di2_u2 = di2l_hk2 * hk2_u2 + di2l_rt2 * kin.rt2_u2
                di2_t2 = di2l_hk2 * hk2_t2 + di2l_rt2 * kin.rt2_t2
                di2_d2 = di2l_hk2 * hk2_d2
                di2_ms = di2l_hk2 * hk2_ms + di2l_rt2 * kin.rt2_ms
                di2_re = di2l_rt2 * kin.rt2_re
            end
        end

        # For wake: check if laminar wake DI is larger
        if ityp_loc == :wake
            di2l, di2l_hk2, di2l_rt2 = _dilw_jac(hk2, kin.rt2)
            if di2l > di2
                di2    = di2l
                di2_s2 = _FT(0)
                di2_u2 = di2l_hk2 * hk2_u2 + di2l_rt2 * kin.rt2_u2
                di2_t2 = di2l_hk2 * hk2_t2 + di2l_rt2 * kin.rt2_t2
                di2_d2 = di2l_hk2 * hk2_d2
                di2_ms = di2l_hk2 * hk2_ms + di2l_rt2 * kin.rt2_ms
                di2_re = di2l_rt2 * kin.rt2_re
            end
        end

        # Wake: double dissipation
        if ityp_loc == :wake
            di2    *= _FT(2); di2_s2 *= _FT(2); di2_u2 *= _FT(2); di2_t2 *= _FT(2)
            di2_d2 *= _FT(2); di2_ms *= _FT(2); di2_re *= _FT(2)
        end

        # DE2 – boundary layer thickness
        hk_safe = max(hk2 - _FT(1), _FT(1e-8))
        de2     = (_FT(3.15) + _FT(1.72) / hk_safe) * kin.t2 + kin.d2
        de2_hk2 = (-_FT(1.72) / hk_safe^2) * kin.t2

        de2_u2 = de2_hk2 * hk2_u2
        de2_t2 = de2_hk2 * hk2_t2 + (_FT(3.15) + _FT(1.72) / hk_safe)
        de2_d2 = de2_hk2 * hk2_d2 + _FT(1)
        de2_ms = de2_hk2 * hk2_ms

        hdmax = _FT(12)
        if de2 > hdmax * kin.t2
            de2    = hdmax * kin.t2
            de2_u2 = _FT(0)
            de2_t2 = hdmax
            de2_d2 = _FT(0)
            de2_ms = _FT(0)
        end

        return (;
            hk2, hk2_u2, hk2_t2, hk2_d2, hk2_ms,
            hs2, hs2_u2, hs2_t2, hs2_d2, hs2_ms, hs2_re,
            hc2, hc2_u2, hc2_t2, hc2_d2, hc2_ms,
            us2, us2_u2, us2_t2, us2_d2, us2_ms, us2_re,
            cf2, cf2_u2, cf2_t2, cf2_d2, cf2_ms, cf2_re,
            cq2, cq2_u2, cq2_t2, cq2_d2, cq2_ms, cq2_re,
            di2, di2_s2, di2_u2, di2_t2, di2_d2, di2_ms, di2_re,
            de2, de2_u2, de2_t2, de2_d2, de2_ms,
            m2 = kin.m2, m2_u2 = kin.m2_u2, m2_ms = kin.m2_ms,
            h2 = kin.h2, h2_t2 = kin.h2_t2, h2_d2 = kin.h2_d2,
            rt2 = kin.rt2, rt2_u2 = kin.rt2_u2, rt2_t2 = kin.rt2_t2, rt2_ms = kin.rt2_ms, rt2_re = kin.rt2_re,
        )
    end

    # ══════════════════════════════════════════════════════════════════════
    #  Compute BLKIN + BLVAR for both stations
    # ══════════════════════════════════════════════════════════════════════

    ityp_closure = SIMI ? :laminar : ityp

    # Use pre-computed station-1 kinematic state if provided (from MRCHDU COM1),
    # otherwise recompute.  Using the pre-computed state ensures SETBL and MRCHDU
    # see exactly the same station-1 derivatives, matching XFoil's COM1=COM2 chain.
    kin1 = isnothing(kin1_pre) ? _blkin(p1.u, p1.t, p1.d) : kin1_pre
    kin2 = _blkin(p2.u, p2.t, p2.d)

    bv1 = _blvar(kin1, ityp_closure, p1.s)
    bv2 = _blvar(kin2, ityp_closure, p2.s)

    # For similarity: station 1 = station 2
    if SIMI
        kin1 = kin2
        bv1  = bv2
    end

    # ══════════════════════════════════════════════════════════════════════
    #  BLMID – midpoint Cf
    # ══════════════════════════════════════════════════════════════════════

    hka_m = _FT(0.5) * (bv1.hk2 + bv2.hk2)
    rta_m = _FT(0.5) * (bv1.rt2 + bv2.rt2)
    ma_m  = _FT(0.5) * (bv1.m2  + bv2.m2)

    if ityp == :wake
        cfm = _FT(0); cfm_hka = _FT(0); cfm_rta = _FT(0); cfm_ma = _FT(0)
    elseif ityp == :similarity || ityp == :laminar
        cfm, cfm_hka, cfm_rta, cfm_ma = _cfl_jac(hka_m, rta_m, ma_m)
    else  # turbulent
        cfmt, cfmt_hka, cfmt_rta, cfmt_ma = _cft_jac(hka_m, rta_m, ma_m)
        cfml, cfml_hka, cfml_rta, cfml_ma = _cfl_jac(hka_m, rta_m, ma_m)
        if cfml > cfmt
            cfm = cfml; cfm_hka = cfml_hka; cfm_rta = cfml_rta; cfm_ma = cfml_ma
        else
            cfm = cfmt; cfm_hka = cfmt_hka; cfm_rta = cfmt_rta; cfm_ma = cfmt_ma
        end
    end

    # CFM sensitivities wrt primary variables
    cfm_u1 = _FT(0.5) * (cfm_hka * bv1.hk2_u2 + cfm_ma * bv1.m2_u2 + cfm_rta * bv1.rt2_u2)
    cfm_t1 = _FT(0.5) * (cfm_hka * bv1.hk2_t2 + cfm_rta * bv1.rt2_t2)
    cfm_d1 = _FT(0.5) * (cfm_hka * bv1.hk2_d2)

    cfm_u2 = _FT(0.5) * (cfm_hka * bv2.hk2_u2 + cfm_ma * bv2.m2_u2 + cfm_rta * bv2.rt2_u2)
    cfm_t2 = _FT(0.5) * (cfm_hka * bv2.hk2_t2 + cfm_rta * bv2.rt2_t2)
    cfm_d2 = _FT(0.5) * (cfm_hka * bv2.hk2_d2)

    cfm_ms_val = _FT(0.5) * (cfm_hka * bv1.hk2_ms + cfm_ma * bv1.m2_ms + cfm_rta * bv1.rt2_ms +
                         cfm_hka * bv2.hk2_ms + cfm_ma * bv2.m2_ms + cfm_rta * bv2.rt2_ms)
    cfm_re_val = _FT(0.5) * (cfm_rta * bv1.rt2_re + cfm_rta * bv2.rt2_re)

    # ══════════════════════════════════════════════════════════════════════
    #  BLDIF – assemble finite-difference equations
    # ══════════════════════════════════════════════════════════════════════

    vs1   = zeros(_FT, 3, 5)
    vs2   = zeros(_FT, 3, 5)
    vsrez = zeros(_FT, 3)
    vsm   = zeros(_FT, 3)
    vsr   = zeros(_FT, 3)

    # ── Logarithmic differences ──
    if SIMI
        xlog = _FT(1)
        ulog = _FT(1)   # BULE = 1.0 default
        tlog = _FT(0)
        hlog = _FT(0)
        ddlog = _FT(0)
    else
        xlog  = log(max(p2.x, eps) / max(p1.x, eps))
        ulog  = log(max(p2.u, eps) / max(p1.u, eps))
        tlog  = log(max(p2.t, eps) / max(p1.t, eps))
        hlog  = log(max(bv2.hs2, eps) / max(bv1.hs2, eps))
        ddlog = _FT(1)
    end

    # ── Upwinding parameter ──
    hk1_up = max(bv1.hk2, _FT(1.0001))
    hk2_up = max(bv2.hk2, _FT(1.0001))

    hk1_clamped_up = (bv1.hk2 < _FT(1.0001))
    hk2_clamped_up = (bv2.hk2 < _FT(1.0001))

    hdcon = _FT(5) / hk2_up^2
    hd_hk1 = _FT(0)
    hd_hk2 = -hdcon * _FT(2) / hk2_up
    if ityp == :wake
        hdcon  = _FT(1) / hk2_up^2
        hd_hk2 = -hdcon * _FT(2) / hk2_up
    end
    if hk2_clamped_up
        hd_hk2 = _FT(0)
    end

    arg_up = abs((hk2_up - _FT(1)) / max(hk1_up - _FT(1), _FT(1e-8)))
    hl     = log(max(arg_up, eps))
    hl_hk1 = hk1_clamped_up ? _FT(0) : -_FT(1) / (hk1_up - _FT(1))
    hl_hk2 = hk2_clamped_up ? _FT(0) :  _FT(1) / (hk2_up - _FT(1))

    hlsq = min(hl^2, _FT(15))
    ehh  = exp(-hlsq * hdcon)
    upw  = _FT(1) - _FT(0.5) * ehh
    upw_hl = ehh * hl * hdcon
    upw_hd = _FT(0.5) * ehh * hlsq
    if hl^2 >= _FT(15)
        upw_hl = _FT(0)  # derivative of min clamp
    end

    upw_hk1 = upw_hl * hl_hk1 + upw_hd * hd_hk1
    upw_hk2 = upw_hl * hl_hk2 + upw_hd * hd_hk2

    upw_u1 = upw_hk1 * bv1.hk2_u2
    upw_t1 = upw_hk1 * bv1.hk2_t2
    upw_d1 = upw_hk1 * bv1.hk2_d2
    upw_u2 = upw_hk2 * bv2.hk2_u2
    upw_t2 = upw_hk2 * bv2.hk2_t2
    upw_d2 = upw_hk2 * bv2.hk2_d2
    upw_ms_val = upw_hk1 * bv1.hk2_ms + upw_hk2 * bv2.hk2_ms

    # ── Equation 1 ──
    if SIMI
        # similarity: zero amplification
        vs2[1, 1] = _FT(1)
        vsrez[1]  = -p2.ampl

    elseif ityp == :laminar
        # Amplification equation via AXSET
        ax1_val, ax1_hk, ax1_th, ax1_rt = _dampl_jac(bv1.hk2, p1.t, bv1.rt2)
        ax2_val, ax2_hk, ax2_th, ax2_rt = _dampl_jac(bv2.hk2, p2.t, bv2.rt2)

        # RMS average
        axsq = _FT(0.5) * (ax1_val^2 + ax2_val^2)
        if axsq <= _FT(0)
            axa = _FT(0); axa_ax1 = _FT(0); axa_ax2 = _FT(0)
        else
            axa = sqrt(axsq)
            axa_ax1 = _FT(0.5) * ax1_val / axa
            axa_ax2 = _FT(0.5) * ax2_val / axa
        end

        # Small additional term near N=Ncrit
        a_avg = _FT(0.5) * (p1.ampl + p2.ampl)
        arg_n = min(_FT(20) * (_FT(acrit) - a_avg), _FT(20))
        if arg_n <= _FT(0)
            exn = _FT(1); exn_a1 = _FT(0); exn_a2 = _FT(0)
        else
            exn    = exp(-arg_n)
            exn_a1 = _FT(20) * _FT(0.5) * exn
            exn_a2 = _FT(20) * _FT(0.5) * exn
        end
        t_sum = max(p1.t + p2.t, eps)
        dax    = exn * _FT(0.002) / t_sum
        dax_a1 = exn_a1 * _FT(0.002) / t_sum
        dax_a2 = exn_a2 * _FT(0.002) / t_sum
        dax_t1 = -dax / t_sum
        dax_t2 = -dax / t_sum

        ax = axa + dax
        ax_hk1 = axa_ax1 * ax1_hk
        ax_t1  = axa_ax1 * ax1_th + dax_t1
        ax_rt1 = axa_ax1 * ax1_rt
        ax_a1  = dax_a1

        ax_hk2 = axa_ax2 * ax2_hk
        ax_t2  = axa_ax2 * ax2_th + dax_t2
        ax_rt2 = axa_ax2 * ax2_rt
        ax_a2  = dax_a2

        dxi = p2.x - p1.x
        rezc = p2.ampl - p1.ampl - ax * dxi
        z_ax = -dxi

        vs1[1, 1] = z_ax * ax_a1 - _FT(1)
        vs1[1, 2] = z_ax * (ax_hk1 * bv1.hk2_t2 + ax_t1 + ax_rt1 * bv1.rt2_t2)
        vs1[1, 3] = z_ax * (ax_hk1 * bv1.hk2_d2)
        vs1[1, 4] = z_ax * (ax_hk1 * bv1.hk2_u2 + ax_rt1 * bv1.rt2_u2)
        vs1[1, 5] = ax
        vs2[1, 1] = z_ax * ax_a2 + _FT(1)
        vs2[1, 2] = z_ax * (ax_hk2 * bv2.hk2_t2 + ax_t2 + ax_rt2 * bv2.rt2_t2)
        vs2[1, 3] = z_ax * (ax_hk2 * bv2.hk2_d2)
        vs2[1, 4] = z_ax * (ax_hk2 * bv2.hk2_u2 + ax_rt2 * bv2.rt2_u2)
        vs2[1, 5] = -ax
        vsm[1] = z_ax * (ax_hk1 * bv1.hk2_ms + ax_rt1 * bv1.rt2_ms +
                          ax_hk2 * bv2.hk2_ms + ax_rt2 * bv2.rt2_ms)
        vsr[1] = z_ax * (ax_rt1 * bv1.rt2_re + ax_rt2 * bv2.rt2_re)
        vsrez[1] = -rezc

    else
        # Turbulent / wake shear-lag equation
        sa  = (_FT(1) - upw) * p1.s + upw * p2.s
        cqa = (_FT(1) - upw) * bv1.cq2 + upw * bv2.cq2
        cfa = (_FT(1) - upw) * bv1.cf2 + upw * bv2.cf2
        hka = (_FT(1) - upw) * bv1.hk2 + upw * bv2.hk2

        usa = _FT(0.5) * (bv1.us2 + bv2.us2)
        rta = _FT(0.5) * (bv1.rt2 + bv2.rt2)
        dea = _FT(0.5) * (bv1.de2 + bv2.de2)
        da  = max(_FT(0.5) * (p1.d + p2.d), eps)

        ald = ityp == :wake ? _FT(model.dlcon) : _FT(1)

        # Equilibrium 1/Ue dUe/dx
        if ityp == :turbulent
            gcc_v = _FT(model.gccon)
            hkc_v     = hka - _FT(1) - gcc_v / max(rta, eps)
            hkc_hka_v = _FT(1)
            hkc_rta_v = gcc_v / max(rta, eps)^2
            if hkc_v < _FT(0.01)
                hkc_v = _FT(0.01); hkc_hka_v = _FT(0); hkc_rta_v = _FT(0)
            end
        else
            gcc_v = _FT(0)
            hkc_v     = hka - _FT(1)
            hkc_hka_v = _FT(1)
            hkc_rta_v = _FT(0)
        end

        hr     = hkc_v / (_FT(model.gacon) * ald * max(hka, eps))
        hr_hka = hkc_hka_v / (_FT(model.gacon) * ald * max(hka, eps)) - hr / max(hka, eps)
        hr_rta = hkc_rta_v / (_FT(model.gacon) * ald * max(hka, eps))

        uq     = (_FT(0.5) * cfa - hr^2) / (_FT(model.gbcon) * da)
        uq_hka = -_FT(2) * hr * hr_hka / (_FT(model.gbcon) * da)
        uq_rta = -_FT(2) * hr * hr_rta / (_FT(model.gbcon) * da)
        uq_cfa = _FT(0.5) / (_FT(model.gbcon) * da)
        uq_da  = -uq / da
        uq_upw = uq_cfa * (bv2.cf2 - bv1.cf2) + uq_hka * (bv2.hk2 - bv1.hk2)

        uq_t1 = (_FT(1) - upw) * (uq_cfa * bv1.cf2_t2 + uq_hka * bv1.hk2_t2) + uq_upw * upw_t1
        uq_d1 = (_FT(1) - upw) * (uq_cfa * bv1.cf2_d2 + uq_hka * bv1.hk2_d2) + uq_upw * upw_d1
        uq_u1 = (_FT(1) - upw) * (uq_cfa * bv1.cf2_u2 + uq_hka * bv1.hk2_u2) + uq_upw * upw_u1
        uq_t2 =      upw  * (uq_cfa * bv2.cf2_t2 + uq_hka * bv2.hk2_t2) + uq_upw * upw_t2
        uq_d2 =      upw  * (uq_cfa * bv2.cf2_d2 + uq_hka * bv2.hk2_d2) + uq_upw * upw_d2
        uq_u2 =      upw  * (uq_cfa * bv2.cf2_u2 + uq_hka * bv2.hk2_u2) + uq_upw * upw_u2
        uq_ms_v = (_FT(1) - upw) * (uq_cfa * bv1.cf2_ms + uq_hka * bv1.hk2_ms) + uq_upw * upw_ms_val +
                        upw  * (uq_cfa * bv2.cf2_ms + uq_hka * bv2.hk2_ms)
        uq_re_v = (_FT(1) - upw) * uq_cfa * bv1.cf2_re + upw * uq_cfa * bv2.cf2_re

        uq_t1 += _FT(0.5) * uq_rta * bv1.rt2_t2
        uq_d1 += _FT(0.5) * uq_da
        uq_u1 += _FT(0.5) * uq_rta * bv1.rt2_u2
        uq_t2 += _FT(0.5) * uq_rta * bv2.rt2_t2
        uq_d2 += _FT(0.5) * uq_da
        uq_u2 += _FT(0.5) * uq_rta * bv2.rt2_u2
        uq_ms_v += _FT(0.5) * uq_rta * bv1.rt2_ms + _FT(0.5) * uq_rta * bv2.rt2_ms
        uq_re_v += _FT(0.5) * uq_rta * bv1.rt2_re + _FT(0.5) * uq_rta * bv2.rt2_re

        scc = _FT(model.sccon) * _FT(1.333) / max(_FT(1) + usa, eps)
        scc_usa = -scc / max(_FT(1) + usa, eps)

        slog = log(max(p2.s, eps) / max(p1.s, eps))
        dxi  = p2.x - p1.x

        rezc = scc * (cqa - sa * ald) * dxi -
               dea * _FT(2) * slog +
               dea * _FT(2) * (uq * dxi - ulog) * _FT(model.duxcon)

        z_cfa = dea * _FT(2) * uq_cfa * dxi * _FT(model.duxcon)
        z_hka = dea * _FT(2) * uq_hka * dxi * _FT(model.duxcon)
        z_da  = dea * _FT(2) * uq_da  * dxi * _FT(model.duxcon)
        z_sl  = -dea * _FT(2)
        z_ul  = -dea * _FT(2) * _FT(model.duxcon)
        z_dxi = scc * (cqa - sa * ald) + dea * _FT(2) * uq * _FT(model.duxcon)
        z_usa = scc_usa * (cqa - sa * ald) * dxi
        z_cqa = scc * dxi
        z_sa  = -scc * dxi * ald
        z_dea = _FT(2) * ((uq * dxi - ulog) * _FT(model.duxcon) - slog)

        z_upw = z_cqa * (bv2.cq2 - bv1.cq2) + z_sa * (p2.s - p1.s) +
                z_cfa * (bv2.cf2 - bv1.cf2) + z_hka * (bv2.hk2 - bv1.hk2)
        z_de1 = _FT(0.5) * z_dea
        z_de2 = _FT(0.5) * z_dea
        z_us1 = _FT(0.5) * z_usa
        z_us2_v = _FT(0.5) * z_usa
        z_d1  = _FT(0.5) * z_da
        z_d2  = _FT(0.5) * z_da
        z_u1  = -z_ul / max(p1.u, eps)
        z_u2  =  z_ul / max(p2.u, eps)
        z_x1  = -z_dxi
        z_x2  =  z_dxi
        z_s1  = (_FT(1) - upw) * z_sa - z_sl / max(p1.s, eps)
        z_s2  =      upw  * z_sa + z_sl / max(p2.s, eps)
        z_cq1 = (_FT(1) - upw) * z_cqa
        z_cq2 =      upw  * z_cqa
        z_cf1 = (_FT(1) - upw) * z_cfa
        z_cf2 =      upw  * z_cfa
        z_hk1 = (_FT(1) - upw) * z_hka
        z_hk2 =      upw  * z_hka

        vs1[1, 1] = z_s1
        vs1[1, 2] = z_upw * upw_t1 + z_de1 * bv1.de2_t2 + z_us1 * bv1.us2_t2
        vs1[1, 3] = z_d1 + z_upw * upw_d1 + z_de1 * bv1.de2_d2 + z_us1 * bv1.us2_d2
        vs1[1, 4] = z_u1 + z_upw * upw_u1 + z_de1 * bv1.de2_u2 + z_us1 * bv1.us2_u2
        vs1[1, 5] = z_x1
        vs2[1, 1] = z_s2
        vs2[1, 2] = z_upw * upw_t2 + z_de2 * bv2.de2_t2 + z_us2_v * bv2.us2_t2
        vs2[1, 3] = z_d2 + z_upw * upw_d2 + z_de2 * bv2.de2_d2 + z_us2_v * bv2.us2_d2
        vs2[1, 4] = z_u2 + z_upw * upw_u2 + z_de2 * bv2.de2_u2 + z_us2_v * bv2.us2_u2
        vs2[1, 5] = z_x2

        vs1[1, 2] += z_cq1 * bv1.cq2_t2 + z_cf1 * bv1.cf2_t2 + z_hk1 * bv1.hk2_t2
        vs1[1, 3] += z_cq1 * bv1.cq2_d2 + z_cf1 * bv1.cf2_d2 + z_hk1 * bv1.hk2_d2
        vs1[1, 4] += z_cq1 * bv1.cq2_u2 + z_cf1 * bv1.cf2_u2 + z_hk1 * bv1.hk2_u2

        vs2[1, 2] += z_cq2 * bv2.cq2_t2 + z_cf2 * bv2.cf2_t2 + z_hk2 * bv2.hk2_t2
        vs2[1, 3] += z_cq2 * bv2.cq2_d2 + z_cf2 * bv2.cf2_d2 + z_hk2 * bv2.hk2_d2
        vs2[1, 4] += z_cq2 * bv2.cq2_u2 + z_cf2 * bv2.cf2_u2 + z_hk2 * bv2.hk2_u2

        vsm[1] = z_upw * upw_ms_val + z_de1 * bv1.de2_ms + z_us1 * bv1.us2_ms +
                  z_de2 * bv2.de2_ms + z_us2_v * bv2.us2_ms +
                  z_cq1 * bv1.cq2_ms + z_cf1 * bv1.cf2_ms + z_hk1 * bv1.hk2_ms +
                  z_cq2 * bv2.cq2_ms + z_cf2 * bv2.cf2_ms + z_hk2 * bv2.hk2_ms
        vsr[1] = z_cq1 * bv1.cq2_re + z_cf1 * bv1.cf2_re +
                 z_cq2 * bv2.cq2_re + z_cf2 * bv2.cf2_re
        vsrez[1] = -rezc
    end

    # ── Equation 2: Momentum equation ──
    ha  = _FT(0.5) * (bv1.h2 + bv2.h2)
    ma  = _FT(0.5) * (bv1.m2 + bv2.m2)
    xa  = _FT(0.5) * (p1.x + p2.x)
    ta  = max(_FT(0.5) * (p1.t + p2.t), eps)
    hwa = _FT(0.5) * (p1.dw / max(p1.t, eps) + p2.dw / max(p2.t, eps))

    t1s = max(p1.t, eps)
    t2s = max(p2.t, eps)

    cfx     = _FT(0.5) * cfm * xa / ta + _FT(0.25) * (bv1.cf2 * p1.x / t1s + bv2.cf2 * p2.x / t2s)
    cfx_xa  = _FT(0.5) * cfm / ta
    cfx_ta  = -_FT(0.5) * cfm * xa / ta^2

    cfx_x1  = _FT(0.25) * bv1.cf2 / t1s + cfx_xa * _FT(0.5)
    cfx_x2  = _FT(0.25) * bv2.cf2 / t2s + cfx_xa * _FT(0.5)
    cfx_t1  = -_FT(0.25) * bv1.cf2 * p1.x / t1s^2 + cfx_ta * _FT(0.5)
    cfx_t2  = -_FT(0.25) * bv2.cf2 * p2.x / t2s^2 + cfx_ta * _FT(0.5)
    cfx_cf1 = _FT(0.25) * p1.x / t1s
    cfx_cf2 = _FT(0.25) * p2.x / t2s
    cfx_cfm = _FT(0.5)  * xa / ta

    btmp = ha + _FT(2) - ma + hwa

    rezt  = tlog + btmp * ulog - xlog * _FT(0.5) * cfx
    z_cfx = -xlog * _FT(0.5)
    z_ha  =  ulog
    z_hwa =  ulog
    z_ma  = -ulog
    z_xl  = -ddlog * _FT(0.5) * cfx
    z_ul  =  ddlog * btmp
    z_tl  =  ddlog

    z_cfm_v = z_cfx * cfx_cfm
    z_cf1_v = z_cfx * cfx_cf1
    z_cf2_v = z_cfx * cfx_cf2

    z_t1 = -z_tl / t1s + z_cfx * cfx_t1 + z_hwa * _FT(0.5) * (-p1.dw / t1s^2)
    z_t2 =  z_tl / t2s + z_cfx * cfx_t2 + z_hwa * _FT(0.5) * (-p2.dw / t2s^2)
    z_x1 = -z_xl / max(p1.x, eps) + z_cfx * cfx_x1
    z_x2 =  z_xl / max(p2.x, eps) + z_cfx * cfx_x2
    z_u1 = -z_ul / max(p1.u, eps)
    z_u2 =  z_ul / max(p2.u, eps)

    vs1[2, 2] = _FT(0.5) * z_ha * bv1.h2_t2 + z_cfm_v * cfm_t1 + z_cf1_v * bv1.cf2_t2 + z_t1
    vs1[2, 3] = _FT(0.5) * z_ha * bv1.h2_d2 + z_cfm_v * cfm_d1 + z_cf1_v * bv1.cf2_d2
    vs1[2, 4] = _FT(0.5) * z_ma * bv1.m2_u2 + z_cfm_v * cfm_u1 + z_cf1_v * bv1.cf2_u2 + z_u1
    vs1[2, 5] = z_x1
    vs2[2, 2] = _FT(0.5) * z_ha * bv2.h2_t2 + z_cfm_v * cfm_t2 + z_cf2_v * bv2.cf2_t2 + z_t2
    vs2[2, 3] = _FT(0.5) * z_ha * bv2.h2_d2 + z_cfm_v * cfm_d2 + z_cf2_v * bv2.cf2_d2
    vs2[2, 4] = _FT(0.5) * z_ma * bv2.m2_u2 + z_cfm_v * cfm_u2 + z_cf2_v * bv2.cf2_u2 + z_u2
    vs2[2, 5] = z_x2

    vsm[2] = _FT(0.5) * z_ma * bv1.m2_ms + z_cfm_v * cfm_ms_val + z_cf1_v * bv1.cf2_ms +
             _FT(0.5) * z_ma * bv2.m2_ms + z_cf2_v * bv2.cf2_ms
    vsr[2] = z_cfm_v * cfm_re_val + z_cf1_v * bv1.cf2_re + z_cf2_v * bv2.cf2_re
    vsrez[2] = -rezt

    # ── Equation 3: Shape parameter equation ──
    xot1 = p1.x / t1s
    xot2 = p2.x / t2s

    hsa = max(_FT(0.5) * (bv1.hs2 + bv2.hs2), eps)
    hca = _FT(0.5) * (bv1.hc2 + bv2.hc2)

    dix     = (_FT(1) - upw) * bv1.di2 * xot1 + upw * bv2.di2 * xot2
    cfx_h   = (_FT(1) - upw) * bv1.cf2 * xot1 + upw * bv2.cf2 * xot2
    dix_upw = bv2.di2 * xot2 - bv1.di2 * xot1
    cfx_upw = bv2.cf2 * xot2 - bv1.cf2 * xot1

    btmp_h = _FT(2) * hca / hsa + _FT(1) - ha - hwa

    rezh  = hlog + btmp_h * ulog + xlog * (_FT(0.5) * cfx_h - dix)
    z_cfx_h = xlog * _FT(0.5)
    z_dix   = -xlog
    z_hca   = _FT(2) * ulog / hsa
    z_ha_h  = -ulog
    z_hwa_h = -ulog
    z_xl_h  = ddlog * (_FT(0.5) * cfx_h - dix)
    z_ul_h  = ddlog * btmp_h
    z_hl    = ddlog

    z_upw_h = z_cfx_h * cfx_upw + z_dix * dix_upw

    z_hs1 = -hca * ulog / hsa^2 - z_hl / max(bv1.hs2, eps)
    z_hs2_v = -hca * ulog / hsa^2 + z_hl / max(bv2.hs2, eps)

    z_cf1_h = (_FT(1) - upw) * z_cfx_h * xot1
    z_cf2_h =      upw  * z_cfx_h * xot2
    z_di1   = (_FT(1) - upw) * z_dix * xot1
    z_di2   =      upw  * z_dix * xot2

    z_t1_h = (_FT(1) - upw) * (z_cfx_h * bv1.cf2 + z_dix * bv1.di2) * (-xot1 / t1s)
    z_t2_h =      upw  * (z_cfx_h * bv2.cf2 + z_dix * bv2.di2) * (-xot2 / t2s)
    z_x1_h = (_FT(1) - upw) * (z_cfx_h * bv1.cf2 + z_dix * bv1.di2) / t1s - z_xl_h / max(p1.x, eps)
    z_x2_h =      upw  * (z_cfx_h * bv2.cf2 + z_dix * bv2.di2) / t2s + z_xl_h / max(p2.x, eps)
    z_u1_h = -z_ul_h / max(p1.u, eps)
    z_u2_h =  z_ul_h / max(p2.u, eps)

    z_t1_h += z_hwa_h * _FT(0.5) * (-p1.dw / t1s^2)
    z_t2_h += z_hwa_h * _FT(0.5) * (-p2.dw / t2s^2)

    vs1[3, 1] = z_di1 * bv1.di2_s2
    vs1[3, 2] = z_hs1 * bv1.hs2_t2 + z_cf1_h * bv1.cf2_t2 + z_di1 * bv1.di2_t2 + z_t1_h
    vs1[3, 3] = z_hs1 * bv1.hs2_d2 + z_cf1_h * bv1.cf2_d2 + z_di1 * bv1.di2_d2
    vs1[3, 4] = z_hs1 * bv1.hs2_u2 + z_cf1_h * bv1.cf2_u2 + z_di1 * bv1.di2_u2 + z_u1_h
    vs1[3, 5] = z_x1_h
    vs2[3, 1] = z_di2 * bv2.di2_s2
    vs2[3, 2] = z_hs2_v * bv2.hs2_t2 + z_cf2_h * bv2.cf2_t2 + z_di2 * bv2.di2_t2 + z_t2_h
    vs2[3, 3] = z_hs2_v * bv2.hs2_d2 + z_cf2_h * bv2.cf2_d2 + z_di2 * bv2.di2_d2
    vs2[3, 4] = z_hs2_v * bv2.hs2_u2 + z_cf2_h * bv2.cf2_u2 + z_di2 * bv2.di2_u2 + z_u2_h
    vs2[3, 5] = z_x2_h

    vsm[3] = z_hs1 * bv1.hs2_ms + z_cf1_h * bv1.cf2_ms + z_di1 * bv1.di2_ms +
             z_hs2_v * bv2.hs2_ms + z_cf2_h * bv2.cf2_ms + z_di2 * bv2.di2_ms
    vsr[3] = z_hs1 * bv1.hs2_re + z_cf1_h * bv1.cf2_re + z_di1 * bv1.di2_re +
             z_hs2_v * bv2.hs2_re + z_cf2_h * bv2.cf2_re + z_di2 * bv2.di2_re

    # Add HC and H contributions to shape parameter equation
    vs1[3, 2] += _FT(0.5) * (z_hca * bv1.hc2_t2 + z_ha_h * bv1.h2_t2) + z_upw_h * upw_t1
    vs1[3, 3] += _FT(0.5) * (z_hca * bv1.hc2_d2 + z_ha_h * bv1.h2_d2) + z_upw_h * upw_d1
    vs1[3, 4] += _FT(0.5) * (z_hca * bv1.hc2_u2) + z_upw_h * upw_u1
    vs2[3, 2] += _FT(0.5) * (z_hca * bv2.hc2_t2 + z_ha_h * bv2.h2_t2) + z_upw_h * upw_t2
    vs2[3, 3] += _FT(0.5) * (z_hca * bv2.hc2_d2 + z_ha_h * bv2.h2_d2) + z_upw_h * upw_d2
    vs2[3, 4] += _FT(0.5) * (z_hca * bv2.hc2_u2) + z_upw_h * upw_u2

    vsm[3] += _FT(0.5) * (z_hca * bv1.hc2_ms) + z_upw_h * upw_ms_val +
              _FT(0.5) * (z_hca * bv2.hc2_ms)

    vsrez[3] = -rezh

    # ── Similarity folding (BLSYS lines 636-644) ─────────────────────────
    # At the similarity station, "1" variables are really "2" variables,
    # so VS1 sensitivities must be folded into VS2.
    if SIMI
        for k in 1:3, l in 1:5
            vs2[k, l] = vs1[k, l] + vs2[k, l]
            vs1[k, l] = zero(_FT)
        end
    end

    return vs1, vs2, vsrez, vsm, vsr
end

# ────────────────────────────────────────────────────────────────────────────
#  _trdif_analytic  –  Transition interval linearization.
#
#  Port of XFoil TRDIF (xblsys.f lines 1218-1572).
#  Splits the transition interval X1→X2 into a laminar part X1→XT and a
#  turbulent part XT→X2, computes BLDIF for each, and sums with proper
#  chain-rule transformation of sensitivities.
#
#  Returns (vs1, vs2, vsrez, vsm, vsr) in the same format as _bldif_analytic.
# ────────────────────────────────────────────────────────────────────────────
function _trdif_analytic(
    p1::BLPrimaryState,
    p2::BLPrimaryState;
    flow::BLFlowParams,
    model::BLModelParams,
    acrit::Real = 9.0f0,
    idampv::Int    = 0,
    precision::Type{FT} = Float32,
) where {FT<:AbstractFloat}
    _FT = FT
    eps = _FT(1e-12)

    x1 = _FT(max(p1.x, 1e-9))
    x2 = _FT(max(p2.x, x1 + _FT(1e-6)))
    dx = x2 - x1

    # ── Run TRCHEK to get transition location XT ──────────────────────
    trstate = _trchek2_core_state(p1, p2, acrit, acrit;
        flow = flow, model = model, idampv = idampv)
    xt = _FT(trstate.xt)

    # Clamp XT within [x1, x2]
    xt = clamp(xt, x1 + eps, x2 - eps)

    # ── Compute XT sensitivities via FD ───────────────────────────────
    fd_eps = 1e-7
    function get_xt(p1m, p2m)
        st = try
            _trchek2_core_state(p1m, p2m, acrit, acrit;
                flow = flow, model = model, idampv = idampv)
        catch
            return Float64(xt)
        end
        return Float64(st.xt)
    end

    xt0 = Float64(xt)
    function fd_xt(p1m, p2m)
        return clamp((get_xt(p1m, p2m) - xt0) / fd_eps, -1e6, 1e6)
    end

    # Station 1 sensitivities
    xt_a1 = _FT(fd_xt(
        BLPrimaryState(p1.x, p1.u, p1.t, p1.d; s=p1.s, ampl=p1.ampl+fd_eps, dw=p1.dw), p2))
    xt_t1 = _FT(fd_xt(
        BLPrimaryState(p1.x, p1.u, p1.t+fd_eps, p1.d; s=p1.s, ampl=p1.ampl, dw=p1.dw), p2))
    xt_d1 = _FT(fd_xt(
        BLPrimaryState(p1.x, p1.u, p1.t, p1.d+fd_eps; s=p1.s, ampl=p1.ampl, dw=p1.dw), p2))
    xt_u1 = _FT(fd_xt(
        BLPrimaryState(p1.x, p1.u+fd_eps, p1.t, p1.d; s=p1.s, ampl=p1.ampl, dw=p1.dw), p2))
    xt_x1 = _FT(fd_xt(
        BLPrimaryState(p1.x+fd_eps, p1.u, p1.t, p1.d; s=p1.s, ampl=p1.ampl, dw=p1.dw), p2))

    # Station 2 sensitivities
    xt_t2 = _FT(fd_xt(p1,
        BLPrimaryState(p2.x, p2.u, p2.t+fd_eps, p2.d; s=p2.s, ampl=p2.ampl, dw=p2.dw)))
    xt_d2 = _FT(fd_xt(p1,
        BLPrimaryState(p2.x, p2.u, p2.t, p2.d+fd_eps; s=p2.s, ampl=p2.ampl, dw=p2.dw)))
    xt_u2 = _FT(fd_xt(p1,
        BLPrimaryState(p2.x, p2.u+fd_eps, p2.t, p2.d; s=p2.s, ampl=p2.ampl, dw=p2.dw)))
    xt_x2 = _FT(fd_xt(p1,
        BLPrimaryState(p2.x+fd_eps, p2.u, p2.t, p2.d; s=p2.s, ampl=p2.ampl, dw=p2.dw)))
    xt_ms = zero(_FT)
    xt_re = zero(_FT)

    # ── Weighting factors (TRDIF lines 1238-1266) ────────────────────
    wf2    = (xt - x1) / dx
    wf2_xt = one(_FT) / dx

    wf2_a1 = wf2_xt * xt_a1
    wf2_x1 = wf2_xt * xt_x1 + (wf2 - one(_FT)) / dx
    wf2_x2 = wf2_xt * xt_x2 - wf2 / dx
    wf2_t1 = wf2_xt * xt_t1
    wf2_t2 = wf2_xt * xt_t2
    wf2_d1 = wf2_xt * xt_d1
    wf2_d2 = wf2_xt * xt_d2
    wf2_u1 = wf2_xt * xt_u1
    wf2_u2 = wf2_xt * xt_u2
    wf2_ms = wf2_xt * xt_ms
    wf2_re = wf2_xt * xt_re

    wf1    = one(_FT) - wf2
    wf1_a1 = -wf2_a1;  wf1_x1 = -wf2_x1;  wf1_x2 = -wf2_x2
    wf1_t1 = -wf2_t1;  wf1_t2 = -wf2_t2
    wf1_d1 = -wf2_d1;  wf1_d2 = -wf2_d2
    wf1_u1 = -wf2_u1;  wf1_u2 = -wf2_u2
    wf1_ms = -wf2_ms;  wf1_re = -wf2_re

    # ── Interpolate primary variables to XT (TRDIF lines 1271-1312) ──
    t1v = _FT(p1.t);  t2v = _FT(p2.t)
    d1v = _FT(p1.d);  d2v = _FT(p2.d)
    u1v = _FT(p1.u);  u2v = _FT(p2.u)

    tt    = t1v*wf1    + t2v*wf2
    tt_a1 = t1v*wf1_a1 + t2v*wf2_a1
    tt_t1 = t1v*wf1_t1 + t2v*wf2_t1 + wf1
    tt_t2 = t1v*wf1_t2 + t2v*wf2_t2 + wf2
    tt_d1 = t1v*wf1_d1 + t2v*wf2_d1
    tt_d2 = t1v*wf1_d2 + t2v*wf2_d2
    tt_u1 = t1v*wf1_u1 + t2v*wf2_u1
    tt_u2 = t1v*wf1_u2 + t2v*wf2_u2
    tt_x1 = t1v*wf1_x1 + t2v*wf2_x1
    tt_x2 = t1v*wf1_x2 + t2v*wf2_x2
    tt_ms = t1v*wf1_ms + t2v*wf2_ms
    tt_re = t1v*wf1_re + t2v*wf2_re

    dt    = d1v*wf1    + d2v*wf2
    dt_a1 = d1v*wf1_a1 + d2v*wf2_a1
    dt_t1 = d1v*wf1_t1 + d2v*wf2_t1
    dt_t2 = d1v*wf1_t2 + d2v*wf2_t2
    dt_d1 = d1v*wf1_d1 + d2v*wf2_d1 + wf1
    dt_d2 = d1v*wf1_d2 + d2v*wf2_d2 + wf2
    dt_u1 = d1v*wf1_u1 + d2v*wf2_u1
    dt_u2 = d1v*wf1_u2 + d2v*wf2_u2
    dt_x1 = d1v*wf1_x1 + d2v*wf2_x1
    dt_x2 = d1v*wf1_x2 + d2v*wf2_x2
    dt_ms = d1v*wf1_ms + d2v*wf2_ms
    dt_re = d1v*wf1_re + d2v*wf2_re

    ut    = u1v*wf1    + u2v*wf2
    ut_a1 = u1v*wf1_a1 + u2v*wf2_a1
    ut_t1 = u1v*wf1_t1 + u2v*wf2_t1
    ut_t2 = u1v*wf1_t2 + u2v*wf2_t2
    ut_d1 = u1v*wf1_d1 + u2v*wf2_d1
    ut_d2 = u1v*wf1_d2 + u2v*wf2_d2
    ut_u1 = u1v*wf1_u1 + u2v*wf2_u1 + wf1
    ut_u2 = u1v*wf1_u2 + u2v*wf2_u2 + wf2
    ut_x1 = u1v*wf1_x1 + u2v*wf2_x1
    ut_x2 = u1v*wf1_x2 + u2v*wf2_x2
    ut_ms = u1v*wf1_ms + u2v*wf2_ms
    ut_re = u1v*wf1_re + u2v*wf2_re

    # ── LAMINAR PART: X1 → XT (TRDIF lines 1314-1403) ────────────────
    pt_lam = BLPrimaryState(_FT(xt), _FT(ut), _FT(tt), _FT(dt);
        s = zero(_FT), ampl = _FT(acrit), dw = zero(_FT))
    p1_ft = BLPrimaryState(_FT(p1.x), _FT(p1.u), _FT(p1.t), _FT(p1.d);
        s = zero(_FT), ampl = _FT(p1.ampl), dw = _FT(p1.dw))

    vs1_l, vs2_l, vsrez_l, vsm_l, vsr_l = _bldif_analytic(
        :laminar, p1_ft, pt_lam;
        flow = flow, model = model, acrit = acrit, idampv = idampv, precision = _FT)

    # Transform VS2 (wrt "T" vars) to BL1,BL2 (wrt "1","2" vars)
    # Equations 2,3 only (equation 1 = amplification handled by turbulent part)
    bl1 = zeros(_FT, 3, 5)
    bl2 = zeros(_FT, 3, 5)
    blrez = zeros(_FT, 3)
    blm = zeros(_FT, 3)
    blr = zeros(_FT, 3)

    for k in 2:3
        blrez[k] = vsrez_l[k]
        blm[k] = vsm_l[k] + vs2_l[k,2]*tt_ms + vs2_l[k,3]*dt_ms +
                  vs2_l[k,4]*ut_ms + vs2_l[k,5]*xt_ms
        blr[k] = vsr_l[k] + vs2_l[k,2]*tt_re + vs2_l[k,3]*dt_re +
                  vs2_l[k,4]*ut_re + vs2_l[k,5]*xt_re

        bl1[k,1] = vs1_l[k,1] + vs2_l[k,2]*tt_a1 + vs2_l[k,3]*dt_a1 +
                    vs2_l[k,4]*ut_a1 + vs2_l[k,5]*xt_a1
        bl1[k,2] = vs1_l[k,2] + vs2_l[k,2]*tt_t1 + vs2_l[k,3]*dt_t1 +
                    vs2_l[k,4]*ut_t1 + vs2_l[k,5]*xt_t1
        bl1[k,3] = vs1_l[k,3] + vs2_l[k,2]*tt_d1 + vs2_l[k,3]*dt_d1 +
                    vs2_l[k,4]*ut_d1 + vs2_l[k,5]*xt_d1
        bl1[k,4] = vs1_l[k,4] + vs2_l[k,2]*tt_u1 + vs2_l[k,3]*dt_u1 +
                    vs2_l[k,4]*ut_u1 + vs2_l[k,5]*xt_u1
        bl1[k,5] = vs1_l[k,5] + vs2_l[k,2]*tt_x1 + vs2_l[k,3]*dt_x1 +
                    vs2_l[k,4]*ut_x1 + vs2_l[k,5]*xt_x1

        bl2[k,1] = zero(_FT)
        bl2[k,2] = vs2_l[k,2]*tt_t2 + vs2_l[k,3]*dt_t2 +
                    vs2_l[k,4]*ut_t2 + vs2_l[k,5]*xt_t2
        bl2[k,3] = vs2_l[k,2]*tt_d2 + vs2_l[k,3]*dt_d2 +
                    vs2_l[k,4]*ut_d2 + vs2_l[k,5]*xt_d2
        bl2[k,4] = vs2_l[k,2]*tt_u2 + vs2_l[k,3]*dt_u2 +
                    vs2_l[k,4]*ut_u2 + vs2_l[k,5]*xt_u2
        bl2[k,5] = vs2_l[k,2]*tt_x2 + vs2_l[k,3]*dt_x2 +
                    vs2_l[k,4]*ut_x2 + vs2_l[k,5]*xt_x2
    end

    # ── TURBULENT PART: XT → X2 (TRDIF lines 1406-1539) ──────────────
    # Compute equilibrium shear coefficient CQ at transition point
    hk_t = hkin(_FT(dt / max(tt, eps)), zero(_FT))
    hk_t = max(hk_t, _FT(1.05))

    # CTR = CTRCON * exp(-CTRCEX/(HK-1))
    ctrcon = _FT(model.ctrcon)
    ctrcex = _FT(model.ctrcex)
    ctr = ctrcon * exp(-ctrcex / max(hk_t - one(_FT), _FT(0.01)))

    # Get CQ at transition from laminar closure (BLVAR(2) in Fortran)
    pt_turb_seed = BLPrimaryState(_FT(xt), _FT(ut), _FT(tt), _FT(dt);
        s = zero(_FT), ampl = zero(_FT), dw = zero(_FT))
    sec_t = blsecondary_state(pt_turb_seed, :turbulent; flow = flow, model = model)
    cq_t = sec_t.cq

    # Initial shear: ST = CTR * CQ
    st = ctr * max(cq_t, _FT(1e-10))

    # Build turbulent primary state at XT with initial shear ST
    pt_turb = BLPrimaryState(_FT(xt), _FT(ut), _FT(tt), _FT(dt);
        s = st, ampl = zero(_FT), dw = zero(_FT))
    p2_ft = BLPrimaryState(_FT(p2.x), _FT(p2.u), _FT(p2.t), _FT(p2.d);
        s = _FT(p2.s), ampl = zero(_FT), dw = _FT(p2.dw))

    vs1_t, vs2_t, vsrez_t, vsm_t, vsr_t = _bldif_analytic(
        :turbulent, pt_turb, p2_ft;
        flow = flow, model = model, acrit = acrit, idampv = idampv, precision = _FT)

    # ST sensitivities wrt "T" variables (via FD on CTR*CQ)
    fd_st = _FT(1e-6)
    function get_st(tt_v, dt_v, ut_v)
        hk = hkin(_FT(dt_v / max(tt_v, eps)), zero(_FT))
        hk = max(hk, _FT(1.05))
        c = ctrcon * exp(-ctrcex / max(hk - one(_FT), _FT(0.01)))
        pt_s = BLPrimaryState(_FT(xt), _FT(ut_v), _FT(tt_v), _FT(dt_v);
            s = zero(_FT), ampl = zero(_FT), dw = zero(_FT))
        sec_s = blsecondary_state(pt_s, :turbulent; flow = flow, model = model)
        return c * max(sec_s.cq, _FT(1e-10))
    end
    st_tt = _FT((get_st(tt+fd_st, dt, ut) - get_st(tt-fd_st, dt, ut)) / (2*fd_st))
    st_dt = _FT((get_st(tt, dt+fd_st, ut) - get_st(tt, dt-fd_st, ut)) / (2*fd_st))
    st_ut = _FT((get_st(tt, dt, ut+fd_st) - get_st(tt, dt, ut-fd_st)) / (2*fd_st))
    st_ms = zero(_FT)
    st_re = zero(_FT)

    # ST sensitivities wrt actual "1" and "2" variables
    st_a1_v = st_tt*tt_a1 + st_dt*dt_a1 + st_ut*ut_a1
    st_x1_v = st_tt*tt_x1 + st_dt*dt_x1 + st_ut*ut_x1
    st_x2_v = st_tt*tt_x2 + st_dt*dt_x2 + st_ut*ut_x2
    st_t1_v = st_tt*tt_t1 + st_dt*dt_t1 + st_ut*ut_t1
    st_t2_v = st_tt*tt_t2 + st_dt*dt_t2 + st_ut*ut_t2
    st_d1_v = st_tt*tt_d1 + st_dt*dt_d1 + st_ut*ut_d1
    st_d2_v = st_tt*tt_d2 + st_dt*dt_d2 + st_ut*ut_d2
    st_u1_v = st_tt*tt_u1 + st_dt*dt_u1 + st_ut*ut_u1
    st_u2_v = st_tt*tt_u2 + st_dt*dt_u2 + st_ut*ut_u2

    # Transform VS1 (wrt "T" vars) to BT1,BT2 (wrt "1","2" vars)
    bt1 = zeros(_FT, 3, 5)
    bt2 = zeros(_FT, 3, 5)
    btrez = zeros(_FT, 3)
    btm = zeros(_FT, 3)
    btr = zeros(_FT, 3)

    for k in 1:3
        btrez[k] = vsrez_t[k]
        btm[k] = vsm_t[k] + vs1_t[k,1]*st_ms + vs1_t[k,2]*tt_ms +
                  vs1_t[k,3]*dt_ms + vs1_t[k,4]*ut_ms + vs1_t[k,5]*xt_ms
        btr[k] = vsr_t[k] + vs1_t[k,1]*st_re + vs1_t[k,2]*tt_re +
                  vs1_t[k,3]*dt_re + vs1_t[k,4]*ut_re + vs1_t[k,5]*xt_re

        bt1[k,1] = vs1_t[k,1]*st_a1_v + vs1_t[k,2]*tt_a1 + vs1_t[k,3]*dt_a1 +
                    vs1_t[k,4]*ut_a1 + vs1_t[k,5]*xt_a1
        bt1[k,2] = vs1_t[k,1]*st_t1_v + vs1_t[k,2]*tt_t1 + vs1_t[k,3]*dt_t1 +
                    vs1_t[k,4]*ut_t1 + vs1_t[k,5]*xt_t1
        bt1[k,3] = vs1_t[k,1]*st_d1_v + vs1_t[k,2]*tt_d1 + vs1_t[k,3]*dt_d1 +
                    vs1_t[k,4]*ut_d1 + vs1_t[k,5]*xt_d1
        bt1[k,4] = vs1_t[k,1]*st_u1_v + vs1_t[k,2]*tt_u1 + vs1_t[k,3]*dt_u1 +
                    vs1_t[k,4]*ut_u1 + vs1_t[k,5]*xt_u1
        bt1[k,5] = vs1_t[k,1]*st_x1_v + vs1_t[k,2]*tt_x1 + vs1_t[k,3]*dt_x1 +
                    vs1_t[k,4]*ut_x1 + vs1_t[k,5]*xt_x1

        bt2[k,1] = vs2_t[k,1]
        bt2[k,2] = vs2_t[k,2] + vs1_t[k,1]*st_t2_v + vs1_t[k,2]*tt_t2 +
                    vs1_t[k,3]*dt_t2 + vs1_t[k,4]*ut_t2 + vs1_t[k,5]*xt_t2
        bt2[k,3] = vs2_t[k,3] + vs1_t[k,1]*st_d2_v + vs1_t[k,2]*tt_d2 +
                    vs1_t[k,3]*dt_d2 + vs1_t[k,4]*ut_d2 + vs1_t[k,5]*xt_d2
        bt2[k,4] = vs2_t[k,4] + vs1_t[k,1]*st_u2_v + vs1_t[k,2]*tt_u2 +
                    vs1_t[k,3]*dt_u2 + vs1_t[k,4]*ut_u2 + vs1_t[k,5]*xt_u2
        bt2[k,5] = vs2_t[k,5] + vs1_t[k,1]*st_x2_v + vs1_t[k,2]*tt_x2 +
                    vs1_t[k,3]*dt_x2 + vs1_t[k,4]*ut_x2 + vs1_t[k,5]*xt_x2
    end

    # ── Sum laminar + turbulent parts (TRDIF lines 1541-1562) ─────────
    vs1 = zeros(_FT, 3, 5)
    vs2 = zeros(_FT, 3, 5)
    vsrez = zeros(_FT, 3)
    vsm = zeros(_FT, 3)
    vsr = zeros(_FT, 3)

    # Equation 1 (shear lag): turbulent part only
    vsrez[1] = btrez[1]
    vsm[1]   = btm[1]
    vsr[1]   = btr[1]
    vs1[1,:] .= bt1[1,:]
    vs2[1,:] .= bt2[1,:]

    # Equations 2,3 (momentum, shape): sum of laminar + turbulent
    for k in 2:3
        vsrez[k] = blrez[k] + btrez[k]
        vsm[k]   = blm[k]   + btm[k]
        vsr[k]   = blr[k]   + btr[k]
        for l in 1:5
            vs1[k,l] = bl1[k,l] + bt1[k,l]
            vs2[k,l] = bl2[k,l] + bt2[k,l]
        end
    end

    return vs1, vs2, vsrez, vsm, vsr
end
