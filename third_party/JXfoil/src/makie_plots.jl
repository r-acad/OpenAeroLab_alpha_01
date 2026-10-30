using CairoMakie

const _PLOT_BLUE = "#1f5aa6"
const _PLOT_ORANGE = "#d97904"
const _PLOT_RED = "#b83b2f"
const _PLOT_GREEN = "#2c7a5a"
const _PLOT_GOLD = "#b39200"
const _PLOT_GRAY = "#5f6b76"
const _PLOT_LIGHT = "#d7dde3"

@inline function _plot_field(row, primary::Symbol, fallback::Symbol = primary)
    if hasproperty(row, primary)
        return float(getproperty(row, primary))
    elseif hasproperty(row, fallback)
        return float(getproperty(row, fallback))
    end
    throw(ArgumentError("row is missing fields $(primary) and $(fallback)"))
end

@inline _row_alpha(row) = _plot_field(row, :alpha_deg)
@inline _row_cl(row) = _plot_field(row, :cl)
@inline _row_cd(row) = _plot_field(row, :cd)
@inline _row_cm(row) = hasproperty(row, :cm_c4) ? float(getproperty(row, :cm_c4)) : _plot_field(row, :cm)
@inline _row_top_xtr(row) = _plot_field(row, :top_xtr)
@inline _row_bot_xtr(row) = _plot_field(row, :bot_xtr)

@inline function _surface_primary(surface::SurfaceMarchResult, field::Symbol)
    return [float(getfield(p, field)) for p in surface.primary]
end

@inline _surface_cp(surface::SurfaceMarchResult) = [1.0 - float(u)^2 for u in surface.ue]

function _apply_publication_theme!()
    set_theme!(
        Theme(
            fontsize = 16,
            backgroundcolor = :white,
            Axis = (
                backgroundcolor = :white,
                xgridcolor = (_PLOT_LIGHT, 0.55),
                ygridcolor = (_PLOT_LIGHT, 0.55),
                leftspinecolor = _PLOT_GRAY,
                bottomspinecolor = _PLOT_GRAY,
                topspinevisible = false,
                rightspinevisible = false,
                xtickcolor = _PLOT_GRAY,
                ytickcolor = _PLOT_GRAY,
                xlabelcolor = _PLOT_GRAY,
                ylabelcolor = _PLOT_GRAY,
                xtrimspine = true,
                ytrimspine = true,
            ),
            Legend = (
                framevisible = false,
                backgroundcolor = :transparent,
            ),
        ),
    )
    return nothing
end

function _save_publication_figure(path::AbstractString, fig::Figure)
    mkpath(dirname(path))
    save(path, fig)
    return path
end

function _leading_edge_window(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    mask = findall(xi -> float(xi) <= 0.12, x)
    isempty(mask) && return (-0.01, 0.12, -0.08, 0.08)
    xx = Float64[x[i] for i in mask]
    yy = Float64[y[i] for i in mask]
    xpad = 0.01
    yspan = max(maximum(yy) - minimum(yy), 0.08)
    ypad = 0.2 * yspan
    return minimum(xx) - xpad, maximum(xx) + xpad, minimum(yy) - ypad, maximum(yy) + ypad
end

function _summary_title(result::CoupledCaseResult)
    return "alpha = $(round(result.alpha_deg; digits = 2)) deg, Re = $(round(result.reynolds; digits = 0)), " *
           "CL = $(round(result.cl; digits = 4)), CD = $(round(result.cd; digits = 5))"
end

function _segment_label(segment::Symbol)
    s = replace(String(segment), "_" => " ")
    return uppercasefirst(s)
end

function _trace_boundaries(samples::Vector{CoupledIterationSample})
    boundaries = Int[]
    isempty(samples) && return boundaries
    prev = samples[1].segment
    for sample in Iterators.drop(samples, 1)
        if sample.segment != prev
            push!(boundaries, sample.global_iteration)
            prev = sample.segment
        end
    end
    return boundaries
end

function _surface_transition_line!(ax::Axis, xtr::Real, color)
    isfinite(float(xtr)) || return
    vlines!(ax, [float(xtr)]; color = (color, 0.4), linestyle = :dot, linewidth = 1.5)
    return nothing
end

function _finite_xy(x::AbstractVector, y::AbstractVector)
    xx = Float64[]
    yy = Float64[]
    for i in eachindex(x, y)
        xi = float(x[i])
        yi = float(y[i])
        if isfinite(xi) && isfinite(yi)
            push!(xx, xi)
            push!(yy, yi)
        end
    end
    return xx, yy
end

function _sweep_segments(
    alpha::AbstractVector,
    cl::AbstractVector,
    cd::AbstractVector;
    alpha_gap_limit::Real = 1.25,
    cl_jump_limit::Real = Inf,
    cd_jump_limit::Real = Inf,
)
    segments = NamedTuple[]
    cur_alpha = Float64[]
    cur_cl = Float64[]
    cur_cd = Float64[]

    function flush_segment!()
        isempty(cur_alpha) && return
        push!(segments, (alpha = copy(cur_alpha), cl = copy(cur_cl), cd = copy(cur_cd)))
        empty!(cur_alpha)
        empty!(cur_cl)
        empty!(cur_cd)
    end

    for i in eachindex(alpha, cl, cd)
        ai = float(alpha[i])
        cli = float(cl[i])
        cdi = float(cd[i])
        if !(isfinite(ai) && isfinite(cli) && isfinite(cdi))
            flush_segment!()
            continue
        end

        if !isempty(cur_alpha)
            alpha_gap = abs(ai - cur_alpha[end])
            cl_jump = abs(cli - cur_cl[end])
            cd_jump = abs(cdi - cur_cd[end])
            if alpha_gap > float(alpha_gap_limit) ||
               cl_jump > float(cl_jump_limit) ||
               cd_jump > float(cd_jump_limit)
                flush_segment!()
            end
        end

        push!(cur_alpha, ai)
        push!(cur_cl, cli)
        push!(cur_cd, cdi)
    end

    flush_segment!()
    return segments
end

function _plot_segmented_cdcl!(
    ax::Axis,
    segments;
    color,
    linewidth::Real = 2.0,
    linestyle = :solid,
    label::Union{Nothing,AbstractString} = nothing,
    show_markers::Bool = false,
    markersize::Real = 6.0,
)
    first_label = true
    for seg in segments
        if length(seg.alpha) >= 2
            if first_label && !isnothing(label)
                lines!(ax, seg.cd, seg.cl; color = color, linewidth = linewidth, linestyle = linestyle, label = label)
                first_label = false
            else
                lines!(ax, seg.cd, seg.cl; color = color, linewidth = linewidth, linestyle = linestyle)
            end
        end

        if show_markers || length(seg.alpha) == 1
            if first_label && !isnothing(label) && length(seg.alpha) == 1
                scatter!(ax, seg.cd, seg.cl; color = color, markersize = markersize, label = label)
                first_label = false
            else
                scatter!(ax, seg.cd, seg.cl; color = color, markersize = markersize)
            end
        end
    end
    return nothing
end

function _plot_segmented_alphacl!(
    ax::Axis,
    segments;
    color,
    linewidth::Real = 2.0,
    linestyle = :solid,
    label::Union{Nothing,AbstractString} = nothing,
    show_markers::Bool = false,
    markersize::Real = 6.0,
)
    first_label = true
    for seg in segments
        if length(seg.alpha) >= 2
            if first_label && !isnothing(label)
                lines!(ax, seg.alpha, seg.cl; color = color, linewidth = linewidth, linestyle = linestyle, label = label)
                first_label = false
            else
                lines!(ax, seg.alpha, seg.cl; color = color, linewidth = linewidth, linestyle = linestyle)
            end
        end

        if show_markers || length(seg.alpha) == 1
            if first_label && !isnothing(label) && length(seg.alpha) == 1
                scatter!(ax, seg.alpha, seg.cl; color = color, markersize = markersize, label = label)
                first_label = false
            else
                scatter!(ax, seg.alpha, seg.cl; color = color, markersize = markersize)
            end
        end
    end
    return nothing
end

function _uiuc_ecdf(values; floor::Real = 1.0e-8)
    vals = sort!([max(abs(float(v)), float(floor)) for v in values if isfinite(float(v))])
    isempty(vals) && return ([float(floor)], [0.0])
    return vals, collect(range(1 / length(vals), 1.0; length = length(vals)))
end

function save_airfoil_geometry_figure(
    path::AbstractString,
    airfoil::Airfoil;
    repanel::Bool = true,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
    title::AbstractString = "Airfoil geometry and XFoil-style panel nodes",
)
    _apply_publication_theme!()
    inv = solve_inviscid_case(
        airfoil,
        0.0;
        repanel = repanel,
        xfoil_load_normalize_digits = xfoil_load_normalize_digits,
    )
    raw_x, raw_y = _xfoil_nodes(airfoil)
    x1, x2, y1, y2 = _leading_edge_window(inv.xnode, inv.ynode)

    fig = Figure(size = (1040, 440))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)

    ax_full = Axis(fig[1, 1]; xlabel = "x/c", ylabel = "y/c", aspect = DataAspect())
    lines!(ax_full, raw_x, raw_y; color = (_PLOT_GRAY, 0.6), linewidth = 2.0, label = "input geometry")
    lines!(ax_full, inv.xnode, inv.ynode; color = _PLOT_BLUE, linewidth = 1.7, label = "paneled geometry")
    scatter!(ax_full, inv.xnode, inv.ynode; color = _PLOT_BLUE, markersize = 5)
    axislegend(ax_full; position = :lb)

    ax_zoom = Axis(fig[1, 2]; xlabel = "x/c", ylabel = "y/c", aspect = DataAspect())
    lines!(ax_zoom, raw_x, raw_y; color = (_PLOT_GRAY, 0.6), linewidth = 2.0)
    lines!(ax_zoom, inv.xnode, inv.ynode; color = _PLOT_BLUE, linewidth = 1.7)
    scatter!(ax_zoom, inv.xnode, inv.ynode; color = _PLOT_BLUE, markersize = 5)
    xlims!(ax_zoom, x1, x2)
    ylims!(ax_zoom, y1, y2)

    return _save_publication_figure(path, fig)
end

function save_airfoil_gallery_figure(
    path::AbstractString,
    airfoils::AbstractVector{<:Airfoil};
    titles::AbstractVector{<:AbstractString} = String[],
    title::AbstractString = "Selected airfoils",
)
    isempty(airfoils) && throw(ArgumentError("airfoils cannot be empty"))
    _apply_publication_theme!()

    n = length(airfoils)
    ncols = min(2, n)
    nrows = cld(n, ncols)
    fig = Figure(size = (1040, 420 * nrows))
    Label(fig[0, 1:ncols], title; fontsize = 18, color = _PLOT_GRAY)

    for (k, airfoil) in enumerate(airfoils)
        row = cld(k, ncols)
        col = ((k - 1) % ncols) + 1
        ax = Axis(
            fig[row, col];
            xlabel = "x/c",
            ylabel = "y/c",
            aspect = DataAspect(),
            title = isempty(titles) ? airfoil.name : String(titles[k]),
        )
        lines!(ax, airfoil.x, airfoil.y; color = _PLOT_BLUE, linewidth = 2.1)
        xlims!(ax, -0.02, 1.02)
    end

    return _save_publication_figure(path, fig)
end

function save_operating_point_summary_figure(
    path::AbstractString,
    result::CoupledCaseResult;
    title::AbstractString = _summary_title(result),
)
    _apply_publication_theme!()
    inv = result.inviscid
    upper = result.upper
    lower = result.lower

    upper_cp = _surface_cp(upper)
    lower_cp = _surface_cp(lower)
    upper_d = _surface_primary(upper, :d)
    lower_d = _surface_primary(lower, :d)
    upper_t = _surface_primary(upper, :t)
    lower_t = _surface_primary(lower, :t)

    cp_all = vcat(upper_cp, lower_cp)
    cp_pad = 0.12 * max(maximum(cp_all) - minimum(cp_all), 1.0)
    thick_all = vcat(upper_d, lower_d, upper_t, lower_t)
    thick_pad = 0.08 * max(maximum(thick_all), 1.0e-4)

    fig = Figure(size = (1040, 840))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)

    ax_geo = Axis(fig[1, 1]; xlabel = "x/c", ylabel = "y/c", aspect = DataAspect(), title = "Paneled geometry")
    lines!(ax_geo, inv.xnode, inv.ynode; color = _PLOT_BLUE, linewidth = 1.8)
    scatter!(ax_geo, inv.xnode, inv.ynode; color = _PLOT_BLUE, markersize = 4)

    ax_cp = Axis(fig[1, 2]; xlabel = "x/c", ylabel = "Cp", title = "Pressure coefficient")
    lines!(ax_cp, upper.xc, upper_cp; color = _PLOT_BLUE, linewidth = 2.2, label = "upper")
    lines!(ax_cp, lower.xc, lower_cp; color = _PLOT_ORANGE, linewidth = 2.2, label = "lower")
    _surface_transition_line!(ax_cp, result.top_xtr, _PLOT_BLUE)
    _surface_transition_line!(ax_cp, result.bot_xtr, _PLOT_ORANGE)
    ylims!(ax_cp, maximum(cp_all) + cp_pad, minimum(cp_all) - cp_pad)
    axislegend(ax_cp; position = :rb)

    ax_ue = Axis(fig[2, 1]; xlabel = "x/c", ylabel = "Ue / Uinf", title = "Edge velocity")
    lines!(ax_ue, upper.xc, upper.ue; color = _PLOT_BLUE, linewidth = 2.2, label = "upper")
    lines!(ax_ue, lower.xc, lower.ue; color = _PLOT_ORANGE, linewidth = 2.2, label = "lower")
    _surface_transition_line!(ax_ue, result.top_xtr, _PLOT_BLUE)
    _surface_transition_line!(ax_ue, result.bot_xtr, _PLOT_ORANGE)
    axislegend(ax_ue; position = :rt)

    ax_bl = Axis(fig[2, 2]; xlabel = "x/c", ylabel = "Thickness / c", title = "Boundary-layer thickness")
    lines!(ax_bl, upper.xc, upper_d; color = _PLOT_BLUE, linewidth = 2.2, label = "upper delta*")
    lines!(ax_bl, upper.xc, upper_t; color = (_PLOT_BLUE, 0.8), linestyle = :dash, linewidth = 2.0, label = "upper theta")
    lines!(ax_bl, lower.xc, lower_d; color = _PLOT_ORANGE, linewidth = 2.2, label = "lower delta*")
    lines!(ax_bl, lower.xc, lower_t; color = (_PLOT_ORANGE, 0.8), linestyle = :dash, linewidth = 2.0, label = "lower theta")
    _surface_transition_line!(ax_bl, result.top_xtr, _PLOT_BLUE)
    _surface_transition_line!(ax_bl, result.bot_xtr, _PLOT_ORANGE)
    ylims!(ax_bl, -0.02 * maximum(thick_all), maximum(thick_all) + thick_pad)
    axislegend(ax_bl; position = :rt)

    return _save_publication_figure(path, fig)
end

function save_operating_point_summary_figure(
    path::AbstractString,
    _airfoil::Airfoil,
    result::CoupledCaseResult;
    title::AbstractString = _summary_title(result),
)
    return save_operating_point_summary_figure(path, result; title = title)
end

function save_low_re_frontier_comparison_figure(
    path::AbstractString,
    cases;
    title::AbstractString = "Low-Reynolds-number frontier comparison",
)
    rows = collect(cases)
    isempty(rows) && throw(ArgumentError("cases cannot be empty"))
    _apply_publication_theme!()

    labels = [String(getproperty(row, :label)) for row in rows]
    xf_status = [String(getproperty(row, :xf_status)) for row in rows]
    jx_status = [String(getproperty(row, :jx_status)) for row in rows]
    cl_xf = [float(getproperty(row, :cl_xf)) for row in rows]
    cl_jx = [float(getproperty(row, :cl_jx)) for row in rows]
    dcd = [max(abs(float(getproperty(row, :dcd))), 1.0e-8) for row in rows]
    dcm = [max(abs(float(getproperty(row, :dcm))), 1.0e-8) for row in rows]
    dxtr = [max(abs(float(getproperty(row, :dxtr))), 1.0e-8) for row in rows]
    idx = collect(1:length(rows))

    ymax = maximum(vcat(cl_xf, cl_jx))
    ymin = minimum(vcat(cl_xf, cl_jx))
    ypad = 0.18 * max(ymax - ymin, 0.2)

    fig = Figure(size = (1080, 860))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)
    Label(
        fig[1, 1:2, Top()],
        "All selected cases are low-Re XFoil nonconverged states from the frontier benchmark; JXfoil values are the recovered reference states.",
        fontsize = 12,
        color = _PLOT_GRAY,
    )

    ax_cl = Axis(fig[1, 1]; ylabel = "CL", title = "Lift coefficient")
    barplot!(ax_cl, idx .- 0.18, cl_xf; color = _PLOT_ORANGE, width = 0.32, label = "XFoil")
    barplot!(ax_cl, idx .+ 0.18, cl_jx; color = _PLOT_BLUE, width = 0.32, label = "JXfoil")
    ax_cl.xticks = (idx, labels)
    ax_cl.xticklabelrotation = pi / 8
    ylims!(ax_cl, ymin - ypad, ymax + ypad)
    axislegend(ax_cl; position = :rt)
    for i in eachindex(idx)
        text!(
            ax_cl,
            idx[i] - 0.18,
            ymax + 0.45 * ypad;
            text = xf_status[i],
            align = (:center, :center),
            fontsize = 10,
            color = _PLOT_ORANGE,
        )
        text!(
            ax_cl,
            idx[i] + 0.18,
            ymax + 0.17 * ypad;
            text = jx_status[i],
            align = (:center, :center),
            fontsize = 10,
            color = _PLOT_BLUE,
        )
    end

    ax_dcd = Axis(fig[1, 2]; ylabel = "|delta CD|", title = "Drag discrepancy", yscale = log10)
    barplot!(ax_dcd, idx, dcd; color = _PLOT_RED, width = 0.55)
    ax_dcd.xticks = (idx, labels)
    ax_dcd.xticklabelrotation = pi / 8

    ax_dcm = Axis(fig[2, 1]; ylabel = "|delta Cm,c/4|", title = "Moment discrepancy")
    barplot!(ax_dcm, idx, dcm; color = _PLOT_GREEN, width = 0.55)
    ax_dcm.xticks = (idx, labels)
    ax_dcm.xticklabelrotation = pi / 8

    ax_dxtr = Axis(fig[2, 2]; ylabel = "|delta xtr|max", title = "Transition discrepancy")
    barplot!(ax_dxtr, idx, dxtr; color = _PLOT_GOLD, width = 0.55)
    ax_dxtr.xticks = (idx, labels)
    ax_dxtr.xticklabelrotation = pi / 8

    return _save_publication_figure(path, fig)
end

function save_low_re_campaign_overview_figure(
    path::AbstractString,
    summary;
    title::AbstractString = "UIUC low-Reynolds-number campaign overview",
)
    _apply_publication_theme!()
    total_cases = Int(getproperty(summary, :total_cases))
    healthy = Int(getproperty(summary, :healthy_references))
    frontier = Int(getproperty(summary, :frontier_cases))
    converged = Int(getproperty(summary, :julia_converged))
    failures = Int(getproperty(summary, :julia_hard_failures))
    shared = Int(getproperty(summary, :shared_load_failures))
    timeout_wins = Int(getproperty(summary, :ideal_xf_timeout))
    xf_safe = Int(getproperty(summary, :xf_converged_jx_nonconverged))
    alt_branches = hasproperty(summary, :alt_xtr_branch) ? Int(getproperty(summary, :alt_xtr_branch)) : 0
    frontier_sanity = hasproperty(summary, :frontier_sanity) ? getproperty(summary, :frontier_sanity) : nothing
    phys_ok = frontier_sanity === nothing ? 0 :
        (hasproperty(frontier_sanity, :phys_ok) ? Int(getproperty(frontier_sanity, :phys_ok)) : 0)
    healthy_preserved = max(healthy - xf_safe, 0)
    frontier_recovered = max(converged - healthy_preserved, 0)
    frontier_unresolved = max(frontier - frontier_recovered, 0)
    pct(x) = 100.0 * float(x) / max(total_cases, 1)

    fig = Figure(size = (1180, 860))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)
    Label(
        fig[1, 1:2],
        "Bars report percentages of the total corpus; count labels are shown in parentheses. Panels 3 and 4 include diagnostic subsets in addition to net outcomes.",
        fontsize = 12,
        color = _PLOT_GRAY,
    )

    ax_split = Axis(fig[2, 1]; ylabel = "Percent of total cases", title = "Reference split")
    split_pct = [pct(healthy), pct(frontier)]
    barplot!(ax_split, [1, 2], split_pct; color = [_PLOT_BLUE, _PLOT_ORANGE], width = 0.6)
    ax_split.xticks = ([1, 2], ["Healthy XFoil refs", "Frontier cases"])
    for (x, count, y) in zip([1, 2], [healthy, frontier], split_pct)
        text!(ax_split, x, y + 1.2; text = string(round(y; digits = 1), "%\n(", count, ")"), align = (:center, :bottom), fontsize = 12)
    end
    ylims!(ax_split, 0.0, 100.0)

    ax_outcomes = Axis(fig[2, 2]; ylabel = "Percent of total cases", title = "Julia outcomes")
    outcome_counts = [converged, failures, shared]
    outcome_pct = [pct(v) for v in outcome_counts]
    barplot!(ax_outcomes, [1, 2, 3], outcome_pct; color = [_PLOT_BLUE, _PLOT_RED, _PLOT_GRAY], width = 0.6)
    ax_outcomes.xticks = ([1, 2, 3], ["Converged", "Hard fail", "Shared load fail"])
    for (x, count, y) in zip([1, 2, 3], outcome_counts, outcome_pct)
        text!(ax_outcomes, x, y + 1.2; text = string(round(y; digits = 1), "%\n(", count, ")"), align = (:center, :bottom), fontsize = 12)
    end
    ylims!(ax_outcomes, 0.0, 100.0)

    ax_frontier = Axis(fig[3, 1]; ylabel = "Percent of total cases", title = "Reliability impact")
    impact_counts = [healthy_preserved, frontier_recovered, timeout_wins, xf_safe]
    impact_pct = [pct(v) for v in impact_counts]
    barplot!(ax_frontier, [1, 2, 3, 4], impact_pct; color = [_PLOT_BLUE, _PLOT_GREEN, _PLOT_GOLD, _PLOT_RED], width = 0.55)
    ax_frontier.xticks = ([1, 2, 3, 4], ["Healthy refs preserved", "Frontier recovered", "XF timeout recoveries", "Healthy regressions"])
    for (x, count, y) in zip([1, 2, 3, 4], impact_counts, impact_pct)
        text!(ax_frontier, x, y + 1.2; text = string(round(y; digits = 1), "%\n(", count, ")"), align = (:center, :bottom), fontsize = 12)
    end
    ylims!(ax_frontier, 0.0, 100.0)

    ax_diag = Axis(fig[3, 2]; ylabel = "Percent of total cases", title = "Frontier diagnostics")
    diag_counts = [frontier_unresolved, phys_ok, alt_branches, shared]
    diag_pct = [pct(v) for v in diag_counts]
    barplot!(ax_diag, [1, 2, 3, 4], diag_pct; color = [_PLOT_ORANGE, _PLOT_GREEN, _PLOT_BLUE, _PLOT_GRAY], width = 0.55)
    ax_diag.xticks = ([1, 2, 3, 4], ["Frontier unresolved", "Physically acceptable", "Alt transition branch", "Shared load fail"])
    for (x, count, y) in zip([1, 2, 3, 4], diag_counts, diag_pct)
        text!(ax_diag, x, y + 1.2; text = string(round(y; digits = 1), "%\n(", count, ")"), align = (:center, :bottom), fontsize = 12)
    end
    ylims!(ax_diag, 0.0, 100.0)

    return _save_publication_figure(path, fig)
end

function save_uiuc_raw_kulfan_comparison_figure(
    path::AbstractString,
    summary;
    title::AbstractString = "Matched UIUC raw-vs-Kulfan parity statistics",
)
    _apply_publication_theme!()
    total_cases = Int(getproperty(summary, :total_cases))
    raw = getproperty(summary, :raw)
    kulfan = getproperty(summary, :kulfan)
    pct(x) = 100.0 * float(x) / max(total_cases, 1)

    fig = Figure(size = (1180, 860))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)
    Label(
        fig[1, 1:2],
        "Both variants use the same executed UIUC corpus; bars report percentages of that matched set.",
        fontsize = 12,
        color = _PLOT_GRAY,
    )

    offsets = (-0.18, 0.18)
    variant_labels = ("Raw mesh", "Kulfan rebuild")
    variant_colors = (_PLOT_BLUE, _PLOT_ORANGE)

    ax_refs = Axis(fig[2, 1]; ylabel = "Percent of matched cases", title = "XFoil reference split")
    ref_categories = ["Healthy refs", "Frontier cases"]
    ref_positions = [1, 2]
    ref_counts = (
        [Int(getproperty(raw, :xf_healthy)), Int(getproperty(raw, :xf_frontier))],
        [Int(getproperty(kulfan, :xf_healthy)), Int(getproperty(kulfan, :xf_frontier))],
    )
    for idx in 1:2
        xpos = ref_positions .+ offsets[idx]
        vals = pct.(ref_counts[idx])
        barplot!(ax_refs, xpos, vals; color = variant_colors[idx], width = 0.32, label = variant_labels[idx])
        for j in eachindex(xpos)
            text!(ax_refs, xpos[j], vals[j] + 1.2; text = string(ref_counts[idx][j]), align = (:center, :bottom), fontsize = 11, color = variant_colors[idx])
        end
    end
    ax_refs.xticks = (ref_positions, ref_categories)
    ylims!(ax_refs, 0.0, 100.0)
    axislegend(ax_refs; position = :rt)

    ax_jx = Axis(fig[2, 2]; ylabel = "Percent of matched cases", title = "JXfoil outcomes")
    jx_categories = ["Converged", "Best", "Fail"]
    jx_positions = [1, 2, 3]
    jx_counts = (
        [Int(getproperty(raw, :jx_converged)), Int(getproperty(raw, :jx_best)), Int(getproperty(raw, :jx_fail))],
        [Int(getproperty(kulfan, :jx_converged)), Int(getproperty(kulfan, :jx_best)), Int(getproperty(kulfan, :jx_fail))],
    )
    for idx in 1:2
        xpos = jx_positions .+ offsets[idx]
        vals = pct.(jx_counts[idx])
        barplot!(ax_jx, xpos, vals; color = variant_colors[idx], width = 0.32, label = variant_labels[idx])
        for j in eachindex(xpos)
            text!(ax_jx, xpos[j], vals[j] + 1.2; text = string(jx_counts[idx][j]), align = (:center, :bottom), fontsize = 11, color = variant_colors[idx])
        end
    end
    ax_jx.xticks = (jx_positions, jx_categories)
    ylims!(ax_jx, 0.0, 100.0)

    ax_frontier = Axis(fig[3, 1]; ylabel = "Percent of matched cases", title = "Frontier handling")
    frontier_categories = ["Recovered frontier", "Healthy regressions"]
    frontier_positions = [1, 2]
    frontier_counts = (
        [Int(getproperty(raw, :frontier_recovered)), Int(getproperty(raw, :healthy_regressions))],
        [Int(getproperty(kulfan, :frontier_recovered)), Int(getproperty(kulfan, :healthy_regressions))],
    )
    for idx in 1:2
        xpos = frontier_positions .+ offsets[idx]
        vals = pct.(frontier_counts[idx])
        barplot!(ax_frontier, xpos, vals; color = variant_colors[idx], width = 0.32, label = variant_labels[idx])
        for j in eachindex(xpos)
            text!(ax_frontier, xpos[j], vals[j] + 1.2; text = string(frontier_counts[idx][j]), align = (:center, :bottom), fontsize = 11, color = variant_colors[idx])
        end
    end
    ax_frontier.xticks = (frontier_positions, frontier_categories)
    ylims!(ax_frontier, 0.0, max(12.0, maximum(vcat(pct.(frontier_counts[1]), pct.(frontier_counts[2]))) + 4.0))

    ax_err = Axis(fig[3, 2]; ylabel = "Healthy-reference maximum", title = "Worst healthy-reference errors", yscale = log10)
    err_categories = ["|dCL|", "|dCD|", "|dXtr|"]
    err_positions = [1, 2, 3]
    err_counts = (
        [max(float(getproperty(raw, :max_dcl_healthy)), 1.0e-8), max(float(getproperty(raw, :max_dcd_healthy)), 1.0e-8), max(float(getproperty(raw, :max_dxtr_healthy)), 1.0e-8)],
        [max(float(getproperty(kulfan, :max_dcl_healthy)), 1.0e-8), max(float(getproperty(kulfan, :max_dcd_healthy)), 1.0e-8), max(float(getproperty(kulfan, :max_dxtr_healthy)), 1.0e-8)],
    )
    for idx in 1:2
        xpos = err_positions .+ offsets[idx]
        barplot!(ax_err, xpos, err_counts[idx]; color = variant_colors[idx], width = 0.32, label = variant_labels[idx])
    end
    ax_err.xticks = (err_positions, err_categories)

    return _save_publication_figure(path, fig)
end

function save_uiuc_raw_kulfan_shift_figure(
    path::AbstractString,
    rows;
    title::AbstractString = "Matched UIUC raw-vs-Kulfan parity shifts",
)
    items = collect(rows)
    isempty(items) && throw(ArgumentError("rows cannot be empty"))
    _apply_publication_theme!()

    raw_dcl = [float(getproperty(row, :raw_dcl)) for row in items if String(getproperty(row, :raw_xf_status)) == "XF" && isfinite(float(getproperty(row, :raw_dcl)))]
    kulfan_dcl = [float(getproperty(row, :kulfan_dcl)) for row in items if String(getproperty(row, :kulfan_xf_status)) == "XF" && isfinite(float(getproperty(row, :kulfan_dcl)))]
    raw_ecdf_x, raw_ecdf_y = _uiuc_ecdf(raw_dcl; floor = 1.0e-7)
    kulfan_ecdf_x, kulfan_ecdf_y = _uiuc_ecdf(kulfan_dcl; floor = 1.0e-7)

    both_finite = [
        row for row in items
        if isfinite(float(getproperty(row, :raw_cl_jx))) && isfinite(float(getproperty(row, :kulfan_cl_jx)))
    ]
    scatter_groups = (
        (
            "Healthy both",
            [row for row in both_finite if String(getproperty(row, :raw_xf_status)) == "XF" && String(getproperty(row, :kulfan_xf_status)) == "XF"],
            _PLOT_BLUE,
        ),
        (
            "Raw frontier only",
            [row for row in both_finite if String(getproperty(row, :raw_xf_status)) != "XF" && String(getproperty(row, :kulfan_xf_status)) == "XF"],
            _PLOT_GREEN,
        ),
        (
            "Kulfan frontier only",
            [row for row in both_finite if String(getproperty(row, :raw_xf_status)) == "XF" && String(getproperty(row, :kulfan_xf_status)) != "XF"],
            _PLOT_ORANGE,
        ),
        (
            "Frontier both",
            [row for row in both_finite if String(getproperty(row, :raw_xf_status)) != "XF" && String(getproperty(row, :kulfan_xf_status)) != "XF"],
            _PLOT_RED,
        ),
    )

    regime_counts = [
        count(row -> String(getproperty(row, :raw_xf_status)) == "XF" && String(getproperty(row, :kulfan_xf_status)) == "XF", items),
        count(row -> String(getproperty(row, :raw_xf_status)) != "XF" && String(getproperty(row, :kulfan_xf_status)) == "XF", items),
        count(row -> String(getproperty(row, :raw_xf_status)) == "XF" && String(getproperty(row, :kulfan_xf_status)) != "XF", items),
        count(row -> String(getproperty(row, :raw_xf_status)) != "XF" && String(getproperty(row, :kulfan_xf_status)) != "XF", items),
    ]

    ranked = sort!(
        [row for row in items if isfinite(float(getproperty(row, :delta_jx_cl)))];
        by = row -> -abs(float(getproperty(row, :delta_jx_cl))),
    )
    top_rows = ranked[1:min(length(ranked), 10)]

    fig = Figure(size = (1180, 860))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)
    Label(
        fig[1, 1:2],
        "The same executed cases are shown twice: once on the raw archival mesh and once on the JNeuralFoil Kulfan reconstruction.",
        fontsize = 12,
        color = _PLOT_GRAY,
    )

    ax_ecdf = Axis(fig[2, 1]; xlabel = "Healthy-reference |dCL|", ylabel = "ECDF", title = "Healthy-reference lift parity", xscale = log10)
    lines!(ax_ecdf, raw_ecdf_x, raw_ecdf_y; color = _PLOT_BLUE, linewidth = 2.3, label = "Raw mesh")
    lines!(ax_ecdf, kulfan_ecdf_x, kulfan_ecdf_y; color = _PLOT_ORANGE, linewidth = 2.3, label = "Kulfan rebuild")
    axislegend(ax_ecdf; position = :rb)
    xlims!(ax_ecdf, minimum(vcat(raw_ecdf_x, kulfan_ecdf_x)), maximum(vcat(raw_ecdf_x, kulfan_ecdf_x)))
    ylims!(ax_ecdf, 0.0, 1.0)

    ax_scatter = Axis(fig[2, 2]; xlabel = "Raw-mesh JXfoil CL", ylabel = "Kulfan JXfoil CL", title = "JXfoil CL shift by geometry treatment")
    scatter_min = Inf
    scatter_max = -Inf
    for (label, group, color) in scatter_groups
        isempty(group) && continue
        x = [float(getproperty(row, :raw_cl_jx)) for row in group]
        y = [float(getproperty(row, :kulfan_cl_jx)) for row in group]
        scatter!(ax_scatter, x, y; color = color, markersize = 9, label = label)
        scatter_min = min(scatter_min, minimum(vcat(x, y)))
        scatter_max = max(scatter_max, maximum(vcat(x, y)))
    end
    if isfinite(scatter_min) && isfinite(scatter_max)
        pad = 0.08 * max(scatter_max - scatter_min, 0.25)
        lines!(ax_scatter, [scatter_min - pad, scatter_max + pad], [scatter_min - pad, scatter_max + pad]; color = (_PLOT_GRAY, 0.5), linestyle = :dash, linewidth = 1.5)
        xlims!(ax_scatter, scatter_min - pad, scatter_max + pad)
        ylims!(ax_scatter, scatter_min - pad, scatter_max + pad)
    end
    axislegend(ax_scatter; position = :rb)

    ax_regime = Axis(fig[3, 1]; ylabel = "Cases", title = "XFoil regime transitions across remeshing")
    regime_labels = ["Healthy both", "Raw frontier only", "Kulfan frontier only", "Frontier both"]
    regime_colors = [_PLOT_BLUE, _PLOT_GREEN, _PLOT_ORANGE, _PLOT_RED]
    regime_pos = 1:4
    barplot!(ax_regime, regime_pos, regime_counts; color = regime_colors, width = 0.6)
    ax_regime.xticks = (collect(regime_pos), regime_labels)
    ax_regime.xticklabelrotation = pi / 10
    for i in eachindex(regime_counts)
        text!(ax_regime, i, regime_counts[i] + 1; text = string(regime_counts[i]), align = (:center, :bottom), fontsize = 11, color = regime_colors[i])
    end

    ax_shift = Axis(fig[3, 2]; ylabel = "|delta CL| in JXfoil", title = "Largest Kulfan-induced JXfoil lift shifts")
    if !isempty(top_rows)
        labels = [replace(String(getproperty(row, :id)), r"\.dat$"i => "") for row in top_rows]
        values = [abs(float(getproperty(row, :delta_jx_cl))) for row in top_rows]
        colors = [
            String(getproperty(row, :kulfan_jx_status)) == "FAIL" ? _PLOT_RED :
            String(getproperty(row, :kulfan_xf_status)) != "XF" ? _PLOT_ORANGE :
            _PLOT_GOLD
            for row in top_rows
        ]
        xpos = collect(1:length(top_rows))
        barplot!(ax_shift, xpos, values; color = colors, width = 0.6)
        ax_shift.xticks = (xpos, labels)
        ax_shift.xticklabelrotation = pi / 6
        for i in eachindex(top_rows)
            text!(
                ax_shift,
                xpos[i],
                values[i] + 0.02 * max(maximum(values), 0.1);
                text = String(getproperty(top_rows[i], :kulfan_xf_status)) * "/" * String(getproperty(top_rows[i], :kulfan_jx_status)),
                align = (:center, :bottom),
                fontsize = 9,
                color = colors[i],
            )
        end
    end

    return _save_publication_figure(path, fig)
end

function save_convergence_trace_figure(
    path::AbstractString,
    trace::CoupledSolveTrace;
    title::AbstractString = "Coupled-solver convergence history",
)
    _apply_publication_theme!()
    isempty(trace.samples) && throw(ArgumentError("trace does not contain any iteration samples"))

    samples = sort(trace.samples; by = s -> s.global_iteration)
    its = [s.global_iteration for s in samples]
    residuals = [max(float(s.residual), 1.0e-10) for s in samples]
    cls = [float(s.cl) for s in samples]
    cds = [float(s.cd) for s in samples]
    top_xtr = [float(s.top_xtr) for s in samples]
    bot_xtr = [float(s.bot_xtr) for s in samples]
    segments = unique(s.segment for s in samples)
    segment_colors = Dict(
        seg => color for (seg, color) in zip(
            segments,
            Iterators.cycle((_PLOT_BLUE, _PLOT_ORANGE, _PLOT_GREEN, _PLOT_RED, _PLOT_GOLD)),
        )
    )
    point_colors = [segment_colors[s.segment] for s in samples]
    boundaries = _trace_boundaries(samples)

    fig = Figure(size = (1040, 960))
    Label(fig[0, 1], title; fontsize = 18, color = _PLOT_GRAY)

    ax_res = Axis(fig[1, 1]; xlabel = "Global iteration", ylabel = "Residual", yscale = log10, title = "Newton residual")
    lines!(ax_res, its, residuals; color = _PLOT_GRAY, linewidth = 1.8)
    scatter!(ax_res, its, residuals; color = point_colors, markersize = 9)

    ax_force = Axis(fig[2, 1]; xlabel = "Global iteration", ylabel = "Coefficient", title = "Force-coefficient evolution")
    lines!(ax_force, its, cls; color = _PLOT_BLUE, linewidth = 2.2, label = "CL")
    lines!(ax_force, its, cds; color = _PLOT_RED, linewidth = 2.0, linestyle = :dash, label = "CD")
    axislegend(ax_force; position = :rb)

    ax_xtr = Axis(fig[3, 1]; xlabel = "Global iteration", ylabel = "xtr / c", title = "Transition-location evolution")
    lines!(ax_xtr, its, top_xtr; color = _PLOT_BLUE, linewidth = 2.2, label = "upper")
    lines!(ax_xtr, its, bot_xtr; color = _PLOT_ORANGE, linewidth = 2.2, linestyle = :dash, label = "lower")
    axislegend(ax_xtr; position = :rb)

    for boundary in boundaries
        for ax in (ax_res, ax_force, ax_xtr)
            vlines!(ax, [boundary - 0.5]; color = (_PLOT_GRAY, 0.35), linestyle = :dot, linewidth = 1.4)
        end
    end

    for seg in segments
        scatter!(ax_res, [NaN], [NaN]; color = segment_colors[seg], markersize = 9, label = _segment_label(seg))
    end
    axislegend(ax_res; position = :rt)

    return _save_publication_figure(path, fig)
end

function save_polar_comparison_figure(
    path::AbstractString,
    julia_rows,
    reference_rows;
    julia_label::AbstractString = "Julia",
    reference_label::AbstractString = "XFoil",
    title::AbstractString = "Polar comparison",
)
    _apply_publication_theme!()
    isempty(julia_rows) && throw(ArgumentError("julia_rows cannot be empty"))
    isempty(reference_rows) && throw(ArgumentError("reference_rows cannot be empty"))

    jrows = sort(collect(julia_rows); by = _row_alpha)
    rrows = sort(collect(reference_rows); by = _row_alpha)

    ja = [_row_alpha(r) for r in jrows]
    ra = [_row_alpha(r) for r in rrows]

    fig = Figure(size = (1040, 840))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)

    ax_cl = Axis(fig[1, 1]; xlabel = "alpha (deg)", ylabel = "CL", title = "Lift polar")
    lines!(ax_cl, ja, [_row_cl(r) for r in jrows]; color = _PLOT_BLUE, linewidth = 2.3, label = julia_label)
    scatter!(ax_cl, ja, [_row_cl(r) for r in jrows]; color = _PLOT_BLUE, markersize = 7)
    lines!(ax_cl, ra, [_row_cl(r) for r in rrows]; color = _PLOT_ORANGE, linewidth = 2.0, linestyle = :dash, label = reference_label)
    scatter!(ax_cl, ra, [_row_cl(r) for r in rrows]; color = _PLOT_ORANGE, markersize = 7)
    axislegend(ax_cl; position = :lt)

    ax_cd = Axis(fig[1, 2]; xlabel = "alpha (deg)", ylabel = "CD", title = "Drag polar")
    lines!(ax_cd, ja, [_row_cd(r) for r in jrows]; color = _PLOT_BLUE, linewidth = 2.3, label = julia_label)
    scatter!(ax_cd, ja, [_row_cd(r) for r in jrows]; color = _PLOT_BLUE, markersize = 7)
    lines!(ax_cd, ra, [_row_cd(r) for r in rrows]; color = _PLOT_ORANGE, linewidth = 2.0, linestyle = :dash, label = reference_label)
    scatter!(ax_cd, ra, [_row_cd(r) for r in rrows]; color = _PLOT_ORANGE, markersize = 7)

    ax_cm = Axis(fig[2, 1]; xlabel = "alpha (deg)", ylabel = "Cm,c/4", title = "Pitching moment")
    lines!(ax_cm, ja, [_row_cm(r) for r in jrows]; color = _PLOT_BLUE, linewidth = 2.3, label = julia_label)
    scatter!(ax_cm, ja, [_row_cm(r) for r in jrows]; color = _PLOT_BLUE, markersize = 7)
    lines!(ax_cm, ra, [_row_cm(r) for r in rrows]; color = _PLOT_ORANGE, linewidth = 2.0, linestyle = :dash, label = reference_label)
    scatter!(ax_cm, ra, [_row_cm(r) for r in rrows]; color = _PLOT_ORANGE, markersize = 7)

    ax_xtr = Axis(fig[2, 2]; xlabel = "alpha (deg)", ylabel = "xtr / c", title = "Transition location")
    lines!(ax_xtr, ja, [_row_top_xtr(r) for r in jrows]; color = _PLOT_BLUE, linewidth = 2.3, label = "$julia_label upper")
    lines!(ax_xtr, ja, [_row_bot_xtr(r) for r in jrows]; color = _PLOT_BLUE, linewidth = 2.3, linestyle = :dash, label = "$julia_label lower")
    lines!(ax_xtr, ra, [_row_top_xtr(r) for r in rrows]; color = _PLOT_ORANGE, linewidth = 2.0, label = "$reference_label upper")
    lines!(ax_xtr, ra, [_row_bot_xtr(r) for r in rrows]; color = _PLOT_ORANGE, linewidth = 2.0, linestyle = :dash, label = "$reference_label lower")
    axislegend(ax_xtr; position = :rb)

    return _save_publication_figure(path, fig)
end

function save_pathological_polar_layout_figure(
    path::AbstractString,
    cases;
    title::AbstractString = "Pathological low-Reynolds-number polars",
    subtitle::AbstractString = "Solid lines show stabilized JXfoil sweeps; dashed lines show the corresponding XFoil sweeps.",
    cl_limits = nothing,
    footnote::Union{Nothing,AbstractString} = nothing,
)
    rows = collect(cases)
    isempty(rows) && throw(ArgumentError("cases cannot be empty"))
    _apply_publication_theme!()

    palette = (_PLOT_BLUE, _PLOT_ORANGE, _PLOT_GREEN, _PLOT_RED, _PLOT_GOLD, _PLOT_GRAY)
    fig = Figure(size = footnote === nothing ? (1280, 700) : (1280, 760))
    Label(fig[0, 1:2], title; fontsize = 18, color = _PLOT_GRAY)
    Label(fig[1, 1:2], subtitle; fontsize = 12, color = _PLOT_GRAY)

    ax_drag = Axis(fig[2, 1]; xlabel = "CD", ylabel = "CL", title = "Cl v Cd")
    ax_alpha = Axis(fig[2, 2]; xlabel = "alpha (deg)", ylabel = "CL", title = "Cl v Alpha")

    all_cl = Float64[]
    all_cd = Float64[]
    all_alpha = Float64[]

    for (k, row) in enumerate(rows)
        label = String(getproperty(row, :label))
        color = palette[mod1(k, length(palette))]
        scatter!(ax_drag, [NaN], [NaN]; color = color, markersize = 6, label = label)

        xf_alpha_raw = getproperty(row, :xfoil_alpha)
        xf_cl_raw = getproperty(row, :xfoil_cl)
        xf_cd_raw = getproperty(row, :xfoil_cd)
        jx_alpha_raw = getproperty(row, :jx_alpha)
        jx_cl_raw = getproperty(row, :jx_cl)
        jx_cd_raw = getproperty(row, :jx_cd)

        xf_cd, xf_cl = _finite_xy(xf_cd_raw, xf_cl_raw)
        xf_alpha, xf_cl_alpha = _finite_xy(xf_alpha_raw, xf_cl_raw)
        jx_cd, jx_cl = _finite_xy(jx_cd_raw, jx_cl_raw)
        jx_alpha, jx_cl_alpha = _finite_xy(jx_alpha_raw, jx_cl_raw)

        xf_segments = hasproperty(row, :xfoil_segments) ?
            getproperty(row, :xfoil_segments) :
            _sweep_segments(xf_alpha_raw, xf_cl_raw, xf_cd_raw; alpha_gap_limit = 1.25)
        jx_segments = hasproperty(row, :jx_segments) ?
            getproperty(row, :jx_segments) :
            _sweep_segments(jx_alpha_raw, jx_cl_raw, jx_cd_raw; alpha_gap_limit = 1.25, cl_jump_limit = 0.35, cd_jump_limit = 0.05)

        isempty(xf_segments) || _plot_segmented_cdcl!(ax_drag, xf_segments; color = (color, 0.65), linewidth = 2.0, linestyle = :dash)
        isempty(jx_segments) || _plot_segmented_cdcl!(ax_drag, jx_segments; color = color, linewidth = 2.5, show_markers = true, markersize = 5.5)
        isempty(xf_segments) || _plot_segmented_alphacl!(ax_alpha, xf_segments; color = (color, 0.65), linewidth = 2.0, linestyle = :dash)
        isempty(jx_segments) || _plot_segmented_alphacl!(ax_alpha, jx_segments; color = color, linewidth = 2.5, show_markers = true, markersize = 5.5)

        append!(all_cl, xf_cl)
        append!(all_cl, jx_cl)
        append!(all_cd, xf_cd)
        append!(all_cd, jx_cd)
        append!(all_alpha, xf_alpha)
        append!(all_alpha, jx_alpha)
    end

    if cl_limits !== nothing
        cl_lo, cl_hi = float(cl_limits[1]), float(cl_limits[2])
        ylims!(ax_drag, cl_lo, cl_hi)
        ylims!(ax_alpha, cl_lo, cl_hi)
    elseif !isempty(all_cl)
        cl_lo = minimum(all_cl)
        cl_hi = maximum(all_cl)
        cl_pad = 0.08 * max(cl_hi - cl_lo, 0.25)
        ylims!(ax_drag, cl_lo - cl_pad, cl_hi + cl_pad)
        ylims!(ax_alpha, cl_lo - cl_pad, cl_hi + cl_pad)
    end
    isempty(all_cd) || begin
        cd_lo = minimum(all_cd)
        cd_hi = maximum(all_cd)
        cd_pad = 0.08 * max(cd_hi - cd_lo, 0.03)
        xlims!(ax_drag, max(0.0, cd_lo - 0.5 * cd_pad), cd_hi + cd_pad)
    end
    isempty(all_alpha) || begin
        alpha_lo = minimum(all_alpha)
        alpha_hi = maximum(all_alpha)
        xlims!(ax_alpha, alpha_lo - 0.5, alpha_hi + 0.5)
    end

    axislegend(ax_drag; position = :rb)
    footnote === nothing || Label(fig[3, 1:2], footnote; fontsize = 12, color = _PLOT_GRAY)
    return _save_publication_figure(path, fig)
end
