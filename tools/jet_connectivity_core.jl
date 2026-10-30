module JetConnectivityCore

export contiguous_runs, wall_jet_diagnostics, interior_jet_diagnostics

"Inclusive index ranges for every contiguous true run in a Boolean vector."
function contiguous_runs(mask::AbstractVector{Bool})
    out = Tuple{Int,Int}[]
    i = 1
    n = length(mask)
    while i <= n
        if mask[i]
            j = i
            while j < n && mask[j + 1]
                j += 1
            end
            push!(out, (i, j))
            i = j + 1
        else
            i += 1
        end
    end
    out
end

"""
Per-frame wall-row x-velocity near-connectivity diagnostic used by `cmesh_cav_movie.jl`.

This is deliberately only the historical wall-row observable: it is not wall-tangential velocity
and cannot see a lifted jet. `alpha`, `ux`, and `x` must describe the same nonempty, increasing wall
strip from the leading-edge sample toward the trailing edge.
"""
function wall_jet_diagnostics(alpha::AbstractVector, ux::AbstractVector,
                              x::AbstractVector, threshold::Real)
    n = length(alpha)
    n > 0 || throw(ArgumentError("wall jet strip is empty"))
    length(ux) == n && length(x) == n ||
        throw(ArgumentError("alpha, ux, and x wall strips must have equal length"))
    isfinite(threshold) || throw(ArgumentError("cavity threshold must be finite"))
    all(v -> v isa Real && isfinite(v), alpha) ||
        throw(ArgumentError("wall alpha strip must contain only finite real values"))
    all(v -> v isa Real && isfinite(v), ux) ||
        throw(ArgumentError("wall x-velocity strip must contain only finite real values"))
    all(v -> v isa Real && isfinite(v), x) ||
        throw(ArgumentError("wall x-coordinate strip must contain only finite real values"))
    all(i -> x[i] < x[i + 1], 1:(n - 1)) ||
        throw(ArgumentError("wall x coordinates must be strictly increasing"))

    cavity_runs = contiguous_runs(alpha .>= threshold)
    # Preserve the archived producer's first wall-vapour-run heuristic. This is not a topology
    # attachment proof: the C-grid's first sampled wall cell can lie below threshold even when the
    # sheet is physically attached. Presence is explicit so x==0 is not overloaded as absence.
    has_wall_cavity = !isempty(cavity_runs)
    cavity_x1 = has_wall_cavity ? x[cavity_runs[1][2]] : 0.0
    jet_runs = contiguous_runs(ux .< 0.0)
    if isempty(jet_runs)
        return (has_wall_cavity=has_wall_cavity,
                cav_x1=cavity_x1, jet_x0=0.0, jet_x1=0.0,
                jet_nruns=0, jet_conn=0, us_min=minimum(ux))
    end

    # Select the reversed run whose downstream end is nearest the first wall-vapour-run endpoint.
    k = argmin([abs(x[r[2]] - cavity_x1) for r in jet_runs])
    x0 = x[jet_runs[k][1]]
    x1 = x[jet_runs[k][2]]
    length_scale = max(cavity_x1 - x[1], 1e-9)
    connected = (has_wall_cavity && abs(x1 - cavity_x1) <= 0.10 * length_scale &&
                 (x0 - x[1]) <= 0.20 * length_scale) ? 1 : 0
    (has_wall_cavity=has_wall_cavity,
     cav_x1=cavity_x1, jet_x0=x0, jet_x1=x1,
     jet_nruns=length(jet_runs), jet_conn=connected, us_min=minimum(ux))
end

"""
Four-connected interior reversed-flow diagnostic used by `cmesh_cav_movie.jl`.

The component nearest the first wall-vapour-run endpoint is measured. Presence is passed explicitly;
`cavity_x1 == 0` is therefore not overloaded as both a valid coordinate and an absence sentinel.
"""
function interior_jet_diagnostics(ux, x, NIW::Integer, first_foil_index::Integer,
                                  NIF::Integer, NJ::Integer,
                                  has_wall_cavity::Bool,
                                  cavity_x1::Real, x_start::Real)
    ndims(ux) == 2 && ndims(x) == 2 ||
        throw(ArgumentError("interior x velocity and coordinates must be matrices"))
    NIW >= 0 || throw(ArgumentError("NIW must be non-negative"))
    NIF > 0 && NJ > 0 || throw(ArgumentError("NIF and NJ must be positive"))
    1 <= first_foil_index <= NIF || throw(ArgumentError("invalid suction-side foil index"))
    NIF <= size(ux, 1) - NIW && NIF <= size(x, 1) - NIW &&
        NJ <= size(ux, 2) && NJ <= size(x, 2) ||
        throw(ArgumentError("interior diagnostic bounds exceed the supplied matrices"))
    isfinite(cavity_x1) && isfinite(x_start) ||
        throw(ArgumentError("wall-cavity endpoint and start coordinates must be finite"))
    !has_wall_cavity && cavity_x1 != 0.0 &&
        throw(ArgumentError("an absent wall cavity must use cav_x1=0"))
    has_wall_cavity && cavity_x1 < x_start &&
        throw(ArgumentError("wall-cavity endpoint precedes its start"))
    for jj in 1:NJ, ii in first_foil_index:NIF
        u = ux[NIW + ii, jj]
        xx = x[NIW + ii, jj]
        u isa Real && isfinite(u) ||
            throw(ArgumentError("interior x velocity must contain only finite real values"))
        xx isa Real && isfinite(xx) ||
            throw(ArgumentError("interior x coordinates must contain only finite real values"))
    end
    labels = zeros(Int, NIF, NJ)
    ncomponents = 0
    stack = Tuple{Int,Int}[]
    @inbounds for jj in 1:NJ, ii in first_foil_index:NIF
        (ux[NIW + ii, jj] < 0.0 && labels[ii, jj] == 0) || continue
        ncomponents += 1
        labels[ii, jj] = ncomponents
        empty!(stack)
        push!(stack, (ii, jj))
        while !isempty(stack)
            a, b = pop!(stack)
            for (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                p, q = a + da, b + db
                (first_foil_index <= p <= NIF && 1 <= q <= NJ) || continue
                (ux[NIW + p, q] < 0.0 && labels[p, q] == 0) || continue
                labels[p, q] = ncomponents
                push!(stack, (p, q))
            end
        end
    end
    ncomponents == 0 && return (n2d=0, x0_2d=0.0, conn2d=0, jmax2d=0)
    !has_wall_cavity &&
        return (n2d=ncomponents, x0_2d=0.0, conn2d=0, jmax2d=0)

    distance = Inf
    seed = 0
    @inbounds for jj in 1:NJ, ii in first_foil_index:NIF
        labels[ii, jj] == 0 && continue
        candidate = abs(x[NIW + ii, jj] - cavity_x1)
        candidate < distance && (distance = candidate; seed = labels[ii, jj])
    end
    seed == 0 && return (n2d=ncomponents, x0_2d=0.0, conn2d=0, jmax2d=0)

    x0 = Inf
    jmax = 0
    @inbounds for jj in 1:NJ, ii in first_foil_index:NIF
        labels[ii, jj] == seed || continue
        x0 = min(x0, x[NIW + ii, jj])
        jmax = max(jmax, jj)
    end
    length_scale = max(cavity_x1 - x_start, 1e-9)
    touches_endpoint = distance <= 0.10 * length_scale
    reaches_upstream = (x0 - x_start) <= 0.20 * length_scale
    (n2d=ncomponents, x0_2d=x0,
     conn2d=(touches_endpoint && reaches_upstream ? 1 : 0), jmax2d=jmax)
end

end # module
