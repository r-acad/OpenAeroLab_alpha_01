# Opt-in pinned-node nested i refinement for the external-flow A-plus C-grid.
# Julia Base only: no SciPy or optimizer is needed at runtime.

function _nested_factors(mesh, n_base::Int, scale::Float64, slope::Float64;
                         leading_edge_spacing::Float64=0.0,
                         minimum_children::Int=2)
    minimum_children in (2,3,4) || throw(ArgumentError("minimum nested children must be 2,3 or 4"))
    lengths = hypot.(diff(mesh.X[:, 1]), diff(mesh.Y[:, 1]))
    all(isfinite, lengths) && minimum(lengths) > 0 ||
        throw(ArgumentError("nested refinement needs positive wall-edge lengths"))
    factors = fill(minimum_children, mesh.NI)
    leading_edge = argmin(mesh.X[:, 1])
    first_surface = mesh.NIW + n_base + 1
    last_surface = mesh.NIW + mesh.NIF - n_base
    first_surface <= leading_edge <= last_surface + 1 ||
        throw(ArgumentError("leading edge is outside the surface interval"))
    target0 = scale * lengths[mesh.NIW]
    arc = 0.0
    for i in first_surface:leading_edge-1
        target = target0 + slope * (arc + 0.5 * lengths[i])
        factors[i] = max(minimum_children, ceil(Int, lengths[i] / target))
        arc += lengths[i]
    end
    arc = 0.0
    for i in last_surface:-1:leading_edge
        target = target0 + slope * (arc + 0.5 * lengths[i])
        factors[i] = max(minimum_children, ceil(Int, lengths[i] / target))
        arc += lengths[i]
    end
    if leading_edge_spacing > 0
        for indices in (leading_edge-1:-1:first_surface,leading_edge:last_surface)
            arc=0.0
            for i in indices
                target=leading_edge_spacing+0.10*(arc+0.5lengths[i])
                factors[i]=max(factors[i],ceil(Int,lengths[i]/target))
                arc+=lengths[i]
            end
        end
    end
    maximum(factors) <= 16 ||
        throw(ArgumentError("nested refinement requires more than 16 children of a source cell"))
    sum(factors) <= 4 * mesh.NI ||
        throw(ArgumentError("nested refinement exceeds four times the source i cells"))
    lengths, factors
end

function _nested_project_lengths(lengths::Vector{Float64}, factors::Vector{Int},
                                 n_wake::Int, n_foil::Int;
                                 ratio::Float64=1.19, max_sweeps::Int=5000,
                                 tolerance::Float64=1e-8,
                                 require_convergence::Bool=true)
    offsets = cumsum(vcat(1, factors))
    children = Vector{Float64}(undef, sum(factors))
    for i in eachindex(factors)
        children[offsets[i]:offsets[i+1]-1] .= lengths[i] / factors[i]
    end
    wake_target = copy(children)
    lo = offsets[n_wake]
    hi = offsets[n_wake+n_foil+2]
    denominator = 1 + ratio^2
    for sweep in 1:max_sweeps
        for parity in 0:1
            for k in lo+parity:2:hi-2
                violation = children[k] - ratio * children[k+1]
                if violation > 0
                    t = violation / denominator
                    children[k] -= t
                    children[k+1] += ratio * t
                end
                violation = children[k+1] - ratio * children[k]
                if violation > 0
                    t = violation / denominator
                    children[k] += ratio * t
                    children[k+1] -= t
                end
            end
        end
        for i in eachindex(factors)
            first = offsets[i]
            last = offsets[i+1] - 1
            correction = (lengths[i] - sum(@view children[first:last])) / factors[i]
            for k in first:last
                children[k] += correction
            end
        end
        # Keep both wake legs exactly half split. This also makes the inserted
        # nodes on the two copies of the C-grid wake cut coincide bitwise.
        for i in 1:n_wake
            for k in offsets[i]:offsets[i+1]-1
                children[k] = wake_target[k]
            end
        end
        for i in n_wake+n_foil+1:length(factors)
            for k in offsets[i]:offsets[i+1]-1
                children[k] = wake_target[k]
            end
        end
        if sweep == 1 || sweep % 10 == 0
            max_ratio = maximum(max(children[k]/children[k+1],
                                    children[k+1]/children[k]) for k in lo:hi-2)
            max_sum_error = maximum(abs(sum(@view children[offsets[i]:offsets[i+1]-1])-
                                        lengths[i]) for i in eachindex(factors))
            if max_ratio <= ratio + tolerance &&
               max_sum_error <= 1e-12 && minimum(children) > 0
                return children, offsets, sweep, max_ratio, max_sum_error
            end
        end
    end
    actual=maximum(max(children[k]/children[k+1],children[k+1]/children[k]) for k in lo:hi-2)
    require_convergence || return children,offsets,max_sweeps,actual,
        maximum(abs(sum(@view children[offsets[i]:offsets[i+1]-1])-lengths[i]) for i in eachindex(factors))
    throw(ArgumentError("nested child-length projection did not meet $ratio after $max_sweeps sweeps (realised $actual)"))
end

function _nested_subdivide(mesh, lengths, factors, children, offsets)
    hasproperty(mesh,:strip) && mesh.strip !== nothing && throw(ArgumentError(
        "single-block nested refinement cannot refine a C+H strip mesh: it would collapse the distinct wake interfaces and discard the H block; use paired block refinement"))
    ni = sum(factors)
    X = Matrix{Float64}(undef, ni+1, mesh.NJ+1)
    Y = similar(X)
    growth = Vector{Float64}(undef, ni+1)
    tags = Vector{UInt8}(undef, ni)
    k = 1
    for i in eachindex(factors)
        partial = 0.0
        for child in 1:factors[i]
            t = partial / lengths[i]
            if child == 1
                X[k, :] .= mesh.X[i, :]
                Y[k, :] .= mesh.Y[i, :]
            else
                X[k, :] .= (1-t) .* mesh.X[i, :] .+ t .* mesh.X[i+1, :]
                Y[k, :] .= (1-t) .* mesh.Y[i, :] .+ t .* mesh.Y[i+1, :]
            end
            growth[k] = (1-t)*mesh.growth[i] + t*mesh.growth[i+1]
            tags[k] = mesh.outer_face_kind[i]
            partial += children[offsets[i]+child-1]
            k += 1
        end
    end
    X[end, :] .= mesh.X[end, :]
    Y[end, :] .= mesh.Y[end, :]
    growth[end] = mesh.growth[end]
    n_wake = sum(@view factors[1:mesh.NIW])
    n_foil = sum(@view factors[mesh.NIW+1:mesh.NIW+mesh.NIF])
    all(factors[i] == factors[end+1-i] for i in 1:mesh.NIW) ||
        error("nested wake subdivision factors must be paired")
    # The two parameterisations traverse the wake in opposite directions.
    # Copy the identical physical cut, avoiding roundoff from complementary
    # interpolation weights such as 1/3 and 2/3.
    for i in 1:n_wake+1
        X[ni+2-i,1]=X[i,1]
        Y[ni+2-i,1]=Y[i,1]
    end
    refined = CMeshGen.ChannelMesh(ni, mesh.NJ, n_wake, n_foil, X, Y,
                                   mesh.dn1, growth, mesh.channel, tags)
    refined, offsets
end

"Uniform second subdivision of an already refined C grid, preserving every parent node."
function _nested_uniform_bisect(mesh)
    hasproperty(mesh,:strip) && mesh.strip !== nothing && throw(ArgumentError(
        "single-block bisection cannot refine a C+H strip mesh; both interfaces and the H block must be refined together"))
    lengths = [hypot(mesh.X[i+1,1]-mesh.X[i,1],
                     mesh.Y[i+1,1]-mesh.Y[i,1]) for i in 1:mesh.NI]
    factors = fill(2,mesh.NI)
    offsets = collect(1:2:2*mesh.NI+1)
    children = Vector{Float64}(undef,2*mesh.NI)
    for i in 1:mesh.NI
        children[2i-1] = 0.5*lengths[i]
        children[2i] = 0.5*lengths[i]
    end
    refined, original_columns = _nested_subdivide(mesh,lengths,factors,children,offsets)
    original_columns == collect(1:2:2*mesh.NI+1) ||
        error("uniform nested subdivision lost a parent column")
    refined
end

"True-centroid wake-cut skew and first-row wake aspect ratio missed by the broad air gate."
function _nested_wake_cut_metrics(mesh)
    hasproperty(mesh,:strip) && mesh.strip !== nothing && throw(ArgumentError(
        "a C+H strip has two distinct interfaces, not a coincident wake cut; measure both blocks and their actual shared faces"))
    Q = CMeshGen.CMeshQualityMetrics
    c = Q.block_cells(mesh.X,mesh.Y)
    max_skew = -Inf
    max_solver_skew = -Inf
    worst_i = 0
    for i in 1:mesh.NIW
        p = mesh.NI+1-i
        f = Q.face_metrics(c.xc[p,1],c.yc[p,1],c.xc[i,1],c.yc[i,1],
            mesh.X[i+1,1],mesh.Y[i+1,1],mesh.X[i,1],mesh.Y[i,1],
            c.xv[p,1],c.yv[p,1],c.xv[i,1],c.yv[i,1])
        if f.skew > max_skew
            max_skew = f.skew
            worst_i = i
        end
        max_solver_skew = max(max_solver_skew,f.skew_v)
    end
    wake_i = vcat(collect(1:mesh.NIW),
        collect(mesh.NIW+mesh.NIF+1:mesh.NI))
    max_ar = maximum(c.ar_p[wake_i,1])
    all(isfinite,(max_skew,max_solver_skew,max_ar)) ||
        error("nonfinite wake-cut quality metric")
    (skew=max_skew,solver_skew=max_solver_skew,ar=max_ar,
     worst_i=worst_i,worst_x=0.5(mesh.X[worst_i,1]+mesh.X[worst_i+1,1]),
     worst_y=0.5(mesh.Y[worst_i,1]+mesh.Y[worst_i+1,1]))
end

function _nested_outlet_patch!(mesh; ncols::Int=4, sweeps::Int=5,
                               gain::Float64=0.7,
                               start_row::Union{Nothing,Int}=nothing,
                               full_row::Union{Nothing,Int}=nothing)
    X, Y = mesh.X, mesh.Y
    ni, nj = size(X)
    start = isnothing(start_row) ? round(Int, 70 * mesh.NJ / 112) : start_row
    full = isnothing(full_row) ? round(Int, 100 * mesh.NJ / 112) : full_row
    2 <= start < full < nj-1 ||
        throw(ArgumentError("nested outlet correction needs enough wall-normal cells for its tapered outlet patch"))
    ncols <= mesh.NIW || throw(ArgumentError("wake has too few cells for outlet correction"))
    for side in (:lower, :upper)
        edge = side === :lower ? 1 : ni
        step = side === :lower ? 1 : -1
        for j in start:nj-1
            a = clamp((j-start)/(full-start), 0.0, 1.0)
            wj = a*a*(3-2a)
            jp = j == nj ? nj-1 : j
            jn = min(nj, j+1)
            tx = X[edge,jn] - X[edge,jp]
            ty = Y[edge,jn] - Y[edge,jp]
            norm = hypot(tx,ty)
            tx /= norm
            ty /= norm
            for k in 1:ncols
                i = edge + k*step
                wcol = cospi(0.5*(k-1)/ncols)^2
                tang = (X[i,j]-X[edge,j])*tx + (Y[i,j]-Y[edge,j])*ty
                X[i,j] -= wj*wcol*tang*tx
                Y[i,j] -= wj*wcol*tang*ty
            end
        end
    end
    Q = CMeshGen.CMeshQualityMetrics
    for _ in 1:sweeps
        cells = Q.block_cells(X,Y)
        for side in (:lower,:upper)
            edge = side === :lower ? 1 : ni
            ci = side === :lower ? 1 : ni-1
            step = side === :lower ? 1 : -1
            sx = zeros(nj)
            sy = zeros(nj)
            count = zeros(Int,nj)
            for j in start:nj-1
                ax, ay = X[edge,j], Y[edge,j]
                bx, by = X[edge,j+1], Y[edge,j+1]
                tx, ty = bx-ax, by-ay
                norm = hypot(tx,ty)
                tx /= norm
                ty /= norm
                fx, fy = 0.5*(ax+bx), 0.5*(ay+by)
                error_t = (fx-cells.xc[ci,j])*tx + (fy-cells.yc[ci,j])*ty
                dx, dy = 2error_t*tx, 2error_t*ty
                sx[j] += dx
                sy[j] += dy
                count[j] += 1
                sx[j+1] += dx
                sy[j+1] += dy
                count[j+1] += 1
            end
            for j in start:nj-1
                a = clamp((j-start)/(full-start), 0.0, 1.0)
                wj = a*a*(3-2a)
                for k in 1:ncols
                    i = edge + k*step
                    wcol = cospi(0.5*(k-1)/ncols)^2
                    X[i,j] += gain*wj*wcol*sx[j]/count[j]
                    Y[i,j] += gain*wj*wcol*sy[j]/count[j]
                end
            end
        end
    end
    (start=start, full=full, ncols=ncols, sweeps=sweeps, gain=gain)
end

"""Build an opt-in nested-i candidate without changing the original grid.

The returned mesh preserves every source i-column at exactly indexed rows.
Wall-node interpolation retains the original airfoil polygon, rather than
resampling it with a spline. The outlet patch is applied separately.
"""
function _nested_refine_candidate(mesh, n_base::Int;
                                  scale::Float64=0.3, slope::Float64=0.15,
                                  ratio::Float64=1.19, uniform_factor::Int=0,
                                  leading_edge_spacing::Float64=0.0,
                                  minimum_children::Int=2)
    lengths, factors = if uniform_factor == 0
        _nested_factors(mesh,n_base,scale,slope;leading_edge_spacing,minimum_children)
    else
        uniform_factor in (3,4) || throw(ArgumentError("uniform nested factor must be 3 or 4"))
        (hypot.(diff(mesh.X[:,1]),diff(mesh.Y[:,1])),fill(uniform_factor,mesh.NI))
    end
    children,offsets,sweeps,actual_ratio,sum_error =
        _nested_project_lengths(lengths,factors,mesh.NIW,mesh.NIF;ratio)
    refined, original_columns =
        _nested_subdivide(mesh, lengths, factors, children, offsets)
    original_foil = mesh.NIW+1:mesh.NIW+mesh.NIF+1
    preserved = all(isequal.(refined.X[original_columns[original_foil], 1:2],
                            mesh.X[original_foil, 1:2])) &&
                all(isequal.(refined.Y[original_columns[original_foil], 1:2],
                            mesh.Y[original_foil, 1:2]))
    preserved || error("nested refinement moved an original foil node")
    paired = all(isequal.(refined.X[1:refined.NIW+1,1],
                         reverse(refined.X[end-refined.NIW:end,1]))) &&
             all(isequal.(refined.Y[1:refined.NIW+1,1],
                         reverse(refined.Y[end-refined.NIW:end,1])))
    paired || error("nested refinement split the wake cut inconsistently")
    metadata = Dict{String,Any}(
        "nested_algorithm"=>"nested_i_projected_v1",
        "source_n_foil"=>mesh.NIF-2n_base,
        "source_n_wake"=>mesh.NIW,
        "source_n_base"=>n_base,
        "refined_n_foil"=>refined.NIF-2sum(factors[mesh.NIW+1:mesh.NIW+n_base]),
        "refined_n_wake"=>refined.NIW,
        "refined_n_base"=>sum(factors[mesh.NIW+1:mesh.NIW+n_base]),
        "nested_scale"=>scale, "nested_slope"=>slope,
        "nested_uniform_factor"=>uniform_factor,
        "nested_minimum_children"=>minimum_children,
        "nested_ratio_limit"=>ratio,
        "nested_projection_sweeps"=>sweeps,
        "nested_projected_max_ratio"=>actual_ratio,
        "nested_parent_sum_error"=>sum_error,
        "original_foil_j1j2_preserved"=>preserved,
        "wake_cut_paired"=>paired)
    refined, metadata, original_columns
end
