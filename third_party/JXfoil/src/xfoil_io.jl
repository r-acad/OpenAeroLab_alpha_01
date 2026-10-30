struct XFoilPlotBlock
    x::Vector{Float64}
    y::Vector{Float64}
end

struct XFoilSurfaceState
    side::Symbol
    x::Vector{Float64}
    ue::Vector{Float64}
    dstar::Vector{Float64}
    theta::Vector{Float64}
    ctau::Vector{Float64}
    mode::Vector{Symbol}
    transition_index::Int
    transition_x::Float64
end

struct XFoilViscousStateDump
    upper::XFoilSurfaceState
    lower::XFoilSurfaceState
end

struct XFoilPanelDump
    iscalars::Dict{String,Int}
    rscalars::Dict{String,Float64}
    iarrays::Dict{String,Vector{Int}}
    arrays::Dict{String,Vector{Float64}}
    matrices::Dict{String,Matrix{Float64}}
end

function parse_xfoil_polar_table(path::AbstractString)
    lines = readlines(path)
    rows = NamedTuple[]
    for line in lines
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        startswith(s, "-") && continue

        fields = split(s)
        length(fields) < 7 && continue
        vals = [tryparse(Float64, f) for f in fields[1:7]]
        any(v -> v === nothing, vals) && continue

        push!(
            rows,
            (
                alpha_deg = vals[1]::Float64,
                cl = vals[2]::Float64,
                cd = vals[3]::Float64,
                cdp = vals[4]::Float64,
                cm = vals[5]::Float64,
                top_xtr = vals[6]::Float64,
                bot_xtr = vals[7]::Float64,
            ),
        )
    end

    isempty(rows) && throw(ArgumentError("No polar data row found in $path"))
    return rows
end

function parse_xfoil_polar(path::AbstractString)
    rows = parse_xfoil_polar_table(path)
    return rows[end]
end

function _xfoil_parse_float(token::AbstractString)
    return parse(Float64, replace(token, 'D' => 'E', 'd' => 'e'))
end

function parse_xfoil_panel_dump(path::AbstractString)
    lines = readlines(path)
    iscalars = Dict{String,Int}()
    rscalars = Dict{String,Float64}()
    iarrays = Dict{String,Vector{Int}}()
    arrays = Dict{String,Vector{Float64}}()
    matrices = Dict{String,Matrix{Float64}}()

    i = 1
    while i <= length(lines)
        s = strip(lines[i])
        if isempty(s)
            i += 1
            continue
        elseif startswith(s, "ISCALAR ")
            fields = split(s)
            length(fields) == 3 ||
                throw(ArgumentError("Malformed ISCALAR line in $path: $(lines[i])"))
            iscalars[fields[2]] = parse(Int, fields[3])
            i += 1
        elseif startswith(s, "RSCALAR ")
            fields = split(s)
            length(fields) == 3 ||
                throw(ArgumentError("Malformed RSCALAR line in $path: $(lines[i])"))
            rscalars[fields[2]] = _xfoil_parse_float(fields[3])
            i += 1
        elseif startswith(s, "IARRAY ")
            fields = split(s)
            length(fields) == 3 ||
                throw(ArgumentError("Malformed IARRAY header in $path: $(lines[i])"))
            name = fields[2]
            n = parse(Int, fields[3])
            vals = Vector{Int}(undef, n)
            for k in 1:n
                i += 1
                i <= length(lines) || throw(
                    ArgumentError("Unexpected EOF while reading IARRAY $name from $path"),
                )
                row = split(strip(lines[i]))
                length(row) >= 2 || throw(
                    ArgumentError("Malformed IARRAY row for $name in $path: $(lines[i])"),
                )
                vals[k] = parse(Int, row[2])
            end
            iarrays[name] = vals
            i += 1
        elseif startswith(s, "ARRAY ")
            fields = split(s)
            length(fields) == 3 ||
                throw(ArgumentError("Malformed ARRAY header in $path: $(lines[i])"))
            name = fields[2]
            n = parse(Int, fields[3])
            vals = Vector{Float64}(undef, n)
            for k in 1:n
                i += 1
                i <= length(lines) || throw(
                    ArgumentError("Unexpected EOF while reading ARRAY $name from $path"),
                )
                row = split(strip(lines[i]))
                length(row) >= 2 || throw(
                    ArgumentError("Malformed ARRAY row for $name in $path: $(lines[i])"),
                )
                vals[k] = _xfoil_parse_float(row[2])
            end
            arrays[name] = vals
            i += 1
        elseif startswith(s, "MATRIX ")
            fields = split(s)
            length(fields) == 4 ||
                throw(ArgumentError("Malformed MATRIX header in $path: $(lines[i])"))
            name = fields[2]
            nr = parse(Int, fields[3])
            nc = parse(Int, fields[4])
            vals = Matrix{Float64}(undef, nr, nc)
            for _ in 1:(nr * nc)
                i += 1
                i <= length(lines) || throw(
                    ArgumentError("Unexpected EOF while reading MATRIX $name from $path"),
                )
                row = split(strip(lines[i]))
                length(row) >= 3 || throw(
                    ArgumentError("Malformed MATRIX row for $name in $path: $(lines[i])"),
                )
                ir = parse(Int, row[1])
                jc = parse(Int, row[2])
                1 <= ir <= nr || throw(
                    ArgumentError("MATRIX row index out of bounds for $name in $path: $(lines[i])"),
                )
                1 <= jc <= nc || throw(
                    ArgumentError("MATRIX column index out of bounds for $name in $path: $(lines[i])"),
                )
                vals[ir, jc] = _xfoil_parse_float(row[3])
            end
            matrices[name] = vals
            i += 1
        else
            throw(ArgumentError("Unrecognized panel dump line in $path: $(lines[i])"))
        end
    end

    return XFoilPanelDump(iscalars, rscalars, iarrays, arrays, matrices)
end

function parse_xfoil_plot_blocks(path::AbstractString)
    lines = readlines(path)
    blocks = XFoilPlotBlock[]
    xb = Float64[]
    yb = Float64[]

    function flush_block!()
        isempty(xb) && return
        push!(blocks, XFoilPlotBlock(copy(xb), copy(yb)))
        empty!(xb)
        empty!(yb)
    end

    for line in lines
        s = strip(line)
        if isempty(s)
            flush_block!()
            continue
        end
        startswith(s, "#") && continue

        fields = split(s)
        length(fields) < 2 && continue

        xv = tryparse(Float64, fields[1])
        yv = tryparse(Float64, fields[2])
        (xv === nothing || yv === nothing) && continue

        push!(xb, xv::Float64)
        push!(yb, yv::Float64)
    end

    flush_block!()
    isempty(blocks) && throw(ArgumentError("No plot blocks found in $path"))
    return blocks
end

function _trim_airfoil_block(
    block::XFoilPlotBlock;
    xmax::Float64 = 1.00005,
)
    n = 0
    for i in eachindex(block.x)
        block.x[i] <= xmax || break
        n = i
    end
    n > 0 || throw(ArgumentError("No airfoil points found in plot block"))
    return XFoilPlotBlock(block.x[1:n], block.y[1:n])
end

function _require_xfoil_blocks(
    path::AbstractString,
    expected::Integer,
)
    blocks = parse_xfoil_plot_blocks(path)
    length(blocks) == expected ||
        throw(ArgumentError("Expected $expected plot blocks in $path, found $(length(blocks))"))
    return blocks
end

function _require_matching_grid(
    side::Symbol,
    name::AbstractString,
    ref::XFoilPlotBlock,
    blk::XFoilPlotBlock;
    atol::Float64 = 1.0e-10,
)
    length(ref.x) == length(blk.x) || throw(
        ArgumentError(
            "XFoil $name block length mismatch on $side: $(length(ref.x)) vs $(length(blk.x))",
        ),
    )

    for i in eachindex(ref.x)
        isapprox(ref.x[i], blk.x[i]; atol = atol, rtol = 0.0) || throw(
            ArgumentError(
                "XFoil $name block x-grid mismatch on $side at index $i: $(ref.x[i]) vs $(blk.x[i])",
            ),
        )
    end
    return nothing
end

function _assemble_xfoil_surface_state(
    side::Symbol,
    ue_block::XFoilPlotBlock,
    dstar_block::XFoilPlotBlock,
    theta_block::XFoilPlotBlock,
    lam_block::XFoilPlotBlock,
    turb_block::XFoilPlotBlock;
    x_tol::Float64 = 1.0e-10,
)
    _require_matching_grid(side, "Ue/Dstar", ue_block, dstar_block; atol = x_tol)
    _require_matching_grid(side, "Ue/Theta", ue_block, theta_block; atol = x_tol)

    nlam = length(lam_block.x)
    nturb = length(turb_block.x)
    nx = length(ue_block.x)
    nlam + nturb == nx || throw(
        ArgumentError(
            "XFoil CTAU assembly mismatch on $side: laminar=$nlam turbulent=$nturb total=$nx",
        ),
    )

    if nlam > 0
        for i in 1:nlam
            isapprox(ue_block.x[i], lam_block.x[i]; atol = x_tol, rtol = 0.0) || throw(
                ArgumentError(
                    "XFoil laminar x-grid mismatch on $side at index $i: $(ue_block.x[i]) vs $(lam_block.x[i])",
                ),
            )
        end
    end
    if nturb > 0
        for i in 1:nturb
            j = nlam + i
            isapprox(ue_block.x[j], turb_block.x[i]; atol = x_tol, rtol = 0.0) || throw(
                ArgumentError(
                    "XFoil turbulent x-grid mismatch on $side at index $j: $(ue_block.x[j]) vs $(turb_block.x[i])",
                ),
            )
        end
    end

    ctau = vcat(lam_block.y, turb_block.y)
    mode = vcat(fill(:laminar, nlam), fill(:turbulent, nturb))
    transition_index = nturb > 0 ? nlam + 1 : nx + 1
    transition_x = nturb > 0 ? turb_block.x[1] : 1.0

    return XFoilSurfaceState(
        side,
        copy(ue_block.x),
        copy(ue_block.y),
        copy(dstar_block.y),
        copy(theta_block.y),
        ctau,
        mode,
        transition_index,
        transition_x,
    )
end

function parse_xfoil_viscous_state_dump(
    ue_path::AbstractString,
    dt_path::AbstractString,
    db_path::AbstractString,
    n_path::AbstractString,
    ct_path::AbstractString;
    xmax::Float64 = 1.00005,
    x_tol::Float64 = 1.0e-10,
)
    ue_blocks = _require_xfoil_blocks(ue_path, 2)
    dt_blocks = _require_xfoil_blocks(dt_path, 2)
    db_blocks = _require_xfoil_blocks(db_path, 2)
    n_blocks = _require_xfoil_blocks(n_path, 2)
    ct_blocks = _require_xfoil_blocks(ct_path, 2)

    upper = _assemble_xfoil_surface_state(
        :upper,
        _trim_airfoil_block(ue_blocks[1]; xmax = xmax),
        _trim_airfoil_block(dt_blocks[1]; xmax = xmax),
        _trim_airfoil_block(dt_blocks[2]; xmax = xmax),
        _trim_airfoil_block(n_blocks[1]; xmax = xmax),
        _trim_airfoil_block(ct_blocks[1]; xmax = xmax);
        x_tol = x_tol,
    )
    lower = _assemble_xfoil_surface_state(
        :lower,
        _trim_airfoil_block(ue_blocks[2]; xmax = xmax),
        _trim_airfoil_block(db_blocks[1]; xmax = xmax),
        _trim_airfoil_block(db_blocks[2]; xmax = xmax),
        _trim_airfoil_block(n_blocks[2]; xmax = xmax),
        _trim_airfoil_block(ct_blocks[2]; xmax = xmax);
        x_tol = x_tol,
    )

    return XFoilViscousStateDump(upper, lower)
end
