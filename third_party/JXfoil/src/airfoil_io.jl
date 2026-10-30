struct Airfoil
    name::String
    x::Vector{Float64}
    y::Vector{Float64}
end

function _parse_xy(line::AbstractString)
    s = strip(line)
    isempty(s) && return nothing
    startswith(s, "#") && return nothing
    startswith(s, "!") && return nothing

    fields = split(s)
    length(fields) < 2 && return nothing
    x = tryparse(Float64, fields[1])
    y = tryparse(Float64, fields[2])
    (x === nothing || y === nothing) && return nothing
    return (x, y)
end

@inline function _integer_like(value::Real; atol::Float64 = 1.0e-9)
    return abs(float(value) - round(float(value))) <= atol
end

function _consume_surface_count_header(
    blocks::AbstractVector{<:AbstractVector{<:Tuple{<:Real,<:Real}}}
)
    isempty(blocks) && return Tuple{Float64,Float64}[][]
    first_block = blocks[1]
    isempty(first_block) && return Vector{Vector{Tuple{Float64,Float64}}}(blocks)

    upper_count_f, lower_count_f = first_block[1]
    _integer_like(upper_count_f) || return Vector{Vector{Tuple{Float64,Float64}}}(blocks)
    _integer_like(lower_count_f) || return Vector{Vector{Tuple{Float64,Float64}}}(blocks)

    upper_count = round(Int, float(upper_count_f))
    lower_count = round(Int, float(lower_count_f))
    upper_count >= 2 || return Vector{Vector{Tuple{Float64,Float64}}}(blocks)
    lower_count >= 2 || return Vector{Vector{Tuple{Float64,Float64}}}(blocks)

    n_remaining = sum(length, blocks) - 1
    expected = upper_count + lower_count
    if !(n_remaining == expected || n_remaining == expected - 1)
        return Vector{Vector{Tuple{Float64,Float64}}}(blocks)
    end

    trimmed = [Tuple{Float64,Float64}.(block) for block in blocks]
    trimmed[1] = trimmed[1][2:end]
    return [block for block in trimmed if !isempty(block)]
end

function _has_surface_count_header(path::AbstractString)
    lines = readlines(path)
    _, blocks = _parse_coordinate_blocks(lines)
    isempty(blocks) && return false
    n_before = sum(length, blocks)
    trimmed = _consume_surface_count_header(blocks)
    n_after = sum(length, trimmed)
    return n_after == n_before - 1
end

function _parse_coordinate_blocks(lines::AbstractVector{<:AbstractString})
    blocks = Vector{Vector{Tuple{Float64,Float64}}}()
    current = Tuple{Float64,Float64}[]
    name = ""

    for line in lines
        xy = _parse_xy(line)
        if xy === nothing
            if !isempty(current)
                push!(blocks, current)
                current = Tuple{Float64,Float64}[]
            end
            if isempty(name) && !isempty(strip(line))
                name = strip(line)
            end
            continue
        end
        push!(current, (xy[1], xy[2]))
    end
    isempty(current) || push!(blocks, current)
    return name, blocks
end

function _stitch_surface_blocks(
    blocks::AbstractVector{<:AbstractVector{<:Tuple{<:Real,<:Real}}}
)
    length(blocks) == 2 || throw(ArgumentError("surface stitching requires exactly two blocks"))

    mean_y = [sum(float(p[2]) for p in block) / length(block) for block in blocks]
    if abs(mean_y[1] - mean_y[2]) > 1.0e-12
        upper_idx = mean_y[1] > mean_y[2] ? 1 : 2
        lower_idx = upper_idx == 1 ? 2 : 1

        function orient_block(block, want_descending_x::Bool)
            forward = Tuple{Float64,Float64}.(block)
            backward = reverse(forward)
            xf = float(forward[1][1]) - float(forward[end][1])
            xb = float(backward[1][1]) - float(backward[end][1])
            if want_descending_x
                return xf >= xb ? forward : backward
            else
                return xf <= xb ? forward : backward
            end
        end

        upper_rows = orient_block(blocks[upper_idx], true)
        lower_rows = orient_block(blocks[lower_idx], false)
        return vcat(upper_rows, lower_rows[2:end])
    end

    best_rows = Tuple{Float64,Float64}[]
    best_score = Inf

    for order in ((1, 2), (2, 1))
        block_a = blocks[order[1]]
        block_b = blocks[order[2]]
        for reverse_a in (false, true)
            rows_a = reverse_a ? reverse(block_a) : collect(block_a)
            for reverse_b in (false, true)
                rows_b = reverse_b ? reverse(block_b) : collect(block_b)
                join_gap = hypot(
                    float(rows_a[end][1]) - float(rows_b[1][1]),
                    float(rows_a[end][2]) - float(rows_b[1][2]),
                )
                close_gap = hypot(
                    float(rows_b[end][1]) - float(rows_a[1][1]),
                    float(rows_b[end][2]) - float(rows_a[1][2]),
                )
                score = join_gap + close_gap
                if score + 1.0e-12 < best_score
                    best_score = score
                    best_rows = vcat(
                        Tuple{Float64,Float64}.(rows_a),
                        Tuple{Float64,Float64}.(rows_b[2:end]),
                    )
                end
            end
        end
    end

    isempty(best_rows) && throw(ArgumentError("unable to stitch airfoil surface blocks"))
    return best_rows
end

function _signed_area(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y must have same length"))
    n < 3 && return 0.0
    area2 = 0.0
    @inbounds for i in 1:(n - 1)
        area2 += x[i] * y[i + 1] - x[i + 1] * y[i]
    end
    area2 += x[n] * y[1] - x[1] * y[n]
    return 0.5 * area2
end

function _ensure_counterclockwise(x::Vector{Float64}, y::Vector{Float64})
    if _signed_area(x, y) < 0.0
        reverse!(x)
        reverse!(y)
    end
    return x, y
end

function _close_curve!(x::Vector{Float64}, y::Vector{Float64}; atol::Float64 = 1.0e-12)
    if length(x) >= 2
        dx = x[1] - x[end]
        dy = y[1] - y[end]
        if hypot(dx, dy) > atol
            push!(x, x[1])
            push!(y, y[1])
        end
    end
    return x, y
end

function load_airfoil(path::AbstractString)
    lines = readlines(path)
    name, blocks = _parse_coordinate_blocks(lines)
    blocks = _consume_surface_count_header(blocks)
    isempty(blocks) && throw(ArgumentError("No coordinate rows found in $path"))
    isempty(name) && (name = splitpath(path)[end])

    rows =
        length(blocks) == 2 ? _stitch_surface_blocks(blocks) :
        reduce(vcat, [Tuple{Float64,Float64}.(block) for block in blocks])

    x = [row[1] for row in rows]
    y = [row[2] for row in rows]
    _ensure_counterclockwise(x, y)
    return Airfoil(name, x, y)
end

function naca4_airfoil(code::Integer; n::Int = 161)
    if code < 0 || code > 9999
        throw(ArgumentError("NACA code must be a 4-digit integer"))
    end
    n < 21 && throw(ArgumentError("n must be at least 21"))
    isodd(n) || throw(ArgumentError("n must be odd so the LE node is unique"))

    m = floor(code / 1000) / 100.0
    p = floor((code % 1000) / 100) / 10.0
    t = (code % 100) / 100.0

    nside = (n + 1) ÷ 2
    β = range(0.0, π; length = nside)
    xc = @. 0.5 * (1.0 - cos(β))

    yt = similar(xc)
    @inbounds for i in eachindex(xc)
        x = xc[i]
        yt[i] = 5.0 * t * (
            0.2969 * sqrt(max(x, 0.0)) -
            0.1260 * x -
            0.3516 * x^2 +
            0.2843 * x^3 -
            0.1015 * x^4
        )
    end

    yc = zeros(Float64, nside)
    dyc = zeros(Float64, nside)
    if p > 0.0
        @inbounds for i in eachindex(xc)
            x = xc[i]
            if x < p
                yc[i] = m / (p^2) * (2.0 * p * x - x^2)
                dyc[i] = 2.0 * m / (p^2) * (p - x)
            else
                yc[i] = m / ((1.0 - p)^2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x^2)
                dyc[i] = 2.0 * m / ((1.0 - p)^2) * (p - x)
            end
        end
    end

    θ = atan.(dyc)
    xu = @. xc - yt * sin(θ)
    yu = @. yc + yt * cos(θ)
    xl = @. xc + yt * sin(θ)
    yl = @. yc - yt * cos(θ)

    x = vcat(reverse(xu), xl[2:end])
    y = vcat(reverse(yu), yl[2:end])

    _ensure_counterclockwise(x, y)
    return Airfoil("NACA$(lpad(code, 4, '0'))", x, y)
end

function _naca4_code_from_name(name::AbstractString)
    m = match(r"^NACA(\d{4})$", strip(name))
    isnothing(m) && return nothing
    return parse(Int, m.captures[1])
end

function _is_builtin_naca4_airfoil(
    airfoil::Airfoil;
    atol::Float64 = 1.0e-12,
)
    code = _naca4_code_from_name(airfoil.name)
    isnothing(code) && return false
    isodd(length(airfoil.x)) || return false

    ref =
        try
            naca4_airfoil(code; n = length(airfoil.x))
        catch
            return false
        end

    return maximum(abs.(ref.x .- airfoil.x)) <= atol &&
           maximum(abs.(ref.y .- airfoil.y)) <= atol
end

"""
    _xfoil_load_normalized_airfoil(airfoil; digits=8) -> Airfoil

Normalize in-memory NACA airfoils to the same coordinate precision XFoil sees
after they are written to a text file and reloaded via `LOAD`. This preserves
generic loaded-coordinate airfoils while tightening parity for the built-in
`naca4_airfoil(...)` generator used throughout the validation harness.
"""
function _xfoil_load_normalized_airfoil(
    airfoil::Airfoil;
    digits::Int = 8,
)
    _is_builtin_naca4_airfoil(airfoil) || return airfoil
    xr = round.(airfoil.x; digits = digits)
    yr = round.(airfoil.y; digits = digits)
    return Airfoil(airfoil.name, xr, yr)
end

# XFoil-exact NACA 4-digit buffer generation (see inviscid_solver.jl for PANGEN).

"""
    _xfoil_naca4_buffer(code; nside=120) -> (xb, yb)

Generate NACA 4-digit buffer airfoil coordinates matching XFoil's NACA4
subroutine exactly (naca.f).  Returns counterclockwise (TE-upper-LE-lower-TE).
"""
function _xfoil_naca4_buffer(code::Integer; nside::Int = 120)
    0 <= code <= 9999 || throw(ArgumentError("NACA code must be 0–9999"))
    T = Float32  # match XFoil single precision

    n4 = code ÷ 1000
    n3 = (code - n4 * 1000) ÷ 100
    n2 = (code - n4 * 1000 - n3 * 100) ÷ 10
    n1 = code - n4 * 1000 - n3 * 100 - n2 * 10

    camber = T(n4) / T(100)
    p      = T(n3) / T(10)
    thick  = T(n2 * 10 + n1) / T(100)

    an = T(1.5)

    xx = Vector{T}(undef, nside)
    yt = Vector{T}(undef, nside)
    yc = Vector{T}(undef, nside)

    for i in 1:nside
        frac = T(i - 1) / T(nside - 1)
        if i == nside
            xx[i] = one(T)
        else
            anp = an + one(T)  # 2.5
            xx[i] = one(T) - anp * frac * (one(T) - frac)^an - (one(T) - frac)^anp
        end

        # Thickness (NACA standard)
        x = xx[i]
        yt[i] = (T(0.29690) * sqrt(max(x, zero(T))) -
                 T(0.12600) * x -
                 T(0.35160) * x^2 +
                 T(0.28430) * x^3 -
                 T(0.10150) * x^4) * thick / T(0.20)

        # Camber
        if p > zero(T) && camber > zero(T)
            if x < p
                yc[i] = camber / p^2 * (T(2) * p * x - x^2)
            else
                yc[i] = camber / (one(T) - p)^2 * ((one(T) - T(2) * p) + T(2) * p * x - x^2)
            end
        else
            yc[i] = zero(T)
        end
    end

    # Assemble: upper from TE to LE, then lower from LE to TE
    nb = 2 * nside - 1
    xb = Vector{T}(undef, nb)
    yb = Vector{T}(undef, nb)

    ib = 0
    for i in nside:-1:1
        ib += 1
        xb[ib] = xx[i]
        yb[ib] = yc[i] + yt[i]
    end
    for i in 2:nside
        ib += 1
        xb[ib] = xx[i]
        yb[ib] = yc[i] - yt[i]
    end

    return Float64.(xb), Float64.(yb)
end


# _xfoil_pangen and xfoil_naca4_airfoil are defined in inviscid_solver.jl
# (after XFoilPanelingParams is available)
