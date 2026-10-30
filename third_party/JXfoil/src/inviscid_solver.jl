struct CaseResult
    alpha_deg::Float64
    xcp::Vector{Float64}
    ycp::Vector{Float64}
    cp::Vector{Float64}
    vt::Vector{Float64}
    cl::Float64
    cm_c4::Float64
    xnode::Vector{Float64}
    ynode::Vector{Float64}
    qnode::Vector{Float64}
end

const _THIN_AIRFOIL_SAMPLES = 801

struct XFoilPanelingParams
    npan::Int
    cvpar::Float64
    cterat::Float64
    ctrrat::Float64
    xsref1::Float64
    xsref2::Float64
    xpref1::Float64
    xpref2::Float64
end

function XFoilPanelingParams(;
    npan::Int = 160,
    cvpar::Real = 1.0,
    cterat::Real = 0.15,
    ctrrat::Real = 0.2,
    xsref1::Real = 1.0,
    xsref2::Real = 1.0,
    xpref1::Real = 1.0,
    xpref2::Real = 1.0,
)
    npan >= 5 || throw(ArgumentError("npan must be at least 5"))
    return XFoilPanelingParams(
        npan,
        float(cvpar),
        float(cterat),
        float(ctrrat),
        float(xsref1),
        float(xsref2),
        float(xpref1),
        float(xpref2),
    )
end

const _XFOIL_DEFAULT_PANELING = XFoilPanelingParams()
const _XFOIL_PANELING_PRECISION = Float32
const _XFOIL_LINEAR_SOLVE_PRECISION = Float32
const _XFOIL_GGCALC_PRECISION = Float32
const _XFOIL_FORCE_PRECISION = Float32

struct XFoilLUFactorization{T<:AbstractFloat}
    a::Matrix{T}
    indx::Vector{Int}
end

function _dedupe_points(x::Vector{Float64}, y::Vector{Float64}; atol::Float64 = 1.0e-12)
    if length(x) != length(y)
        throw(ArgumentError("x and y lengths must match"))
    end
    xo = Float64[]
    yo = Float64[]
    for i in eachindex(x)
        if isempty(xo) || hypot(x[i] - xo[end], y[i] - yo[end]) > atol
            push!(xo, x[i])
            push!(yo, y[i])
        end
    end
    if length(xo) >= 2 && hypot(xo[1] - xo[end], yo[1] - yo[end]) > atol
        push!(xo, xo[1])
        push!(yo, yo[1])
    end
    return xo, yo
end

function _dedupe_x_surface(
    x::Vector{Float64},
    y::Vector{Float64};
    atol::Float64 = 1.0e-12,
)
    length(x) == length(y) || throw(ArgumentError("x and y lengths must match"))
    n = length(x)
    n >= 2 || throw(ArgumentError("surface needs at least two points"))

    xo = Float64[]
    yo = Float64[]
    i = 1
    while i <= n
        xv = x[i]
        ysum = 0.0
        cnt = 0
        while i <= n && abs(x[i] - xv) <= atol
            ysum += y[i]
            cnt += 1
            i += 1
        end
        push!(xo, xv)
        push!(yo, ysum / max(cnt, 1))
    end

    length(xo) >= 2 || throw(ArgumentError("surface collapsed after x-deduplication"))
    return xo, yo
end

function _interp_linear(
    xdata::AbstractVector{<:Real},
    ydata::AbstractVector{<:Real},
    xq::Real,
)
    n = length(xdata)
    n == length(ydata) || throw(ArgumentError("x and y lengths must match"))
    n >= 2 || throw(ArgumentError("interpolant needs at least two points"))

    if xq <= xdata[1]
        return float(ydata[1])
    elseif xq >= xdata[end]
        return float(ydata[end])
    end

    i = searchsortedlast(xdata, xq)
    i = clamp(i, 1, n - 1)
    x1 = float(xdata[i])
    x2 = float(xdata[i + 1])
    y1 = float(ydata[i])
    y2 = float(ydata[i + 1])
    dx = x2 - x1
    if abs(dx) <= 1.0e-14
        return 0.5 * (y1 + y2)
    end
    t = (float(xq) - x1) / dx
    return (1.0 - t) * y1 + t * y2
end

function _split_surfaces(airfoil::Airfoil)
    x = Float64[]
    y = Float64[]
    for i in eachindex(airfoil.x)
        xi = float(airfoil.x[i])
        yi = float(airfoil.y[i])
        if isempty(x) || hypot(xi - x[end], yi - y[end]) > 1.0e-12
            push!(x, xi)
            push!(y, yi)
        end
    end

    n = length(x)
    n >= 5 || throw(ArgumentError("airfoil needs at least 5 points for surface split"))

    ile = argmin(x)
    2 <= ile <= n - 1 || throw(ArgumentError("unable to identify LE split index"))

    upper_x = copy(x[1:ile])
    upper_y = copy(y[1:ile])
    lower_x = copy(x[ile:end])
    lower_y = copy(y[ile:end])

    if upper_x[1] > upper_x[end]
        reverse!(upper_x)
        reverse!(upper_y)
    end
    if lower_x[1] > lower_x[end]
        reverse!(lower_x)
        reverse!(lower_y)
    end

    upper_x, upper_y = _dedupe_x_surface(upper_x, upper_y)
    lower_x, lower_y = _dedupe_x_surface(lower_x, lower_y)
    return upper_x, upper_y, lower_x, lower_y
end

function _trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y lengths must match"))
    n >= 2 || throw(ArgumentError("trapz needs at least two points"))

    s = 0.0
    @inbounds for i in 1:(n - 1)
        dx = float(x[i + 1] - x[i])
        s += 0.5 * (float(y[i]) + float(y[i + 1])) * dx
    end
    return s
end

function _thin_airfoil_cl_cm(
    airfoil::Airfoil,
    alpha_deg::Real;
    ntheta::Int = _THIN_AIRFOIL_SAMPLES,
)
    ntheta >= 11 || throw(ArgumentError("ntheta must be at least 11"))

    upper_x, upper_y, lower_x, lower_y = _split_surfaces(airfoil)
    theta = collect(range(0.0, pi; length = ntheta))
    xcos = @. 0.5 * (1.0 - cos(theta))

    zc = Vector{Float64}(undef, ntheta)
    @inbounds for i in eachindex(xcos)
        xu = _interp_linear(upper_x, upper_y, xcos[i])
        xl = _interp_linear(lower_x, lower_y, xcos[i])
        zc[i] = 0.5 * (xu + xl)
    end

    dzdx = Vector{Float64}(undef, ntheta)
    dzdx[1] = (zc[2] - zc[1]) / max(xcos[2] - xcos[1], 1.0e-14)
    dzdx[end] = (zc[end] - zc[end - 1]) / max(xcos[end] - xcos[end - 1], 1.0e-14)
    @inbounds for i in 2:(ntheta - 1)
        dx = max(xcos[i + 1] - xcos[i - 1], 1.0e-14)
        dzdx[i] = (zc[i + 1] - zc[i - 1]) / dx
    end

    i0 = _trapz(theta, dzdx)
    i1 = _trapz(theta, @. dzdx * cos(theta))
    i2 = _trapz(theta, @. dzdx * cos(2.0 * theta))

    alpha = deg2rad(float(alpha_deg))
    a0 = alpha - i0 / pi
    a1 = 2.0 * i1 / pi
    a2 = 2.0 * i2 / pi

    cl = 2.0 * pi * (a0 + 0.5 * a1)
    cm_c4 = -0.25 * pi * (a1 - a2)
    return cl, cm_c4
end

function _blend_force_coefficients(
    cl_panel::Real,
    cm_panel::Real,
    _airfoil::Airfoil,
    _alpha_deg::Real,
    ;
    context::Symbol = :inviscid,
)
    # Algorithmic path only: no parity-calibration surrogate blend.
    context == :inviscid || context == :viscous || throw(ArgumentError("invalid context"))
    return float(cl_panel), float(cm_panel)
end

function _xfoil_nodes(airfoil::Airfoil; atol::Float64 = 1.0e-12)
    x = Float64[]
    y = Float64[]
    for i in eachindex(airfoil.x)
        xi = float(airfoil.x[i])
        yi = float(airfoil.y[i])
        if isempty(x) || hypot(xi - x[end], yi - y[end]) > atol
            push!(x, xi)
            push!(y, yi)
        end
    end
    length(x) >= 3 || throw(ArgumentError("airfoil needs at least 3 non-duplicate nodes"))
    return x, y
end

function _xfoil_buffer_nodes(airfoil::Airfoil; atol::Float64 = 1.0e-12)
    x = Float64[]
    y = Float64[]
    for i in eachindex(airfoil.x)
        xi = float(airfoil.x[i])
        yi = float(airfoil.y[i])
        if isempty(x) || hypot(xi - x[end], yi - y[end]) > atol
            push!(x, xi)
            push!(y, yi)
        end
    end
    length(x) >= 3 || throw(ArgumentError("buffer airfoil needs at least 3 non-duplicate nodes"))
    return x, y
end

function _scalc(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y lengths must match"))
    n >= 2 || throw(ArgumentError("need at least 2 points for arclength"))
    T = _panel_float_type(eltype(x), eltype(y))
    s = Vector{T}(undef, n)
    s[1] = zero(T)
    for i in 2:n
        dx = T(x[i] - x[i - 1])
        dy = T(y[i] - y[i - 1])
        s[i] = s[i - 1] + sqrt(dx * dx + dy * dy)
    end
    return s
end

function _trisol!(
    a::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(a)
    n == length(b) == length(c) == length(d) ||
        throw(ArgumentError("trisol array size mismatch"))
    n >= 2 || throw(ArgumentError("trisol needs at least 2 rows"))

    for k in 2:n
        km = k - 1
        c[km] /= a[km]
        d[km] /= a[km]
        a[k] -= b[k] * c[km]
        d[k] -= b[k] * d[km]
    end

    d[n] /= a[n]
    for k in (n - 1):-1:1
        d[k] -= c[k] * d[k + 1]
    end
    return d
end

function _xfoil_ludcmp!(a::Matrix{T}) where {T<:AbstractFloat}
    n = size(a, 1)
    n == size(a, 2) || throw(ArgumentError("LU matrix must be square"))
    vv = Vector{T}(undef, n)
    indx = Vector{Int}(undef, n)

    for i in 1:n
        aamax = zero(T)
        for j in 1:n
            aamax = max(abs(a[i, j]), aamax)
        end
        vv[i] = one(T) / aamax
    end

    for j in 1:n
        for i in 1:(j - 1)
            s = a[i, j]
            for k in 1:(i - 1)
                s -= a[i, k] * a[k, j]
            end
            a[i, j] = s
        end

        aamax = zero(T)
        imax = j
        for i in j:n
            s = a[i, j]
            for k in 1:(j - 1)
                s -= a[i, k] * a[k, j]
            end
            a[i, j] = s

            dum = vv[i] * abs(s)
            if dum >= aamax
                imax = i
                aamax = dum
            end
        end

        if j != imax
            for k in 1:n
                dum = a[imax, k]
                a[imax, k] = a[j, k]
                a[j, k] = dum
            end
            vv[imax] = vv[j]
        end

        indx[j] = imax
        if j != n
            dum = one(T) / a[j, j]
            for i in (j + 1):n
                a[i, j] *= dum
            end
        end
    end

    return indx
end

function _xfoil_lu(
    a::AbstractMatrix{<:Real};
    precision::Type{T} = Float64,
) where {T<:AbstractFloat}
    n = size(a, 1)
    n == size(a, 2) || throw(ArgumentError("LU matrix must be square"))
    af = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:n
            af[i, j] = T(a[i, j])
        end
    end
    indx = _xfoil_ludcmp!(af)
    return XFoilLUFactorization(af, indx)
end

function _xfoil_baksub!(
    a::AbstractMatrix{T},
    indx::AbstractVector{Int},
    b::AbstractVector{T},
) where {T<:AbstractFloat}
    n = size(a, 1)
    n == size(a, 2) == length(indx) == length(b) ||
        throw(ArgumentError("LU solve size mismatch"))

    ii = 0
    for i in 1:n
        ll = indx[i]
        s = b[ll]
        b[ll] = b[i]
        if ii != 0
            for j in ii:(i - 1)
                s -= a[i, j] * b[j]
            end
        elseif s != zero(T)
            ii = i
        end
        b[i] = s
    end

    for i in n:-1:1
        s = b[i]
        if i < n
            for j in (i + 1):n
                s -= a[i, j] * b[j]
            end
        end
        b[i] = s / a[i, i]
    end
    return b
end

function _xfoil_lu_solve!(
    fac::XFoilLUFactorization{T},
    b::AbstractVector{T},
) where {T<:AbstractFloat}
    return _xfoil_baksub!(fac.a, fac.indx, b)
end

function _panel_float_type(types::Type...)
    t = promote_type(types...)
    return t <: AbstractFloat ? t : Float64
end

function _spline_interval(ss::Real, s::AbstractVector{<:Real})
    n = length(s)
    n >= 2 || throw(ArgumentError("need at least two spline nodes"))
    ilow = 1
    i = n
    while i - ilow > 1
        imid = (i + ilow) ÷ 2
        if float(ss) < float(s[imid])
            i = imid
        else
            ilow = imid
        end
    end
    return i
end

function _splind_segment(
    x::AbstractVector{<:Real},
    s::AbstractVector{<:Real};
    xs1::Real = -999.0,
    xs2::Real = -999.0,
)
    n = length(x)
    n == length(s) || throw(ArgumentError("x and s lengths must match"))
    n >= 2 || throw(ArgumentError("spline segment needs at least 2 points"))

    T = _panel_float_type(eltype(x), eltype(s))
    a = zeros(T, n)
    b = zeros(T, n)
    c = zeros(T, n)
    rhs = zeros(T, n)
    xs1t = T(xs1)
    xs2t = T(xs2)
    two = T(2.0)
    three = T(3.0)
    one = T(1.0)
    zero = T(0.0)
    eps = T(1.0e-15)

    for i in 2:(n - 1)
        dsm = T(s[i] - s[i - 1])
        dsp = T(s[i + 1] - s[i])
        b[i] = dsp
        a[i] = two * (dsm + dsp)
        c[i] = dsm
        rhs[i] =
            three * (
                T(x[i + 1] - x[i]) * dsm / max(dsp, eps) +
                T(x[i] - x[i - 1]) * dsp / max(dsm, eps)
            )
    end

    if xs1t == T(999.0)
        a[1] = two
        c[1] = one
        rhs[1] = three * T(x[2] - x[1]) / max(T(s[2] - s[1]), eps)
    elseif xs1t == T(-999.0)
        a[1] = one
        c[1] = one
        rhs[1] = two * T(x[2] - x[1]) / max(T(s[2] - s[1]), eps)
    else
        a[1] = one
        c[1] = zero
        rhs[1] = xs1t
    end

    if xs2t == T(999.0)
        b[n] = one
        a[n] = two
        rhs[n] = three * T(x[n] - x[n - 1]) / max(T(s[n] - s[n - 1]), eps)
    elseif xs2t == T(-999.0)
        b[n] = one
        a[n] = one
        rhs[n] = two * T(x[n] - x[n - 1]) / max(T(s[n] - s[n - 1]), eps)
    else
        a[n] = one
        b[n] = zero
        rhs[n] = xs2t
    end

    if n == 2 && xs1t == T(-999.0) && xs2t == T(-999.0)
        b[n] = one
        a[n] = two
        rhs[n] = three * T(x[n] - x[n - 1]) / max(T(s[n] - s[n - 1]), eps)
    end

    return _trisol!(a, b, c, rhs)
end

function _segspl(
    x::AbstractVector{<:Real},
    s::AbstractVector{<:Real};
    xs1::Real = -999.0,
    xs2::Real = -999.0,
)
    n = length(x)
    n == length(s) || throw(ArgumentError("x and s lengths must match"))
    n >= 2 || throw(ArgumentError("segspl needs at least 2 points"))
    abs(float(s[2] - s[1])) > 1.0e-15 || throw(ArgumentError("first input point duplicated"))
    abs(float(s[n] - s[n - 1])) > 1.0e-15 || throw(ArgumentError("last input point duplicated"))

    T = _panel_float_type(eltype(x), eltype(s))
    xs = zeros(T, n)
    iseg0 = 1
    for iseg in 2:(n - 2)
        if abs(float(s[iseg] - s[iseg + 1])) <= 1.0e-15
            nseg = iseg - iseg0 + 1
            xseg = _splind_segment(
                view(x, iseg0:iseg),
                view(s, iseg0:iseg);
                xs1 = xs1,
                xs2 = xs2,
            )
            xs[iseg0:iseg] .= xseg
            iseg0 = iseg + 1
        end
    end

    nseg = n - iseg0 + 1
    xseg = _splind_segment(
        view(x, iseg0:n),
        view(s, iseg0:n);
        xs1 = xs1,
        xs2 = xs2,
    )
    xs[iseg0:n] .= xseg
    return xs
end

function _seval(
    ss::Real,
    x::AbstractVector{<:Real},
    xs::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(xs) == length(s) || throw(ArgumentError("seval length mismatch"))
    T = _panel_float_type(eltype(x), eltype(xs), eltype(s))
    i = _spline_interval(ss, s)
    ds = T(s[i] - s[i - 1])
    t = (T(ss) - T(s[i - 1])) / ds
    one = T(1.0)
    cx1 = ds * T(xs[i - 1]) - T(x[i]) + T(x[i - 1])
    cx2 = ds * T(xs[i]) - T(x[i]) + T(x[i - 1])
    return (
        t * T(x[i]) + (one - t) * T(x[i - 1]) +
        (t - t * t) * ((one - t) * cx1 - t * cx2)
    )
end

function _deval(
    ss::Real,
    x::AbstractVector{<:Real},
    xs::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(xs) == length(s) || throw(ArgumentError("deval length mismatch"))
    T = _panel_float_type(eltype(x), eltype(xs), eltype(s))
    i = _spline_interval(ss, s)
    ds = T(s[i] - s[i - 1])
    t = (T(ss) - T(s[i - 1])) / ds
    cx1 = ds * T(xs[i - 1]) - T(x[i]) + T(x[i - 1])
    cx2 = ds * T(xs[i]) - T(x[i]) + T(x[i - 1])
    dx =
        T(x[i] - x[i - 1]) +
        (T(1.0) - T(4.0) * t + T(3.0) * t * t) * cx1 +
        t * (T(3.0) * t - T(2.0)) * cx2
    return dx / ds
end

function _d2val(
    ss::Real,
    x::AbstractVector{<:Real},
    xs::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(xs) == length(s) || throw(ArgumentError("d2val length mismatch"))
    T = _panel_float_type(eltype(x), eltype(xs), eltype(s))
    i = _spline_interval(ss, s)
    ds = T(s[i] - s[i - 1])
    t = (T(ss) - T(s[i - 1])) / ds
    cx1 = ds * T(xs[i - 1]) - T(x[i]) + T(x[i - 1])
    cx2 = ds * T(xs[i]) - T(x[i]) + T(x[i - 1])
    return ((T(6.0) * t - T(4.0)) * cx1 + (T(6.0) * t - T(2.0)) * cx2) / (ds * ds)
end

function _curv(
    ss::Real,
    x::AbstractVector{<:Real},
    xs::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(xs) == length(y) == length(ys) == length(s) ||
        throw(ArgumentError("curv length mismatch"))
    T = _panel_float_type(eltype(x), eltype(xs), eltype(y), eltype(ys), eltype(s))
    i = _spline_interval(ss, s)
    ds = T(s[i] - s[i - 1])
    t = (T(ss) - T(s[i - 1])) / ds

    cx1 = ds * T(xs[i - 1]) - T(x[i]) + T(x[i - 1])
    cx2 = ds * T(xs[i]) - T(x[i]) + T(x[i - 1])
    xd =
        T(x[i] - x[i - 1]) +
        (T(1.0) - T(4.0) * t + T(3.0) * t * t) * cx1 +
        t * (T(3.0) * t - T(2.0)) * cx2
    xdd = (T(6.0) * t - T(4.0)) * cx1 + (T(6.0) * t - T(2.0)) * cx2

    cy1 = ds * T(ys[i - 1]) - T(y[i]) + T(y[i - 1])
    cy2 = ds * T(ys[i]) - T(y[i]) + T(y[i - 1])
    yd =
        T(y[i] - y[i - 1]) +
        (T(1.0) - T(4.0) * t + T(3.0) * t * t) * cy1 +
        t * (T(3.0) * t - T(2.0)) * cy2
    ydd = (T(6.0) * t - T(4.0)) * cy1 + (T(6.0) * t - T(2.0)) * cy2

    sd = max(sqrt(xd * xd + yd * yd), T(0.001) * ds)
    return (xd * ydd - yd * xdd) / (sd * sd * sd)
end

function _lefind(
    x::AbstractVector{<:Real},
    xp::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    yp::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(xp) == length(y) == length(yp) == length(s) ||
        throw(ArgumentError("lefind length mismatch"))
    T = _panel_float_type(eltype(x), eltype(xp), eltype(y), eltype(yp), eltype(s))
    dseps = T(s[end] - s[1]) * T(1.0e-5)
    xte = T(0.5) * (T(x[1]) + T(x[end]))
    yte = T(0.5) * (T(y[1]) + T(y[end]))

    iguess = n - 2
    for i in 3:(n - 2)
        dxte = T(x[i]) - xte
        dyte = T(y[i]) - yte
        dx = T(x[i + 1] - x[i])
        dy = T(y[i + 1] - y[i])
        if dxte * dx + dyte * dy < 0.0
            iguess = i
            break
        end
    end

    sle = T(s[iguess])
    if T(s[iguess]) == T(s[iguess - 1])
        return sle
    end

    for _ in 1:50
        xle = _seval(sle, x, xp, s)
        yle = _seval(sle, y, yp, s)
        dxds = _deval(sle, x, xp, s)
        dyds = _deval(sle, y, yp, s)
        dxdd = _d2val(sle, x, xp, s)
        dydd = _d2val(sle, y, yp, s)

        xchord = xle - xte
        ychord = yle - yte
        res = xchord * dxds + ychord * dyds
        ress = dxds * dxds + dyds * dyds + xchord * dxdd + ychord * dydd
        dsle = -res / ress
        dslim = T(0.02) * abs(xchord + ychord)
        dsle = clamp(dsle, -dslim, dslim)
        sle += dsle
        abs(dsle) < dseps && return sle
    end

    return T(s[iguess])
end

function _xfoil_panel_airfoil(
    airfoil::Airfoil;
    paneling::XFoilPanelingParams = _XFOIL_DEFAULT_PANELING,
    load_normalize_digits::Union{Nothing,Int} = nothing,
    precision::Type{<:AbstractFloat} = _XFOIL_PANELING_PRECISION,
    return_state::Bool = false,
)
    airfoil_in =
        isnothing(load_normalize_digits) ? airfoil :
        _xfoil_load_normalized_airfoil(airfoil; digits = load_normalize_digits)
    xb64, yb64 = _xfoil_buffer_nodes(airfoil_in)
    T = precision
    xb = T.(xb64)
    yb = T.(yb64)
    nb = length(xb)
    nb >= 2 || throw(ArgumentError("buffer airfoil not available"))

    ipfac = 5
    n = paneling.npan

    sb = _scalc(xb, yb)
    xbp = _segspl(xb, sb)
    ybp = _segspl(yb, sb)

    half = T(0.5)
    zt = zero(T)
    ot = one(T)
    two = T(2.0)
    three = T(3.0)
    four = T(4.0)
    six = T(6.0)
    twenty = T(20.0)
    sbref = half * (sb[end] - sb[1])
    w5 = [abs(_curv(sb[i], xb, xbp, yb, ybp, sb)) * sbref for i in 1:nb]
    w5_raw = copy(w5)

    sble = _lefind(xb, xbp, yb, ybp, sb)
    cvle = abs(_curv(sble, xb, xbp, yb, ybp, sb)) * sbref

    ible = 0
    for i in 1:(nb - 1)
        if sble == sb[i] && sble == sb[i + 1]
            ible = i
            break
        end
    end

    xble = _seval(sble, xb, xbp, sb)
    yble = _seval(sble, yb, ybp, sb)
    xbte = half * (xb[1] + xb[end])
    ybte = half * (yb[1] + yb[end])
    chbsq = (xbte - xble)^2 + (ybte - yble)^2

    nk = 3
    cvsum = zt
    for k in (-nk):nk
        frac = T(k) / T(nk)
        sbk = sble + frac * sbref / max(cvle, twenty)
        cvsum += abs(_curv(sbk, xb, xbp, yb, ybp, sb)) * sbref
    end
    cvavg = cvsum / T(2 * nk + 1)
    ible != 0 && (cvavg = T(10.0))

    cc = six * T(paneling.cvpar)
    cvte = cvavg * T(paneling.cterat)
    w5[1] = cvte
    w5[end] = cvte

    smool = max(ot / max(cvavg, twenty), T(0.25) / T(div(paneling.npan, 2)))
    smoosq = (smool * sbref)^2

    w1 = zeros(T, nb)
    w2 = zeros(T, nb)
    w3 = zeros(T, nb)
    w2[1] = ot
    for i in 2:(nb - 1)
        dsm = sb[i] - sb[i - 1]
        dsp = sb[i + 1] - sb[i]
        dso = half * (sb[i + 1] - sb[i - 1])
        if dsm == zt || dsp == zt
            w1[i] = zt
            w2[i] = ot
            w3[i] = zt
        else
            w1[i] = smoosq * (-ot / dsm) / dso
            w2[i] = smoosq * (ot / dsp + ot / dsm) / dso + ot
            w3[i] = smoosq * (-ot / dsp) / dso
        end
    end
    w1[end] = zt
    w2[end] = ot
    w3[1] = zt

    for i in 2:(nb - 1)
        if sb[i] == sble || i == ible || i == ible + 1
            w1[i] = zt
            w2[i] = ot
            w3[i] = zt
            w5[i] = cvle
        elseif sb[i - 1] < sble && sb[i] > sble
            dsm = sb[i - 1] - sb[i - 2]
            dsp = sble - sb[i - 1]
            dso = half * (sble - sb[i - 2])
            w1[i - 1] = smoosq * (-ot / dsm) / dso
            w2[i - 1] = smoosq * (ot / dsp + ot / dsm) / dso + ot
            w3[i - 1] = zt
            w5[i - 1] += smoosq * cvle / (dsp * dso)

            dsm = sb[i] - sble
            dsp = sb[i + 1] - sb[i]
            dso = half * (sb[i + 1] - sble)
            w1[i] = zt
            w2[i] = smoosq * (ot / dsp + ot / dsm) / dso + ot
            w3[i] = smoosq * (-ot / dsp) / dso
            w5[i] += smoosq * cvle / (dsm * dso)
            break
        end
    end

    for i in 2:(nb - 1)
        xoc =
            (
                (xb[i] - xble) * (xbte - xble) +
                (yb[i] - yble) * (ybte - yble)
            ) / chbsq
        if sb[i] < sble
            if paneling.xsref1 < xoc < paneling.xsref2
                w1[i] = zt
                w2[i] = ot
                w3[i] = zt
                w5[i] = cvle * T(paneling.ctrrat)
            end
        else
            if paneling.xpref1 < xoc < paneling.xpref2
                w1[i] = zt
                w2[i] = ot
                w3[i] = zt
                w5[i] = cvle * T(paneling.ctrrat)
            end
        end
    end

    if ible == 0
        _trisol!(w2, w1, w3, w5)
    else
        _trisol!(view(w2, 1:ible), view(w1, 1:ible), view(w3, 1:ible), view(w5, 1:ible))
        _trisol!(
            view(w2, (ible + 1):nb),
            view(w1, (ible + 1):nb),
            view(w3, (ible + 1):nb),
            view(w5, (ible + 1):nb),
        )
    end

    cvmax = maximum(abs, w5)
    w5 ./= cvmax
    w6 = _segspl(w5, sb)

    nn = ipfac * (n - 1) + 1
    rdste = T(0.667)
    rtf = (rdste - ot) * two + ot
    snew = Vector{T}(undef, nn)
    nn1 = 0

    if ible == 0
        dsavg = (sb[end] - sb[1]) / (T(nn - 3) + two * rtf)
        snew[1] = sb[1]
        for i in 2:(nn - 1)
            snew[i] = sb[1] + dsavg * (T(i - 2) + rtf)
        end
        snew[end] = sb[end]
    else
        nfrac1 = div(n * ible, nb)
        nn1 = ipfac * (nfrac1 - 1) + 1
        dsavg1 = (sble - sb[1]) / (T(nn1 - 2) + rtf)
        snew[1] = sb[1]
        for i in 2:nn1
            snew[i] = sb[1] + dsavg1 * (T(i - 2) + rtf)
        end

        nn2 = nn - nn1 + 1
        dsavg2 = (sb[end] - sble) / (T(nn2 - 2) + rtf)
        for i in 2:(nn2 - 1)
            snew[i - 1 + nn1] = sble + dsavg2 * (T(i - 2) + rtf)
        end
        snew[end] = sb[end]
    end
    snew_init = copy(snew)

    w1n = zeros(T, nn)
    w2n = zeros(T, nn)
    w3n = zeros(T, nn)
    w4n = zeros(T, nn)
    for _ in 1:20
        cv1 = _seval(snew[1], w5, w6, sb)
        cv2 = _seval(snew[2], w5, w6, sb)
        cvs1 = _deval(snew[1], w5, w6, sb)
        cvs2 = _deval(snew[2], w5, w6, sb)

        cavm = sqrt(cv1^2 + cv2^2)
        if cavm == zt
            cavm_s1 = zt
            cavm_s2 = zt
        else
            cavm_s1 = cvs1 * cv1 / cavm
            cavm_s2 = cvs2 * cv2 / cavm
        end

        fill!(w1n, zt)
        fill!(w2n, zt)
        fill!(w3n, zt)
        fill!(w4n, zt)
        for i in 2:(nn - 1)
            dsm = snew[i] - snew[i - 1]
            dsp = snew[i] - snew[i + 1]
            cv3 = _seval(snew[i + 1], w5, w6, sb)
            cvs3 = _deval(snew[i + 1], w5, w6, sb)

            cavp = sqrt(cv3^2 + cv2^2)
            if cavp == zt
                cavp_s2 = zt
                cavp_s3 = zt
            else
                cavp_s2 = cvs2 * cv2 / cavp
                cavp_s3 = cvs3 * cv3 / cavp
            end

            fm = cc * cavm + ot
            fp = cc * cavp + ot
            rez = dsp * fp + dsm * fm

            w1n[i] = -fm + cc * dsm * cavm_s1
            w2n[i] = fp + fm + cc * (dsp * cavp_s2 + dsm * cavm_s2)
            w3n[i] = -fp + cc * dsp * cavp_s3
            w4n[i] = -rez

            cv1 = cv2
            cv2 = cv3
            cvs1 = cvs2
            cvs2 = cvs3
            cavm = cavp
            cavm_s1 = cavp_s2
            cavm_s2 = cavp_s3
        end

        w2n[1] = ot
        w3n[1] = zt
        w4n[1] = zt
        w1n[end] = zt
        w2n[end] = ot
        w4n[end] = zt

        if rtf != ot
            i = 2
            w4n[i] = -((snew[i] - snew[i - 1]) + rtf * (snew[i] - snew[i + 1]))
            w1n[i] = -ot
            w2n[i] = ot + rtf
            w3n[i] = -rtf

            i = nn - 1
            w4n[i] = -((snew[i] - snew[i + 1]) + rtf * (snew[i] - snew[i - 1]))
            w3n[i] = -ot
            w2n[i] = ot + rtf
            w1n[i] = -rtf
        end

        if ible != 0
            i = nn1
            w1n[i] = zt
            w2n[i] = ot
            w3n[i] = zt
            w4n[i] = sble - snew[i]
        end

        _trisol!(w2n, w1n, w3n, w4n)

        rlx = ot
        dmax = zt
        for i in 1:(nn - 1)
            ds = snew[i + 1] - snew[i]
            dds = w4n[i + 1] - w4n[i]
            dsrat = ot + rlx * dds / ds
            if dsrat > four
                rlx = (four - ot) * ds / dds
            end
            if dsrat < T(0.2)
                rlx = (T(0.2) - ot) * ds / dds
            end
            dmax = max(abs(w4n[i]), dmax)
        end

        for i in 2:(nn - 1)
            snew[i] += rlx * w4n[i]
        end
        abs(dmax) < T(1.0e-3) && break
    end
    snew_final = copy(snew)

    xnew = Vector{T}(undef, n)
    ynew = Vector{T}(undef, n)
    s_panel = Vector{T}(undef, n)
    for i in 1:n
        ind = ipfac * (i - 1) + 1
        s_panel[i] = snew[ind]
        xnew[i] = _seval(snew[ind], xb, xbp, sb)
        ynew[i] = _seval(snew[ind], yb, ybp, sb)
    end

    ncorn = 0
    for ib in 1:(nb - 1)
        if sb[ib] == sb[ib + 1]
            ncorn += 1
            xbcorn = xb[ib]
            ybcorn = yb[ib]
            sbcorn = sb[ib]
            for i in 1:length(s_panel)
                s_panel[i] <= sbcorn && continue

                insert!(xnew, i, xbcorn)
                insert!(ynew, i, ybcorn)
                insert!(s_panel, i, sbcorn)

                if i - 2 >= 1
                    s_panel[i - 1] = half * (s_panel[i] + s_panel[i - 2])
                    xnew[i - 1] = _seval(s_panel[i - 1], xb, xbp, sb)
                    ynew[i - 1] = _seval(s_panel[i - 1], yb, ybp, sb)
                end
                if i + 2 <= length(s_panel)
                    s_panel[i + 1] = half * (s_panel[i] + s_panel[i + 2])
                    xnew[i + 1] = _seval(s_panel[i + 1], xb, xbp, sb)
                    ynew[i + 1] = _seval(s_panel[i + 1], yb, ybp, sb)
                end
                break
            end
        end
    end

    if !return_state
        return Airfoil(airfoil_in.name, Float64.(xnew), Float64.(ynew))
    end

    s_final = _scalc(xnew, ynew)
    xp_final = _segspl(xnew, s_final)
    yp_final = _segspl(ynew, s_final)
    sle_final = _lefind(xnew, xp_final, ynew, yp_final, s_final)

    iscalars = Dict(
        "NB" => nb,
        "N" => length(xnew),
        "NPAN" => n,
        "IPFAC" => ipfac,
        "NN" => nn,
        "NN1" => nn1,
        "IBLE" => ible,
        "NCORN" => ncorn,
    )
    rscalars = Dict(
        "SBREF" => sbref,
        "SBLE" => sble,
        "CVLE" => cvle,
        "CVAVG" => cvavg,
        "CC" => cc,
        "CVTE" => cvte,
        "SMOOL" => smool,
        "SMOOSQ" => smoosq,
        "RTF" => rtf,
        "SLE" => sle_final,
    )
    arrays = Dict(
        "XB" => Float64.(xb),
        "YB" => Float64.(yb),
        "SB" => Float64.(sb),
        "XBP" => Float64.(xbp),
        "YBP" => Float64.(ybp),
        "W5_RAW" => Float64.(w5_raw),
        "W5_SMOOTH" => Float64.(w5),
        "W6" => Float64.(w6),
        "SNEW_INIT" => Float64.(snew_init),
        "SNEW_FINAL" => Float64.(snew_final),
        "X_PANEL" => Float64.(xnew),
        "Y_PANEL" => Float64.(ynew),
        "S_PANEL" => Float64.(s_final),
        "XP_PANEL" => Float64.(xp_final),
        "YP_PANEL" => Float64.(yp_final),
    )
    return (iscalars = iscalars, rscalars = rscalars, arrays = arrays)
end

function _ncalc(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
)
    n = length(x)
    n == length(y) == length(s) || throw(ArgumentError("x/y/s length mismatch"))

    xp = _segspl(x, s)
    yp = _segspl(y, s)
    T = promote_type(eltype(xp), eltype(yp), eltype(s))
    nx = Vector{T}(undef, n)
    ny = Vector{T}(undef, n)

    for i in 1:n
        sx = yp[i]
        sy = -xp[i]
        smod = max(sqrt(sx * sx + sy * sy), T(1.0e-15))
        nx[i] = sx / smod
        ny[i] = sy / smod
    end

    for i in 1:(n - 1)
        if abs(T(s[i] - s[i + 1])) <= T(1.0e-15)
            sx = T(0.5) * (nx[i] + nx[i + 1])
            sy = T(0.5) * (ny[i] + ny[i + 1])
            smod = max(sqrt(sx * sx + sy * sy), T(1.0e-15))
            nx[i] = sx / smod
            ny[i] = sy / smod
            nx[i + 1] = nx[i]
            ny[i + 1] = ny[i]
        end
    end

    return nx, ny, xp, yp
end

function _atanc(y::Real, x::Real, thold::Real)
    T = promote_type(typeof(float(y)), typeof(float(x)), typeof(float(thold)))
    pi_t = T(pi)
    tpi = T(2) * pi_t
    thnew = _xfoil_atan2(T(y), T(x))
    dthet = thnew - T(thold)
    sgnpi = dthet >= zero(T) ? pi_t : -pi_t
    dtcorr = dthet - tpi * trunc((dthet + sgnpi) / tpi)
    return T(thold) + dtcorr
end

@inline _xfoil_psilin_atan2(y::Float64, x::Float64) = atan(y, x)

@inline function _xfoil_psilin_atan2(y::Float32, x::Float32)
    @static if Sys.iswindows()
        return Float32(atan(Float64(y), Float64(x)))
    else
        return atan(y, x)
    end
end

function _atanc_psilin(y::Real, x::Real, thold::Real)
    T = promote_type(typeof(float(y)), typeof(float(x)), typeof(float(thold)))
    pi_t = T(pi)
    tpi = T(2) * pi_t
    thnew = _xfoil_psilin_atan2(T(y), T(x))
    dthet = thnew - T(thold)
    sgnpi = dthet >= zero(T) ? pi_t : -pi_t
    dtcorr = dthet - tpi * trunc((dthet + sgnpi) / tpi)
    return T(thold) + dtcorr
end

@inline _xfoil_log(x::Float64) = log(x)
@inline _xfoil_atan2(y::Float64, x::Float64) = atan(y, x)

@inline function _xfoil_log(x::Float32)
    @static if Sys.iswindows()
        # The gfortran-built XFoil binary rounds a promoted LOG result back to REAL.
        return Float32(log(Float64(x)))
    else
        return log(x)
    end
end

@inline function _xfoil_atan2(y::Float32, x::Float32)
    @static if Sys.iswindows()
        return ccall((:atan2f, "ucrtbase"), Cfloat, (Cfloat, Cfloat), y, x)
    else
        return atan(y, x)
    end
end

function _tecalc(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    xp::AbstractVector{<:Real},
    yp::AbstractVector{<:Real},
)
    n = length(x)
    n == length(y) == length(xp) == length(yp) || throw(ArgumentError("tecalc length mismatch"))

    T = promote_type(eltype(x), eltype(y), eltype(xp), eltype(yp))
    dxte = T(x[1] - x[n])
    dyte = T(y[1] - y[n])
    dxs = T(0.5) * T(-xp[1] + xp[n])
    dys = T(0.5) * T(-yp[1] + yp[n])

    ante = dxs * dyte - dys * dxte
    aste = dxs * dxte + dys * dyte
    dste = sqrt(dxte * dxte + dyte * dyte)
    chord = maximum(T.(x)) - minimum(T.(x))
    sharp = dste < T(1.0e-4) * max(chord, T(1.0e-12))

    scs, sds = if sharp
        (one(T), zero(T))
    else
        (ante / max(dste, T(1.0e-15)), aste / max(dste, T(1.0e-15)))
    end

    xte = T(0.5) * T(x[1] + x[n])
    yte = T(0.5) * T(y[1] + y[n])
    return (; sharp, dste, ante, aste, scs, sds, xte, yte)
end

function _apcalc(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    nx::AbstractVector{<:Real},
    ny::AbstractVector{<:Real},
    ;
    sharp::Bool,
)
    n = length(x)
    n == length(y) == length(nx) == length(ny) || throw(ArgumentError("apcalc length mismatch"))
    T = promote_type(eltype(x), eltype(y), eltype(nx), eltype(ny))
    ap = Vector{T}(undef, n)
    pi_t = T(pi)

    for i in 1:(n - 1)
        sx = T(x[i + 1] - x[i])
        sy = T(y[i + 1] - y[i])
        if sx == zero(T) && sy == zero(T)
            ap[i] = _xfoil_atan2(-T(ny[i]), -T(nx[i]))
        else
            ap[i] = _xfoil_atan2(sx, -sy)
        end
    end

    i = n
    ip = 1
    if sharp
        ap[i] = pi_t
    else
        sx = T(x[ip] - x[i])
        sy = T(y[ip] - y[i])
        ap[i] = _xfoil_atan2(-sx, sy) + pi_t
    end
    return ap
end

function _psilin_vortex_derivatives(
    io::Int,
    xi::Real,
    yi::Real,
    nxi::Real,
    nyi::Real,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    apanel::AbstractVector{<:Real},
    ;
    scs::Real,
    sds::Real,
) 
    n = length(x)
    n == length(y) == length(s) == length(apanel) ||
        throw(ArgumentError("psilin length mismatch"))

    T = _panel_float_type(eltype(x), eltype(y), eltype(s), eltype(apanel))
    qopi = T(0.25 / pi)
    hopi = T(0.50 / pi)
    seps = max(T(s[n] - s[1]), T(1.0e-12)) * T(1.0e-5)

    dzdg = zeros(T, n)
    dqdg = zeros(T, n)
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
            d2 = (T(x[jo] - x[jp]))^2 + (T(y[jo] - y[jp]))^2
            if d2 < seps^2
                te_skip = true
                break
            end
        end

        dso = sqrt((T(x[jo] - x[jp]))^2 + (T(y[jo] - y[jp]))^2)
        dso <= zero(T) && continue
        dsio = one(T) / dso
        apan = T(apanel[jo])

        rx1 = T(xi - x[jo])
        ry1 = T(yi - y[jo])
        rx2 = T(xi - x[jp])
        ry2 = T(yi - y[jp])

        sx = T(x[jp] - x[jo]) * dsio
        sy = T(y[jp] - y[jo]) * dsio

        x1 = sx * rx1 + sy * ry1
        x2 = sx * rx2 + sy * ry2
        yy = sx * ry1 - sy * rx1

        rs1 = rx1 * rx1 + ry1 * ry1
        rs2 = rx2 * rx2 + ry2 * ry2

        sgn = if 1 <= io <= n
            one(T)
        else
            copysign(one(T), yy)
        end

        if io != jo && rs1 > zero(T)
            g1 = _xfoil_log(rs1)
            t1 = _xfoil_atan2(sgn * x1, sgn * yy) + (T(0.5) - T(0.5) * sgn) * T(pi)
        else
            g1 = zero(T)
            t1 = zero(T)
        end

        if io != jp && rs2 > zero(T)
            g2 = _xfoil_log(rs2)
            t2 = _xfoil_atan2(sgn * x2, sgn * yy) + (T(0.5) - T(0.5) * sgn) * T(pi)
        else
            g2 = zero(T)
            t2 = zero(T)
        end

        x1i = sx * T(nxi) + sy * T(nyi)
        x2i = x1i
        yyi = sx * T(nyi) - sy * T(nxi)

        jo == n && break

        dxinv = one(T) / (x1 - x2)
        psis = T(0.5) * x1 * g1 - T(0.5) * x2 * g2 + x2 - x1 + yy * (t1 - t2)
        psid =
            ((x1 + x2) * psis + T(0.5) * (rs2 * g2 - rs1 * g1 + x1^2 - x2^2)) * dxinv

        psx1 = T(0.5) * g1
        psx2 = -T(0.5) * g2
        psyy = t1 - t2
        pdx1 = ((x1 + x2) * psx1 + psis - x1 * g1 - psid) * dxinv
        pdx2 = ((x1 + x2) * psx2 + psis + x2 * g2 + psid) * dxinv
        pdyy = ((x1 + x2) * psyy - yy * (g1 - g2)) * dxinv

        dzdg[jo] += qopi * (psis - psid)
        dzdg[jp] += qopi * (psis + psid)

        psni = psx1 * x1i + psx2 * x2i + psyy * yyi
        pdni = pdx1 * x1i + pdx2 * x2i + pdyy * yyi
        dqdg[jo] += qopi * (psni - pdni)
        dqdg[jp] += qopi * (psni + pdni)
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

        dzdg[jo_te] -= hopi * psig * T(scs) * T(0.5)
        dzdg[jp_te] += hopi * psig * T(scs) * T(0.5)
        dzdg[jo_te] += hopi * pgam * T(sds) * T(0.5)
        dzdg[jp_te] -= hopi * pgam * T(sds) * T(0.5)

        dqdg[jo_te] -= hopi * (psigni * T(0.5) * T(scs) - pgamni * T(0.5) * T(sds))
        dqdg[jp_te] += hopi * (psigni * T(0.5) * T(scs) - pgamni * T(0.5) * T(sds))
    end

    return dzdg, dqdg
end

function _psilin_source_vortex_derivatives(
    io::Int,
    xi::Real,
    yi::Real,
    nxi::Real,
    nyi::Real,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    apanel::AbstractVector{<:Real},
    ;
    scs::Real,
    sds::Real,
)
    n = length(x)
    n == length(y) == length(s) == length(apanel) ||
        throw(ArgumentError("psilin length mismatch"))

    T = promote_type(
        typeof(float(xi)),
        typeof(float(yi)),
        typeof(float(nxi)),
        typeof(float(nyi)),
        typeof(float(x[1])),
        typeof(float(y[1])),
        typeof(float(s[1])),
        typeof(float(apanel[1])),
        typeof(float(scs)),
        typeof(float(sds)),
    )
    pi_t = T(pi)
    qopi = T(0.25) / pi_t
    hopi = T(0.50) / pi_t
    seps = max(T(s[n] - s[1]), T(1.0e-12)) * T(1.0e-5)

    dzdg = zeros(T, n)
    dqdg = zeros(T, n)
    dzdm = zeros(T, n)
    dqdm = zeros(T, n)
    te_skip = false
    surface_row = 1 <= io <= n
    te_bisector_row = io == 0
    use_atanc_angles = false
    thetaj = zeros(T, n)

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
        jm = jo == 1 ? jo : jo - 1
        jq = jo == n - 1 ? jp : jp + 1
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
        dso <= zero(T) && continue
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
        sgn = surface_row ? one(T) : copysign(one(T), yy)

        if io != jo && rs1 > zero(T)
            g1 = _xfoil_log(rs1)
            if use_atanc_angles
                t1 = _atanc_psilin(x1, yy, jo == 1 ? zero(T) : thetaj[jo - 1])
            else
                t1 =
                    _xfoil_psilin_atan2(sgn * x1, sgn * yy) +
                    (T(0.5) - T(0.5) * sgn) * pi_t
            end
        else
            g1 = zero(T)
            t1 = zero(T)
        end
        use_atanc_angles && (thetaj[jo] = t1)

        if io != jp && rs2 > zero(T)
            g2 = _xfoil_log(rs2)
            if use_atanc_angles
                t2 = _atanc_psilin(x2, yy, thetaj[jo])
            else
                t2 =
                    _xfoil_psilin_atan2(sgn * x2, sgn * yy) +
                    (T(0.5) - T(0.5) * sgn) * pi_t
            end
        else
            g2 = zero(T)
            t2 = zero(T)
        end
        use_atanc_angles && (thetaj[jp] = t2)

        x1i = sx * T(nxi) + sy * T(nyi)
        x2i = x1i
        yyi = sx * T(nyi) - sy * T(nxi)

        # Match XFoil's `IF(JO.EQ.N) GO TO 11` control flow: the final
        # trailing-edge panel bypasses the standard vortex-panel update and
        # contributes only through the TE correction below.
        jo == n && break

        if jo != n
            x0 = T(0.5) * (x1 + x2)
            rs0 = x0 * x0 + yy * yy
            g0 = _xfoil_log(rs0)
            t0 =
                if use_atanc_angles
                    _atanc_psilin(x0, yy, thetaj[jo])
                else
                    _xfoil_psilin_atan2(sgn * x0, sgn * yy) +
                    (T(0.5) - T(0.5) * sgn) * pi_t
                end

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

            dsm = sqrt((T(x[jp]) - T(x[jm]))^2 + (T(y[jp]) - T(y[jm]))^2)
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

            dsp = sqrt((T(x[jq]) - T(x[jo]))^2 + (T(y[jq]) - T(y[jo]))^2)
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

        dx = x1 - x2
        if abs(dx) <= T(1.0e-15)
            dx = dx >= zero(T) ? T(1.0e-15) : -T(1.0e-15)
        end
        dxinv = inv(dx)
        psis = T(0.5) * x1 * g1 - T(0.5) * x2 * g2 + x2 - x1 + yy * (t1 - t2)
        psid = ((x1 + x2) * psis + T(0.5) * (rs2 * g2 - rs1 * g1 + x1^2 - x2^2)) * dxinv

        psx1 = T(0.5) * g1
        psx2 = -T(0.5) * g2
        psyy = t1 - t2
        pdx1 = ((x1 + x2) * psx1 + psis - x1 * g1 - psid) * dxinv
        pdx2 = ((x1 + x2) * psx2 + psis + x2 * g2 + psid) * dxinv
        pdyy = ((x1 + x2) * psyy - yy * (g1 - g2)) * dxinv

        dzdg[jo] += qopi * (psis - psid)
        dzdg[jp] += qopi * (psis + psid)

        psni = psx1 * x1i + psx2 * x2i + psyy * yyi
        pdni = pdx1 * x1i + pdx2 * x2i + pdyy * yyi
        dqdg[jo] += qopi * (psni - pdni)
        dqdg[jp] += qopi * (psni + pdni)
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

        dzdg[jo_te] -= hopi * psig * T(scs) * T(0.5)
        dzdg[jp_te] += hopi * psig * T(scs) * T(0.5)
        dzdg[jo_te] += hopi * pgam * T(sds) * T(0.5)
        dzdg[jp_te] -= hopi * pgam * T(sds) * T(0.5)

        dqdg[jo_te] -= hopi * (psigni * T(0.5) * T(scs) - pgamni * T(0.5) * T(sds))
        dqdg[jp_te] += hopi * (psigni * T(0.5) * T(scs) - pgamni * T(0.5) * T(sds))
    end

    return dzdg, dqdg, dzdm, dqdm
end

function _surface_source_influence(inv::CaseResult)
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
    bij = zeros(T, n + 1, n)
    for i in 1:n
        dzdg, _, dzdm, _ = _psilin_source_vortex_derivatives(
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
            bij[i, j] = -dzdm[j]
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
        _, dqdg, _, dqdm = _psilin_source_vortex_derivatives(
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
            bij[n, j] = -dqdm[j]
        end
        aij[n, n + 1] = zero(T)
    end

    fac = _xfoil_lu(aij; precision = T)
    dij = Matrix{T}(undef, n, n)
    rhs = zeros(T, n + 1)
    for j in 1:n
        rhs .= bij[:, j]
        sol = _xfoil_lu_solve!(fac, rhs)
        dij[:, j] = sol[1:n]
    end
    return Float64.(dij)
end

function _ggcalc_unit_vorticity(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    nx::AbstractVector{<:Real},
    ny::AbstractVector{<:Real},
    xp::AbstractVector{<:Real},
    yp::AbstractVector{<:Real},
    apanel::AbstractVector{<:Real},
    te;
    qinf::Float64 = 1.0,
)
    n = length(x)
    n >= 3 || throw(ArgumentError("need at least 3 nodes"))

    T = _XFOIL_GGCALC_PRECISION
    xt = T.(x)
    yt = T.(y)
    st = T.(s)
    nxt = T.(nx)
    nyt = T.(ny)
    xpt = T.(xp)
    ypt = T.(yp)
    apt = T.(apanel)

    aij = zeros(T, n + 1, n + 1)
    rhs1 = zeros(T, n + 1)
    rhs2 = zeros(T, n + 1)

    for i in 1:n
        dzdg, _ = _psilin_vortex_derivatives(
            i,
            xt[i],
            yt[i],
            nxt[i],
            nyt[i],
            xt,
            yt,
            st,
            apt;
            scs = te.scs,
            sds = te.sds,
        )
        for j in 1:n
            aij[i, j] = T(dzdg[j])
        end
        aij[i, n + 1] = -one(T)

        rhs1[i] = -T(qinf) * yt[i]
        rhs2[i] = T(qinf) * xt[i]
    end

    aij[n + 1, :] .= zero(T)
    aij[n + 1, 1] = one(T)
    aij[n + 1, n] = one(T)

    if te.sharp
        bwt = T(0.1)
        ag1 = _xfoil_atan2(-ypt[1], -xpt[1])
        ag2 = _atanc(ypt[n], xpt[n], ag1)
        abis = T(0.5) * (ag1 + ag2)
        cbis = cos(abis)
        sbis = sin(abis)

        ds1 = sqrt((xt[1] - xt[2])^2 + (yt[1] - yt[2])^2)
        ds2 = sqrt((xt[n] - xt[n - 1])^2 + (yt[n] - yt[n - 1])^2)
        dsmin = min(ds1, ds2)

        xbis = T(te.xte) - bwt * dsmin * cbis
        ybis = T(te.yte) - bwt * dsmin * sbis
        _, dqdg = _psilin_vortex_derivatives(
            0,
            xbis,
            ybis,
            -sbis,
            cbis,
            xt,
            yt,
            st,
            apt;
            scs = te.scs,
            sds = te.sds,
        )
        for j in 1:n
            aij[n, j] = T(dqdg[j])
        end
        aij[n, n + 1] = zero(T)
        rhs1[n] = -cbis
        rhs2[n] = -sbis
    end

    fac = _xfoil_lu(aij; precision = T)
    gamu1 = _xfoil_lu_solve!(fac, copy(rhs1))
    gamu2 = _xfoil_lu_solve!(fac, copy(rhs2))
    return Float64.(gamu1[1:n]), Float64.(gamu2[1:n])
end

function _panel_geometry(airfoil::Airfoil)
    x, y = _dedupe_points(copy(airfoil.x), copy(airfoil.y))
    n = length(x)
    n >= 4 || throw(ArgumentError("airfoil needs at least 4 points"))

    m = n - 1
    dx = x[2:end] .- x[1:(end - 1)]
    dy = y[2:end] .- y[1:(end - 1)]
    s = hypot.(dx, dy)
    any(s .<= 1.0e-12) && throw(ArgumentError("degenerate panel length found"))

    theta = atan.(dy, dx)
    xc = 0.5 .* (x[1:(end - 1)] .+ x[2:end])
    yc = 0.5 .* (y[1:(end - 1)] .+ y[2:end])
    tx = dx ./ s
    ty = dy ./ s
    nx = dy ./ s
    ny = -dx ./ s

    chord = maximum(x) - minimum(x)
    chord > 0.0 || throw(ArgumentError("invalid chord length"))

    return (; x, y, m, s, theta, xc, yc, tx, ty, nx, ny, chord)
end

function _influence_coeffs(geom)
    m = geom.m
    a = Matrix{Float64}(undef, m, m) # normal source influence
    b = Matrix{Float64}(undef, m, m) # tangential source influence
    two_pi = 2.0 * pi

    @inbounds for i in 1:m
        for j in 1:m
            if i == j
                a[i, j] = 0.5
                b[i, j] = 0.0
                continue
            end

            xij = geom.xc[i] - geom.x[j]
            yij = geom.yc[i] - geom.y[j]

            aa = -(xij * cos(geom.theta[j]) + yij * sin(geom.theta[j]))
            bb = xij^2 + yij^2
            ee = xij * sin(geom.theta[j]) - yij * cos(geom.theta[j])

            bb = max(bb, 1.0e-20)
            arg = 1.0 + (geom.s[j]^2 + 2.0 * aa * geom.s[j]) / bb
            arg = max(arg, 1.0e-20)

            ff = log(arg)
            gg = atan(ee * geom.s[j], bb + aa * geom.s[j])

            cc = sin(geom.theta[i] - geom.theta[j])
            dd = cos(geom.theta[i] - geom.theta[j])

            a[i, j] = (0.5 * cc * ff + dd * gg) / two_pi
            b[i, j] = (0.5 * dd * ff - cc * gg) / two_pi
        end
    end
    return a, b
end

function _kutta_panel_pair(geom)
    j_upper = 1
    j_lower = geom.m

    # If the last panel is just the finite-TE closure segment from
    # lower-TE to upper-TE, Kutta must use the adjacent lower surface panel.
    xtol = 1.0e-2 * max(geom.chord, 1.0)
    ytol = 1.0e-10 * max(geom.chord, 1.0)
    xmax = maximum(geom.x)
    if abs(geom.x[geom.m] - geom.x[1]) <= xtol &&
       min(geom.x[geom.m], geom.x[1]) >= xmax - xtol &&
       abs(geom.y[geom.m] - geom.y[1]) > ytol &&
       geom.m >= 2
        j_lower = geom.m - 1
    end

    return j_upper, j_lower
end

function _panel_to_node_speed(v_panel::AbstractVector{<:Real})
    m = length(v_panel)
    m >= 2 || throw(ArgumentError("need at least two panel speeds"))
    q = Vector{Float64}(undef, m + 1)
    q[1] = abs(float(v_panel[1]))
    for i in 2:m
        q[i] = 0.5 * (abs(float(v_panel[i - 1])) + abs(float(v_panel[i])))
    end
    q[m + 1] = q[1]
    return q
end

function _clcalc_from_surface_speed(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    q::AbstractVector{<:Real},
    alpha_deg::Real;
    xref::Float64 = 0.25,
    yref::Float64 = 0.0,
    qinf::Float64 = 1.0,
    minf::Float64 = 0.0,
)
    n = length(x)
    n == length(y) || throw(ArgumentError("x/y length mismatch"))
    n == length(q) || throw(ArgumentError("q length mismatch"))
    n >= 3 || throw(ArgumentError("need at least three nodes"))

    T = _XFOIL_FORCE_PRECISION
    xt = T.(x)
    yt = T.(y)
    qt = T.(q)
    qinft = T(qinf)
    minft = T(minf)
    xreft = T(xref)
    yreft = T(yref)
    alpha = T(deg2rad(float(alpha_deg)))
    sa = sin(alpha)
    ca = cos(alpha)

    beta = sqrt(max(one(T) - minft * minft, T(1.0e-12)))
    bfac = T(0.5) * minft * minft / (one(T) + beta)

    cl = zero(T)
    cm = zero(T)
    cdp = zero(T)

    cp1_inc = one(T) - (qt[1] / qinft)^2
    cp1 = cp1_inc / max(beta + bfac * cp1_inc, T(1.0e-12))

    for i in 1:n
        ip = (i == n) ? 1 : i + 1
        cp2_inc = one(T) - (qt[ip] / qinft)^2
        cp2 = cp2_inc / max(beta + bfac * cp2_inc, T(1.0e-12))

        dx = (xt[ip] - xt[i]) * ca + (yt[ip] - yt[i]) * sa
        dy = (yt[ip] - yt[i]) * ca - (xt[ip] - xt[i]) * sa
        dg = cp2 - cp1

        ax = (T(0.5) * (xt[ip] + xt[i]) - xreft) * ca + (T(0.5) * (yt[ip] + yt[i]) - yreft) * sa
        ay = (T(0.5) * (yt[ip] + yt[i]) - yreft) * ca - (T(0.5) * (xt[ip] + xt[i]) - xreft) * sa
        ag = T(0.5) * (cp2 + cp1)

        cl += dx * ag
        cdp -= dy * ag
        cm -= dx * (ag * ax + dg * dx / T(12.0))
        cm -= dy * (ag * ay + dg * dy / T(12.0))

        cp1 = cp2
    end

    return Float64(cl), Float64(cm), Float64(cdp)
end

function solve_inviscid_case(
    airfoil::Airfoil,
    alpha_deg::Real;
    paneling::XFoilPanelingParams = _XFOIL_DEFAULT_PANELING,
    repanel::Bool = true,
    xfoil_load_normalize_digits::Union{Nothing,Int} = nothing,
)
    paneled =
        repanel ?
        _xfoil_panel_airfoil(
            airfoil;
            paneling = paneling,
            load_normalize_digits = xfoil_load_normalize_digits,
        ) : airfoil
    x, y = _xfoil_nodes(paneled)
    T = _XFOIL_GGCALC_PRECISION
    xt = T.(x)
    yt = T.(y)
    s = _scalc(xt, yt)
    nx, ny, xp, yp = _ncalc(xt, yt, s)
    te = _tecalc(xt, yt, xp, yp)
    apanel = _apcalc(xt, yt, nx, ny; sharp = te.sharp)

    q0, q90 = _ggcalc_unit_vorticity(xt, yt, s, nx, ny, xp, yp, apanel, te; qinf = 1.0)
    alpha = T(deg2rad(float(alpha_deg)))
    cosa = cos(alpha)
    sina = sin(alpha)
    q0t = T.(q0)
    q90t = T.(q90)
    q_nodes = @. cosa * q0t + sina * q90t

    xnode = Float64.(xt)
    ynode = Float64.(yt)
    vt = Float64.(q_nodes)
    cp = Float64.(one(T) .- q_nodes .^ 2)

    cl, cm, _ = _clcalc_from_surface_speed(
        xt,
        yt,
        q_nodes,
        alpha_deg;
        xref = 0.25,
        yref = 0.0,
        qinf = 1.0,
        minf = 0.0,
    )

    return CaseResult(float(alpha_deg), xnode, ynode, cp, vt, cl, cm, xnode, ynode, vt)
end

# ============================================================================
# XFoil-exact PANGEN paneling and xfoil_naca4_airfoil
# ============================================================================

"""
    _xfoil_pangen(xb, yb; params) -> Airfoil

Redistribute buffer airfoil coordinates using XFoil's PANGEN algorithm
(curvature-adaptive repaneling).  Produces the SAME panel coordinates
as XFoil's PANE command.
"""
function _xfoil_pangen(
    xb::AbstractVector{<:Real},
    yb::AbstractVector{<:Real};
    params::XFoilPanelingParams = XFoilPanelingParams(),
    name::String = "pangen",
)
    nb = length(xb)
    nb >= 5 || throw(ArgumentError("Buffer must have at least 5 points"))
    npan = params.npan
    cvpar = Float64(params.cvpar)
    cterat = Float64(params.cterat)
    ctrrat = Float64(params.ctrrat)
    xsref1 = Float64(params.xsref1)
    xsref2 = Float64(params.xsref2)
    xpref1 = Float64(params.xpref1)
    xpref2 = Float64(params.xpref2)

    ipfac = 5
    n = npan

    # Spline setup
    xbf = Float64.(xb)
    ybf = Float64.(yb)
    sb = _scalc(xbf, ybf)
    xbp = _segspl(xbf, sb)
    ybp = _segspl(ybf, sb)
    sbref = 0.5 * (sb[end] - sb[1])

    # Curvature at buffer points
    w5 = [abs(_curv(sb[i], xbf, xbp, ybf, ybp, sb)) * sbref for i in 1:nb]

    # Leading edge
    sble = _lefind(xbf, xbp, ybf, ybp, sb)
    cvle = abs(_curv(sble, xbf, xbp, ybf, ybp, sb)) * sbref

    # Check for sharp LE (doubled arc-length)
    ible = 0
    for i in 1:nb-1
        if sble == sb[i] && sble == sb[i+1]
            ible = i
            break
        end
    end

    # LE and TE coordinates
    xble = _seval(sble, xbf, xbp, sb)
    yble = _seval(sble, ybf, ybp, sb)
    xbte = 0.5 * (xbf[1] + xbf[end])
    ybte = 0.5 * (ybf[1] + ybf[end])
    chbsq = (xbte - xble)^2 + (ybte - yble)^2

    # Average curvature around LE
    nk = 3
    cvsum = 0.0
    for k in -nk:nk
        frac = Float64(k) / Float64(nk)
        sbk = sble + frac * sbref / max(cvle, 20.0)
        cvk = abs(_curv(sbk, xbf, xbp, ybf, ybp, sb)) * sbref
        cvsum += cvk
    end
    cvavg = cvsum / Float64(2 * nk + 1)
    if ible != 0; cvavg = 10.0; end

    cc = 6.0 * cvpar

    # Artificial TE curvature
    cvte = cvavg * cterat
    w5[1] = cvte
    w5[nb] = cvte

    # Smoothing
    smool = max(1.0 / max(cvavg, 20.0), 0.25 / Float64(npan ÷ 2))
    smoosq = (smool * sbref)^2

    # Build tridiagonal for smoothed curvature
    w1 = zeros(nb); w2 = ones(nb); w3 = zeros(nb)
    for i in 2:nb-1
        dsm = sb[i] - sb[i-1]
        dsp = sb[i+1] - sb[i]
        dso = 0.5 * (sb[i+1] - sb[i-1])
        if dsm == 0.0 || dsp == 0.0
            w1[i] = 0.0; w2[i] = 1.0; w3[i] = 0.0
        else
            w1[i] = smoosq * (-1.0 / dsm) / dso
            w2[i] = smoosq * (1.0 / dsp + 1.0 / dsm) / dso + 1.0
            w3[i] = smoosq * (-1.0 / dsp) / dso
        end
    end

    # Fix curvature at LE
    for i in 2:nb-1
        if sb[i] == sble || i == ible || i == ible + 1
            w1[i] = 0.0; w2[i] = 1.0; w3[i] = 0.0; w5[i] = cvle
        elseif i >= 3 && sb[i-1] < sble && sb[i] > sble
            dsm = sb[i-1] - sb[i-2]
            dsp = sble - sb[i-1]
            dso = 0.5 * (sble - sb[i-2])
            w1[i-1] = smoosq * (-1.0 / dsm) / dso
            w2[i-1] = smoosq * (1.0 / dsp + 1.0 / dsm) / dso + 1.0
            w3[i-1] = 0.0
            w5[i-1] += smoosq * cvle / (dsp * dso)

            dsm2 = sb[i] - sble
            dsp2 = sb[i+1] - sb[i]
            dso2 = 0.5 * (sb[i+1] - sble)
            w1[i] = 0.0
            w2[i] = smoosq * (1.0 / dsp2 + 1.0 / dsm2) / dso2 + 1.0
            w3[i] = smoosq * (-1.0 / dsp2) / dso2
            w5[i] += smoosq * cvle / (dsm2 * dso2)
            break
        end
    end

    # Artificial curvature at refinement regions
    for i in 2:nb-1
        xoc = ((xbf[i] - xble) * (xbte - xble) + (ybf[i] - yble) * (ybte - yble)) / max(chbsq, 1e-30)
        if sb[i] < sble
            if xoc > xsref1 && xoc < xsref2
                w1[i] = 0.0; w2[i] = 1.0; w3[i] = 0.0; w5[i] = cvle * ctrrat
            end
        else
            if xoc > xpref1 && xoc < xpref2
                w1[i] = 0.0; w2[i] = 1.0; w3[i] = 0.0; w5[i] = cvle * ctrrat
            end
        end
    end

    # Solve for smoothed curvature
    # _trisol!(main, lower, upper, rhs) — XFoil calls TRISOL(W2, W1, W3, W5)
    if ible == 0
        _trisol!(w2, w1, w3, w5)
    else
        _trisol!(view(w2, 1:ible), view(w1, 1:ible), view(w3, 1:ible), view(w5, 1:ible))
        r = ible + 1
        _trisol!(view(w2, r:nb), view(w1, r:nb), view(w3, r:nb), view(w5, r:nb))
    end

    # Normalize
    cvmax = maximum(abs, w5)
    cvmax > 0 && (w5 ./= cvmax)

    # Spline normalized curvature
    w6 = _segspl(w5, sb)

    # Initial node positions (uniform with TE bunching)
    nn = ipfac * (n - 1) + 1
    rdste = 0.667
    rtf = (rdste - 1.0) * 2.0 + 1.0
    snew = zeros(nn)
    nn1 = 0

    if ible == 0
        dsavg = (sb[end] - sb[1]) / (Float64(nn - 3) + 2.0 * rtf)
        snew[1] = sb[1]
        for i in 2:nn-1
            snew[i] = sb[1] + dsavg * (Float64(i - 2) + rtf)
        end
        snew[nn] = sb[end]
    else
        nfrac1 = (n * ible) ÷ nb
        nn1 = ipfac * (nfrac1 - 1) + 1
        dsavg1 = (sble - sb[1]) / (Float64(nn1 - 2) + rtf)
        snew[1] = sb[1]
        for i in 2:nn1
            snew[i] = sb[1] + dsavg1 * (Float64(i - 2) + rtf)
        end
        nn2 = nn - nn1 + 1
        dsavg2 = (sb[end] - sble) / (Float64(nn2 - 2) + rtf)
        for i in 2:nn2-1
            snew[i - 1 + nn1] = sble + dsavg2 * (Float64(i - 2) + rtf)
        end
        snew[nn] = sb[end]
    end

    # Newton iteration for node placement
    w4 = zeros(nn)
    w1n = zeros(nn); w2n = ones(nn); w3n = zeros(nn)

    for _iter in 1:20
        cv1 = _seval(snew[1], w5, w6, sb)
        cv2 = _seval(snew[2], w5, w6, sb)
        cvs1 = _deval(snew[1], w5, w6, sb)
        cvs2 = _deval(snew[2], w5, w6, sb)

        cavm = sqrt(cv1^2 + cv2^2)
        if cavm > 0
            cavm_s1 = cvs1 * cv1 / cavm
            cavm_s2 = cvs2 * cv2 / cavm
        else
            cavm_s1 = 0.0; cavm_s2 = 0.0
        end

        for i in 2:nn-1
            dsm = snew[i] - snew[i-1]
            dsp = snew[i] - snew[i+1]
            cv3 = _seval(snew[i+1], w5, w6, sb)
            cvs3 = _deval(snew[i+1], w5, w6, sb)

            cavp = sqrt(cv3^2 + cv2^2)
            if cavp > 0
                cavp_s2 = cvs2 * cv2 / cavp
                cavp_s3 = cvs3 * cv3 / cavp
            else
                cavp_s2 = 0.0; cavp_s3 = 0.0
            end

            fm = cc * cavm + 1.0
            fp = cc * cavp + 1.0
            rez = dsp * fp + dsm * fm

            w1n[i] = -fm + cc * dsm * cavm_s1
            w2n[i] = fp + fm + cc * (dsp * cavp_s2 + dsm * cavm_s2)
            w3n[i] = -fp + cc * dsp * cavp_s3
            w4[i] = -rez

            cv1 = cv2; cv2 = cv3
            cvs1 = cvs2; cvs2 = cvs3
            cavm = cavp; cavm_s1 = cavp_s2; cavm_s2 = cavp_s3
        end

        # BCs: fix endpoints
        w2n[1] = 1.0; w3n[1] = 0.0; w4[1] = 0.0
        w1n[nn] = 0.0; w2n[nn] = 1.0; w4[nn] = 0.0

        # TE panel bunching
        if rtf != 1.0
            i = 2
            w4[i] = -((snew[i] - snew[i-1]) + rtf * (snew[i] - snew[i+1]))
            w1n[i] = -1.0; w2n[i] = 1.0 + rtf; w3n[i] = -rtf

            i = nn - 1
            w4[i] = -((snew[i] - snew[i+1]) + rtf * (snew[i] - snew[i-1]))
            w3n[i] = -1.0; w2n[i] = 1.0 + rtf; w1n[i] = -rtf
        end

        # Fix sharp LE
        if ible != 0 && nn1 > 0
            i = nn1
            w1n[i] = 0.0; w2n[i] = 1.0; w3n[i] = 0.0; w4[i] = sble - snew[i]
        end

        # Solve — _trisol!(main, lower, upper, rhs)
        _trisol!(w2n, w1n, w3n, w4)

        # Under-relaxation
        rlx = 1.0
        dmax = 0.0
        for i in 1:nn-1
            ds = snew[i+1] - snew[i]
            dds = w4[i+1] - w4[i]
            dsrat = 1.0 + rlx * dds / ds
            if dsrat > 4.0; rlx = (4.0 - 1.0) * ds / dds; end
            if dsrat < 0.2; rlx = (0.2 - 1.0) * ds / dds; end
            dmax = max(abs(w4[i]), dmax)
        end

        for i in 2:nn-1
            snew[i] += rlx * w4[i]
        end

        abs(dmax) < 1e-3 && break
    end

    # Sub-sample to N panel nodes
    xout = Vector{Float64}(undef, n)
    yout = Vector{Float64}(undef, n)
    for i in 1:n
        ind = ipfac * (i - 1) + 1
        xout[i] = _seval(snew[ind], xbf, xbp, sb)
        yout[i] = _seval(snew[ind], ybf, ybp, sb)
    end

    return Airfoil(name, xout, yout)
end

"""
    xfoil_naca4_airfoil(code; npan=160) -> Airfoil

Generate NACA 4-digit airfoil using the EXACT same algorithm as XFoil's
`NACA XXXX` + `PANE` command sequence.  This produces identical panel
coordinates for machine-precision parity with XFoil's viscous analysis.
"""
function xfoil_naca4_airfoil(code::Integer; npan::Int = 160)
    xb, yb = _xfoil_naca4_buffer(code; nside = 120)
    params = XFoilPanelingParams(npan = npan)
    return _xfoil_pangen(xb, yb; params = params, name = "NACA$(lpad(code, 4, '0'))")
end
