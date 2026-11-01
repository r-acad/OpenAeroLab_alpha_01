module EquationResidualProbe

using Printf: @sprintf

export reference_scales, assess, history_header, history_row

const COMPONENTS = (:mass, :mom_x, :mom_y, :energy)
const METRICS = (:linf_raw, :l2_raw, :linf_convective, :l2_convective,
                 :signed_flux_fraction, :peak_i, :peak_j)

"""
    reference_scales(; rho_inf, ux_inf, uy_inf, temp_inf, eint, chord)

References for a two-dimensional nondimensional air solve. `eint` is the
solver's internal-energy coefficient in E = rho*(eint*T + |u|^2/2).
Each local residual is `(dc/vol) * (chord/U_inf) / Q_inf`, where Q_inf is
rho, rho*U, rho*U, or E for mass, x/y momentum, and energy respectively.
The signed net is `sum(dc)/(Q_inf*U_inf*chord)` per unit span.
"""
function reference_scales(; rho_inf::Real, ux_inf::Real, uy_inf::Real,
                           temp_inf::Real, eint::Real, chord::Real)
    vals = Float64.((rho_inf, ux_inf, uy_inf, temp_inf, eint, chord))
    all(isfinite, vals) || throw(ArgumentError("nonfinite freestream reference"))
    rho, ux, uy, temp, cv, c = vals
    rho > 0 && temp > 0 && cv > 0 && c > 0 ||
        throw(ArgumentError("rho, temperature, eint, and chord must be positive"))
    u = hypot(ux, uy)
    u > 0 || throw(ArgumentError("positive freestream speed required"))
    e = rho*(cv*temp + 0.5*u*u)
    (speed=u, chord=c, convective_time=c/u,
     quantity=(mass=rho, mom_x=rho*u, mom_y=rho*u, energy=e),
     flux=(mass=rho*u*c, mom_x=rho*u*u*c,
           mom_y=rho*u*u*c, energy=e*u*c))
end

function _component(dc::AbstractMatrix, vol::AbstractMatrix, qref::Float64,
                    fluxref::Float64, convective_time::Float64)
    size(dc) == size(vol) || throw(ArgumentError("residual/volume shape mismatch"))
    isempty(dc) && throw(ArgumentError("empty residual array"))
    raw_max = 0.0
    raw_square = 0.0
    conv_max = 0.0
    conv_square = 0.0
    signed = 0.0
    correction = 0.0
    peak = first(eachindex(dc))
    conv_factor = convective_time/qref
    for idx in eachindex(dc,vol)
        r = Float64(dc[idx])
        v = Float64(vol[idx])
        isfinite(r) && isfinite(v) && v > 0 ||
            throw(ArgumentError("nonfinite residual or nonpositive cell volume"))
        raw = abs(r/v)
        normalized = raw*conv_factor
        raw_square += raw*raw
        conv_square += normalized*normalized
        if normalized > conv_max
            conv_max = normalized
            raw_max = raw
            peak = idx
        end
        # Compensated signed sum matters when large opposing boundary fluxes
        # leave a small net. This does not replace a face-telescoping check.
        y = r-correction
        t = signed+y
        correction = (t-signed)-y
        signed = t
    end
    i,j = Tuple(CartesianIndices(dc)[peak])
    (linf_raw=raw_max, l2_raw=sqrt(raw_square/length(dc)),
     linf_convective=conv_max,
     l2_convective=sqrt(conv_square/length(dc)),
     signed_flux_fraction=signed/fluxref, peak_i=i, peak_j=j)
end

"""Pure host reduction of *raw post-step face divergence*, with no state write."""
function assess(residuals::NamedTuple, vol::AbstractMatrix, scales::NamedTuple)
    keys(residuals) == COMPONENTS ||
        throw(ArgumentError("residual keys must be $(COMPONENTS)"))
    all(size(getproperty(residuals,k)) == size(vol) for k in COMPONENTS) ||
        throw(ArgumentError("all residual arrays must match cell volumes"))
    parts = map(COMPONENTS) do k
        _component(getproperty(residuals,k), vol,
                   getproperty(scales.quantity,k),
                   getproperty(scales.flux,k), scales.convective_time)
    end
    NamedTuple{COMPONENTS}(parts)
end

history_header() = join(vcat("iteration",
    [string(k,"_",m) for k in COMPONENTS for m in METRICS]), ",")

function history_row(iteration::Integer, report::NamedTuple)
    iteration >= 1 || throw(ArgumentError("iteration must be positive"))
    keys(report) == COMPONENTS || throw(ArgumentError("component mismatch"))
    values = String[string(iteration)]
    for k in COMPONENTS, metric in METRICS
        value = getproperty(getproperty(report,k),metric)
        push!(values, metric in (:peak_i,:peak_j) ? string(value) :
              @sprintf("%.12g",value))
    end
    join(values,",")
end

end # module EquationResidualProbe
