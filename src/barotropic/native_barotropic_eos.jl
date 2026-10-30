module NativeBarotropicEOS

export Material, phase_state

const Scalar = Union{Float32,Float64}

"""Explicit numerical material for the lq=0 barotropic phase-volume EOS."""
struct Material{T<:Scalar}
    rl::T
    rv::T
    pv::T
    A::T
    psm::T
    function Material(rl::T, rv::T, pv::T, A::T, psm::T) where {T<:Scalar}
        all(isfinite, (rl, rv, pv, A, psm)) || throw(ArgumentError("finite material required"))
        rl > rv > zero(T) || throw(ArgumentError("require rl > rv > 0"))
        pv > zero(T) && A > zero(T) && psm >= zero(T) ||
            throw(ArgumentError("require pv,A > 0 and psm >= 0"))
        new{T}(rl, rv, pv, A, psm)
    end
end

@inline function invalid_state(::Type{T}, status::UInt8) where {T<:Scalar}
    n = T(NaN)
    (; valid=false, status, raw=n, p=n, c2=n, Y=n, alpha=n, beta=n,
       Al=n, volume_residual=n, dpdq=n, ordered=false)
end

"""
    phase_state(material::Material{T}, rho::T, q::T)

Derive pressure and fixed-Y sound speed from authoritative total density rho
and vapor partial density q=rho*Y. Alpha is the reference composition; beta is
the actual vapor phase volume. No density, pressure, sound-speed or phase floor
is applied. Invalid states return valid=false and NaN derived fields.

Status 1 denotes invalid conservative inputs, 2 an invalid positive raw root,
3 an invalid phase volume, and 4 an invalid derivative/derived state. Phase
ordering is reported separately; a phase-source algorithm needs its own
whole-interval ordering admission. This function supplies no thermal energy law.
"""
@inline function phase_state(m::Material{T}, rho::T, q::T) where {T<:Scalar}
    z0, o = zero(T), one(T)
    isfinite(rho) && isfinite(q) && rho > z0 && z0 <= q <= rho ||
        return invalid_state(T, 0x01)
    liquid = rho - q
    L = liquid / (m.rl * m.rl * m.A)
    raw = if q == z0
        m.pv + m.rl * m.A * ((rho - m.rl) / rho)
    elseif q == rho
        (m.pv / m.rv) * rho
    else
        # Group the near-liquid difference before dividing. This is a new
        # arithmetic contract, not a bitwise alias of the old alpha helper.
        B = ((m.rl - rho) + q) / m.rl - m.pv * L
        Cabs = q * (m.pv / m.rv)
        discroot = hypot(B, T(2) * sqrt(L) * sqrt(Cabs))
        B >= z0 ? T(2) * Cabs / (B + discroot) :
                  (discroot - B) / (T(2) * L)
    end
    isfinite(raw) && raw > z0 || return invalid_state(T, 0x02)
    Al = o - (raw - m.pv) / (m.rl * m.A)
    Al > z0 && isfinite(Al) || return invalid_state(T, 0x03)
    vapor_volume_per_mass = m.pv / (m.rv * raw)
    beta = q * vapor_volume_per_mass
    volume_residual = liquid * Al / m.rl + beta - o
    guardnorm = hypot(raw, m.psm)
    p = raw / T(2) + guardnorm / T(2)
    guardprime = (o + raw / guardnorm) / T(2)
    den = L + beta / raw
    c2 = guardprime / (rho * den)
    K = vapor_volume_per_mass - Al / m.rl
    dpdq = guardprime * K / den
    Y = q / rho
    alpha = (q * m.rl) / (liquid * m.rv + q * m.rl)
    isfinite(p) && isfinite(c2) && c2 > z0 && isfinite(dpdq) &&
        isfinite(alpha) && isfinite(beta) && isfinite(volume_residual) ||
        return invalid_state(T, 0x04)
    (; valid=true, status=0x00, raw, p, c2, Y, alpha, beta, Al,
       volume_residual, dpdq, ordered=K > z0)
end

end # module
