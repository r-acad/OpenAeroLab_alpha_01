module NativeBarotropicSource

using ..NativeBarotropicEOS: Material, phase_state
export SourceModel, admit_interval, source_step, interval_certificates

# Outward enclosures of real arithmetic on the exact Float64 inputs. These
# operations require ordinary IEEE round-to-nearest arithmetic, without
# fast-math reassociation or flush-to-zero. Uncertain/nonfinite bounds reject.
struct Bounds
    lo::Float64
    hi::Float64
end
point(x::Float64) = Bounds(x,x)
function badd(a::Bounds,b::Bounds)
    lo,hi = a.lo+b.lo,a.hi+b.hi
    isnan(lo) || isnan(hi) ? Bounds(-Inf,Inf) : Bounds(prevfloat(lo),nextfloat(hi))
end
function bsub(a::Bounds,b::Bounds)
    lo,hi = a.lo-b.hi,a.hi-b.lo
    isnan(lo) || isnan(hi) ? Bounds(-Inf,Inf) : Bounds(prevfloat(lo),nextfloat(hi))
end
function bmul(a::Bounds,b::Bounds)
    products = (a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi)
    any(isnan,products) && return Bounds(-Inf,Inf)
    Bounds(prevfloat(minimum(products)),nextfloat(maximum(products)))
end
finite_positive(b::Bounds) = isfinite(b.lo) && isfinite(b.hi) && b.lo>0
ieee_environment() = Base.JLOptions().fast_math in (0,2) &&
                     !get_zero_subnormals() && rounding(Float64)==RoundNearest

# Exact rational -> binary64 nearest, ties to even. Integer quotient/remainder
# determines rounding; no ambient BigFloat precision or rounding mode is used.
function round_exact(x::Rational{BigInt})
    x == 0 && return 0.0
    n,d = abs(numerator(x)),denominator(x)
    e = ndigits(n,base=2)-ndigits(d,base=2)
    (e >= 0 ? n < (d << e) : (n << (-e)) < d) && (e -= 1)
    e > 1023 && return (x < 0 ? -Inf : Inf)
    shift = e < -1022 ? 1074 : 52-e
    a,b = shift >= 0 ? (n << shift,d) : (n,d << (-shift))
    sig,rem = divrem(a,b)
    twice = 2rem
    (twice > b || (twice == b && isodd(sig))) && (sig += 1)
    y = ldexp(Float64(sig),-shift)
    x < 0 ? -y : y
end
exact(x::Float64) = Rational{BigInt}(x)

"""Outward sign certificates for a fixed-rho closed q interval.
The certificate is separate from the native EOS's computed-state validity.
"""
function interval_certificates(m::Material{Float64},rho::Float64,qmin::Float64,qmax::Float64)
    r,qh = point(rho),point(qmax)
    rl,rv,pv,A = point(m.rl),point(m.rv),point(m.pv),point(m.A)
    C = badd(pv,bmul(rl,A))
    D = bmul(bmul(rl,rl),A)
    # Exact positive common-phase domain: qmax*pv < rv*C. If the interval
    # touches q=0, require rho*C>D as well for a strictly positive liquid root.
    phase_volume = bsub(bmul(rv,C),bmul(qh,pv))
    liquid_root = bsub(bmul(r,C),D)
    # F = rv*[D*q + rho*z*(rho-q)]*K at every admitted phase-volume root.
    # Its positive bracket makes the ordering sign independent of q.
    F = badd(bmul(bmul(pv,r),bsub(r,rv)),
             bmul(bmul(bmul(rv,rl),A),bsub(rl,r)))
    # Julia 1.12 jl_options: fast_math 0=default, 1=fast, 2=IEEE.
    ieee = ieee_environment()
    inputs = ieee && isfinite(rho) && isfinite(qmin) && isfinite(qmax) &&
             rho>0 && 0<=qmin<=qmax<=rho
    valid = inputs && finite_positive(phase_volume) &&
            (qmin>0 || finite_positive(liquid_root)) && finite_positive(F)
    (;valid, ieee_environment=ieee, phase_volume, liquid_root, ordering=F,
      touches_liquid=qmin==0, domain_certified=inputs && finite_positive(phase_volume) &&
          (qmin>0 || finite_positive(liquid_root)),
      ordering_certified=inputs && finite_positive(F))
end

"""CPU Float64 compatibility source with an explicitly rounded equilibrium.
Exact rational threshold caches classify the physical endpoint regimes. Interior
relaxation uses the nearest Float64 equilibrium; its physical source discrepancy
must be assessed separately. No physical rate calibration is implied.
"""
struct SourceModel
    material::Material{Float64}
    kp::Float64
    kd::Float64
    zstar::Float64
    rholstar::Float64
    rhovstar::Float64
    Kstar::Float64
    vlstar_exact::Rational{BigInt}
    Kstar_exact::Rational{BigInt}
    function SourceModel(m::Material{Float64}, kp::Float64, kd::Float64)
        ieee_environment() || throw(ArgumentError("source construction requires IEEE nearest arithmetic with subnormals"))
        isfinite(kp) && isfinite(kd) && kp >= 0 && kd >= 0 ||
            throw(ArgumentError("finite nonnegative source rates required"))
        rl,rv,pv,A,psm = exact.((m.rl,m.rv,m.pv,m.A,m.psm))
        psm < 2pv || throw(ArgumentError("source requires psm < 2pv"))
        zs_exact = pv-psm^2/(4pv)
        vl_exact = (1-(zs_exact-pv)/(rl*A))/rl
        vv_exact = pv/(rv*zs_exact)
        ks_exact = vv_exact-vl_exact
        vl_exact > 0 && ks_exact > 0 ||
            throw(ArgumentError("invalid or unordered exact source-threshold phases"))
        zs,rls,rvs = round_exact(zs_exact),round_exact(1/vl_exact),round_exact(1/vv_exact)
        isfinite(zs) && zs > 0 && isfinite(rls) && isfinite(rvs) && rls > rvs > 0 ||
            throw(ArgumentError("invalid or unordered source-threshold phases"))
        ks = round_exact(ks_exact)
        isfinite(ks) && ks > 0 || throw(ArgumentError("invalid source threshold compliance"))
        new(m, kp, kd, zs, rls, rvs, ks, vl_exact, ks_exact)
    end
end

@inline function phase_polynomial(m, z)
    # z*K(z), whose convexity permits an endpoint/vertex interval certificate.
    m.pv/m.rv - z/m.rl + z*(z-m.pv)/(m.rl*m.rl*m.A)
end

function admit_interval(s::SourceModel, rho::Float64, q0::Float64)
    fail(reason) = (; valid=false, reason, target=NaN, regime=:invalid,
                     zlo=NaN, zhi=NaN, min_zK=NaN, certificates=nothing,
                     target_hi=NaN, target_lo=NaN, target_remainder=nothing,
                     target_remainder_bounds=nothing, target_rounding_error=nothing)
    a = phase_state(s.material, rho, q0)
    a.valid || return fail(:initial_eos)
    # Classify against the exact affine equilibrium, before any rounding.
    r_exact = exact(rho)
    qt_exact = (1-r_exact*s.vlstar_exact)/s.Kstar_exact
    target, regime = if qt_exact <= 0
        (0.0, :liquid_endpoint)
    elseif qt_exact >= r_exact
        (rho, :vapor_endpoint)
    else
        (round_exact(qt_exact), :interior)
    end
    target_hi = round_exact(qt_exact)
    isfinite(target_hi) || return fail(:unclipped_target_representation)
    target_lo = round_exact(qt_exact-exact(target_hi))
    isfinite(target_lo) || return fail(:unclipped_target_representation)
    target_remainder = qt_exact-exact(target_hi)-exact(target_lo)
    rem_rounded = round_exact(target_remainder)
    isfinite(rem_rounded) || return fail(:unclipped_target_representation)
    target_remainder_bounds = Bounds(prevfloat(rem_rounded),nextfloat(rem_rounded))
    target_rounding_error = exact(target)-clamp(qt_exact,zero(r_exact),r_exact)
    0 <= target <= rho && isfinite(target) || return fail(:target_bounds)
    b = phase_state(s.material, rho, target)
    b.valid || return fail(:target_eos)
    qmin,qmax = minmax(q0,target)
    certificates = interval_certificates(s.material,rho,qmin,qmax)
    certificates.domain_certified || return fail(:domain_not_certified)
    certificates.ordering_certified || return fail(:ordering_not_certified)
    zl, zh = minmax(a.raw, b.raw)
    m = s.material
    vertex = (m.rl*m.A + m.pv)/2
    zK = min(phase_polynomial(m,zl), phase_polynomial(m,zh))
    if zl < vertex < zh
        zK = min(zK,phase_polynomial(m,vertex))
    end
    isfinite(zK) && zK > 0 || return fail(:phase_ordering)
    (; valid=true, reason=:accepted, target, regime, zlo=zl, zhi=zh, min_zK=zK,
       certificates, target_hi, target_lo, target_remainder,
       target_remainder_bounds, target_rounding_error)
end

function rate(s::SourceModel, rho::Float64, q::Float64, interval)
    m = s.material
    a = phase_state(m,rho,q)
    a.valid || return NaN
    D = rho*m.rv + q*(m.rl-m.rv)
    L = (rho-q)/(m.rl*m.rl*m.A)
    den = L*s.zstar + q*m.pv/(m.rv*a.raw)
    guarddiv = (1+(a.raw+s.zstar)/(hypot(a.raw,m.psm)+hypot(s.zstar,m.psm)))/2
    H = (s.zstar*s.Kstar/den)*guarddiv
    isfinite(H) && H > 0 || return NaN
    if interval.regime === :liquid_endpoint
        gap = (q-interval.target_hi)-interval.target_lo
        gap_bounds = bsub(bsub(bsub(point(q),point(interval.target_hi)),
                              point(interval.target_lo)),interval.target_remainder_bounds)
        finite_positive(gap_bounds) || return NaN
        s.kd*H*gap*D/(rho*m.rv)
    elseif interval.regime === :vapor_endpoint
        gap = (interval.target_hi-q)+interval.target_lo
        gap_bounds = badd(badd(bsub(point(interval.target_hi),point(q)),
                              point(interval.target_lo)),interval.target_remainder_bounds)
        finite_positive(gap_bounds) || return NaN
        s.kp*H*gap*D/(rho*m.rl)
    else
        q <= interval.target ? H*s.kp*(rho-q)*D/(rho*m.rl) :
                               H*s.kd*q*D/(rho*m.rv)
    end
end

"""
    source_step(model, rho::Float64, q0::Float64, h::Float64)

Validate the whole source interval before any identity return. The stored
Float64 target defines the representable equilibrium; the small EOS residual
of that rounded target must be recorded separately against an exact oracle.
Solve the
backward-Euler half predictor by at most 100 bracket iterations, then apply
exponential relaxation of the target distance. Returns diagnostics and a new q;
no caller state is mutated, clipped, retried or silently precision-converted.
This barotropic scalar method has not been admitted for coupled flow integration.
"""
function source_step(s::SourceModel, rho::Float64, q0::Float64, h::Float64)
    failed(reason, n=0, width=NaN, residual=NaN) =
        (; valid=false, reason, q=q0, iterations=n, bracket_width=width,
           predictor_residual=residual, target=NaN, rate=NaN, hk=NaN)
    isfinite(h) && h >= 0 || return failed(:invalid_time)
    interval = admit_interval(s,rho,q0)
    interval.valid || return failed(interval.reason)
    qt = interval.target
    if q0 == qt
        return (; valid=true, reason=:representable_equilibrium, q=q0, iterations=0,
                bracket_width=0.0, predictor_residual=0.0, target=qt, rate=0.0, hk=0.0)
    end
    k0 = rate(s,rho,q0,interval)
    isfinite(k0) && k0 >= 0 || return failed(:invalid_initial_rate)
    if h == 0 || k0 == 0
        return (; valid=true, reason=:identity, q=q0, iterations=0,
                bracket_width=0.0, predictor_residual=0.0, target=qt, rate=k0, hk=h*k0)
    end
    lo, hi = minmax(q0,qt)
    # Resolve the actually stored partial-density scale, including dilute q.
    # A rho-scaled stopping band can terminate before a trace-species predictor
    # has taken any iterations. This is a new API arithmetic contract.
    tol = 8*eps(max(abs(q0),abs(qt)))
    n = 0
    while hi-lo > tol && n < 100
        mid = lo+(hi-lo)/2
        (mid == lo || mid == hi) && break
        km = rate(s,rho,mid,interval)
        isfinite(km) && km >= 0 || return failed(:invalid_predictor_rate,n,hi-lo)
        g = (mid-q0) - (h/2)*km*(qt-mid)
        isfinite(g) || return failed(:predictor_overflow,n,hi-lo,g)
        if g > 0
            hi = mid
        else
            lo = mid
        end
        n += 1
    end
    hi-lo <= tol || return failed(:predictor_iteration_cap,n,hi-lo)
    qh = lo+(hi-lo)/2
    kh = rate(s,rho,qh,interval)
    isfinite(kh) && kh >= 0 || return failed(:invalid_corrector_rate,n,hi-lo)
    residual = (qh-q0) - (h/2)*kh*(qt-qh)
    hk = h*kh
    isfinite(residual) && isfinite(hk) || return failed(:corrector_overflow,n,hi-lo,residual)
    q1 = hk <= log(2.0) ? q0+(qt-q0)*(-expm1(-hk)) : qt+(q0-qt)*exp(-hk)
    min(q0,qt) <= q1 <= max(q0,qt) && phase_state(s.material,rho,q1).valid ||
        return failed(:corrector_bounds_or_eos,n,hi-lo,residual)
    (; valid=true, reason=:accepted, q=q1, iterations=n, bracket_width=hi-lo,
       predictor_residual=residual, target=qt, rate=kh, hk)
end

end # module
