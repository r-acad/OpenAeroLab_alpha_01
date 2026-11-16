# Collision operators and subgrid models on one cell (tuples of the 19 populations). DESIGN.md section 6.
# Populations are stored shifted by their rest weight (f_s = f - w_i), so every formula here is in shifted form:
# rho = 1 + sum f_s, momentum = sum c f_s, feq_s = feq - w_i, and f_s' = feq_s + (1 - omega) fneq.

"Density and velocity of shifted populations."
@inline function macros_shifted(f::NTuple{Q,T}) where {T<:AbstractFloat}
    rho = one(T); mx = zero(T); my = zero(T); mz = zero(T)
    @inbounds for i in 1:Q
        rho += f[i]; mx += T(CX[i]) * f[i]; my += T(CY[i]) * f[i]; mz += T(CZ[i]) * f[i]
    end
    inv = one(T) / rho
    rho, mx * inv, my * inv, mz * inv
end

"Shifted equilibrium of direction i."
@inline feq_shifted(i::Int, rho::T, ux::T, uy::T, uz::T, usq::T) where {T<:AbstractFloat} = feq(i, rho, ux, uy, uz, usq) - T(WT[i])

# Blend a population tuple toward its equilibrium only when a direction has
# become non-positive. Both endpoints have the same conserved moments for the
# unforced RR collision and AMR reconstruction, so the blend adds local
# dissipation without changing density or momentum. This is a last-resort
# kinetic safeguard; the count is reported by the run loop.
const POSITIVITY_LIMITED_CELLS = Threads.Atomic{Int64}(0)
@inline function positive_shifted(out::NTuple{Q,T}, eq::NTuple{Q,T}) where {T<:AbstractFloat}
    alpha = one(T)
    @inbounds for i in 1:Q
        w = T(WT[i]); floor = T(1e-6) * w
        actual = out[i] + w
        if actual < floor
            equilibrium = eq[i] + w
            (isfinite(equilibrium) && equilibrium > floor) || return out
            candidate = (equilibrium - floor) / (equilibrium - actual)
            alpha = min(alpha, candidate)
        end
    end
    if alpha < one(T)
        Threads.atomic_add!(POSITIVITY_LIMITED_CELLS, Int64(1))
        safe_alpha = max(zero(T), T(0.99) * alpha)
        return ntuple(i -> eq[i] + safe_alpha * (out[i] - eq[i]), Val(Q))
    end
    out
end

# The Guo body force changes momentum during collision. If limiting is needed,
# blend with the equilibrium of the *post-force* moments rather than with the
# pre-force equilibrium used by the collision itself.
@inline function positive_shifted_forced(out::NTuple{Q,T}) where {T<:AbstractFloat}
    needs_limit = false
    @inbounds for i in 1:Q
        w = T(WT[i])
        needs_limit |= out[i] + w < T(1e-6) * w
    end
    needs_limit || return out
    rho, ux, uy, uz = macros_shifted(out)
    usq = ux*ux + uy*uy + uz*uz
    eq = ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q))
    positive_shifted(out, eq)
end

"""
Recursive-regularised collision of shifted populations: returns the post-collision populations and (rho, u).
`sigma` blends the measured second-order moments with `pi_fd` (HRR); sigma = 1 is plain RR. `F` is a body force
per unit volume in lattice units (Guo et al. 2002): the velocity is the force-corrected one, u = (m + F/2) / rho.
The arithmetic is `collide_q19` (kernel.jl), followed by the positivity safeguard.
"""
@inline function collide_rr(f::NTuple{Q,T}, omega::T, sigma::T, pi_fd::NTuple{6,T}, F::NTuple{3,T}=(zero(T), zero(T), zero(T))) where {T<:AbstractFloat}
    forced = !(F[1] == zero(T) && F[2] == zero(T) && F[3] == zero(T))
    out, drho, rho, ux, uy, uz = collide_q19(f, omega, sigma, pi_fd, F, forced)
    limit_positive(out, drho, rho, ux, uy, uz, forced), rho, ux, uy, uz
end

"The positivity safeguard of a post-collision tuple: untouched unless a population is at the floor."
@inline function limit_positive(out::NTuple{Q,T}, drho::T, rho::T, ux::T, uy::T, uz::T, forced::Bool) where {T<:AbstractFloat}
    below_floor(out) || return out
    forced ? positive_shifted_forced(out) : positive_shifted(out, equilibrium_q19(drho, rho, ux, uy, uz))
end

# The subgrid models below take a scalar gradient or the eight lanes of an oct (kernel.jl).
"WALE eddy viscosity (lattice units) from the velocity gradient g[a, b] = d u_b / d x_a."
@inline function nu_wale(g::NTuple{9,V}, cw) where V
    T = _eltype(V)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    s11 = g11; s22 = g22; s33 = g33
    s12 = T(0.5) * (g12 + g21); s13 = T(0.5) * (g13 + g31); s23 = T(0.5) * (g23 + g32)
    q11 = g11 * g11 + g12 * g21 + g13 * g31; q12 = g11 * g12 + g12 * g22 + g13 * g32; q13 = g11 * g13 + g12 * g23 + g13 * g33
    q21 = g21 * g11 + g22 * g21 + g23 * g31; q22 = g21 * g12 + g22 * g22 + g23 * g32; q23 = g21 * g13 + g22 * g23 + g23 * g33
    q31 = g31 * g11 + g32 * g21 + g33 * g31; q32 = g31 * g12 + g32 * g22 + g33 * g32; q33 = g31 * g13 + g32 * g23 + g33 * g33
    tr = (q11 + q22 + q33) / T(3)
    d11 = q11 - tr; d22 = q22 - tr; d33 = q33 - tr
    d12 = T(0.5) * (q12 + q21); d13 = T(0.5) * (q13 + q31); d23 = T(0.5) * (q23 + q32)
    ss = s11 * s11 + s22 * s22 + s33 * s33 + T(2) * (s12 * s12 + s13 * s13 + s23 * s23)
    dd = d11 * d11 + d22 * d22 + d33 * d33 + T(2) * (d12 * d12 + d13 * d13 + d23 * d23)
    den = ss * ss * sqrt(ss) + dd * sqrt(sqrt(dd)) + T(1e-20)
    cw * cw * dd * sqrt(dd) / den
end

"Smagorinsky eddy viscosity from the strain magnitude of the gradient."
@inline function nu_smagorinsky(g::NTuple{9,V}, cs) where V
    T = _eltype(V)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    s12 = T(0.5) * (g12 + g21); s13 = T(0.5) * (g13 + g31); s23 = T(0.5) * (g23 + g32)
    ss = g11 * g11 + g22 * g22 + g33 * g33 + T(2) * (s12 * s12 + s13 * s13 + s23 * s23)
    cs * cs * sqrt(T(2) * ss)
end

"Vreman eddy viscosity (Vreman 2004), c = 2.5 Cs^2."
@inline function nu_vreman(g::NTuple{9,V}, c) where V
    T = _eltype(V)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    # beta_ij = sum_m g_mi g_mj
    b11 = g11 * g11 + g21 * g21 + g31 * g31; b22 = g12 * g12 + g22 * g22 + g32 * g32; b33 = g13 * g13 + g23 * g23 + g33 * g33
    b12 = g11 * g12 + g21 * g22 + g31 * g32; b13 = g11 * g13 + g21 * g23 + g31 * g33; b23 = g12 * g13 + g22 * g23 + g32 * g33
    bb = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23
    aa = b11 + b22 + b33
    off = (aa <= T(1e-20)) | (bb <= zero(T))
    _select(off, zero(V), c * sqrt(_select(off, one(V), bb / aa)))
end

@inline omega_of(nu::V) where V = (T = _eltype(V); one(T) / (T(3) * nu + T(0.5)))

"The finite-difference second-order moment for HRR: Pi_fd = -rho cs^2 tau S (lattice units) from the gradient."
@inline function pi_fd_from_gradient(g::NTuple{9,V}, rho::V, tau::V) where V
    T = _eltype(V)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    c = -rho * tau / T(3)
    (c * T(2) * g11, c * T(2) * g22, c * T(2) * g33, c * (g12 + g21), c * (g13 + g31), c * (g23 + g32))
end
