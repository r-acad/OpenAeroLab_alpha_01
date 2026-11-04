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

"Second-order non-equilibrium moments of shifted populations about the equilibrium at (rho, u)."
@inline function pi_neq_shifted(f::NTuple{Q,T}, rho::T, ux::T, uy::T, uz::T) where {T<:AbstractFloat}
    usq = ux * ux + uy * uy + uz * uz
    pi_neq_shifted(f, ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q)))
end
"The same from the shifted equilibria already computed (the step computes them once per cell)."
@inline function pi_neq_shifted(f::NTuple{Q,T}, feqs::NTuple{Q,T}) where {T<:AbstractFloat}
    pxx = zero(T); pyy = zero(T); pzz = zero(T); pxy = zero(T); pxz = zero(T); pyz = zero(T)
    @inbounds for i in 2:Q
        fn = f[i] - feqs[i]
        cx = T(CX[i]); cy = T(CY[i]); cz = T(CZ[i])
        pxx += cx * cx * fn; pyy += cy * cy * fn; pzz += cz * cz * fn
        pxy += cx * cy * fn; pxz += cx * cz * fn; pyz += cy * cz * fn
    end
    (pxx, pyy, pzz, pxy, pxz, pyz)
end

"""
Recursive-regularised collision of shifted populations: returns the post-collision populations and (rho, u).
`sigma` blends the measured second-order moments with `pi_fd` (HRR); sigma = 1 is plain RR. `F` is a body force
per unit volume in lattice units (Guo et al. 2002): the velocity is the force-corrected one, u = (m + F/2) / rho,
and the forcing term (1 - omega/2) w_i [3 (c - u) . F + 9 (c . u)(c . F)] is added.
"""
@inline function collide_rr(f::NTuple{Q,T}, omega::T, sigma::T, pi_fd::NTuple{6,T}, F::NTuple{3,T}=(zero(T), zero(T), zero(T))) where {T<:AbstractFloat}
    rho, mx, my, mz = macros_shifted(f)
    forced = !(F[1] == zero(T) && F[2] == zero(T) && F[3] == zero(T))
    # the velocity bindings are assigned once: a captured variable assigned twice is boxed by the ntuple closures
    # below (the prototype's trap, 27 MB per step on a 32^3 box)
    h = forced ? T(0.5) / rho : zero(T)
    ux = mx + h * F[1]; uy = my + h * F[2]; uz = mz + h * F[3]
    usq = ux * ux + uy * uy + uz * uz
    # the equilibria and the third-order coefficients once per cell (computing them per direction cost the kernel
    # a third of its rate)
    feqs = ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q))
    p = pi_neq_shifted(f, feqs)
    a2 = sigma == one(T) ? p : ntuple(n -> sigma * p[n] + (one(T) - sigma) * pi_fd[n], Val(6))
    a3 = a3_terms(a2, ux, uy, uz)
    keep = one(T) - omega
    if forced
        # Guo's source on the Hermite basis: the regularised non-equilibrium carries no momentum (the post-collision
        # momentum is rho u = m + F/2 by construction), so the first-order part of the source carries the other F/2
        # (prefactor 1/2, not BGK's 1 - omega/2, which left the momentum input at (3 - omega)/2 F: G1.3 read 7/8 of
        # the body force at omega 1.25); the second-order part keeps 1 - omega/2
        pre = one(T) - T(0.5) * omega
        out = ntuple(Val(Q)) do i
            cx = T(CX[i]); cy = T(CY[i]); cz = T(CZ[i])
            cu = cx * ux + cy * uy + cz * uz; cF = cx * F[1] + cy * F[2] + cz * F[3]; uF = ux * F[1] + uy * F[2] + uz * F[3]
            @inbounds feqs[i] + keep * fneq_regularised(i, a2, a3) + T(WT[i]) * (T(1.5) * cF + pre * (T(9) * cu * cF - T(3) * uF))
        end
    else
        out = ntuple(i -> (@inbounds feqs[i] + keep * fneq_regularised(i, a2, a3)), Val(Q))
    end
    out, rho, ux, uy, uz
end

"WALE eddy viscosity (lattice units) from the velocity gradient g[a, b] = d u_b / d x_a."
@inline function nu_wale(g::NTuple{9,T}, cw::T) where {T<:AbstractFloat}
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
@inline function nu_smagorinsky(g::NTuple{9,T}, cs::T) where {T<:AbstractFloat}
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    s12 = T(0.5) * (g12 + g21); s13 = T(0.5) * (g13 + g31); s23 = T(0.5) * (g23 + g32)
    ss = g11 * g11 + g22 * g22 + g33 * g33 + T(2) * (s12 * s12 + s13 * s13 + s23 * s23)
    cs * cs * sqrt(T(2) * ss)
end

"Vreman eddy viscosity (Vreman 2004), c = 2.5 Cs^2."
@inline function nu_vreman(g::NTuple{9,T}, c::T) where {T<:AbstractFloat}
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    # beta_ij = sum_m g_mi g_mj
    b11 = g11 * g11 + g21 * g21 + g31 * g31; b22 = g12 * g12 + g22 * g22 + g32 * g32; b33 = g13 * g13 + g23 * g23 + g33 * g33
    b12 = g11 * g12 + g21 * g22 + g31 * g32; b13 = g11 * g13 + g21 * g23 + g31 * g33; b23 = g12 * g13 + g22 * g23 + g32 * g33
    bb = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23
    aa = b11 + b22 + b33
    (aa <= T(1e-20) || bb <= zero(T)) ? zero(T) : c * sqrt(bb / aa)
end

@inline omega_of(nu::T) where {T<:AbstractFloat} = one(T) / (T(3) * nu + T(0.5))

"The finite-difference second-order moment for HRR: Pi_fd = -rho cs^2 tau S (lattice units) from the gradient."
@inline function pi_fd_from_gradient(g::NTuple{9,T}, rho::T, tau::T) where {T<:AbstractFloat}
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = g
    c = -rho * tau / T(3)
    (c * T(2) * g11, c * T(2) * g22, c * T(2) * g33, c * (g12 + g21), c * (g13 + g31), c * (g23 + g32))
end
