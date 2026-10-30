# Consistent potential-flow initial field for the confined foil (validation ladder tier 2b, 14 September 2026).
#
# The movie driver starts every run from the uniform freestream around a foil that blocks it; in a compressible liquid
# that impulsive start is a water-hammer event (a +0.5 q pressure wave at c/U 5) whose reflections ring the channel for
# passes. This module evaluates the linear-strength vortex panel solution of Kuethe and Chow (section 5.10), with the
# image system of the two parallel channel walls, at arbitrary points: the driver uses it (`TOPO_INIT=panel`) to start
# from the confined potential flow (velocity and Bernoulli pressure), so that the impulsive no-slip start produces
# vorticity but no compression. The panel algebra is that of the base-flow reference `panel_confined.jl`
# (S0903_RUNS/species_base_flow_20260913), which gave the confined inviscid lift of the 13 September readings.
#
# Frames: the foil is unrotated in the C-mesh and the channel is rotated by alpha about (0.5, 0); the panel problem is
# solved in the freestream frame (foil rotated by -alpha about (0.5, 0), freestream along +x, walls at y = +-H), and
# the velocity is rotated back to the foil frame. Velocities are in units of the freestream speed.
module PanelInit

using LinearAlgebra
export PanelSolution, solve_confined_panels, potential_velocity, potential_field, blunt_panel_loop

struct PanelSolution
    alpha_deg::Float64
    H::Float64                 # channel half-height (Inf: unconfined)
    M::Int                     # images per side
    X::Vector{Float64}         # closed loop of nodes in the freestream frame (first == last)
    Y::Vector{Float64}
    S::Vector{Float64}         # panel lengths
    th::Vector{Float64}        # panel angles
    gam::Vector{Float64}       # nodal vortex strengths gamma/(2 pi V)
    Cp::Vector{Float64}        # surface pressure coefficient at the control points
    Cl::Float64                # lift coefficient from the surface pressure (freestream frame)
    Gamma::Float64             # circulation in units of V c
end

# Kuethe-Chow influence coefficients of panel (Xj,Yj)->(Xj1,Yj1) with angle thj, length Sj at a point (xi,yi) whose
# local tangent has angle thi (thi = 0: Cn is the y-velocity, Ct the x-velocity).
@inline function coeffs(xi, yi, thi, Xj, Yj, thj, Sj)
    A = -(xi - Xj) * cos(thj) - (yi - Yj) * sin(thj)
    B = (xi - Xj)^2 + (yi - Yj)^2
    C = sin(thi - thj); D = cos(thi - thj)
    E = (xi - Xj) * sin(thj) - (yi - Yj) * cos(thj)
    Fl = log(1 + (Sj^2 + 2A * Sj) / B)
    G = atan(E * Sj, B + A * Sj)
    P = (xi - Xj) * sin(thi - 2thj) + (yi - Yj) * cos(thi - 2thj)
    Q = (xi - Xj) * cos(thi - 2thj) - (yi - Yj) * sin(thi - 2thj)
    Cn2 = D + 0.5 * Q * Fl / Sj - (A * C + D * E) * G / Sj
    Cn1 = 0.5 * D * Fl + C * G - Cn2
    Ct2 = C + 0.5 * P * Fl / Sj + (A * D - C * E) * G / Sj
    Ct1 = 0.5 * C * Fl - D * G - Ct2
    Cn1, Cn2, Ct1, Ct2
end

# foil frame -> freestream frame about (0.5, 0) (rotation by -alpha) and back
rotate_to_freestream(x, y, alpha_deg) = (a = -deg2rad(alpha_deg); c = cos(a); s = sin(a);
                                         (0.5 + (x - 0.5) * c - y * s, (x - 0.5) * s + y * c))
rotate_vector_to_foil(u, v, alpha_deg) = (a = deg2rad(alpha_deg); c = cos(a); s = sin(a); (u * c - v * s, u * s + v * c))

# image m of a point at height y: strength factor (-1)^m, position 2mH + (-1)^m y
@inline image_sign(m) = isodd(m) ? -1.0 : 1.0
@inline image_y(y, m, H) = m == 0 ? y : 2m * H + (isodd(m) ? -y : y)   # m == 0 is the panel itself (H may be Inf)

"""Solve the confined panel problem for the foil loop `x, y` (foil frame, clockwise from the trailing edge, closed, or
open by at most 1e-8, the generator's own trailing-edge tolerance) at incidence `alpha_deg`, channel half-height `H`
(Inf: unconfined) with `M` images per side. A section with a finite trailing edge is refused (2 Oct 2026): closing it
here would add one base panel and put the Kutta node at the lower corner (C_L 2.79 against 0.64 measured); pass
`blunt_panel_loop(x, y)` instead, which closes it through the base midpoint."""
function solve_confined_panels(x::AbstractVector, y::AbstractVector; alpha_deg::Real, H::Real=Inf, M::Integer=60)
    x0 = collect(Float64, x); y0 = collect(Float64, y)
    hypot(x0[end] - x0[1], y0[end] - y0[1]) <= 1e-8 ||
        throw(ArgumentError("open (blunt) panel loop, end gap $(hypot(x0[end] - x0[1], y0[end] - y0[1])): pass blunt_panel_loop(x, y)"))
    (x0[1] == x0[end] && y0[1] == y0[end]) || (push!(x0, x0[1]); push!(y0, y0[1]))
    X = similar(x0); Y = similar(y0)
    for k in eachindex(x0); X[k], Y[k] = rotate_to_freestream(x0[k], y0[k], alpha_deg); end
    N = length(X) - 1
    S = [hypot(X[j+1] - X[j], Y[j+1] - Y[j]) for j in 1:N]
    all(>(0), S) || throw(ArgumentError("degenerate panel in the foil loop"))
    th = [atan(Y[j+1] - Y[j], X[j+1] - X[j]) for j in 1:N]
    xc = [(X[j] + X[j+1]) / 2 for j in 1:N]; yc = [(Y[j] + Y[j+1]) / 2 for j in 1:N]
    An = zeros(N + 1, N + 1); At = zeros(N, N + 1); rhs = zeros(N + 1)
    images = isfinite(H) ? (-M:M) : (0:0)
    cn1 = zeros(N); cn2 = zeros(N); ct1 = zeros(N); ct2 = zeros(N)
    for i in 1:N
        fill!(cn1, 0.0); fill!(cn2, 0.0); fill!(ct1, 0.0); fill!(ct2, 0.0)
        for j in 1:N, m in images
            if m == 0 && i == j
                cn1[j] += -1.0; cn2[j] += 1.0; ct1[j] += pi / 2; ct2[j] += pi / 2
            else
                Yj = image_y(Y[j], m, H); Yj1 = image_y(Y[j+1], m, H)
                thj = atan(Yj1 - Yj, X[j+1] - X[j])
                a, b, c, d = coeffs(xc[i], yc[i], th[i], X[j], Yj, thj, S[j])
                sg = image_sign(m)
                cn1[j] += sg * a; cn2[j] += sg * b; ct1[j] += sg * c; ct2[j] += sg * d
            end
        end
        An[i, 1] = cn1[1]; An[i, N+1] = cn2[N]; At[i, 1] = ct1[1]; At[i, N+1] = ct2[N]
        for j in 2:N; An[i, j] = cn1[j] + cn2[j-1]; At[i, j] = ct1[j] + ct2[j-1]; end
        rhs[i] = sin(th[i])
    end
    An[N+1, 1] = 1.0; An[N+1, N+1] = 1.0; rhs[N+1] = 0.0            # Kutta condition
    gam = An \ rhs
    V = [cos(th[i]) + sum(At[i, j] * gam[j] for j in 1:N+1) for i in 1:N]
    Cp = 1 .- V .^ 2
    ny = [cos(th[j]) for j in 1:N]                                    # outward normal of the clockwise loop
    Cl = -sum(Cp[j] * ny[j] * S[j] for j in 1:N)
    Gamma = 2pi * sum((gam[j] + gam[j+1]) / 2 * S[j] for j in 1:N)
    PanelSolution(Float64(alpha_deg), Float64(H), Int(M), X, Y, S, th, gam, Cp, Cl, Gamma)
end

"""
    blunt_panel_loop(x, y; n_half=8) -> (X, Y)

The panel loop of a section with a finite (blunt) trailing edge, for `solve_confined_panels` (blunt trailing edge plan,
memory/project_docs/2026-10-01_BLUNT_TRAILING_EDGE_PLAN.md, section 5.3; measured 2 Oct 2026 as variant V6d in
memory/project_docs/_records/2026-10-02_blunt_te/map_panel/). `x, y` is the open section of `read_airfoil_dat` (lower corner first,
upper corner last, clockwise). The loop starts and ends at the base midpoint m, runs down the lower half of the base in
`n_half` equal panels to the lower corner, around the section, and down the upper half of the base in `n_half` panels
back to m. The base is thereby a solid panel boundary, and the unchanged solver's Kutta condition (first node = last
node) sits at the base midpoint, where the two shear layers of the real flow leave a stagnant base.

Measured on the Caltech 0.006 c edge at 3 deg, H/c 5 (60 images): confined C_L 0.6405 against 0.6369 for the sharp
section, and no start-field pressure below the vapour line at sigma 6. On the stage-0 mesh it gave 2000 explicit species
steps without a rejected attempt at dn1 1e-4; at dn1 5e-5 it does so only with the wall-distance omega start
(memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/README.md). The alternative of closing the loop with one base panel and the Kutta condition at the
lower corner, which an open loop gets by default, gives C_L 2.79 and p_min/q -55. A closed (sharp) loop is refused here:
its panel problem is the unchanged default path.
"""
function blunt_panel_loop(x::AbstractVector, y::AbstractVector; n_half::Integer=8)
    n_half >= 1 || throw(ArgumentError("n_half must be >= 1 (got $n_half)"))
    length(x) == length(y) >= 3 || throw(ArgumentError("section needs at least three points"))
    xl, yl = Float64(x[1]), Float64(y[1]); xu, yu = Float64(x[end]), Float64(y[end])
    hypot(xu - xl, yu - yl) > 1e-8 || throw(ArgumentError("blunt_panel_loop needs an open (blunt) section; a sharp loop uses solve_confined_panels directly"))
    xm, ym = 0.5*(xl + xu), 0.5*(yl + yu)
    X = [xm]; Y = [ym]
    for k in 1:n_half-1; push!(X, xm + k/n_half*(xl - xm)); push!(Y, ym + k/n_half*(yl - ym)); end
    append!(X, Float64.(x)); append!(Y, Float64.(y))
    for k in 1:n_half-1; push!(X, xu + k/n_half*(xm - xu)); push!(Y, yu + k/n_half*(ym - yu)); end
    push!(X, xm); push!(Y, ym)
    (X, Y)
end

"""Velocity (units of V) at the point (x, y) of the FOIL frame: freestream plus all panels and their images, rotated back."""
function potential_velocity(sol::PanelSolution, x::Real, y::Real)
    xf, yf = rotate_to_freestream(Float64(x), Float64(y), sol.alpha_deg)
    N = length(sol.S); u = 1.0; v = 0.0
    images = isfinite(sol.H) ? (-sol.M:sol.M) : (0:0)
    @inbounds for j in 1:N, m in images
        Yj = image_y(sol.Y[j], m, sol.H); Yj1 = image_y(sol.Y[j+1], m, sol.H)
        thj = atan(Yj1 - Yj, sol.X[j+1] - sol.X[j])
        a, b, c, d = coeffs(xf, yf, 0.0, sol.X[j], Yj, thj, sol.S[j])     # tangent angle 0: Cn = y-velocity, Ct = x-velocity
        sg = image_sign(m)
        v += sg * (a * sol.gam[j] + b * sol.gam[j+1])
        u += sg * (c * sol.gam[j] + d * sol.gam[j+1])
    end
    rotate_vector_to_foil(u, v, sol.alpha_deg)
end

"""Velocity components (units of V, foil frame) and Cp = 1 - |u|^2 at every point of the arrays `xc, yc` (foil frame)."""
function potential_field(sol::PanelSolution, xc::AbstractMatrix, yc::AbstractMatrix)
    size(xc) == size(yc) || throw(DimensionMismatch("xc and yc"))
    u = zeros(Float64, size(xc)); v = zeros(Float64, size(xc))
    Threads.@threads for k in eachindex(xc)
        @inbounds u[k], v[k] = potential_velocity(sol, xc[k], yc[k])
    end
    (u=u, v=v, Cp=1 .- (u .^ 2 .+ v .^ 2))
end

"""Largest normal velocity on the channel walls (freestream frame) sampled between x = -1 and 2.5: the wall check."""
function wall_residual(sol::PanelSolution; nsample::Integer=41)
    isfinite(sol.H) || return NaN
    worst = 0.0
    for yw in (sol.H, -sol.H), xs in range(-1.0, 2.5; length=nsample)
        # a point of the freestream frame: undo the rotation to use potential_velocity, then read the y-velocity there
        a = deg2rad(sol.alpha_deg); c = cos(a); s = sin(a)
        x = 0.5 + (xs - 0.5) * c - yw * s; y = (xs - 0.5) * s + yw * c
        uf, vf = potential_velocity(sol, x, y)
        # back to the freestream frame
        vn = -uf * s + vf * c
        worst = max(worst, abs(vn))
    end
    worst
end

end # module
