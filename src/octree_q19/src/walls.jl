# The wall function: Reichardt's law of the wall with an optional adverse-pressure-gradient term, its inversion
# for the friction velocity, and the geometric tables of the wall cells. The step applies it (stream.jl,
# `wall_slip`) through the Bouzidi rule as a slip velocity of the wall:
#   1. the tangential speed U is sampled from the previous step's velocity at a point on the wall normal
#      (`sampling`: the wall cell itself, or the fluid cell one or two cells further out), at wall distance y_s;
#   2. the friction velocity solves U = u_law(y_s; u_tau, P), with P the adverse part of the pressure gradient
#      along the flow, (1/rho) dp/ds, when `pressure_gradient` is on (zero otherwise);
#   3. at the wall cell's own distance y the law gives the total viscosity that carries the wall shear with the
#      law's gradient, nu / (du+/dy+)(y+) (kappa y u_tau in the log layer); the cell's collision uses the larger
#      of that and its molecular plus subgrid viscosity, nu_cell (with the law's value alone, lower than the subgrid
#      one in places, a sheared channel lost stability);
#   4. the wall velocity of the cell's cut links is the slip u_slip = u_law(y) - u_tau^2 y / nu_cell along the cell's
#      own tangential velocity (between zero and the cell's own speed): the wall velocity for which a linear
#      profile between the wall and a cell on the law carries the shear nu_cell (u_law - u_slip) / y = u_tau^2
#      (zero on a resolved wall, whatever the sampling point);
#      the slip is then made normal to the cell's link sum d (see `wall_model_tables`), so that the wall term adds
#      no mass in any cell: on a curved surface the cut links of a cell are not symmetric about its normal, and a
#      slip tangential to the true surface pumped mass between neighbouring wall cells (an aircraft run settled at
#      a lift far above attached flow);
#   5. nu_cell, the slip velocity and P are exponential time averages over `filter_steps` steps (50 by default), the
#      usual practice of wall-modelled LES: unfiltered, the slip followed the cell's velocity direction step by
#      step and a coarse turbulent channel lost stability within a few thousand steps.
# Where the wall cell is in the viscous sublayer the slip tends to zero and the viscosity to the molecular one:
# the model switches itself off on a resolved wall. Everything is in lattice units of the cell's level.
#
# The law. u+ = R(y+) + D(y+, p+), with Reichardt's R (viscous sublayer to log layer) and, for an adverse gradient
# p+ = nu P / u_tau^3 > 0, the additive term of the pressure-gradient log law (Afzal 1996),
#   D = (1/kappa) [2 (s - 1) - 2 ln((s + 1)/2)],  s = sqrt(1 + p+ y+),
# which vanishes with p+ and tends to the half-power profile u = (2/kappa) sqrt(P y) as u_tau tends to zero: where
# the sampled speed is at or below that profile the wall carries no friction (the flow is at or beyond separation).
#
# Status (14 Nov 2026): not validated; off by default in Aero Lab. Measurements:
# - Turbulent channel at friction Reynolds number 2000 (half-width 20 cells, wall cells at y+ 50, WALE, speed
#   sampled one cell out, 120,000 steps; reference U_bulk / u_tau = 21.7, Hoyas and Jimenez 2006): this wall
#   function gives 16.7, no-slip walls give 19.1. The wall takes about three times the friction the law asks for
#   (modelled u_tau 0.55 of the real one): the cut links also carry to the wall the momentum of the cell's own
#   motion, which the slip of step 4 does not account for.
# - Coarse channel of tools/tests/q19_wall_model.jl (8.6 cells across, wall cells at y+ 5 to 17): the wall
#   receives 1.2 to 1.3 times the modelled friction, against 3.5 to 4.4 times with the no-slip rule.
# Forms tried and dropped:
# - adding -u_cell u_n to the slip's target, or lowering the wall velocity by 3 u_n u_t: unstable, or the channel
#   accelerated without limit;
# - returning the measured tangential link momentum to the cell as a force: lost stability within 200 steps;
# - a wall velocity solved each step so that the links take exactly the law's stress (with the mass of the
#   returns balanced per cell): exact and mass-conserving on walls along the lattice, but the channel above then
#   overshoots (U_bulk / u_tau 25.8 and rising after 120,000 steps: the law sampled at the second cell over a
#   coarse interior), and on plates tilted 30 degrees against the lattice it drove cross-flows of 0.3 lattice
#   units (a link-wise slip on stair-step cells has no control of the wall-normal momentum).
# Open: a wall treatment that rebuilds the wall cell's populations from the law (velocity and gradients), as the
# wall-modelled LBM literature does, and a check of the interior's resolved turbulence at this resolution.

const KAPPA = 0.41

"Reichardt's law u+(y+), valid from the viscous sublayer to the log layer (Reichardt 1951)."
@inline function uplus_reichardt(yp::T) where {T<:AbstractFloat}
    # expm1 avoids losing significant digits in 1-exp(-y+/11) close to a wall.
    a = yp / T(11); b = yp / T(3)
    T(1 / KAPPA) * log1p(T(KAPPA) * yp) + T(7.8) * (-expm1(-a) - a * exp(-b))
end
"du+/dy+ of Reichardt's law."
@inline function duplus_reichardt(yp::T) where {T<:AbstractFloat}
    one(T) / (one(T) + T(KAPPA) * yp) + T(7.8) * (exp(-yp / T(11)) / T(11) - exp(-yp / T(3)) / T(11) + (yp / T(33)) * exp(-yp / T(3)))
end

"""
The law in dimensional form at wall distance `y` for friction velocity `ut > 0`, viscosity `nu` and adverse
pressure gradient `P` >= 0: the speed u = ut [R(y+) + D(y+, p+)], its derivative with respect to ut at fixed y,
and its wall-normal slope in wall units, (du/dy) nu / ut^2.
"""
@inline function law_speed(ut::T, y::T, nu::T, P::T) where {T<:AbstractFloat}
    yp = y * ut / nu
    up = uplus_reichardt(yp); dup = duplus_reichardt(yp)
    P > zero(T) || return ut * up, up + yp * dup, dup
    a = P * y
    s = sqrt(one(T) + a / (ut * ut))
    phi = T(2) * (s - one(T)) - T(2) * log((s + one(T)) / T(2))
    k = T(1 / KAPPA)
    ut * up + ut * k * phi,
    up + yp * dup + k * phi - T(2) * k * a / (ut * ut * (s + one(T))),
    dup + k * nu * P / (ut * ut * ut * (s + one(T)))
end

"The half-power profile of a wall without friction under the adverse pressure gradient P: u = (2/kappa) sqrt(P y)."
@inline separation_speed(y::T, P::T) where {T<:AbstractFloat} = T(2 / KAPPA) * sqrt(max(zero(T), P) * y)

"""
Friction velocity from the tangential speed `u` at wall distance `y`, viscosity `nu` and adverse pressure gradient
`P` (lattice units): Newton on u_law(y; u_tau, P) = u kept inside a bracket, warm-started from `guess`. Without a
pressure gradient the bracket is around [sqrt(u nu / y), max(u, that)] (u+ about y+ gives the lower end, u+ >= 1
beyond y+ = 1 the upper one; Reichardt exceeds y+ by a percent near y+ = 2, hence the margins); with one the root is
lower and the bracket starts at zero. Returns 0 for u <= 0 and where the speed is at or below the half-power
profile (no friction). Never throws.
"""
@inline function u_tau_step(u::T, y::T, nu::T, P::T, guess::T) where {T<:AbstractFloat}
    (u > zero(T) && y > zero(T) && isfinite(u)) || return zero(T)
    adverse = P > zero(T)
    adverse && u <= separation_speed(y, P) && return zero(T)
    visc = sqrt(u * nu / y); lo = adverse ? zero(T) : T(0.8) * visc; hi = max(u, T(1.25) * visc)
    ut = (guess > lo && guess < hi) ? guess : adverse ? min(visc, T(0.5) * hi) : sqrt(lo * hi)
    for _ in 1:16
        speed, slope, _ = law_speed(ut, y, nu, adverse ? P : zero(T))
        g = speed - u
        g > zero(T) ? (hi = ut) : (lo = ut)
        next = slope > zero(T) ? ut - g / slope : T(-1)
        next = (next > lo && next < hi && isfinite(next)) ? next : T(0.5) * (lo + hi)
        done = abs(next - ut) <= T(1e-5) * ut
        ut = next
        done && break
    end
    ut
end

"""
The wall model of one cut cell (lattice units): from the tangential speed `us` sampled at wall distance `ys`, the
molecular viscosity `nu`, the cell's molecular plus subgrid viscosity `nusgs`, the cell's wall distance `y` and the
adverse pressure gradient `P`: the friction velocity (warm-started from `guess`), the speed the law gives at the
cell, and the viscosity of the cell, the larger of `nusgs` and the law's total viscosity at y, nu / (du+/dy+).
Without friction (no flow, or at or beyond separation) the law's speed is the half-power profile and the viscosity
is `nusgs`.
"""
@inline function wall_model(us::T, ys::T, nu::T, nusgs::T, y::T, P::T, guess::T) where {T<:AbstractFloat}
    ut = u_tau_step(us, ys, nu, P, guess)
    ut > zero(T) || return zero(T), separation_speed(y, P), nusgs
    speed, _, slope = law_speed(ut, y, nu, P)
    # the slope is kept between that of the sublayer (Reichardt exceeds it by a percent near y+ = 2) and a floor
    ut, speed, max(nusgs, nu / clamp(slope, T(1e-6), one(T)))
end

"The slip speed of the wall for which a cell at the law's speed `ulaw`, wall distance `y` and viscosity `nucell` carries the shear u_tau^2 on a linear profile: ulaw - u_tau^2 y / nucell, not below zero."
@inline wall_slip_speed(ulaw::T, ut::T, y::T, nucell::T) where {T<:AbstractFloat} = max(zero(T), ulaw - ut * ut * y / nucell)

"""
The wall-model tables of a level: per cut cell, the unit wall normal pointing into the fluid (zero where none can be
defined: no triangle, or opposing surfaces within the cell, where the cell stays a plain no-slip cell), the wall
distance of the cell centre (cells), the slot whose velocity is sampled (0 = the cell itself), its distance, and
the link sum d = sum over the cut links of k_i w_i c_i (k_i the factor the link rule applies to the wall term): the
wall term adds the mass -6 rho u_w . d to the cell, so the slip velocity is kept normal to d.
The normal is the distance-weighted mean of the normals of the triangles its cut links meet, each turned toward the
cell; the distance is the smallest one of the link crossings along that normal.
"""
function wall_model_tables(mesh::OctMesh, Lv::OctLevel, sampling::Symbol, ::Type{T}) where T
    nc = length(Lv.cut_slot)
    normal = zeros(T, 3, nc); dist = zeros(T, nc); ref = zeros(Int32, nc); dref = zeros(T, nc); dlink = zeros(T, 3, nc)
    unit(t) = (n = vcross(vsub(t[2], t[1]), vsub(t[3], t[1])); l = vnorm(n); l > 0 ? vmul(n, 1 / l) : (0.0, 0.0, 0.0))
    steps = sampling == :cell ? 0 : sampling == :normal_2 ? 2 : 1
    for ci in 1:nc
        acc = (0.0, 0.0, 0.0); wsum = 0.0
        for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
            tr = Int(Lv.cut_tri[li]); (1 <= tr <= length(mesh.triangles)) || continue
            i = Int(Lv.cut_dir[li]); c = (Float64(CX[i]), Float64(CY[i]), Float64(CZ[i]))
            n = unit(mesh.triangles[tr]); cn = vdot(c, n)
            abs(cn) < 1e-9 && continue
            cn > 0 && (n = vmul(n, -1.0); cn = -cn)                 # toward the cell: the link points into the wall
            w = 1 / (Float64(Lv.cut_q[li]) * abs(cn) + 0.1)
            acc = vadd(acc, vmul(n, w)); wsum += w
        end
        (wsum > 0 && vnorm(acc) / wsum >= 0.5) || continue
        n = vmul(acc, 1 / vnorm(acc))
        y = Inf
        for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
            i = Int(Lv.cut_dir[li]); cn = CX[i] * n[1] + CY[i] * n[2] + CZ[i] * n[3]
            cn < -1e-9 && (y = min(y, Float64(Lv.cut_q[li]) * (-cn)))
        end
        isfinite(y) || continue
        ytrue = y; y = clamp(y, 0.02, 1.5)        # the model's distance of the cell; the sample distance uses the true one
        normal[1, ci] = T(n[1]); normal[2, ci] = T(n[2]); normal[3, ci] = T(n[3]); dist[ci] = T(y); dref[ci] = T(y)
        # the sum over the cut links of (the factor of the wall term) w_i c_i: a wall velocity normal to it adds no mass
        dl = (0.0, 0.0, 0.0)
        for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
            i = Int(Lv.cut_dir[li]); q = Float64(Lv.cut_q[li])
            factor = (!Lv.cut_partner[li] || q < 0.5) ? 1.0 : 1 / (2q)
            dl = vadd(dl, vmul((Float64(CX[i]), Float64(CY[i]), Float64(CZ[i])), factor * WT[i]))
        end
        dlink[1, ci] = T(dl[1]); dlink[2, ci] = T(dl[2]); dlink[3, ci] = T(dl[3])
        steps == 0 && continue
        # the lattice direction closest to the normal whose `steps` cells outward are fluid
        cell = slot_cell(Lv, Int(Lv.cut_slot[ci])); best = 0.6; chosen = 0
        for d in 2:Q
            a = (CX[d] * n[1] + CY[d] * n[2] + CZ[d] * n[3]) / sqrt(CX[d]^2 + CY[d]^2 + CZ[d]^2)
            a > best || continue
            ok = true
            for m in 1:steps
                s = find_slot(Lv, cell[1] + m * CX[d], cell[2] + m * CY[d], cell[3] + m * CZ[d])
                (s != 0 && Lv.kind[slot_k(s), slot_oct(s)] == K_FLUID) || (ok = false)
            end
            ok && (best = a; chosen = d)
        end
        chosen == 0 && continue
        ref[ci] = Int32(find_slot(Lv, cell[1] + steps * CX[chosen], cell[2] + steps * CY[chosen], cell[3] + steps * CZ[chosen]))
        dref[ci] = T(ytrue + steps * (CX[chosen] * n[1] + CY[chosen] * n[2] + CZ[chosen] * n[3]))
    end
    normal, dist, ref, dref, dlink
end

"""
The state of the wall model on the finest level that has one: the number of wall cells with a model, the quartiles
of their y+ (the cell's own wall distance times the last friction velocity over the viscosity) among the cells
with friction, and the number of cells where the model found none (at rest, or at or beyond separation).
`nothing` when no level has a wall model.
"""
function wall_model_summary(st)
    for l in length(st.levels):-1:1
        S = st.levels[l]; S.wm_on || continue
        Lv = st.mesh.levels[l]
        model = [ci for ci in eachindex(Lv.cut_slot) if !(S.wm_n[1, ci] == 0 && S.wm_n[2, ci] == 0 && S.wm_n[3, ci] == 0)]
        yplus = sort!([Float64(S.wm_y[ci]) * Float64(S.tau_w[ci]) / Float64(S.nu) for ci in model if S.tau_w[ci] > 0])
        q(p) = isempty(yplus) ? 0.0 : yplus[clamp(round(Int, p * length(yplus)), 1, length(yplus))]
        return (level=l, cut_cells=length(Lv.cut_slot), model_cells=length(model), friction_cells=length(yplus),
                yplus_q10=q(0.1), yplus_median=q(0.5), yplus_q90=q(0.9), sampled_away=count(ci -> S.wm_ref[ci] != 0, model))
    end
    nothing
end
