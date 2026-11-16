# The wall function of the CPU octree D3Q19 solver (src/octree_q19/src/walls.jl, applied in stream.jl): Reichardt's
# law and its inversion, the adverse-pressure-gradient term, the limits of the model (it vanishes on a resolved wall; mixing-length
# viscosity in the log layer), the geometric tables on tilted plates, a resolved channel that must stay Poiseuille,
# and a coarse turbulent channel that must stay stable, keep its mass and deliver a friction near the modelled one.
#   julia -t 8 --project=src/octree_q19 tools/tests/q19_wall_model.jl
using Test, Printf, Random, Statistics
using OctreeQ19
const G = OctreeQ19

function box_tris(lo, hi)
    v = [(x, y, z) for z in (lo[3], hi[3]) for y in (lo[2], hi[2]) for x in (lo[1], hi[1])]
    faces = ((1, 3, 7, 5), (2, 6, 8, 4), (1, 5, 6, 2), (3, 4, 8, 7), (1, 2, 4, 3), (5, 7, 8, 6))
    tri = G.Tri[]
    for f in faces; push!(tri, (v[f[1]], v[f[2]], v[f[3]])); push!(tri, (v[f[1]], v[f[3]], v[f[4]])); end
    tri
end

"The force-driven channel of gate G1.3 (two slabs, half-width H cells, tilted about the flow axis), with a wall specification."
function channel(n::Int, H::Float64, angle_deg::Float64, nu::Float64, Fx::Float64, walls; precision=Float64, seed=1, sgs=:none)
    Random.seed!(seed)
    a = deg2rad(angle_deg); nrm = (0.0, -sin(a), cos(a))
    Lx = 2 * (n ÷ 4); Ly = 2n; Lz = 2n
    c = (0.0, Ly / 2 + (rand() - 0.5), Lz / 2 + (rand() - 0.5))
    thick = 4.0 * n
    rot(p) = (p[1], cos(a) * p[2] - sin(a) * p[3], sin(a) * p[2] + cos(a) * p[3])
    slab(sign) = [Tuple((rot(p) .+ c) for p in t) for t in box_tris((-Lx, -4n, sign > 0 ? H : -H - thick), (2Lx, 4n, sign > 0 ? H + thick : -H))]
    spec = G.MeshSpec(domain=(Float64(Lx), Float64(Ly), Float64(Lz)), periodic=(true, false, false), levels=1, dx_finest=1.0, faces=(:periodic, :periodic, :slip, :slip, :slip, :slip))
    mesh = G.generate(vcat(slab(+1), slab(-1)), spec)
    cfg = G.test_case_config(spec; nu_lattice=nu, precision=precision, body_force=(Fx, 0.0, 0.0), walls=walls, sgs=sgs)
    mesh, cfg, G.State(mesh, cfg, precision), c, nrm
end

@testset "Reichardt's law and its inversion" begin
    # u+ ~ y+ in the sublayer, the log law far out; monotone; the derivative matches a finite difference
    @test isapprox(G.uplus_reichardt(0.5), 0.5; rtol=0.05)
    @test isapprox(G.uplus_reichardt(1000.0), log(1000) / 0.41 + 5.2; rtol=0.02)
    @test all(diff(G.uplus_reichardt.(10 .^ range(-1, 4; length=200))) .> 0)
    for yp in (0.3, 5.0, 30.0, 500.0)
        fd = (G.uplus_reichardt(yp * 1.0001) - G.uplus_reichardt(yp * 0.9999)) / (yp * 0.0002)
        @test isapprox(G.duplus_reichardt(yp), fd; rtol=1e-4)
    end
    # the friction velocity of the step satisfies the law, from any starting value, in both precisions
    worst = 0.0
    for u in (1e-5, 1e-3, 0.02, 0.08), y in (0.03, 0.5, 1.4, 2.5), nu in (1e-6, 1e-4, 1e-2)
        ref = G.u_tau_step(u, y, nu, 0.0, 0.0)
        @test ref > 0
        worst = max(worst, abs(ref * G.uplus_reichardt(y * ref / nu) / u - 1))
        for guess in (ref, 3ref, ref / 5)
            worst = max(worst, abs(G.u_tau_step(u, y, nu, 0.0, guess) / ref - 1))
        end
        @test isapprox(G.u_tau_step(Float32(u), Float32(y), Float32(nu), 0f0, 0f0), ref; rtol=2e-3)
    end
    @printf("largest relative residual of the step's friction velocity: %.2e\n", worst)
    @test worst < 1e-4
    @test G.u_tau_step(0.0, 0.5, 1e-4, 0.0, 0.0) == 0.0 && G.u_tau_step(-1.0, 0.5, 1e-4, 0.0, 0.0) == 0.0
    @test isfinite(G.u_tau_step(1e-30, 0.5, 1e-4, 0.0, 0.0)) && isfinite(G.u_tau_step(0.3, 1e-3, 1e-7, 0.0, 0.0))
end

@testset "adverse pressure gradient" begin
    nu = 2e-5; y = 1.5; u = 0.05
    ut0 = G.u_tau_step(u, y, nu, 0.0, 0.0)
    # the term vanishes with the gradient, lowers the friction as the gradient grows, and removes it where the
    # speed is at or below the half-power profile (2/kappa) sqrt(P y)
    last = ut0
    for P in (1e-9, 1e-7, 1e-6, 1e-5, 5e-5)
        ut = G.u_tau_step(u, y, nu, P, 0.0)
        speed, slope, dudy = G.law_speed(ut, y, nu, P)
        @test isapprox(speed, u; rtol=2e-4)
        @test 0 < ut <= last * (1 + 1e-9)
        # the derivative of the law's speed with respect to the friction velocity matches a finite difference
        fd = (G.law_speed(ut * 1.0001, y, nu, P)[1] - G.law_speed(ut * 0.9999, y, nu, P)[1]) / (ut * 0.0002)
        @test isapprox(slope, fd; rtol=1e-3)
        # and its wall-normal slope in wall units matches one too
        fy = (G.law_speed(ut, y * 1.0001, nu, P)[1] - G.law_speed(ut, y * 0.9999, nu, P)[1]) / (y * 0.0002) * nu / ut^2
        @test isapprox(dudy, fy; rtol=1e-3)
        last = ut
    end
    @test isapprox(G.u_tau_step(u, y, nu, 1e-9, 0.0), ut0; rtol=1e-3)
    @test G.u_tau_step(u, y, nu, 5e-5, 0.0) < 0.8 * ut0
    Psep = (u * G.KAPPA / 2)^2 / y                                    # the gradient whose half-power profile reaches u at y
    @test G.separation_speed(y, Psep) ≈ u
    @test G.u_tau_step(u, y, nu, 1.01 * Psep, ut0) == 0.0
    @test G.u_tau_step(u, y, nu, 0.9 * Psep, ut0) > 0
    # as the friction vanishes the law tends to the half-power profile
    @test isapprox(G.law_speed(1e-7, y, nu, 1e-5)[1], G.separation_speed(y, 1e-5); rtol=1e-3)
    # without friction the model returns the half-power speed at the cell and the subgrid viscosity
    @test G.wall_model(u, y, nu, 3nu, 0.5, 2 * Psep, ut0) == (0.0, G.separation_speed(0.5, 2 * Psep), 3nu)
    # a favourable or zero gradient is the plain law
    @test G.wall_model(u, y, nu, nu, 0.5, 0.0, 0.0) == G.wall_model(u, y, nu, nu, 0.5, -1e-5, 0.0)
    # single precision
    @test isapprox(G.u_tau_step(Float32(u), Float32(y), Float32(nu), 1f-5, 0f0), G.u_tau_step(u, y, nu, 1e-5, 0.0); rtol=2e-3)
end

@testset "limits of the model" begin
    nu = 1e-4
    # resolved wall: y+ about 0.45; no slip (to 1 % of the speed: the law differs from u+ = y+ by that much), molecular viscosity
    ut, ulaw, nucell = G.wall_model(4e-5, 0.5, nu, nu, 0.5, 0.0, 0.0)
    @test 0.5 * ut / nu < 0.5 && isapprox(nucell, nu; rtol=1e-2) && G.wall_slip_speed(ulaw, ut, 0.5, nucell) < 1e-2 * 4e-5
    for u in (0.01, 0.05), ys in (0.6, 1.5), y in (0.1, 0.5, 0.9), P in (0.0, 1e-6)
        ut, ulaw, nucell = G.wall_model(u, ys, nu, nu, y, P, 0.0)
        @test ut > 0 && nucell >= nu
        @test ulaw == G.law_speed(ut, y, nu, P)[1]
        # the wall velocity for which a linear profile to a cell at the law's own speed carries u_tau^2
        slip = G.wall_slip_speed(ulaw, ut, y, nucell)
        @test 0 <= slip <= ulaw
        slip > 0 && @test isapprox(nucell * (ulaw - slip) / y, ut^2; rtol=1e-9)
        # the subgrid viscosity is a floor
        @test G.wall_model(u, ys, nu, 1e3 * nu, y, P, 0.0)[3] == 1e3 * nu
    end
    @test G.wall_slip_speed(1e-4, 1e-2, 0.5, nu) == 0                 # a cell too slow to carry the shear keeps a no-slip wall
    @test G.wall_model(0.0, 0.5, nu, 3nu, 0.5, 0.0, 0.0) == (0.0, 0.0, 3nu)
    # log layer (y+ about 600): mixing-length viscosity kappa y u_tau and a slip of u - u_tau / kappa
    ut, ulaw, nucell = G.wall_model(0.05, 0.5, 2e-6, 2e-6, 0.5, 0.0, 0.0)
    slip = G.wall_slip_speed(ulaw, ut, 0.5, nucell); yp = 0.5 * ut / 2e-6
    @printf("log layer: y+ %.0f, u_tau %.5f, slip/u %.3f, nu_cell/nu %.1f (kappa y+ = %.1f)\n", yp, ut, slip / 0.05, nucell / 2e-6, G.KAPPA * yp)
    @test 300 < yp < 2000 && isapprox(nucell, G.KAPPA * 0.5 * ut; rtol=0.03) && isapprox(slip, 0.05 - ut / G.KAPPA; rtol=0.03)
end

@testset "wall-model tables on tilted plates" begin
    for angle in (0.0, 30.0), sampling in (:cell, :normal_1, :normal_2)
        H = 6.3
        mesh, cfg, st, c, nrm = channel(36, H, angle, 0.01, 1e-6, G.WallSpec(wall_function=true, sampling=sampling))
        Lv = mesh.levels[1]; S = st.levels[1]
        @test S.wm_on
        nmodel = 0; worst_n = 0.0; worst_y = 0.0; worst_ref = 0.0; nref = 0
        for (ci, s) in enumerate(Lv.cut_slot)
            n = (S.wm_n[1, ci], S.wm_n[2, ci], S.wm_n[3, ci]); all(==(0), n) && continue
            p = G.slot_centre(Lv, Int(s)); abs(p[2] - mesh.domain[2] / 2) < mesh.domain[2] / 4 || continue
            sd = (p[2] - c[2]) * nrm[2] + (p[3] - c[3]) * nrm[3]          # signed distance from the centre plane
            inward = sd > 0 ? (-nrm[1], -nrm[2], -nrm[3]) : nrm           # the normal of the nearer plate, into the fluid
            nmodel += 1
            worst_n = max(worst_n, maximum(abs.(n .- inward)))
            worst_y = max(worst_y, abs(S.wm_y[ci] - clamp(H - abs(sd), 0.02, 1.5)))
            r = Int(S.wm_ref[ci])
            if r != 0
                nref += 1
                pr = G.slot_centre(Lv, r); sr = (pr[2] - c[2]) * nrm[2] + (pr[3] - c[3]) * nrm[3]
                worst_ref = max(worst_ref, abs(S.wm_yref[ci] - (H - abs(sr))))
                @test Lv.kind[G.slot_k(r), G.slot_oct(r)] == G.K_FLUID
            else
                @test S.wm_yref[ci] == S.wm_y[ci]
            end
        end
        @printf("angle %4.1f sampling %-8s: %d wall cells with a model, %d sampled away from the cell; normal error %.1e, distance error %.1e, sample distance error %.1e\n",
                angle, sampling, nmodel, nref, worst_n, worst_y, worst_ref)
        @test nmodel > 100 && worst_n < 1e-5 && worst_y < 1e-4 && worst_ref < 1e-4
        sampling == :cell ? (@test nref == 0) : (@test nref > 0.9 * nmodel)
    end
end

"Velocity profile in the central quarter of y: signed distance from the centre plane and u_x of every fluid cell."
function profile(st, c, nrm)
    Lv = st.mesh.levels[1]; S = st.levels[1]; Ly = st.mesh.domain[2]
    d = Float64[]; u = Float64[]
    for o in S.real_octs, k in 1:8
        Lv.kind[k, o] == G.K_FLUID || continue
        p = G.slot_centre(Lv, G.slot_id(k, o)); abs(p[2] - Ly / 2) < Ly / 8 || continue
        push!(d, (p[2] - c[2]) * nrm[2] + (p[3] - c[3]) * nrm[3]); push!(u, Float64(S.u[k, 1, o]))
    end
    d, u
end

@testset "a resolved channel stays Poiseuille with the wall function on" begin
    nu = 0.1; Fx = 1e-5; n = 36; H = 6.0
    for angle in (0.0, 30.0), sampling in (:cell, :normal_1)
        err = map((false, true)) do on
            mesh, cfg, st, c, nrm = channel(n, H, angle, nu, Fx, G.WallSpec(wall_function=on, sampling=sampling); seed=7)
            G.initialise!(st, p -> (1.0, 0.0, 0.0, 0.0))
            for s in 1:round(Int, 5 * H^2 / nu); G.advance!(st; collect=false); end
            d, u = profile(st, c, nrm)
            maximum(abs.(u .- Fx / (2nu) .* (H^2 .- d .^ 2))) / (Fx * H^2 / (2nu))
        end
        @printf("angle %4.1f sampling %-8s: profile error without the wall function %.3e, with it %.3e (wall y+ about %.3f)\n", angle, sampling, err[1], err[2], 0.5 * sqrt(Fx * H) / nu)
        @test err[2] < max(1.25 * err[1], err[1] + 2e-3)
    end
end

@testset "under-resolved turbulent channel: stable, mass kept, friction near the model's" begin
    # Plates aligned with the lattice, 8.6 cells across, wall cells at y+ of about 10 to 25, Smagorinsky viscosity in
    # the interior: a coarse wall-modelled channel that is turbulent from its start. Readings over the second half:
    #   the run stays in range and keeps its mass (the unfiltered slip lost 20 % of it within 4000 steps);
    #   the wall force collected by the momentum exchange against the sum of rho u_tau^2 over the wall cells (one cell
    #   of wall area each). The model's known limit: the links also absorb momentum that the cell's resolved motion
    #   carries to the wall, so the ratio reads 1.3 to 1.5 here (3.5 to 4.4 with the no-slip rule on the same flow).
    H = 4.3; nu = 5e-5; ut0 = 1.5e-3; Fx = ut0^2 / H; cs = 0.1
    for (pressure_gradient, precision) in ((false, Float64), (true, Float32))
        mesh, cfg, st, c, nrm = channel(16, H, 0.0, nu, Fx, G.WallSpec(wall_function=true, sampling=:cell, pressure_gradient=pressure_gradient); precision=precision, seed=3, sgs=:smagorinsky)
        Lv = mesh.levels[1]; S = st.levels[1]
        # start near the expected state: the law's velocity at the wall cells, and above them the profile of a
        # Smagorinsky interior under the linear stress, du/dy = sqrt(tau) / cs
        G.initialise!(st, function (p)
            y = H - abs(p[3] - c[3]); y <= 0 && return (1.0, 0.0, 0.0, 0.0)
            (1.0, ut0 * (G.uplus_reichardt(0.5 * ut0 / nu) + (2H / (3cs)) * ((1 - 0.5 / H)^1.5 - (1 - min(y, H) / H)^1.5)), 0.0, 0.0)
        end)
        wall = [ci for ci in eachindex(Lv.cut_slot) if !(S.wm_n[1, ci] == 0 && S.wm_n[2, ci] == 0 && S.wm_n[3, ci] == 0)]
        @test length(wall) == 2 * Lv.n[1] * Lv.n[2]                                  # one layer of wall cells per plate
        steps = 20000; ratios = Float64[]; m0 = G.total_mass(st)
        for s in 1:steps
            sample = s > steps ÷ 2 && s % 100 == 0
            G.advance!(st; collect=sample)
            sample && push!(ratios, st.forces[1] / sum(Float64(S.rho[G.slot_k(Lv.cut_slot[ci]), G.slot_oct(Lv.cut_slot[ci])]) * Float64(S.tau_w[ci])^2 for ci in wall))
        end
        yplus = [Float64(S.wm_y[ci]) * Float64(S.tau_w[ci]) / nu for ci in wall]
        rmin, rmax, umax = G.state_extrema(st); drift = G.total_mass(st) / m0 - 1
        @printf("pressure-gradient term %-5s %s: wall y+ %.1f to %.1f, mean u_tau %.3e, momentum exchange / sum rho u_tau^2 = %.3f (std %.3f over %d samples), mass change %+.2e, |u|max %.4f\n",
                pressure_gradient, precision, minimum(yplus), maximum(yplus), mean(Float64.(S.tau_w[wall])), mean(ratios), std(ratios), length(ratios), drift, umax)
        @test all(isfinite, (rmin, rmax, umax)) && umax < 0.1 && 0.97 < rmin <= rmax < 1.03
        @test abs(drift) < 5e-3
        @test minimum(yplus) > 2 && maximum(yplus) < 60
        @test 0.9 < mean(ratios) < 1.8
    end
end
@testset "tilted plates: the slip adds no mass" begin
    # On plates tilted against the lattice the cut links of a wall cell are not symmetric about its normal; the slip
    # velocity is kept normal to the cell link sum, so that the wall term adds no mass in any cell.
    H = 4.3; nu = 5e-5; ut0 = 1.5e-3; Fx = ut0^2 / H
    mesh, cfg, st, c, nrm = channel(16, H, 30.0, nu, Fx, G.WallSpec(wall_function=true, sampling=:normal_1); precision=Float64, seed=5, sgs=:smagorinsky)
    Lv = mesh.levels[1]; S = st.levels[1]
    G.initialise!(st, function (p)
        y = H - abs((p[2] - c[2]) * nrm[2] + (p[3] - c[3]) * nrm[3]); y <= 0 && return (1.0, 0.0, 0.0, 0.0)
        (1.0, ut0 * (10 + 20 * (1 - (1 - min(y, H) / H)^1.5)), 0.0, 0.0)
    end)
    # the link sum of every wall cell points into the wall, and the wall term of any slip normal to it sums to zero
    worst = 0.0
    for ci in eachindex(Lv.cut_slot)
        n = (S.wm_n[1, ci], S.wm_n[2, ci], S.wm_n[3, ci]); all(==(0), n) && continue
        d = (S.wm_d[1, ci], S.wm_d[2, ci], S.wm_d[3, ci])
        @test sum(d .* n) < 0
        u = (0.01, 0.02, -0.015); a = sum(u .* d) / sum(d .* d); w = u .- a .* d
        total = 0.0
        for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
            i = Int(Lv.cut_dir[li]); q = Float64(Lv.cut_q[li]); factor = (!Lv.cut_partner[li] || q < 0.5) ? 1.0 : 1 / (2q)
            total += factor * 6 * G.WT[i] * (G.CX[i] * w[1] + G.CY[i] * w[2] + G.CZ[i] * w[3])
        end
        worst = max(worst, abs(total))
    end
    @test worst < 1e-12
    m0 = G.total_mass(st)
    for s in 1:6000; G.advance!(st); end
    rmin, rmax, umax = G.state_extrema(st); drift = G.total_mass(st) / m0 - 1
    # the interpolated link rule is not mass-conserving by itself: the same run without the wall function is the reference
    mesh0, cfg0, st0, c0, nrm0 = channel(16, H, 30.0, nu, Fx, G.WallSpec(wall_function=false); precision=Float64, seed=5, sgs=:smagorinsky)
    G.initialise!(st0, function (p)
        y = H - abs((p[2] - c0[2]) * nrm0[2] + (p[3] - c0[3]) * nrm0[3]); y <= 0 && return (1.0, 0.0, 0.0, 0.0)
        (1.0, ut0 * (10 + 20 * (1 - (1 - min(y, H) / H)^1.5)), 0.0, 0.0)
    end)
    m00 = G.total_mass(st0)
    for s in 1:6000; G.advance!(st0); end
    drift0 = G.total_mass(st0) / m00 - 1
    @printf("plates at 30 degrees, 6000 steps: mass change %+.2e with the wall function, %+.2e without; rho %.4f..%.4f, |u|max %.4f; largest wall-term mass of a slip normal to the link sum %.1e
", drift, drift0, rmin, rmax, umax, worst)
    @test all(isfinite, (rmin, rmax, umax)) && umax < 0.1
    @test abs(drift) < max(1.5 * abs(drift0), abs(drift0) + 2e-3)
end
println("q19_wall_model: done")
