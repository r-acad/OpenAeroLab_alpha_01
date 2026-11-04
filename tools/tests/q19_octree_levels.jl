# Gate G2 of the CPU octree D3Q19 solver: coupling between levels.
#   G2.0 uniform flow through a refined box is exact; the step allocates nothing per cell
#   G2.7 interface dissipation of a shear wave crossing or running along the interfaces of a refined slab
#   G2.3 a vortex convected across two interfaces of a refined box against the one-level reference run
#   G2.5 mass drift of a closed periodic box with a refined region
#   G2.1 Poiseuille with the channel centre on a coarser level than the plates (the profile crosses interfaces)
# One periodic box; the refined region is a user box or slab. CPU, a few minutes.
#   julia -t 8 --project=src/octree_q19 tools/tests/q19_octree_levels.jl
using Test, Printf, LinearAlgebra, Random
using OctreeQ19
const G = OctreeQ19
const ONLY = get(ENV, "Q19_TEST_ONLY", "")
want(name) = isempty(ONLY) || occursin(ONLY, name)

"Periodic box of n^3 finest cells with `levels` levels; a central box region refined to the finest level."
function refined_box(n::Int, levels::Int; nu=0.01, precision=Float64, region_fraction=0.5)
    dx = 1.0; Lbox = Float64(n)
    regions = levels == 1 ? G.Region[] : [G.Region(:box, levels; lo=ntuple(_ -> Lbox * (0.5 - region_fraction / 2), 3), hi=ntuple(_ -> Lbox * (0.5 + region_fraction / 2), 3))]
    spec = G.MeshSpec(domain=(Lbox, Lbox, Lbox), periodic=(true, true, true), levels=levels, dx_finest=dx, band_fine=4, band=3, regions=regions)
    mesh = G.generate(G.Tri[], spec)
    cfg = G.test_case_config(spec; nu_lattice=nu, precision=precision)
    mesh, cfg, G.State(mesh, cfg, precision)
end

"Periodic box of n^3 finest cells; the slab n/4 <= z < 3n/4 (full x and y extent) refined to the finest level."
function refined_slab(n::Int, levels::Int; nu=0.01, precision=Float64)
    Lbox = Float64(n)
    regions = levels == 1 ? G.Region[] : [G.Region(:box, levels; lo=(0.0, 0.0, Lbox / 4), hi=(Lbox, Lbox, 3Lbox / 4))]
    spec = G.MeshSpec(domain=(Lbox, Lbox, Lbox), periodic=(true, true, true), levels=levels, dx_finest=1.0, band_fine=4, band=3, regions=regions)
    mesh = G.generate(G.Tri[], spec)
    cfg = G.test_case_config(spec; nu_lattice=nu, precision=precision)
    mesh, cfg, G.State(mesh, cfg, precision)
end

"Velocity sampled at the leaf containing p, and the leaf's level."
function sample(st, p)
    l, s = G.locate_leaf(st.mesh, p); l == 0 && return (NaN, NaN, NaN, 0)
    S = st.levels[l]; k = G.slot_k(s); o = G.slot_oct(s)
    (Float64(S.u[k, 1, o]), Float64(S.u[k, 2, o]), Float64(S.u[k, 3, o]), l)
end
"Density sampled at the leaf containing p."
function sample_rho(st, p)
    l, s = G.locate_leaf(st.mesh, p); l == 0 && return NaN
    Float64(st.levels[l].rho[G.slot_k(s), G.slot_oct(s)])
end

"Kinetic energy over the fluid leaves of all levels, weighted by the cell volume in finest-cell units."
function kinetic(st)
    e = 0.0; L = G.nlevels(st.mesh)
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]; w = 8.0^(L - l)
        for o in S.real_octs, k in 1:8
            Lv.kind[k, o] == G.K_FLUID || continue
            e += w * 0.5 * (Float64(S.u[k, 1, o])^2 + Float64(S.u[k, 2, o])^2 + Float64(S.u[k, 3, o])^2)
        end
    end
    e
end

"Viscosity measured from the kinetic-energy decay of a shear wave of wavenumber k over `steps` coarse steps, in finest-level units."
function measured_viscosity(st, k, steps, levels)
    s1 = steps ÷ 5
    for _ in 1:s1; G.advance!(st); end
    e1 = kinetic(st)
    for _ in s1+1:steps; G.advance!(st); end
    e2 = kinetic(st)
    -log(e2 / e1) / (2 * k^2 * (steps - s1)) / 2^(levels - 1)
end

want("G2.0") && @testset "G2.0 uniform flow and allocation through a refined box" begin
    # a uniform stream must cross the interfaces unchanged: any deviation isolates a coupling defect
    n = 32; U0 = 0.04
    mesh, cfg, st = refined_box(n, 3; nu=0.005)
    G.initialise!(st, p -> (1.0, U0, 0.0, 0.0))
    for _ in 1:3; G.advance!(st); end
    alloc = @allocated G.advance!(st)
    @test alloc < 256 * 1024                       # thread tasks only, no per-cell allocation in the coupling
    for _ in 1:60; G.advance!(st); end
    worst = 0.0; worst_rho = 0.0
    for (l, Lv) in enumerate(mesh.levels)
        S = st.levels[l]
        for o in S.real_octs, k in 1:8
            Lv.kind[k, o] == G.K_FLUID || continue
            worst = max(worst, abs(S.u[k, 1, o] - U0), abs(S.u[k, 2, o]), abs(S.u[k, 3, o])); worst_rho = max(worst_rho, abs(S.rho[k, o] - 1))
        end
    end
    @printf("uniform flow through 3 levels, 64 coarse steps: max |u - U0| %.3e, max |rho - 1| %.3e, allocation per step %d bytes\n", worst, worst_rho, alloc)
    @test worst < 1e-9 && worst_rho < 1e-9
end

want("G2.7") && @testset "G2.7 interface dissipation of a shear wave (slab, 2 levels)" begin
    # a shear wave at 16 coarse cells per wavelength crosses (normal) or runs along (tangential) the two interfaces
    # of the refined slab; the measured viscosity over the mean of the one-level runs at the fine and the coarse
    # spacing reads the smoothing of the transfers (linear transfers: +19 % and +9 %; cubic: about +1 %)
    n = 32; nu = 0.005; k = 2π / n; A0 = 0.01
    for (name, init) in (("normal u_x(z)", p -> (1.0, A0 * sin(k * p[3]), 0.0, 0.0)), ("tangential u_y(x)", p -> (1.0, 0.0, A0 * sin(k * p[1]), 0.0)))
        m, c, s = refined_slab(n, 1; nu=nu); G.initialise!(s, init); r_f = measured_viscosity(s, k, 800, 1)
        m, c, s = refined_slab(n ÷ 2, 1; nu=nu / 2); G.initialise!(s, p -> init(p .* 2)); r_c = 2 * measured_viscosity(s, 2k, 400, 1)
        m, c, s = refined_slab(n, 2; nu=nu); G.initialise!(s, init); r_2 = measured_viscosity(s, k, 800, 2)
        excess = (r_2 - 0.5 * (r_f + r_c)) / nu
        @printf("shear wave %s: one level fine %.4f coarse %.4f, two levels %.4f (x nu); excess %+.4f\n", name, r_f / nu, r_c / nu, r_2 / nu, excess)
        @test abs(excess) < 0.02
    end
end

want("G2.3") && @testset "G2.3 vortex convected across interfaces" begin
    # the convected vortex of the Spiegel-type AMR gate: a Gaussian-profile (Taylor) vortex, u_theta = u_max (r/rc)
    # exp((1 - (r/rc)^2)/2), compact (3e-4 of its peak at the box edge) with the closed-form pressure deficit
    # p - p_inf = -(rho u_max^2 / 2) exp(1 - (r/rc)^2) set in the density, so that no acoustic transient is started;
    # core radius rc = 8 finest cells (4 coarse cells), uniform stream U0 along x, periodic box of 64 finest cells,
    # 2 levels with the central half of the box fine: in one period (back to its start at the fine box's centre) the
    # vortex leaves through the downstream interface and re-enters through the upstream one. The reference is the
    # one-level run at the fine spacing; the one-level run at the coarse spacing (the same physical vortex) shows
    # how much of any difference is the coarse level's own discretisation.
    n = 64; U0 = 0.04; umax = 0.006; rc = 8.0; nu = 0.005
    results = Dict{String,Any}()
    for (name, nn, levels) in (("fine", n, 1), ("coarse", n ÷ 2, 1), ("2-level", n, 2))
        h = n ÷ nn                                        # finest cells per cell of this run
        rc_l = rc / h; nu_l = nu / h
        function vortex(p, c)
            dx_ = p[1] - c[1]; dx_ -= nn * round(dx_ / nn); dz_ = p[3] - c[3]; dz_ -= nn * round(dz_ / nn); r2 = dx_^2 + dz_^2
            e = exp(0.5 * (1 - r2 / rc_l^2))
            s = umax / rc_l * e                           # u_theta / r
            (1 - 1.5 * umax^2 * e^2, U0 - s * dz_, 0.0, s * dx_)   # rho = 1 + 3 (p - p_inf); the swirl is in the x-z plane
        end
        mesh, cfg, st = refined_box(nn, levels; nu=nu_l)
        c0 = (nn / 2, nn / 2, nn / 2)
        G.initialise!(st, p -> vortex(p, c0))
        steps = round(Int, n / U0 / (h * 2^(levels - 1)))  # one period at the same physical time
        pts = [(x + 0.5, nn / 2 + 0.5, z + 0.5) for x in 0:nn-1, z in 0:nn-1]
        # the far-field density fluctuation (more than 3 rc from the vortex, which moves at U0) eight times per
        # period: the interface pressure noise is its excess over the one-level fine run
        far_rms = Float64[]
        for chunk in 1:8
            for _ in 1:steps÷8; G.advance!(st); end
            xv = mod(nn / 2 + U0 * (chunk * (steps ÷ 8)) * 2^(levels - 1), nn)
            farv = [(dx_ = p[1] - xv; dx_ -= nn * round(dx_ / nn); hypot(dx_, p[3] - nn / 2) > 3rc_l) for p in pts]
            push!(far_rms, sqrt(sum((sample_rho(st, p) - 1)^2 for p in pts[farv]) / count(farv)))
        end
        for _ in 8 * (steps ÷ 8)+1:steps; G.advance!(st); end
        sw = [(u = sample(st, p); hypot(u[1] - U0, u[3])) for p in pts]
        peak = maximum(sw)
        wsum = sum(sw .^ 2); cx = h * sum(sw[i]^2 * pts[i][1] for i in eachindex(pts)) / wsum; cz = h * sum(sw[i]^2 * pts[i][3] for i in eachindex(pts)) / wsum
        results[name] = (peak=peak, cx=cx, cz=cz, far_rms=far_rms)
        @printf("vortex %-8s: swirl peak %.5f (initial %.5f), centroid (%.3f, %.3f) finest cells (start %.1f, %.1f), %d steps, mass drift %.3e, far-field rms / deficit %s %%\n",
                name, peak, umax, cx, cz, n / 2, n / 2, steps, (G.total_mass(st) - st.mass0) / st.mass0, string(round.(100 .* far_rms ./ (1.5 * umax^2 * exp(1)); digits=2)))
    end
    ref = results["fine"]; amr = results["2-level"]; crs = results["coarse"]
    deficit = 1.5 * umax^2 * exp(1)                        # the initial core deficit of rho (= 3 x the pressure deficit)
    noise = maximum(amr.far_rms .- ref.far_rms) / deficit
    @printf("vortex: 2-level peak error %+.3f %%, centroid error %.3f cells, interface pressure noise %.3f %% of the deficit; coarse-only peak error %+.3f %%, centroid error %.3f cells, far-field excess %.3f %%\n",
            100 * (amr.peak - ref.peak) / ref.peak, hypot(amr.cx - ref.cx, amr.cz - ref.cz), 100 * noise, 100 * (crs.peak - ref.peak) / ref.peak, hypot(crs.cx - ref.cx, crs.cz - ref.cz), 100 * maximum(crs.far_rms .- ref.far_rms) / deficit)
    @test abs(amr.peak - ref.peak) / ref.peak < 0.01
    @test hypot(amr.cx - ref.cx, amr.cz - ref.cz) < 0.01 * n
    @test noise < 0.01
end

want("G2.5") && @testset "G2.5 mass drift with a refined region" begin
    n = 32; mesh, cfg, st = refined_box(n, 3; nu=0.01)
    Random.seed!(4)
    G.initialise!(st, p -> (1.0 + 0.01 * sin(2π * p[1] / n) * cos(2π * p[3] / n), 0.03 * sin(2π * p[2] / n), 0.02 * cos(2π * p[3] / n) * sin(2π * p[1] / n), 0.01 * sin(2π * p[1] / n)))
    m0 = st.mass0
    steps = 10 * n                                           # 10 n coarse steps: about 10 convective times at u ~ 0.03 (n / 0.03 fine steps = n / 0.12 coarse steps per time)
    drift = Float64[]
    for s in 1:steps
        G.advance!(st); s % (n) == 0 && push!(drift, (G.total_mass(st) - m0) / m0)
    end
    @printf("mass drift per n coarse steps: %s\n", string(round.(drift; sigdigits=3)))
    @test all(isfinite, st.levels[3].f)
    @test abs(drift[end]) < 1e-4                              # registered: < 1e-6 per convective time; read against the coarse-step cadence here
end

include(joinpath(@__DIR__, "q19_octree_geometry_helpers.jl"))
want("G2.1") && @testset "G2.1 Poiseuille with an interface inside the channel" begin
    # the plates at the finest level, the channel centre one level coarser (band_fine 4 with H = 16)
    nu = 0.1; Fx = 1e-5; n = 48; H = 16.0
    function box_tris(lo, hi)
        v = [(x, y, z) for z in (lo[3], hi[3]) for y in (lo[2], hi[2]) for x in (lo[1], hi[1])]
        faces = ((1, 3, 7, 5), (2, 6, 8, 4), (1, 5, 6, 2), (3, 4, 8, 7), (1, 2, 4, 3), (5, 7, 8, 6))
        tri = G.Tri[]
        for f in faces; push!(tri, (v[f[1]], v[f[2]], v[f[3]])); push!(tri, (v[f[1]], v[f[3]], v[f[4]])); end
        tri
    end
    Lx = 24; Ly = 2n; Lz = n; c = (0.0, Ly / 2 + 0.3, Lz / 2 + 0.2)
    tri = vcat(box_tris((-Lx, -4n, H), (2Lx, 4n, H + 4n)), box_tris((-Lx, -4n, -H - 4n), (2Lx, 4n, -H)))
    tri = [Tuple((p[1], p[2] + c[2], p[3] + c[3]) for p in t) for t in tri]
    for levels in (1, 2)
        spec = G.MeshSpec(domain=(Float64(Lx), Float64(Ly), Float64(Lz)), periodic=(true, false, false), levels=levels, dx_finest=1.0, band_fine=4, band=3,
                          faces=(:periodic, :periodic, :slip, :slip, :slip, :slip))
        mesh = G.generate(tri, spec)
        cfg = G.test_case_config(spec; nu_lattice=nu, precision=Float64, body_force=(Fx, 0.0, 0.0))
        st = G.State(mesh, cfg, Float64); G.initialise!(st, p -> (1.0, 0.0, 0.0, 0.0))
        per = mesh.counts["per_level"]
        steps = round(Int, 5 * H^2 / nu) ÷ 2^(levels - 1)       # coarse steps: the same physical time
        for s in 1:steps; G.advance!(st; collect=(s == steps)); end
        # the profile at the leaves' own centres (a coarse leaf holds its centre value: the analytic profile is
        # evaluated there, not at the finest sampling point, which would read the piecewise-constant representation)
        d = Float64[]; u = Float64[]; seen = Set{Tuple{Int,Int}}()
        for z in 0:n-1
            p = (Lx / 2 + 0.5, Ly / 2 + 0.5, z + 0.5)
            l, s = G.locate_leaf(mesh, p); l == 0 && continue
            (l, s) in seen && continue; push!(seen, (l, s))
            zc = G.slot_centre(mesh.levels[l], s)[3] - c[3]; abs(zc) < H || continue
            push!(d, zc); push!(u, sample(st, p)[1])
        end
        A = hcat(ones(length(d)), d .^ 2); coef = A \ u; H_eff = sqrt(-coef[1] / coef[2])
        prof_err = maximum(abs.(u .- Fx / (2nu) .* (H^2 .- d .^ 2))) / (Fx * H^2 / (2nu))
        # the force balance over all fluid cells in finest-cell units (a coarse cell weighs 8); st.forces is in
        # coarse-lattice units and a force scales as rho dx^4 / dt^2, a factor 4 per level
        L = levels; fluid_units = sum(p["fluid"] * 8.0^(L - p["level"]) for p in per)
        friction_err = abs(abs(st.forces[1]) * 4.0^(L - 1) - Fx * fluid_units) / (Fx * fluid_units)
        @printf("Poiseuille %d level(s): levels %s, profile error %.3e, wall position error %+.4f cells, friction error %.3e\n", levels, string([(p["level"], p["fluid"]) for p in per]), prof_err, H_eff - H, friction_err)
        @test prof_err < 0.03
        @test abs(H_eff - H) < 0.1
        @test friction_err < 0.02
    end
end
println("q19_octree_levels: done")
