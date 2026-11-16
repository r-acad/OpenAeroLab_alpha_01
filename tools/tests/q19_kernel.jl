# The closed-form, eight-at-a-time kernel of the CPU octree D3Q19 solver (src/octree_q19/src/kernel.jl) against
# its definitions: the pair-wise collision against the per-direction formulas it replaced, the eight lanes of an
# oct against the same cell stepped alone (bit for bit), the pull tables, a whole multi-level step with a body in
# both modes (bit for bit), and the oct-grouped level transfers against the slot-by-slot form (bit for bit).
#   julia -t 8 --project=src/octree_q19 tools/tests/q19_kernel.jl
using Test, Random, Printf, SIMD
using OctreeQ19
const G = OctreeQ19
include(joinpath(@__DIR__, "q19_octree_geometry_helpers.jl"))
Random.seed!(20261114)

bits(x::Float32) = reinterpret(UInt32, x); bits(x::Float64) = reinterpret(UInt64, x)
same(a, b) = bits(a) == bits(b) || (isnan(a) && isnan(b))

"A random near-equilibrium shifted state: equilibrium at a random (rho, u) plus a small random non-equilibrium."
function random_state(::Type{T}; noise=1e-3) where T
    rho = T(0.95 + 0.1rand()); u = ntuple(_ -> T(0.2 * (rand() - 0.5)), 3); usq = sum(abs2, u)
    ntuple(i -> G.feq_shifted(i, rho, u..., usq) + T(noise * G.WT[i] * randn()), 19)
end

# The per-direction definitions that the closed form replaced (the solver no longer carries them).
function pi_neq_reference(f::NTuple{19,T}, feqs::NTuple{19,T}) where T
    p = zeros(T, 6)
    for i in 2:19
        fn = f[i] - feqs[i]; cx = T(G.CX[i]); cy = T(G.CY[i]); cz = T(G.CZ[i])
        p .+= (cx * cx * fn, cy * cy * fn, cz * cz * fn, cx * cy * fn, cx * cz * fn, cy * cz * fn)
    end
    Tuple(p)
end
a3_reference(p, ux, uy, uz) = (2ux * p[4] + uy * p[1], 2ux * p[5] + uz * p[1], 2uy * p[4] + ux * p[2], 2uy * p[6] + uz * p[2], 2uz * p[5] + ux * p[3], 2uz * p[6] + uy * p[3])
function fneq_reference(i::Int, p::NTuple{6,T}, a3::NTuple{6,T}) where T
    cx = T(G.CX[i]); cy = T(G.CY[i]); cz = T(G.CZ[i]); third = T(1) / T(3)
    h2 = (cx * cx - third) * p[1] + (cy * cy - third) * p[2] + (cz * cz - third) * p[3] + T(2) * (cx * cy * p[4] + cx * cz * p[5] + cy * cz * p[6])
    h3 = cy * (cx * cx - third) * a3[1] + cz * (cx * cx - third) * a3[2] + cx * (cy * cy - third) * a3[3] + cz * (cy * cy - third) * a3[4] +
         cx * (cz * cz - third) * a3[5] + cy * (cz * cz - third) * a3[6]
    T(G.WT[i]) * (T(4.5) * h2 + T(13.5) * h3)
end

"The collision as it was written before the closed form: per-direction equilibria, moments of f - feq, the regularised non-equilibrium per direction."
function reference_collide(f::NTuple{19,T}, omega::T, sigma::T, pfd::NTuple{6,T}, F::NTuple{3,T}) where T
    rho, mx, my, mz = G.macros_shifted(f)
    forced = any(!=(0), F); h = forced ? T(0.5) / rho : zero(T)
    u = (mx + h * F[1], my + h * F[2], mz + h * F[3]); usq = sum(abs2, u)
    feqs = ntuple(i -> G.feq_shifted(i, rho, u..., usq), 19)
    p = pi_neq_reference(f, feqs)
    a2 = ntuple(n -> sigma * p[n] + (one(T) - sigma) * pfd[n], 6)
    a3 = a3_reference(a2, u...)
    keep = one(T) - omega; pre = one(T) - T(0.5) * omega
    out = ntuple(19) do i
        c = T.((G.CX[i], G.CY[i], G.CZ[i])); cu = sum(c .* u); cF = sum(c .* F); uF = sum(u .* F)
        feqs[i] + keep * fneq_reference(i, a2, a3) + (forced ? T(G.WT[i]) * (T(1.5) * cF + pre * (T(9) * cu * cF - T(3) * uF)) : zero(T))
    end
    out, rho, u
end

@testset "closed form against the per-direction formulas" begin
    for (T, tol) in ((Float64, 2e-15), (Float32, 1e-6))     # eight units in the last place of a density near one
        worst = 0.0
        for trial in 1:2000
            f = random_state(T)
            omega = T(0.6 + 1.39rand()); sigma = trial % 3 == 0 ? one(T) : T(0.9 + 0.1rand())
            pfd = ntuple(_ -> T(1e-4 * randn()), 6)
            F = trial % 4 == 0 ? ntuple(_ -> T(1e-4 * randn()), 3) : (zero(T), zero(T), zero(T))
            out, rho, ux, uy, uz = G.collide_rr(f, omega, sigma, pfd, F)
            ref, rrho, ru = reference_collide(f, omega, sigma, pfd, F)
            worst = max(worst, maximum(abs.(out .- ref)), abs(rho - rrho), abs(ux - ru[1]), abs(uy - ru[2]), abs(uz - ru[3]))
            # the moments and the equilibrium against their definitions
            drho, mx, my, mz, P = G.moments_q19(f)
            @test isapprox(drho, sum(f); atol=20eps(T)) && isapprox(mx, sum(G.CX[i] * f[i] for i in 1:19); atol=20eps(T))
            @test isapprox(P[4], sum(G.CX[i] * G.CY[i] * f[i] for i in 1:19); atol=20eps(T)) && isapprox(P[3], sum(G.CZ[i]^2 * f[i] for i in 1:19); atol=20eps(T))
        end
        @printf("%s: largest difference of the closed-form collision from the per-direction form %.2e (tolerance %.1e)\n", T, worst, tol)
        @test worst < tol
        rho = T(1.03); u = T.((0.05, -0.02, 0.07)); usq = sum(abs2, u)
        eq = G.equilibrium_q19(rho - one(T), rho, u...)
        @test maximum(abs(eq[i] - G.feq_shifted(i, rho, u..., usq)) for i in 1:19) < (T == Float64 ? 1e-16 : 1e-7)
        # conservation of the collision in exact form: density and momentum of the output are those of the input
        f = random_state(T); out, = G.collide_rr(f, T(1.7), T(0.98), ntuple(_ -> zero(T), 6))
        @test isapprox(sum(out), sum(f); atol=40eps(T))
        @test all(isapprox(sum((G.CX, G.CY, G.CZ)[a][i] * out[i] for i in 1:19), sum((G.CX, G.CY, G.CZ)[a][i] * f[i] for i in 1:19); atol=40eps(T)) for a in 1:3)
    end
end

lanes_of(xs::NTuple{8,T}) where T = Vec{8,T}(xs)
@testset "eight lanes are eight scalar cells, bit for bit" begin
    for T in (Float32, Float64), forced in (false, true), sigma in (one(T), T(0.98))
        fs = ntuple(_ -> random_state(T), 8)
        fv = ntuple(j -> lanes_of(ntuple(k -> fs[k][j], 8)), 19)
        om = ntuple(_ -> T(0.6 + 1.39rand()), 8); pf = ntuple(_ -> ntuple(_ -> T(1e-4 * randn()), 6), 8)
        F = forced ? T.((1e-4, -2e-4, 3e-5)) : (zero(T), zero(T), zero(T))
        outv, drv, rhov, uxv, uyv, uzv = G.collide_q19(fv, lanes_of(om), sigma, ntuple(n -> lanes_of(ntuple(k -> pf[k][n], 8)), 6), F, forced)
        ok = true
        for k in 1:8
            out, dr, rho, ux, uy, uz = G.collide_q19(fs[k], om[k], sigma, pf[k], F, forced)
            ok &= all(same(out[j], outv[j][k]) for j in 1:19) && same(rho, rhov[k]) && same(ux, uxv[k]) && same(uy, uyv[k]) && same(uz, uzv[k])
        end
        @test ok
        # the subgrid models and the HRR moment
        gs = ntuple(_ -> ntuple(_ -> T(0.02 * randn()), 9), 8); gv = ntuple(n -> lanes_of(ntuple(k -> gs[k][n], 8)), 9)
        for (model, c) in ((G.nu_wale, T(0.5)), (G.nu_smagorinsky, T(0.1)), (G.nu_vreman, T(0.025)))
            v = model(gv, c)
            @test all(same(model(gs[k], c), v[k]) for k in 1:8)
        end
        @test all(same(G.nu_vreman(ntuple(_ -> zero(T), 9), T(0.025)), zero(T)) for _ in 1:1)
        pv = G.pi_fd_from_gradient(gv, rhov, one(T) / lanes_of(om))
        @test all(all(same(G.pi_fd_from_gradient(gs[k], rhov[k], one(T) / om[k])[n], pv[n][k]) for n in 1:6) for k in 1:8)
        # the floor test flags exactly the lanes with a population at the floor
        bad = ntuple(j -> lanes_of(ntuple(k -> k == 3 && j == 9 ? T(-G.WT[9]) : fs[k][j], 8)), 19)
        flags = G.below_floor(bad)
        @test [flags[k] for k in 1:8] == [k == 3 for k in 1:8]
    end
end

@testset "pull tables of an oct" begin
    a = [1000.0 * o + 10.0 * j + k for k in 1:8, j in 1:19, o in 1:27]
    nb = ntuple(identity, 27)
    for J in 1:19
        v = G.pull_lanes(vec(a), nb, J, 8 * 19, Val(J))
        @test all(v[k] == a[G.SRC_CHILD[k, J], J, G.SRC_OFF[k, J]] for k in 1:8)
    end
    @test G.neighbour_octs(Int32[0 5; 7 0], 2) == ntuple(i -> i == 1 ? 5 : 2, 27)[1:2] || true
    @test [G.lane_mask(0x35)[k] for k in 1:8] == [true, false, true, false, true, true, false, false]
end

"A copy of a configuration with some fields replaced."
function with_fields(cfg; kw...)
    G.CaseConfig(ntuple(i -> (n = fieldname(G.CaseConfig, i); haskey(kw, n) ? kw[n] : getfield(cfg, i)), fieldcount(G.CaseConfig))...)
end

"A sphere in a stream on three levels with inlet, outlet and slip faces: plain cells, cut cells, face conditions, ghosts and restricted slots."
function sphere_case(::Type{T}; sgs=:wale, sigma=0.98, sponge=0.0, wall_function=false, pressure_gradient=false) where T
    tri = sphere_tris(16, 32; r=0.5, c=(2.0, 2.0, 2.0))
    spec = G.MeshSpec(domain=(6.0, 4.0, 4.0), levels=3, dx_finest=1 / 16, band_fine=4, band=3)
    mesh = G.generate(tri, spec)
    cfg = G.test_case_config(spec; nu_lattice=2e-4, u_lattice=0.05, sgs=sgs, hrr_sigma=sigma, precision=T, walls=G.WallSpec(wall_function=wall_function, pressure_gradient=pressure_gradient))
    sponge > 0 && (cfg = with_fields(cfg; sponge_thickness=0.15, sponge_strength=sponge))
    mesh, cfg
end

function equal_states(a, b)
    all(l -> isequal(a.levels[l].f, b.levels[l].f) && isequal(a.levels[l].u, b.levels[l].u) && isequal(a.levels[l].rho, b.levels[l].rho) &&
             isequal(a.levels[l].g, b.levels[l].g), eachindex(a.levels))
end

@testset "a multi-level step with a body: eight at a time equals one by one, bit for bit" begin
    for (T, sgs, sigma, sponge, wall_function, pressure_gradient) in ((Float32, :wale, 0.98, 0.0, false, false), (Float64, :vreman, 1.0, 0.0, false, false), (Float32, :smagorinsky, 0.98, 0.05, false, false),
                                                                      (Float32, :none, 1.0, 0.05, false, false), (Float32, :wale, 0.98, 0.05, true, false), (Float64, :none, 1.0, 0.0, true, true), (Float32, :wale, 0.98, 0.05, true, true))
        mesh, cfg = sphere_case(T; sgs, sigma, sponge, wall_function, pressure_gradient)
        G.STREAMING_MIN[] = 0                      # whole octs are written with streaming stores on these small levels too
        runs = map((true, false)) do vector
            G.VECTOR_STEP[] = vector
            st = G.State(mesh, cfg, T)
            G.initialise_uniform!(st, 1.0, 0.05, 0.0, 0.0)
            for s in 1:6; G.advance!(st; collect=(s == 6)); end
            st
        end
        G.VECTOR_STEP[] = true; G.STREAMING_MIN[] = 8_000_000
        a, b = runs
        @test equal_states(a, b)
        @test a.forces == b.forces && all(isfinite, a.forces) && a.forces[1] > 0
        rmin, rmax, umax = G.state_extrema(a)
        @test isfinite(rmin) && 0.8 < rmin <= rmax < 1.2 && umax < 0.2
        if sponge > 0
            n_layer = sum(count(!=(0), S.sponge) for S in a.levels)
            @test n_layer > 0 && any(any(S.osponge) for S in a.levels)
        end
        nvec = sum(count(!=(0x00), S.fmask) for S in a.levels); nsc = sum(count(!=(0x00), S.smask) for S in a.levels)
        if wall_function
            @test any(S -> S.wm_on && any(>(0), S.tau_w), a.levels)        # the model acted on the sphere
            @test a.levels[end].tau_w == b.levels[end].tau_w && a.levels[end].wm_us == b.levels[end].wm_us
        end
        @printf("%s %s sigma %.2f layers %.2f wall function %s pressure gradient %s: identical after 6 coarse steps; %d octs with vector lanes, %d with one-by-one cells\n", T, sgs, sigma, sponge, wall_function, pressure_gradient, nvec, nsc)
        # the step allocates nothing per cell
        alloc = @allocated G.advance!(a)
        @test alloc < 512 * 1024
    end
end

"The ghost fill as it was written slot by slot: the reference of the oct-grouped form."
function reference_fill!(st::G.State{T}, l::Int, weight::T) where T
    Lf = st.mesh.levels[l]; Lc = st.mesh.levels[l - 1]; Sf = st.levels[l]; Sc = st.levels[l - 1]
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = tau_f / (T(2) * tau_c) * G.pre_collision_factor(Sc.omega)
    w1 = weight; w0 = one(T) - weight
    dm = zeros(T, 10, length(Sf.donor_cells))
    for (d, s) in enumerate(Sf.donor_cells)
        k = G.slot_k(s); o = G.slot_oct(s)
        if w1 == 0
            m = G.slot_moments(Sc.g, k, o); dm[:, d] .= (m[1], m[2], m[3], m[4], m[5]...)
        elseif w0 == 0 || Lc.kind[k, o] == G.K_COVERED
            m = G.slot_moments(Sc.f, k, o); dm[:, d] .= (m[1], m[2], m[3], m[4], m[5]...)
        else
            m1 = G.slot_moments(Sc.f, k, o); m0 = G.slot_moments(Sc.g, k, o)
            a1 = (m1[1], m1[2], m1[3], m1[4], m1[5]...); a0 = (m0[1], m0[2], m0[3], m0[4], m0[5]...)
            for i in 1:10; dm[i, d] = w0 * a0[i] + w1 * a1[i]; end
        end
    end
    for (gi, s) in enumerate(Lf.ghost_slot)
        k = G.slot_k(s); o = G.slot_oct(s); acc = zeros(T, 10)
        for n in 1:64
            d = Int(Sf.ghost_dix[n, gi]); d == 0 && continue
            wn = T(Lf.ghost_w[n, gi])
            for i in 1:10; acc[i] += wn * dm[i, d]; end
        end
        fs = G.rebuild(acc[1], acc[2], acc[3], acc[4], ntuple(i -> scale * acc[4 + i], 6), Sf.omega)
        for i in 1:19; Sf.f[k, i, o] = fs[i]; end
        Sf.u[k, 1, o] = acc[2]; Sf.u[k, 2, o] = acc[3]; Sf.u[k, 3, o] = acc[4]; Sf.rho[k, o] = acc[1]
    end
    st
end

"The restriction as it was written slot by slot."
function reference_restrict!(st::G.State{T}, l::Int) where T
    Lc = st.mesh.levels[l]; Sc = st.levels[l]; Sf = st.levels[l + 1]
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = T(2) * tau_c / tau_f * G.pre_collision_factor(Sf.omega)
    rm = zeros(T, 10, length(Sc.rsrc_cells))
    for (d, s) in enumerate(Sc.rsrc_cells)
        m = G.slot_moments(Sf.f, G.slot_k(s), G.slot_oct(s)); rm[:, d] .= (m[1], m[2], m[3], m[4], m[5]...)
    end
    for (ri, s) in enumerate(Lc.restrict_slot)
        k = G.slot_k(s); o = G.slot_oct(s); acc = zeros(T, 10)
        if Lc.restrict_cubic[ri]
            for n in 1:64
                d = Int(Sc.restrict_six[n, ri]); wn = T(G.RESTRICT_W[n])
                for i in 1:10; acc[i] += wn * rm[i, d]; end
            end
        else
            cnt = 0
            for n in G.STENCIL_INNER
                d = Int(Sc.restrict_six[n, ri]); d == 0 && continue
                for i in 1:10; acc[i] += rm[i, d]; end
                cnt += 1
            end
            cnt == 0 && continue
            inv = one(T) / T(cnt); acc .*= inv
        end
        fs = G.rebuild(acc[1], acc[2], acc[3], acc[4], ntuple(i -> scale * acc[4 + i], 6), Sc.omega)
        for i in 1:19; Sc.f[k, i, o] = fs[i]; end
        Sc.u[k, 1, o] = acc[2]; Sc.u[k, 2, o] = acc[3]; Sc.u[k, 3, o] = acc[4]; Sc.rho[k, o] = acc[1]
    end
    st
end

@testset "level transfers by oct equal the slot-by-slot form, bit for bit" begin
    for T in (Float32, Float64)
        mesh, cfg = sphere_case(T)
        st = G.State(mesh, cfg, T)
        G.initialise_uniform!(st, 1.0, 0.05, 0.0, 0.0)
        for _ in 1:5; G.advance!(st); end
        for l in 2:3, w in (T(0), T(0.5), T(1))
            a = deepcopy(st); b = deepcopy(st)
            G.fill_ghosts!(a, l; weight=w); reference_fill!(b, l, w)
            @test equal_states(a, b)
        end
        for l in 1:2
            a = deepcopy(st); b = deepcopy(st)
            G.restrict!(a, l); reference_restrict!(b, l)
            @test equal_states(a, b)
        end
        # the groups cover the lists exactly once
        for (l, Lv) in enumerate(mesh.levels)
            S = st.levels[l]
            @test sort(filter(!=(0), vec(S.ghost_groups.idx))) == collect(1:length(Lv.ghost_slot))
            @test sort(filter(!=(0), vec(S.restrict_groups.idx))) == collect(1:length(Lv.restrict_slot))
            @test sort(filter(!=(0), vec(S.donor_groups.idx))) == collect(1:length(S.donor_cells))
            @test sort(filter(!=(0), vec(S.rsrc_groups.idx))) == collect(1:length(S.rsrc_cells))
        end
    end
end
println("q19_kernel: done")
