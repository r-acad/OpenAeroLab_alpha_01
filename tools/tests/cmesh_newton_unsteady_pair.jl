# Test of BDF dual time stepping on the two-block (C + strip) Newton march of a blunt trailing edge
# (src/compressible/cmesh_unsteady_newton_pair_ka.jl; the `dual` keyword of newton_pair_flow_step! and newton_pair_step!,
# the `bdf` keyword of newton_pair_sst_step!; 18 Nov 2026), on the small blunt section of tools/tests/cmesh_steady_newton_pair.jl
# (NACA 0012 cut at 95 % chord, Mach 0.3, Re 1e5, SST):
#   1. a physical step of 1e30 reproduces the steady coupled pair step (to round-off: the BDF terms vanish);
#   2. restart exactness: 3 + 3 steps with a save and a load into a fresh pair in between equal 6 steps bit for bit;
#   3. second order in time: the converged 2-deg field rotated by 2 deg is marched at 4 deg; the RMS
#      errors of lift and drag at t = 0.5, ..., 1.0 against dt = 0.00625 fall by about five from dt = 0.025 to 0.0125. The
#      blunt base answers the rotation within about 0.05 time units; dt = 0.05 does not resolve it (measured 18 Nov at
#      0.05 -> 0.025: 3.66 and 3.64 on t >= 0.5, as the single block with the same kind of start, 3.84 and 3.66).
#   julia --startup-file=no --project=src tools/tests/cmesh_newton_unsteady_pair.jl
using Test, KernelAbstractions, StaticArrays, LinearAlgebra, Serialization
const LBM_F = Float32
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
withenv("LUDWIG_KINETIC_FLUX" => nothing, "LUDWIG_KINETIC_SHEAR_FLOOR" => nothing, "LUDWIG_SST_VARIANT" => nothing,
        "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS" => nothing) do
    include(joinpath(ROOT, "src", "collision", "lattice.jl"))
    include(joinpath(ROOT, "src", "cmesh", "cmesh_gen.jl"))
    include(joinpath(ROOT, "src", "cmesh", "cmesh_metrics.jl"))
    include(joinpath(ROOT, "src", "compressible", "cmesh_solver_ka.jl"))
end
const M = CMeshSolverKA
# scratch files: the private tree's TEMP_DELETEABLE when it is known, else the system's temporary directory
scratch() = (d = get(ENV, "LUDWIG_PRIVATE_ROOT", ""); isdir(d) ? mktempdir(mkpath(joinpath(d, "TEMP_DELETEABLE"))) : mktempdir())

"NACA 0012 truncated at 95 % chord (a blunt base of 1.7 % chord), unit chord."
function foil(; cut=0.95, n=40)
    yt(x) = 5 * 0.12 * (0.2969 * sqrt(x) - 0.126x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    xs = [cut * 0.5 * (1 - cospi(k / n)) for k in 0:n]; y = [x == 1.0 ? 0.0 : yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs), xs[2:end]) ./ cut, vcat(-reverse(y), y[2:end]) ./ cut)
end
const MESH = CMeshGen.marched_airfoil_mesh(foil(); n_foil=96, n_wake=24, n_normal=32, dn1=1e-3, alpha_deg=2.0, far=20.0,
                                           wake_aspect=100.0).mesh
function block(nd, alpha)
    T = Float64
    met = CMeshMetrics.build_metrics(nd.X, nd.Y, nd.NIW, nd.NIF; trip_upper=0.0, trip_lower=0.0, scalar_type=T)
    cx, cy, cz, w, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=w, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, nd.X, nd.Y; limiter=:venkat_chord, smooth_coefficient=1f0)
    M.set_freestream_ka!(s, 0.3, alpha); s.nu_lam = hypot(s.uxi, s.uyi) / 1e5
    M.set_sst_ka!(s); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0)
    M.sst_step_ka!(s; corrected_wall_metric=true)
    s
end
function pair(alpha)
    m = MESH; cn = (X=m.X, Y=m.Y, NIW=m.NIW, NIF=m.NIF, NI=m.NI, NJ=m.NJ); hn = M.newton_pair_strip_nodes(m.strip)
    M.newton_pair_build(block(cn, alpha), block(hn, alpha), cn, hn)
end
const KW = (delta=0.3, lintol=0.05, maxit=60)
fields(s) = (s.rho, s.ux, s.uy, s.T, s.tk, s.tom, s.nut)
statevec(p) = vcat((vec(Array(f)) for s in (p.c, p.h) for f in fields(s))...)

# converged state at 2 deg (as tools/tests/cmesh_steady_newton_pair.jl, longer)
const S2 = let p = pair(2.0), m = M.newton_pair_init(p; cfl=5.0, agg=(4, 4))
    M.newton_pair_march!(p, m; nsteps=80, cflmax=1e3, growth=1.5, every=20, quiet=true, rre=(3, 8), load_tol=1e-9, window=10, KW...)
    [map(copy, fields(p.c)), map(copy, fields(p.h))]
end
# The converged 2-deg state in a pair whose freestream is at `alpha` (set first: set_freestream_ka! also refills the field
# with uniform flow, so it must not be called after the copy).
function from_steady(alpha)
    p = pair(alpha)
    for (s, st) in ((p.c, S2[1]), (p.h, S2[2])), (f, v) in zip(fields(s), st); copyto!(f, v); end
    M.newton_pair_bc!(p); M.newton_pair_sst_bc!(p)
    p
end

@testset "pair dual time with an infinite step is the steady pair step" begin
    a = from_steady(4.0); ma = M.newton_pair_init(a; cfl=50.0, agg=(4, 4))
    b = from_steady(4.0); mb = M.newton_pair_init(b; cfl=50.0, agg=(4, 4))
    tb = M.newton_pair_time_init(b; dt=1e30)
    ra = M.newton_pair_step!(a, ma; KW...)
    rb = M.newton_pair_step!(b, mb; dual=(tb.c, tb.h), KW...)
    @test ra.ok && rb.ok && haskey(rb, :l2t) && !haskey(ra, :l2t)
    va, vb = statevec(a), statevec(b)
    @test maximum(abs.(va .- vb) ./ max.(abs.(va), 1e-12)) < 1e-9
end

@testset "pair restart exactness" begin
    path = joinpath(scratch(), "restart_pair.jls"); kw = (quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    a = from_steady(4.0); ma = M.newton_pair_init(a; cfl=50.0, agg=(4, 4)); ta = M.newton_pair_time_init(a; dt=0.1)
    ra = M.newton_pair_time_march!(a, ma, ta; nsteps=6, kw...)
    b = from_steady(4.0); mb = M.newton_pair_init(b; cfl=50.0, agg=(4, 4)); tb = M.newton_pair_time_init(b; dt=0.1)
    M.newton_pair_time_march!(b, mb, tb; nsteps=3, checkpoint=path, kw...)
    c = pair(4.0); tc = M.newton_pair_time_load!(path, c); mc = M.newton_pair_init(c; cfl=50.0, agg=(4, 4))
    M.newton_pair_time_march!(c, mc, tc; nsteps=3, kw...)
    @test all(r -> r.ok, ra.records)
    @test tc.c.step == 6 && tc.h.step == 6 && tc.c.t ≈ ta.c.t
    @test statevec(c) == statevec(a)
    println("restart: inner iterations per step ", [r.inner for r in ra.records], ", drop ", [round(r.drop; sigdigits=2) for r in ra.records])
end

# A smooth transient on the body: the converged 2-deg field rotated by 2 deg (a smooth field at the new incidence carrying the
# circulation of 2 deg) is marched at 4 deg. (A change of the far-field incidence alone reaches the body only after acoustic
# times: over t <= 1 the lift moved by 1e-3, measured 18 Nov. The first version also called set_freestream_ka! after copying
# the steady state and so started impulsively from uniform flow: drag error ratio 15.8, lift 3.5.)
function rotated(alpha)
    p = from_steady(alpha); d = alpha - 2.0; cs, sn = cosd(d), sind(d)
    for s in (p.c, p.h)
        u = copy(s.ux); v = copy(s.uy); @. s.ux = cs * u - sn * v; @. s.uy = sn * u + cs * v
    end
    M.newton_pair_bc!(p); M.newton_pair_sst_bc!(p)
    p
end
function history(dt)
    p = rotated(4.0); m = M.newton_pair_init(p; cfl=50.0, agg=(4, 4)); tm = M.newton_pair_time_init(p; dt)
    r = M.newton_pair_time_march!(p, m, tm; nsteps=round(Int, 1.0 / dt), quiet=true, cfl=1e4, inner_max=40, inner_tol=1e-8,
                                  inner_abs=1e-13, KW...)
    k = round(Int, 0.1 / dt)
    all(x -> x.ok, r.records), [(x.cl, x.cd) for x in r.records if x.step % k == 0 && x.t >= 0.5 - 1e-9]
end
rms(h, ref, f) = sqrt(sum(abs2, f.(h) .- f.(ref)) / length(h))

@testset "pair second order in time" begin
    okr, ref = history(0.00625); ok1, h1 = history(0.025); ok2, h2 = history(0.0125)
    @test okr && ok1 && ok2
    rl = rms(h1, ref, first) / rms(h2, ref, first); rd = rms(h1, ref, last) / rms(h2, ref, last)
    println("pair error ratios dt 0.025 -> 0.0125: lift ", rl, ", drag ", rd, " (5 for second order); RMS errors lift ",
            rms(h1, ref, first), " -> ", rms(h2, ref, first), ", drag ", rms(h1, ref, last), " -> ", rms(h2, ref, last))
    @test 3.5 < rl < 6.5
    @test 3.5 < rd < 6.5
    @test abs(first(ref[end]) - first(ref[1])) > 1e-2      # a real transient over the compared window, not a steady state
end

@testset "pair inner CFL control (cfl_adapt, 19 Nov 2026)" begin
    # On this transient every inner update is full, so the control keeps the requested CFL and the march is the default one bit
    # for bit. (Its back-off on rejected updates is tested on the single block in tools/tests/cmesh_newton_cfl_adapt.jl.)
    kw = (quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    a = from_steady(4.0); ma = M.newton_pair_init(a; cfl=50.0, agg=(4, 4)); ta = M.newton_pair_time_init(a; dt=0.1)
    ra = M.newton_pair_time_march!(a, ma, ta; nsteps=3, kw...)
    b = from_steady(4.0); mb = M.newton_pair_init(b; cfl=50.0, agg=(4, 4)); tb = M.newton_pair_time_init(b; dt=0.1)
    rb = M.newton_pair_time_march!(b, mb, tb; nsteps=3, cfl_adapt=true, kw...)
    @test all(r -> r.cfl == 1e4, ra.records) && all(r -> r.cfl == 1e4, rb.records)
    @test statevec(b) == statevec(a)
end
