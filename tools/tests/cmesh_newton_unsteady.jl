# Test of BDF dual time stepping on the Newton-Krylov airfoil march (src/compressible/cmesh_unsteady_newton_ka.jl; the
# `dual` keyword of newton_flow_step!, newton_rans_step! and the `bdf` keyword of newton_sst_implicit_lines!; 18 Nov 2026),
# on the small airfoil of tools/tests/cmesh_steady_newton.jl (48 surface cells, Mach 0.3, Re 1e5, default flux, SST):
#   1. a physical step of 1e30 reproduces the steady coupled step (to round-off: the BDF terms vanish);
#   2. second order in time: the converged 4-deg field rotated by 2 deg is marched at 6 deg; the RMS errors of lift and drag
#      at t = 0.1, ..., 1.0 against dt = 0.0125 fall by about five from dt = 0.05 to 0.025;
#   3. restart exactness: 3 + 3 steps with a save and a load into a fresh solver in between equal 6 steps bit for bit.
#   julia --startup-file=no --project=src tools/tests/cmesh_newton_unsteady.jl
using Test, KernelAbstractions, StaticArrays, LinearAlgebra, TOML, SHA, Dates, Serialization
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

function small_nodes()
    xs = [0.5 * (1 - cospi(k / 24)) for k in 0:24]
    yt(x) = x == 1.0 ? 0.0 : 5 * 0.12 * (0.2969 * sqrt(x) - 0.126x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    foil = CMeshGen.AirfoilPts(vcat(reverse(xs), xs[2:end]), vcat(-reverse(yt.(xs)), yt.(xs[2:end])))
    mesh = CMeshGen.generate_cmesh(foil; n_foil=48, n_wake=12, n_normal=20, dn1=0.003, growth_max=1.8, R_ff=20.0, L_wake=20.0,
                                   smooth_iters=60, bl_freeze=5)
    (X=mesh.X, Y=mesh.Y, NIW=mesh.NIW, NIF=mesh.NIF, NI=mesh.NI, NJ=mesh.NJ)
end
const ND = small_nodes()
function fresh(alpha=4.0)
    T = Float64
    met = CMeshMetrics.build_metrics(ND.X, ND.Y, ND.NIW, ND.NIF; trip_upper=0.0, trip_lower=0.0, scalar_type=T)
    cx, cy, cz, w, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(KernelAbstractions.CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=w, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, ND.X, ND.Y; limiter=:venkat, smooth_coefficient=0.05f0); M.newton_unlimited_reconstruction!(s)
    M.set_freestream_ka!(s, 0.3, alpha); s.nu_lam = hypot(s.uxi, s.uyi) / 1e5
    M.set_sst_ka!(s); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0); M.sst_step_ka!(s; corrected_wall_metric=true)
    s
end
const KW = (blocks=:roe, delta=0.3, lintol=0.05, maxit=60)
statevec(s) = vcat(vec(Array(s.rho)), vec(Array(s.ux)), vec(Array(s.uy)), vec(Array(s.T)), vec(Array(s.tk)), vec(Array(s.tom)))

# converged steady state at 4 deg (as tools/tests/cmesh_newton_res_tol.jl, 80 steps)
function steady4()
    s = fresh(4.0); m = M.newton_rans_init(s; cfl=5.0, agg=(4, 4))
    M.newton_rans_march!(s, m; nsteps=80, cflmax=1e3, growth=1.5, every=10, quiet=true, rre=(3, 8), load_tol=1e-9, window=10, KW...)
    s
end
const S4 = statevec(steady4())
# The converged 4-deg state in a solver whose freestream is at `alpha` (set by `fresh` first: set_freestream_ka! also refills
# the field with uniform flow, so it must not be called after the copy; until 18 Nov 2026 it was, and every case of this
# file started impulsively from uniform flow).
function from_steady(alpha)
    s = fresh(alpha); v = S4; n = length(s.rho)
    for (k, f) in enumerate((s.rho, s.ux, s.uy, s.T, s.tk, s.tom)); copyto!(f, reshape(v[(k-1)*n+1:k*n], size(f))); end
    M.applyBC_ka!(s, Int32(2)); M._newton_sst_bc!(s)
    s
end

@testset "dual time with an infinite step is the steady step" begin
    a = from_steady(6.0); ma = M.newton_rans_init(a; cfl=50.0, agg=(4, 4))
    b = from_steady(6.0); mb = M.newton_rans_init(b; cfl=50.0, agg=(4, 4))
    tm = M.newton_time_init(b; dt=1e30)
    M.newton_rans_step!(a, ma; KW...)
    M.newton_rans_step!(b, mb; dual=tm, KW...)
    va, vb = statevec(a), statevec(b)
    @test maximum(abs.(va .- vb) ./ max.(abs.(va), 1e-12)) < 1e-9
end

# A smooth transient on the body: the converged 4-deg field rotated by 2 deg (a smooth field at the new incidence carrying
# the circulation of 4 deg), marched at 6 deg. A change of the far-field incidence alone reaches the body only after
# acoustic times (the lift moved by 1e-3 over t <= 2, measured 18 Nov).
function rotated(alpha)
    s = from_steady(alpha); d = alpha - 4.0; cs, sn = cosd(d), sind(d)
    u = copy(s.ux); v = copy(s.uy); @. s.ux = cs * u - sn * v; @. s.uy = sn * u + cs * v
    M.applyBC_ka!(s, Int32(2)); M._newton_sst_bc!(s)
    s
end
# Lift and drag at t = 0.1, 0.2, ..., 1.0, every step's inner iterations converged tightly.
function history(dt)
    s = rotated(6.0); m = M.newton_rans_init(s; cfl=50.0, agg=(4, 4)); tm = M.newton_time_init(s; dt)
    r = M.newton_time_march!(s, m, tm; nsteps=round(Int, 1.0 / dt), quiet=true, cfl=1e4, inner_max=40, inner_tol=1e-8,
                             inner_abs=1e-13, KW...)
    k = round(Int, 0.1 / dt)
    all(x -> x.ok, r.records), [(x.cl, x.cd) for x in r.records if x.step % k == 0]
end
rms(h, ref, f) = sqrt(sum(abs2, f.(h) .- f.(ref)) / length(h))

@testset "second order in time" begin
    # Against a reference at dt = 0.0125, the RMS errors at dt = 0.05 and 0.025 fall by (1/4 - 1/64)/(1/16 - 1/64) = 5 for
    # second order (measured 18 Nov 2026: 5.13 in lift, 6.19 in drag. Until a fix on 18 Nov this test started impulsively from
    # uniform flow, through a set_freestream_ka! call after the copy of the steady state, and read 4.4 and 4.9 over t <= 2.)
    okr, ref = history(0.0125); ok1, h1 = history(0.05); ok2, h2 = history(0.025)
    @test okr && ok1 && ok2
    rl = rms(h1, ref, first) / rms(h2, ref, first); rd = rms(h1, ref, last) / rms(h2, ref, last)
    println("error ratios dt 0.05 -> 0.025: lift ", rl, ", drag ", rd, " (5 for second order)")
    @test 3.5 < rl < 6.5
    @test 3.5 < rd < 6.5
    @test abs(first(ref[end]) - first(ref[1])) > 1e-2      # a real transient, not a steady state
end

@testset "restart exactness" begin
    path = joinpath(mktempdir(), "restart.jls")
    a = from_steady(6.0); ma = M.newton_rans_init(a; cfl=50.0, agg=(4, 4)); ta = M.newton_time_init(a; dt=0.1)
    M.newton_time_march!(a, ma, ta; nsteps=6, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    b = from_steady(6.0); mb = M.newton_rans_init(b; cfl=50.0, agg=(4, 4)); tb = M.newton_time_init(b; dt=0.1)
    M.newton_time_march!(b, mb, tb; nsteps=3, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, checkpoint=path, KW...)
    c = fresh(6.0); tc = M.newton_time_load!(path, c); mc = M.newton_rans_init(c; cfl=50.0, agg=(4, 4))
    M.newton_time_march!(c, mc, tc; nsteps=3, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    @test tc.step == 6 && tc.t ≈ ta.t
    @test statevec(c) == statevec(a)
end
