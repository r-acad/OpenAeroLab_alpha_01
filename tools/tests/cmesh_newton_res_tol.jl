# Test of the opt-in residual condition of the Newton march's stop test (`res_tol` of newton_rans_march! and
# newton_cold_run, src/compressible/cmesh_steady_newton_ka.jl; 17 Nov 2026), on the small airfoil of
# tools/tests/cmesh_steady_newton.jl (48 surface cells, Mach 0.3, Re 1e5, default flux):
#   1. res_tol = Inf is the default exactly: same steps, same state bit for bit;
#   2. with a loose loads tolerance the march stops on settled loads; with res_tol far below the reachable residual it
#      does not stop on them and runs to its step limit;
#   3. with res_tol above the residual reached when the loads settle, it stops at the same step as without it.
#   julia --startup-file=no --project=src tools/tests/cmesh_newton_res_tol.jl
using Test, KernelAbstractions, StaticArrays, LinearAlgebra, TOML, SHA, Dates, Serialization
const LBM_F = Float32
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
withenv("LUDWIG_KINETIC_FLUX" => nothing, "LUDWIG_KINETIC_SHEAR_FLOOR" => nothing, "LUDWIG_SST_VARIANT" => nothing) do
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
function fresh()
    T = Float64
    met = CMeshMetrics.build_metrics(ND.X, ND.Y, ND.NIW, ND.NIF; trip_upper=0.0, trip_lower=0.0, scalar_type=T)
    cx, cy, cz, w, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(KernelAbstractions.CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=w, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, ND.X, ND.Y; limiter=:venkat, smooth_coefficient=0.05f0); M.newton_unlimited_reconstruction!(s)
    M.set_freestream_ka!(s, 0.3, 4.0); s.nu_lam = hypot(s.uxi, s.uyi) / 1e5
    M.set_sst_ka!(s); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0); M.sst_step_ka!(s; corrected_wall_metric=true)
    d = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    s, [d.flow_l2..., d.turb_k_equation_l2, d.turb_omega_equation_l2]
end
function march(; kw...)
    s, ref = fresh(); m = M.newton_rans_init(s; cfl=5.0, agg=(4, 4))
    r = M.newton_rans_march!(s, m; nsteps=80, cflmax=1e3, growth=1.5, every=10, quiet=true, rre=(3, 8), blocks=:roe, delta=0.3,
                             lintol=0.3, maxit=60, load_tol=5e-2, window=10, ref, kw...)
    d = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    r, maximum(collect(d.flow_l2) ./ ref[1:4]), (Array(s.rho), Array(s.ux), Array(s.uy), Array(s.T))
end

@testset "res_tol of the Newton march stop test" begin
    r0, f0, st0 = march()
    ri, fi, sti = march(res_tol=Inf)
    @test r0.converged && r0.steps < 80                       # the loose loads test stops early
    @test ri.steps == r0.steps && all(sti .== st0)           # 1. res_tol = Inf: unchanged, bit for bit
    rs, fs, _ = march(res_tol=1e-14)
    @test !rs.converged && rs.steps == 80                    # 2. strict residual condition: no stop on loads alone
    @test isfinite(fs)                                       #    (on this 48-cell airfoil the residual cycles near 1e-3)
    rl, fl, stl = march(res_tol=10 * f0 + 1.0)
    @test rl.converged && rl.steps == r0.steps && all(stl .== st0)   # 3. loose residual condition: same stop
    println("steps: default ", r0.steps, " (residual ", f0, "), strict ", rs.steps, " (", fs, ")")
end
