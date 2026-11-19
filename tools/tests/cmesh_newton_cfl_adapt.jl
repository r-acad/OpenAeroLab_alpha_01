# Test of the opt-in inner CFL control of the dual time step (`cfl_adapt` of newton_time_step! and newton_pair_time_step!,
# src/compressible/cmesh_unsteady_newton_ka.jl; 19 Nov 2026):
#   1. the rule (newton_time_next_cfl): the steady march's growth and back-off, capped at the requested CFL, floored;
#   2. an impulsive Mach 2 start on a laminar cylinder (Re 300, 224 x 48 C mesh, flux lowmach_shear_tscaled): at a fixed
#      inner CFL of 1e4 every inner update of the third step is rejected and the state stays frozen while the clock
#      advances; with cfl_adapt the inner CFL backs off, the second step's inner iterations converge, and the state moves.
#   julia --startup-file=no --project=src tools/tests/cmesh_newton_cfl_adapt.jl
using Test, KernelAbstractions, StaticArrays, LinearAlgebra, TOML, SHA, Dates, Serialization
const LBM_F = Float32
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
withenv("LUDWIG_KINETIC_FLUX" => "lowmach_shear_tscaled", "LUDWIG_KINETIC_SHEAR_FLOOR" => "0.15", "LUDWIG_SST_VARIANT" => nothing,
        "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS" => nothing) do
    include(joinpath(ROOT, "src", "collision", "lattice.jl"))
    include(joinpath(ROOT, "src", "cmesh", "cmesh_gen.jl"))
    include(joinpath(ROOT, "src", "cmesh", "cmesh_metrics.jl"))
    include(joinpath(ROOT, "src", "compressible", "cmesh_solver_ka.jl"))
end
const M = CMeshSolverKA

@testset "inner CFL rule" begin
    full = (ok=true, alpha=1.0); lim = (ok=true, alpha=0.6); rej = (ok=false, alpha=0.1)
    @test M.newton_time_next_cfl(100.0, full, 1e4, 1.0, 1.5) == 150.0
    @test M.newton_time_next_cfl(9e3, full, 1e4, 1.0, 1.5) == 1e4            # capped at the requested CFL
    @test M.newton_time_next_cfl(1e4, full, 1e4, 1.0, 1.5) == 1e4            # full updates keep the requested CFL
    @test M.newton_time_next_cfl(100.0, lim, 1e4, 1.0, 1.5) ≈ 60.0
    @test M.newton_time_next_cfl(100.0, (ok=true, alpha=0.2), 1e4, 1.0, 1.5) ≈ 50.0
    @test M.newton_time_next_cfl(100.0, rej, 1e4, 1.0, 1.5) ≈ 25.0
    @test M.newton_time_next_cfl(2.0, rej, 1e4, 1.0, 1.5) == 1.0             # floor
end

function cylinder_nodes()
    th = range(0, 2pi; length=2049)
    circ = CMeshGen.AirfoilPts([0.5 + 0.5cos(t) for t in th], [-0.5sin(t) for t in th])
    m = CMeshGen.generate_cmesh(circ; n_foil=128, n_wake=48, n_normal=48, dn1=4e-3, growth_max=1.12, R_ff=30.0, L_wake=30.0,
                                le_ds=pi / 128, cluster_le=0.0, cluster_te=0.0, smooth_iters=400, bl_freeze=0)
    (X=m.X, Y=m.Y, NIW=m.NIW, NIF=m.NIF, NI=m.NI, NJ=m.NJ)
end
function cylinder(nd)
    T = Float64
    met = CMeshMetrics.build_metrics(nd.X, nd.Y, nd.NIW, nd.NIF; trip_upper=-10.0, trip_lower=-10.0, trip_side=:y_sign, scalar_type=T)
    cx, cy, cz, qw, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(KernelAbstractions.CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=qw, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, nd.X, nd.Y; limiter=:venkat_chord, smooth_coefficient=0.2f0)
    M.set_freestream_ka!(s, 2.0, 0.0); s.nu_lam = hypot(s.uxi, s.uyi) / 300
    M.set_air_sutherland_ka!(s, 300.0); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0)
    s
end
const KW = (blocks=:roe, delta=0.3, lintol=0.3, maxit=60)

@testset "impulsive Mach 2 start: fixed inner CFL freezes, cfl_adapt recovers" begin
    nd = cylinder_nodes()
    # fixed inner CFL (the default): by the third step every inner update is rejected
    s = cylinder(nd); U = hypot(s.uxi, s.uyi); tm = M.newton_time_init(s; dt=0.05 / U); m = M.newton_rans_init(s; cfl=50.0, agg=(4, 4))
    r = [M.newton_time_step!(s, m, tm; cfl=1e4, inner_max=8, inner_tol=1e-3, KW...) for _ in 1:3]
    println("fixed: ", [(x.cd, x.first, x.final, x.ok) for x in r])
    @test !r[3].ok && r[3].final == r[3].first && r[3].cd == r[2].cd       # frozen while the clock advances
    @test all(x -> x.cfl == 1e4, r)
    # cfl_adapt: the inner CFL backs off and the second step converges (the driver's settings: 20 inner iterations)
    s = cylinder(nd); tm = M.newton_time_init(s; dt=0.05 / U); m = M.newton_rans_init(s; cfl=50.0, agg=(4, 4))
    r = [M.newton_time_step!(s, m, tm; cfl=1e4, inner_max=20, inner_tol=1e-3, cfl_adapt=true, KW...) for _ in 1:2]
    println("cfl_adapt: ", [(x.cd, x.first, x.final, x.ok, x.cfl) for x in r])
    @test r[1].final < 0.5 * r[1].first
    @test r[2].ok && r[2].final < 1e-2 * r[2].first && r[2].cd != r[1].cd
    @test all(isfinite, (r[2].cl, r[2].cd))
end
