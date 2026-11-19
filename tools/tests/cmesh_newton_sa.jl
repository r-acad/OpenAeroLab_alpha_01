# Test of the Spalart-Allmaras model of the Newton-Krylov march (src/compressible/cmesh_steady_newton_sa_ka.jl; 19 Nov 2026)
# on the small airfoil of tools/tests/cmesh_newton_unsteady.jl (48 surface cells, Mach 0.3, Re 1e5) and the small blunt
# pair of tools/tests/cmesh_newton_unsteady_pair.jl:
#   1. registration: free-stream values and boundary ghosts; refusals; unregistered solvers keep SST;
#   2. the steady SA march converges (SA equation residual, flow residuals, loads) and gives a turbulent boundary layer;
#   3. the SA source terms of the coefficient kernel against an independent evaluation in the TMR notation, at every cell;
#      every neighbour coefficient non-negative (an M-matrix with the diagonal);
#   4. BDF dual time with an infinite step is the steady SA step; restart exactness with SA;
#   5. the C+H pair: the steady SA march converges, and the pair's turbulence residual entry is SA's.
#   julia --startup-file=no --project=src tools/tests/cmesh_newton_sa.jl
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
scratch() = (d = get(ENV, "LUDWIG_PRIVATE_ROOT", ""); isdir(d) ? mktempdir(mkpath(joinpath(d, "TEMP_DELETEABLE"))) : mktempdir())

function small_nodes()
    xs = [0.5 * (1 - cospi(k / 24)) for k in 0:24]
    yt(x) = x == 1.0 ? 0.0 : 5 * 0.12 * (0.2969 * sqrt(x) - 0.126x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    foil = CMeshGen.AirfoilPts(vcat(reverse(xs), xs[2:end]), vcat(-reverse(yt.(xs)), yt.(xs[2:end])))
    mesh = CMeshGen.generate_cmesh(foil; n_foil=48, n_wake=12, n_normal=20, dn1=0.003, growth_max=1.8, R_ff=20.0, L_wake=20.0,
                                   smooth_iters=60, bl_freeze=5)
    (X=mesh.X, Y=mesh.Y, NIW=mesh.NIW, NIF=mesh.NIF, NI=mesh.NI, NJ=mesh.NJ)
end
const ND = small_nodes()
function base(alpha)
    T = Float64
    met = CMeshMetrics.build_metrics(ND.X, ND.Y, ND.NIW, ND.NIF; trip_upper=0.0, trip_lower=0.0, scalar_type=T)
    cx, cy, cz, w, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(KernelAbstractions.CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=w, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, ND.X, ND.Y; limiter=:venkat, smooth_coefficient=0.05f0); M.newton_unlimited_reconstruction!(s)
    M.set_freestream_ka!(s, 0.3, alpha); s.nu_lam = hypot(s.uxi, s.uyi) / 1e5
    M.set_sst_ka!(s); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0)
    s
end
fresh_sa(alpha) = M.set_newton_sa!(base(alpha))
fresh_sst(alpha) = (s = base(alpha); M.sst_step_ka!(s; corrected_wall_metric=true); s)
const KW = (blocks=:roe, delta=0.3, lintol=0.05, maxit=60)
fields(s) = (s.rho, s.ux, s.uy, s.T, s.tk, s.tom, s.nut)
statevec(s) = vcat((vec(Array(f)) for f in fields(s))...)
inner(s, a) = Array(view(a, s.g+1:s.g+s.NI, s.g+1:s.g+s.NJ))
fv1(chi) = chi^3 / (chi^3 + 7.1^3)

@testset "registration" begin
    s = fresh_sa(4.0); nu = Float64(s.nu_lam)
    @test M.newton_sa_on(s) && !M.newton_sa_on(fresh_sst(4.0))
    @test s.k_inf ≈ 3nu
    @test s.k_inf / s.w_inf ≈ 3nu * fv1(3.0)                     # far-field nu_t of the boundary kernels
    g = s.g; NJ = s.NJ; i = 2
    @test s.nut[i+g, NJ+g+1] ≈ 3nu * fv1(3.0) && s.tk[i+g, NJ+g+1] ≈ 3nu
    iw = s.NIW + 3                                               # a wall cell: ghost nt = -interior, ghost nu_t = 0
    @test s.tk[iw+g, g] == -s.tk[iw+g, g+1] && s.nut[iw+g, g] == 0
    t = base(4.0); t.rhok = Int32(1); @test_throws ArgumentError M.set_newton_sa!(t)
    t = base(4.0); t.sst = false; @test_throws ArgumentError M.set_newton_sa!(t)
    @test_throws ArgumentError M.set_newton_sa!(base(4.0); nut_ratio=0)
end

# converged steady SA state at 4 deg, and the SST state for comparison
function steady(s)
    m = M.newton_rans_init(s; cfl=5.0, agg=(4, 4))
    M.newton_rans_march!(s, m; nsteps=120, cflmax=1e3, growth=1.5, every=10, quiet=true, rre=(3, 8), load_tol=1e-9, window=10, KW...)
    s, m
end
const SA4, MSA4 = let s = fresh_sa(4.0), d0 = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    global SA_COLD = (collect(d0.flow_l2), d0.turb_k_equation_l2)
    steady(s)
end
const SST4 = steady(fresh_sst(4.0))[1]

@testset "steady SA march" begin
    s = SA4; d = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    flow_drop = maximum(collect(d.flow_l2) ./ SA_COLD[1]); sa_drop = d.turb_k_equation_l2 / SA_COLD[2]
    cl, cd = M.loads_ka(s); cls, cds = M.loads_ka(SST4)
    nu = Float64(s.nu_lam); nut = inner(s, s.nut)
    println("SA 4 deg: CL ", cl, " CD ", cd, " (SST ", cls, " ", cds, "); residual drops flow ", flow_drop, ", SA ", sa_drop,
            "; max nu_t/nu ", maximum(nut) / nu)
    @test flow_drop < 1e-4
    @test sa_drop < 1e-6
    @test d.turb_omega_equation_l2 == 0
    @test all(isfinite, nut) && minimum(nut) >= 0 && maximum(nut) / nu > 5     # a turbulent boundary layer and wake
    @test minimum(inner(s, s.tk)) >= M._SA_FLOOR
    # The two models on the same coarse mesh (Re 1e5, first cell y+ near 7) differ: measured 19 Nov CL 0.304 against 0.370,
    # CD 0.0522 against 0.0516. A loose check only; the accuracy test is the NACA 0012 against the TMR's SA results.
    @test abs(cl - cls) / cls < 0.25 && abs(cd - cds) / cds < 0.25
    @test all(inner(s, s.tom) .== s.w_inf)                                       # omega is not a field under SA
end

# The SA source terms in the TMR notation (incompressible form, standard model with ft2 and note 1(c)), from the cell
# values of the same state. Returns (production, destruction).
function sa_terms(nt, nu, d, Om)
    cb1 = 0.1355; cb2 = 0.622; σ = 2 / 3; κ = 0.41; cw1 = cb1 / κ^2 + (1 + cb2) / σ; cw2 = 0.3; cw3 = 2.0
    χ = nt / nu; f1 = fv1(χ); f2 = 1 - χ / (1 + χ * f1); ft2 = 1.2 * exp(-0.5χ^2)
    Sb = nt / (κ^2 * d^2) * f2
    St = Sb >= -0.7Om ? Om + Sb : Om + Om * (0.7^2 * Om + 0.9Sb) / ((0.9 - 2 * 0.7) * Om - Sb)
    r = min(nt / (St * κ^2 * d^2), 10.0); g = r + cw2 * (r^6 - r); fw = g * ((1 + cw3^6) / (g^6 + cw3^6))^(1 / 6)
    cb1 * (1 - ft2) * St * nt, (cw1 * fw - cb1 / κ^2 * ft2) * (nt / d)^2
end

@testset "SA source terms against an independent evaluation" begin
    s = SA4; st = M._NEWTON_SA[s]; c = st.coef; q = st.geo; g = s.g
    M._newton_sst_bc!(s); M._newton_sa_coeffs!(s, c, q)
    worst = 0.0; nneg = 0
    for j in 1:s.NJ, i in 1:s.NI
        ia, ja = i + g, j + g
        nu = M._air_nu_local(s.rho[ia, ja], s.T[ia, ja], s.nu_lam, s.rho_inf, s.suth, s.air_sutherland)
        @test c.sf2[i, j] == nu
        J = q.xxi[i, j] * q.yet[i, j] - q.xet[i, j] * q.yxi[i, j]
        grad(f) = (a = (f[ia+1, ja] - f[ia-1, ja]) / 2; b = (f[ia, ja+1] - f[ia, ja-1]) / 2;
                   ((a * q.yet[i, j] - b * q.yxi[i, j]) / J, (-a * q.xet[i, j] + b * q.xxi[i, j]) / J))
        (_, dudy), (dvdx, _) = grad(s.ux), grad(s.uy); (nx, ny) = grad(s.tk)
        n0 = s.tk[ia, ja]; d = s.wdist[i, j]
        P, D = sa_terms(n0, nu, d, abs(dudy - dvdx)); C = 0.622 / (2 / 3) * (nx^2 + ny^2)
        ref = P * s.gammat[i, j] - D + C
        got = c.sk[i, j] - c.dk[i, j] * n0              # the source and the Newton-form destruction at the current nt
        worst = max(worst, abs(got - ref) / (abs(P) + abs(D) + C + 1e-300))
        nneg += count(<(0), (c.akW[i, j], c.akE[i, j], c.akS[i, j], c.akN[i, j], c.dk[i, j], c.sk[i, j]))
    end
    println("SA source terms: largest relative difference ", worst)
    @test worst < 1e-10
    @test nneg == 0
end

@testset "cb2 split form (CFL3D/FUN3D)" begin
    s = M.set_newton_sa!(base(4.0); cb2=:split); @test M._NEWTON_SA[s].cb2 === :split
    @test_throws ArgumentError M.set_newton_sa!(base(4.0); cb2=:other)
    d0 = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    steady(s); d = M.steady_airfoil_diagnostics_ka(s; corrected_wall_metric=true)
    cl, cd = M.loads_ka(s)
    println("SA split 4 deg: CL ", cl, " CD ", cd, " (source form ", M.loads_ka(SA4), "); SA residual drop ",
            d.turb_k_equation_l2 / d0.turb_k_equation_l2)
    @test maximum(collect(d.flow_l2) ./ collect(d0.flow_l2)) < 1e-4 && d.turb_k_equation_l2 / d0.turb_k_equation_l2 < 1e-6
    st = M._NEWTON_SA[s]; M._newton_sst_bc!(s); M._newton_sa_coeffs!(s, st.coef, st.geo); c = st.coef
    @test all(>=(0), c.akW) && all(>=(0), c.akE) && all(>=(0), c.akS) && all(>=(0), c.akN) && all(>=(0), c.dk)
    @test abs(cl - M.loads_ka(SA4)[1]) / cl < 0.05                  # the same model: a discretisation difference only
end

function from_sa(alpha)
    s = fresh_sa(alpha)
    for (f, v) in zip(fields(s), fields(SA4)); copyto!(f, v); end
    M.applyBC_ka!(s, Int32(2)); M._newton_sst_bc!(s)
    s
end

@testset "SA dual time: infinite step and restart" begin
    a = from_sa(6.0); ma = M.newton_rans_init(a; cfl=50.0, agg=(4, 4))
    b = from_sa(6.0); mb = M.newton_rans_init(b; cfl=50.0, agg=(4, 4)); tm = M.newton_time_init(b; dt=1e30)
    M.newton_rans_step!(a, ma; KW...); M.newton_rans_step!(b, mb; dual=tm, KW...)
    va, vb = statevec(a), statevec(b)
    @test maximum(abs.(va .- vb) ./ max.(abs.(va), 1e-12)) < 1e-9
    path = joinpath(scratch(), "restart.jls")
    a = from_sa(6.0); ma = M.newton_rans_init(a; cfl=50.0, agg=(4, 4)); ta = M.newton_time_init(a; dt=0.1)
    M.newton_time_march!(a, ma, ta; nsteps=6, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    b = from_sa(6.0); mb = M.newton_rans_init(b; cfl=50.0, agg=(4, 4)); tb = M.newton_time_init(b; dt=0.1)
    M.newton_time_march!(b, mb, tb; nsteps=3, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, checkpoint=path, KW...)
    c = fresh_sa(6.0); tc = M.newton_time_load!(path, c); mc = M.newton_rans_init(c; cfl=50.0, agg=(4, 4))
    M.newton_time_march!(c, mc, tc; nsteps=3, quiet=true, cfl=1e4, inner_max=10, inner_tol=1e-6, KW...)
    @test tc.step == 6 && statevec(c) == statevec(a)
    @test abs(M.loads_ka(a)[1] - M.loads_ka(SA4)[1]) > 1e-3        # the incidence change moved the lift
end

# ---- the C+H pair --------------------------------------------------------------------------------------------------------
function blunt_foil(; cut=0.95, n=40)
    yt(x) = 5 * 0.12 * (0.2969 * sqrt(x) - 0.126x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    xs = [cut * 0.5 * (1 - cospi(k / n)) for k in 0:n]; y = [x == 1.0 ? 0.0 : yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs), xs[2:end]) ./ cut, vcat(-reverse(y), y[2:end]) ./ cut)
end
const PMESH = CMeshGen.marched_airfoil_mesh(blunt_foil(); n_foil=96, n_wake=24, n_normal=32, dn1=1e-3, alpha_deg=2.0, far=20.0,
                                            wake_aspect=100.0).mesh
function block(nd, alpha, sa)
    T = Float64
    met = CMeshMetrics.build_metrics(nd.X, nd.Y, nd.NIW, nd.NIF; trip_upper=0.0, trip_lower=0.0, scalar_type=T)
    cx, cy, cz, w, _, _ = build_d3q39_static(T)
    s = M.build_sol_ka(CPU(), met; drot=T(2), cxv=cx, cyv=cy, czv=cz, wv=w, scalar_type=T)
    s.wall_flux = true; s.fullvisc = Int32(1)
    M.set_airfoil_muscl_ka!(s, nd.X, nd.Y; limiter=:venkat_chord, smooth_coefficient=1f0)
    M.set_freestream_ka!(s, 0.3, alpha); s.nu_lam = hypot(s.uxi, s.uyi) / 1e5
    M.set_sst_ka!(s); M.applyBC_ka!(s, Int32(2)); M.prepare_dt_ka!(s, 0.45f0)
    sa ? M.set_newton_sa!(s) : M.sst_step_ka!(s; corrected_wall_metric=true)
    s
end
function pair(alpha, sa)
    m = PMESH; cn = (X=m.X, Y=m.Y, NIW=m.NIW, NIF=m.NIF, NI=m.NI, NJ=m.NJ); hn = M.newton_pair_strip_nodes(m.strip)
    M.newton_pair_build(block(cn, alpha, sa), block(hn, alpha, sa), cn, hn)
end

@testset "pair SA march" begin
    loads = Dict{Bool,Tuple{Float64,Float64}}()
    for sa in (true, false)
        p = pair(2.0, sa); m = M.newton_pair_init(p; cfl=5.0, agg=(4, 4))
        cold = M.newton_pair_six(p, m)
        M.newton_pair_march!(p, m; nsteps=80, cflmax=1e3, growth=1.5, every=20, quiet=true, rre=(3, 8), load_tol=1e-9, window=10,
                             delta=0.3, lintol=0.05, maxit=60)
        six = M.newton_pair_six(p, m); loads[sa] = M.newton_pair_loads(p)
        if sa
            println("pair SA: CL ", loads[sa][1], " CD ", loads[sa][2], "; residual drops ", six ./ cold)
            @test maximum(six[1:4] ./ cold[1:4]) < 1e-4
            @test six[5] / cold[5] < 1e-4 && six[6] == 0
            @test all(s -> minimum(inner(s, s.nut)) >= 0 && maximum(inner(s, s.nut)) / Float64(s.nu_lam) > 5, (p.c, p.h))
        end
    end
    println("pair SST: CL ", loads[false][1], " CD ", loads[false][2])
    @test abs(loads[true][1] - loads[false][1]) / loads[false][1] < 0.25
end
