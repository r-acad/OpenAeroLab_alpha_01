# Tests of the opt-in kinetic face-flux settings of the airfoil reconstruction (src/cmesh/cmesh_flux_core.jl):
# LUDWIG_KINETIC_FLUX = lowmach / lowmach_shear, the shear floor LUDWIG_KINETIC_SHEAR_FLOOR, and LUDWIG_SST_VARIANT.
#   1. Properties of the face states of `_lowmach_shear_face_states` (consistency, kept means, untouched density and
#      temperature, the scale factors of each velocity component, the floor-1 limit, rotation invariance), Float64 and
#      Float32.
#   2. The tangential momentum flux of the 39-velocity half-range flux on a pure shear jump across a face scales with
#      the factor applied to the jump (the mechanism the setting exists for).
#   3. Fresh processes: the environment parsing at load (defaults, valid, invalid, clamped), and the solver's face flux
#      `_airfoil_thermal_face_flux` dispatching to exactly those states (default: the plain flux, unchanged).
#   4. A fresh process: the Newton march of tools/tests/cmesh_steady_newton.jl under lowmach_shear (floor 0.15) and the
#      TMR SST, to that test's criteria with 100 steps instead of 40 (the setting needs about three times the steps there).
#   julia --startup-file=no --project=src tools/tests/cmesh_kinetic_flux_variants.jl
using Test, StaticArrays
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const LATTICE = joinpath(ROOT, "src", "collision", "lattice.jl")
const CORE = joinpath(ROOT, "src", "cmesh", "cmesh_flux_core.jl")
const VARS = ("LUDWIG_KINETIC_FLUX", "LUDWIG_KINETIC_SHEAR_FLOOR", "LUDWIG_SST_VARIANT")

module FC end
withenv((v => nothing for v in VARS)...) do
    Core.eval(FC, :(const LBM_F = Float64))
    Base.include(FC, LATTICE); Base.include(FC, CORE)
end

const DROT = 2.0                                   # diatomic gas, as the airfoil solver
gcs2(T) = (1 + 1 / (1.5 + 0.5DROT)) * FC._gas_cs2(T)   # gamma cs2
sound(T, Tgas=1) = sqrt(gcs2(T) * Tgas)

@testset "lowmach_shear face states ($T)" for T in (Float64, Float32)
    a = sound(T); tol = T == Float64 ? 1e-13 : 2e-6; d = T(DROT)
    st(r, ux, uy, t) = (T(r), T(ux), T(uy), T(t))
    nx, ny = T(0.6), T(0.8)
    l = st(1.02, 0.11a, -0.04a, 1.01); r = st(0.97, 0.13a, 0.02a, 0.99)
    for c in (0.0, 0.1, 0.15, 0.3, 1.0)
        L, R = FC._lowmach_shear_face_states(l, r, d, nx, ny, c)
        @test all(x -> x isa T, (L..., R...))                                     # no promotion
        @test L[1] === l[1] && R[1] === r[1] && L[4] === l[4] && R[4] === r[4]    # density, temperature untouched
        @test isapprox(L[2] + R[2], l[2] + r[2]; atol=tol) && isapprox(L[3] + R[3], l[3] + r[3]; atol=tol)  # means kept
        E, F = FC._lowmach_shear_face_states(l, l, d, nx, ny, c)
        @test all(isapprox.(E, l; atol=tol)) && all(isapprox.(F, l; atol=tol))  # equal states unchanged
    end
    # floor 1 is the isotropic low-Mach scaling exactly
    L1, R1 = FC._lowmach_shear_face_states(l, r, d, nx, ny, 1.0); L2, R2 = FC._lowmach_face_states(l, r, d)
    @test all(isapprox.(L1, L2; atol=tol)) && all(isapprox.(R1, R2; atol=tol))
    # the factors on the normal and tangential jumps, read back from the returned states, against the specification
    # z = min(1, max M), zt = min(z, sqrt(max Mn^2 + (c z)^2 + 1e-6)) evaluated in Float64 from the same states
    un(s) = s[2] * nx + s[3] * ny; ut(s) = -s[2] * ny + s[3] * nx
    function factors(l, r, c)
        L, R = FC._lowmach_shear_face_states(l, r, d, nx, ny, c)
        ((un(R) - un(L)) / (un(r) - un(l)), (ut(R) - ut(L)) / (ut(r) - ut(l)))
    end
    function spec(l, r, c)
        g = gcs2(Float64); m2 = max((Float64(l[2])^2 + Float64(l[3])^2) / g, (Float64(r[2])^2 + Float64(r[3])^2) / g)
        mn2 = max(Float64(un(l))^2 / g, Float64(un(r))^2 / g); z = min(1, sqrt(m2))
        (z, min(z, sqrt(mn2 + (c * z)^2 + 1e-6)))
    end
    vel(m_n, m_t, d_n, d_t) = (st(1, (m_n - d_n) * a * nx - (m_t - d_t) * a * ny, (m_n - d_n) * a * ny + (m_t - d_t) * a * nx, 1),
                               st(1, (m_n + d_n) * a * nx - (m_t + d_t) * a * ny, (m_n + d_n) * a * ny + (m_t + d_t) * a * nx, 1))
    rt = T == Float64 ? 1e-9 : 2e-3
    # flow along the face at Mach 0.15 (a tiny normal component): normal jump x z, tangential jump x about c z
    for c in (0.1, 0.15, 0.3)
        l, r = vel(0.0, 0.15, 0.001, 0.002)
        fn, ft = factors(l, r, c); z, zt = spec(l, r, c)
        @test isapprox(fn, z; rtol=rt) && isapprox(ft, zt; rtol=rt)
        @test isapprox(zt, c * z; rtol=0.03) && ft < fn                          # the floor binds
    end
    # face-normal Mach >= 1: both jumps unscaled
    l, r = vel(1.2, 0.3, 0.01, 0.01); fn, ft = factors(l, r, 0.15)
    @test isapprox(fn, 1; atol=tol) && isapprox(ft, 1; atol=tol)
    # supersonic flow along the face: the normal jump unscaled, the tangential one still scaled by about the floor
    l, r = vel(0.0, 1.5, 0.01, 0.01); fn, ft = factors(l, r, 0.15); z, zt = spec(l, r, 0.15)
    @test isapprox(fn, 1; atol=tol) && isapprox(ft, zt; rtol=rt) && zt < 0.16
    # flow normal to the face below Mach 1: the tangential factor is z (the min(z, .) branch), as the isotropic scaling
    l, r = vel(0.4, 0.0, 0.005, 0.005); fn, ft = factors(l, r, 0.15)
    @test isapprox(ft, fn; rtol=rt)
    # rotation invariance: rotating states and normal rotates the returned states
    l = st(1.02, 0.11a, -0.04a, 1.01); r = st(0.97, 0.13a, 0.02a, 0.99)
    for θ in (0.3, 1.9, -2.4)
        cθ, sθ = T(cos(θ)), T(sin(θ)); rot(s) = (s[1], cθ * s[2] - sθ * s[3], sθ * s[2] + cθ * s[3], s[4])
        A = FC._lowmach_shear_face_states(rot(l), rot(r), d, cθ * nx - sθ * ny, sθ * nx + cθ * ny, 0.15)
        B = rot.(FC._lowmach_shear_face_states(l, r, d, nx, ny, 0.15))
        @test all(isapprox.(A[1], B[1]; atol=tol)) && all(isapprox.(A[2], B[2]; atol=tol))
    end
end

@testset "shear dissipation of the lattice flux scales with the tangential factor" begin
    cx, cy, cz, w, _, _ = FC.build_d3q39_static(Float64)
    a = sound(Float64); U = 0.15a; δ = 1e-3U
    nx, ny = 0.0, 1.0                                # a wall-parallel face: x is along the face
    l = (1.0, U - δ, 0.0, 1.0); r = (1.0, U + δ, 0.0, 1.0)
    flux(l, r) = FC._fvflux_normal6(l..., l..., l..., r..., r..., r..., nx, ny, 1.0, cx, cy, cz, w, DROT, Int32(0))
    base = flux(l, r)[2]                             # x momentum flux = minus the tangential one: purely dissipative here
    @test base < 0                                   # momentum diffuses from the faster side to the slower one
    z = sqrt((U + δ)^2 / gcs2(Float64))
    for (name, states, factor) in (("lowmach", FC._lowmach_face_states(l, r, DROT), z),
                                   ("lowmach_shear c=0.3", FC._lowmach_shear_face_states(l, r, DROT, nx, ny, 0.3), sqrt((0.3z)^2 + 1e-6)),
                                   ("lowmach_shear c=0.15", FC._lowmach_shear_face_states(l, r, DROT, nx, ny, 0.15), sqrt((0.15z)^2 + 1e-6)))
        @test isapprox(flux(states...)[2] / base, factor; rtol=1e-3)
    end
    # equal states: every setting gives the same (exact) flux
    e = (1.0, U, 0.02a, 1.0)
    f0 = flux(e, e)
    @test all(isapprox.(flux(FC._lowmach_shear_face_states(e, e, DROT, nx, ny, 0.15)...), f0; atol=1e-14))
end

# ---- fresh processes: constants are read from the environment when the file is loaded ----
function run_julia(code; env...)
    e = Dict{String,String}(k => v for (k, v) in ENV if !(k in VARS))
    for (k, v) in env; e[String(k)] = v; end
    readchomp(setenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, e))
end
const LOAD = "const LBM_F=Float64; using StaticArrays; include(raw\"$LATTICE\"); include(raw\"$CORE\"); "
parsed(; env...) = split(run_julia(LOAD * "print(KINETIC_FLUX_VARIANT, ' ', KINETIC_SHEAR_FLOOR, ' ', SST_VARIANT)"; env...))

@testset "environment parsing at load" begin
    @test parsed() == ["0", "0.3", "0"]                                                         # defaults
    @test parsed(LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15") == ["5", "0.15", "0"]
    @test parsed(LUDWIG_KINETIC_FLUX="lowmach", LUDWIG_SST_VARIANT="tmr1994") == ["2", "0.3", "1"]
    @test parsed(LUDWIG_KINETIC_SHEAR_FLOOR="abc", LUDWIG_SST_VARIANT="other") == ["0", "0.3", "0"]  # invalid: defaults
    @test parsed(LUDWIG_KINETIC_SHEAR_FLOOR="2.0")[2] == "1.0"                                   # clamped to [0, 1]
    @test parsed(LUDWIG_KINETIC_FLUX="unknown")[1] == "0"
end

const SOLVER = """
using KernelAbstractions, StaticArrays, LinearAlgebra, TOML, SHA, Dates, Serialization
const LBM_F = Float32
include(raw"$(joinpath(ROOT, "src", "collision", "lattice.jl"))")
include(raw"$(joinpath(ROOT, "src", "cmesh", "cmesh_gen.jl"))")
include(raw"$(joinpath(ROOT, "src", "cmesh", "cmesh_metrics.jl"))")
include(raw"$(joinpath(ROOT, "src", "compressible", "cmesh_solver_ka.jl"))")
const M = CMeshSolverKA
cx, cy, cz, w, _, _ = build_d3q39_static(Float64)
l = (1.02, 0.11, -0.04, 1.01); r = (0.97, 0.13, 0.02, 0.99); nx, ny, S = 0.6, 0.8, 1.3
plain(l, r) = M._fvflux_normal6(l..., l..., l..., r..., r..., r..., nx, ny, S, cx, cy, cz, w, 2.0, Int32(0))
got = M._airfoil_thermal_face_flux(l, r, nx, ny, S, cx, cy, cz, w, 2.0)
want = M.KINETIC_FLUX_VARIANT == 5 ? plain(M._lowmach_shear_face_states(l, r, 2.0, nx, ny, M.KINETIC_SHEAR_FLOOR)...) : plain(l, r)
print(M.KINETIC_FLUX_VARIANT, ' ', got == want, ' ', got == plain(l, r))
"""

@testset "solver face flux dispatch" begin
    @test split(run_julia(SOLVER)) == ["0", "true", "true"]                                      # default: plain flux
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["5", "true", "false"]
end

# The Newton march of tools/tests/cmesh_steady_newton.jl (same small airfoil, solver and march settings) under the
# paper's settings, with the criteria of that test but 100 steps instead of 40: the low-Mach face states remove most of
# the flux's dissipation, and on this 48-cell airfoil the march needs about three times the steps (measured 16 Nov 2026:
# 40 steps leave the flow residuals at 1e-2 of cold, 80 steps at 4e-5, 120 steps at 2e-8; with the preconditioner blocks
# differentiating the variant's own first-order flux, no faster). Loads are not compared with the default flux: on
# this coarse mesh the default flux's dissipation lowers the lift (CL 0.35 against 0.46).
const MARCH = SOLVER[1:findfirst("const M = CMeshSolverKA", SOLVER)[end]] * """

function small_nodes()
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,dn1=.003,growth_max=1.8,R_ff=20.,L_wake=20.,smooth_iters=60,bl_freeze=5)
    (X=mesh.X,Y=mesh.Y,NIW=mesh.NIW,NIF=mesh.NIF,NI=mesh.NI,NJ=mesh.NJ)
end
nd=small_nodes();T=Float64
met=CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=0.,trip_lower=0.,scalar_type=T)
cx,cy,cz,w,_,_=build_d3q39_static(T)
s=M.build_sol_ka(KernelAbstractions.CPU(),met;drot=T(2),cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type=T)
s.wall_flux=true;s.fullvisc=Int32(1)
M.set_airfoil_muscl_ka!(s,nd.X,nd.Y;limiter=:venkat,smooth_coefficient=.05f0);M.newton_unlimited_reconstruction!(s)
M.set_freestream_ka!(s,.3,4.);s.nu_lam=hypot(s.uxi,s.uyi)/1e5
M.set_sst_ka!(s);M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0);M.sst_step_ka!(s;corrected_wall_metric=true)
six(s)=(d=M.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true);[d.flow_l2...,d.turb_k_equation_l2,d.turb_omega_equation_l2])
cold=six(s);m=M.newton_rans_init(s;cfl=5.,agg=(4,4))
M.newton_rans_march!(s,m;nsteps=100,cflmax=1e3,growth=1.5,every=10,quiet=true,rre=(3,8),blocks=:roe,delta=.3,lintol=.3,maxit=60,load_tol=1e-5,window=10)
warm=six(s);d=M.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true)
g=s.g;inner(a)=Array(a)[g+1:g+s.NI,g+1:g+s.NJ]
ok=(M.KINETIC_FLUX_VARIANT==5, M.SST_VARIANT==1, M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T), all(warm[1:4].<1e-3 .*cold[1:4]),
    all(isfinite,warm), abs(d.normalized_mass_imbalance)<1e-4, 0.2<d.cl<0.7&&0<d.cd<0.1, minimum(inner(s.tk))>0&&minimum(inner(s.tom))>0)
print(join(ok, ' '))
"""

@testset "Newton march on the small airfoil under the paper settings (100 steps)" begin
    r = split(run_julia(MARCH; LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15", LUDWIG_SST_VARIANT="tmr1994"))
    names = ("variant 5", "TMR SST", "physical state", "flow residuals 1e-3 of cold", "finite residuals", "mass imbalance",
             "loads in range", "positive k and omega")
    for (n, v) in zip(names, r)
        @test v == "true"
    end
    @test length(r) == length(names)
end
