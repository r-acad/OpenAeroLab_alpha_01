# Test of the opt-in shear floor per level of the Newton march's mesh sequencing (LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS,
# src/cmesh/cmesh_flux_core.jl; set per level by newton_cold_run and newton_pair_cold_run; 17 Nov 2026). Each case runs in
# a fresh Julia process, because the switches are constants read at load:
#   1. parsing: unset leaves the constant path; a list is read and set per level; a wrong level count, a value outside
#      [0, 1] or a non-number is refused;
#   2. the face flux of the scheduled variants uses the floor of the current level;
#   3. a schedule whose value equals LUDWIG_KINETIC_SHEAR_FLOOR gives the same Newton march bit for bit;
#   4. newton_cold_run sets each level's floor before building that level, and leaves the last level's value.
#   julia --startup-file=no --project=src tools/tests/cmesh_shear_floor_levels.jl
using Test
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const VARS = ("LUDWIG_KINETIC_FLUX", "LUDWIG_KINETIC_SHEAR_FLOOR", "LUDWIG_SST_VARIANT", "LUDWIG_AUSM_MINF",
              "LUDWIG_ROE_ENTROPY_FLOOR", "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS")
function run_julia(code; env...)
    e = Dict{String,String}(k => v for (k, v) in ENV if !(k in VARS))
    for (k, v) in env; e[String(k)] = v; end
    readchomp(setenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, e))
end
const LATTICE = joinpath(ROOT, "src", "collision", "lattice.jl")
const CORE = joinpath(ROOT, "src", "cmesh", "cmesh_flux_core.jl")
const LOAD = "const LBM_F=Float64; using StaticArrays; include(raw\"$LATTICE\"); "

@testset "parsing of LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS" begin
    q = LOAD * "include(raw\"$CORE\"); print(KINETIC_SHEAR_FLOOR_SCHEDULED, ' ', _kinetic_shear_floor() === KINETIC_SHEAR_FLOOR, ' ', " *
        "_set_level_shear_floor!(2, 3), ' ', _kinetic_shear_floor())"
    @test split(run_julia(q)) == ["false", "true", "0.3", "0.3"]                 # unset: the constant, setter a no-op
    @test split(run_julia(q; LUDWIG_KINETIC_SHEAR_FLOOR="0.15", LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1, 0.5,0.15")) ==
          ["true", "true", "0.5", "0.5"]           # starts at the constant; level 2 of 3 takes the second value
    w = LOAD * "include(raw\"$CORE\"); _set_level_shear_floor!(3, 3); a = _kinetic_shear_floor(); " *
        "b = try _set_level_shear_floor!(1, 2); \"no\" catch e; e isa ArgumentError ? \"refused\" : \"other\" end; print(a, ' ', b)"
    @test split(run_julia(w; LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1,0.5,0.15")) == ["0.15", "refused"]   # wrong level count
    bad = LOAD * "try include(raw\"$CORE\"); print(\"loaded\") catch e; x = e isa LoadError ? e.error : e; print(x isa ArgumentError) end"
    @test run_julia(bad; LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1,2") == "true"       # outside [0, 1]
    @test run_julia(bad; LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1,abc") == "true"     # not a number
    @test run_julia(bad; LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS=" ") == "loaded"       # blank: unset
end

const SOLVER = """
using KernelAbstractions, StaticArrays, LinearAlgebra, TOML, SHA, Dates, Serialization
const LBM_F = Float32
include(raw"$(joinpath(ROOT, "src", "collision", "lattice.jl"))")
include(raw"$(joinpath(ROOT, "src", "cmesh", "cmesh_gen.jl"))")
include(raw"$(joinpath(ROOT, "src", "cmesh", "cmesh_metrics.jl"))")
include(raw"$(joinpath(ROOT, "src", "compressible", "cmesh_solver_ka.jl"))")
const M = CMeshSolverKA
"""

const DISPATCH = SOLVER * """
cx, cy, cz, w, _, _ = build_d3q39_static(Float64)
l = (1.02, 0.11, -0.04, 1.01); r = (0.97, 0.13, 0.02, 0.99); nx, ny, S = 0.6, 0.8, 1.3
plain(l, r) = M._fvflux_normal6(l..., l..., l..., r..., r..., r..., nx, ny, S, cx, cy, cz, w, 2.0, Int32(0))
want(c) = M.KINETIC_FLUX_VARIANT == 5 ? plain(M._lowmach_shear_face_states(l, r, 2.0, nx, ny, c)...) :
    (L = M._lowmach_normal_face_states(l, r, 2.0, nx, ny); M._fvflux_roe(L[1]..., L[2]..., nx, ny, S, 2.0, c, M.ROE_ENTROPY_FLOOR))
got() = M._airfoil_thermal_face_flux(l, r, nx, ny, S, cx, cy, cz, w, 2.0)
M._set_level_shear_floor!(1, 2); a = got() == want(1.0)
M._set_level_shear_floor!(2, 2); b = got() == want(0.5); c = got() != want(0.15)
print(a, ' ', b, ' ', c)
"""

@testset "face flux uses the floor of the current level" begin
    for v in ("lowmach_shear", "roe_lowmach_shear_entropy")
        @test split(run_julia(DISPATCH; LUDWIG_KINETIC_FLUX=v, LUDWIG_KINETIC_SHEAR_FLOOR="0.15",
                              LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1,0.5")) == ["true", "true", "true"]
    end
end

# The small airfoil of tools/tests/cmesh_steady_newton.jl (48 surface cells, Mach 0.3, Re 1e5) under lowmach_shear with
# the floor 0.15 and the TMR form of SST.
const SMALL = SOLVER * """
function small_nodes()
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,dn1=.003,growth_max=1.8,R_ff=20.,L_wake=20.,smooth_iters=60,bl_freeze=5)
    (X=mesh.X,Y=mesh.Y,NIW=mesh.NIW,NIF=mesh.NIF,NI=mesh.NI,NJ=mesh.NJ)
end
seen = Float64[]
function make_solver(nd; T=Float64)
    push!(seen, M._kinetic_shear_floor())
    met=CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=0.,trip_lower=0.,scalar_type=T)
    cx,cy,cz,w,_,_=build_d3q39_static(T)
    s=M.build_sol_ka(KernelAbstractions.CPU(),met;drot=T(2),cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type=T)
    s.wall_flux=true;s.fullvisc=Int32(1)
    M.set_airfoil_muscl_ka!(s,nd.X,nd.Y;limiter=:venkat,smooth_coefficient=.05f0);M.newton_unlimited_reconstruction!(s)
    M.set_freestream_ka!(s,.3,4.);s.nu_lam=hypot(s.uxi,s.uyi)/1e5
    M.set_sst_ka!(s);M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0);M.sst_step_ka!(s;corrected_wall_metric=true)
    s
end
nd = small_nodes()
"""

const MARCH = SMALL * """
M._set_level_shear_floor!(1, 1)
s = make_solver(nd); m = M.newton_rans_init(s; cfl=5., agg=(4,4))
M.newton_rans_march!(s,m;nsteps=20,cflmax=1e3,growth=1.5,every=10,quiet=true,rre=(3,8),blocks=:roe,delta=.3,lintol=.3,maxit=60,load_tol=1e-9,window=10)
print(bytes2hex(sha256(reinterpret(UInt8, vcat(vec(Array(s.rho)), vec(Array(s.ux)), vec(Array(s.uy)), vec(Array(s.T)), vec(Array(s.tk)), vec(Array(s.tom)))))))
"""

@testset "a schedule equal to the constant floor: same march bit for bit" begin
    env = (LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15", LUDWIG_SST_VARIANT="tmr1994")
    h0 = run_julia(MARCH; env...)
    h1 = run_julia(MARCH; env..., LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="0.15")
    h2 = run_julia(MARCH; env..., LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="0.3")
    @test length(h0) == 64 && h1 == h0     # the Ref path computes the same numbers
    @test h2 != h0                         # and a different value changes them
end

const COLD = SMALL * """
r = M.newton_cold_run(nd, make_solver; levels=(2,1), nsteps=(6,6), cflmax=(1e3,1e3), load_tol=(1e-9,1e-9), window=(10,10),
                      lintol=.3, maxit=60, quiet=true)
print(join(seen, ','), ' ', M._kinetic_shear_floor(), ' ', isfinite(r.summary["final_CL"]))
"""

@testset "newton_cold_run sets each level's floor" begin
    env = (LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15", LUDWIG_SST_VARIANT="tmr1994")
    @test split(run_julia(COLD; env..., LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS="1,0.15")) == ["1.0,0.15", "0.15", "true"]
    @test split(run_julia(COLD; env...)) == ["0.15,0.15", "0.15", "true"]       # unset: the constant on both levels
end
