# Tests of the opt-in kinetic face-flux settings of the airfoil reconstruction (src/cmesh/cmesh_flux_core.jl):
# LUDWIG_KINETIC_FLUX = lowmach / lowmach_shear, the shear floor LUDWIG_KINETIC_SHEAR_FLOOR, and LUDWIG_SST_VARIANT.
#   1. Properties of the face states of `_lowmach_shear_face_states` (consistency, kept means, untouched density and
#      temperature, the scale factors of each velocity component, the floor-1 limit, rotation invariance), Float64 and
#      Float32.
#   2. The tangential momentum flux of the 39-velocity half-range flux on a pure shear jump across a face scales with
#      the factor applied to the jump (the mechanism the setting exists for).
#   2b. The reference fluxes of the comparison, LUDWIG_KINETIC_FLUX = roe / ausmup (`_fvflux_roe`, `_fvflux_ausmup`):
#      exact flux of equal states, conservation, the upwind side's flux on supersonic faces (for Roe, the Roe
#      property), a stationary contact, no tangential flux on a shear jump along a face, rotation invariance.
#   2c. The shear floor on them, LUDWIG_KINETIC_FLUX = roe_shear / roe_lowmach_shear / ausmup_shear: floor 0 is the plain
#      flux bit for bit; exact flux of equal states, conservation, stationary contact, rotation invariance; on a shear jump
#      along a face the tangential flux of the floor, matched to the 39-velocity flux under lowmach_shear (0.5 %), and the
#      lattice-to-EFM ratio (5/18) sqrt(3 pi) of that dissipation; the normal-jump scaling `_lowmach_normal_face_states`.
#   3. Fresh processes: the environment parsing at load (defaults, valid, invalid, clamped; LUDWIG_AUSM_MINF), and the
#      solver's face flux `_airfoil_thermal_face_flux` dispatching to exactly those states or fluxes (default: the plain
#      flux, unchanged).
#   4. A fresh process: the Newton march of tools/tests/cmesh_steady_newton.jl under lowmach_shear (floor 0.15) and the
#      TMR SST, to that test's criteria with 100 steps instead of 40 (the setting needs about three times the steps there).
#   julia --startup-file=no --project=src tools/tests/cmesh_kinetic_flux_variants.jl
using Test, StaticArrays
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const LATTICE = joinpath(ROOT, "src", "collision", "lattice.jl")
const CORE = joinpath(ROOT, "src", "cmesh", "cmesh_flux_core.jl")
const VARS = ("LUDWIG_KINETIC_FLUX", "LUDWIG_KINETIC_SHEAR_FLOOR", "LUDWIG_SST_VARIANT", "LUDWIG_AUSM_MINF",
              "LUDWIG_ROE_ENTROPY_FLOOR", "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS")

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

# The two non-kinetic reference fluxes of the flux comparison (LUDWIG_KINETIC_FLUX = roe / ausmup)
@testset "Roe and AUSM+-up reference fluxes ($T)" for T in (Float64, Float32)
    a = sound(T); d = T(DROT); tol = T == Float64 ? 1e-12 : 2e-5
    st(r, ux, uy, t) = (T(r), T(ux), T(uy), T(t))
    nx, ny, S = T(0.6), T(0.8), T(1.3)
    function euler(s, nx, ny, S)                     # the exact flux of one state
        cs2 = FC._gas_cs2(T); un = s[2]*nx + s[3]*ny; p = s[1]*cs2*s[4]
        H = (1.5 + 0.5DROT + 1)*cs2*s[4] + (s[2]^2 + s[3]^2)/2
        T.((s[1]*un*S, (s[1]*un*s[2] + p*nx)*S, (s[1]*un*s[3] + p*ny)*S, s[1]*un*H*S))
    end
    near(x, y) = all(isapprox.(x, y; rtol=tol, atol=tol))
    for (name, F) in (("roe", (l, r, nx, ny, S) -> FC._fvflux_roe(l..., r..., nx, ny, S, d)),
                      ("ausmup M 0.15", (l, r, nx, ny, S) -> FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0225)),
                      ("ausmup M 0", (l, r, nx, ny, S) -> FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0)))
        l = st(1.02, 0.11a, -0.04a, 1.01); r = st(0.97, 0.13a, 0.02a, 0.99)
        @test all(x -> x isa T, F(l, r, nx, ny, S))                                # no promotion
        @test near(F(l, l, nx, ny, S), euler(l, nx, ny, S))                        # equal states: exact flux
        @test near(F(l, r, nx, ny, S), .-F(r, l, -nx, -ny, S))                     # conservative: F(L,R,n) = -F(R,L,-n)
        # faces supersonic in one direction: the upwind side's flux (for Roe this is the Roe property FR - FL = A dU)
        sup(m) = (st(1.02, (m*nx - 0.1ny)*a, (m*ny + 0.1nx)*a, 1.01), st(0.95, (1.05m*nx + 0.2ny)*a, (1.05m*ny - 0.2nx)*a, 0.97))
        l2, r2 = sup(2.0); @test near(F(l2, r2, nx, ny, S), euler(l2, nx, ny, S))
        l3, r3 = sup(-2.0); @test near(F(l3, r3, nx, ny, S), euler(r3, nx, ny, S))
        # stationary contact (density and temperature jump at equal pressure, no velocity): no mass flux, pressure only
        c1 = st(1.2, 0, 0, 1/1.2); c2 = st(0.9, 0, 0, 1/0.9); f = F(c1, c2, nx, ny, S); p = FC._gas_cs2(T)*T(1)
        @test abs(f[1]) < tol && abs(f[4]) < tol && isapprox(f[2], p*nx*S; rtol=tol) && isapprox(f[3], p*ny*S; rtol=tol)
        # a shear jump across a face along the flow (no normal velocity): no tangential momentum flux
        U = T(0.15)*a; δ = T(1e-3)*U
        s1 = (T(1), -(U - δ)*ny, (U - δ)*nx, T(1)); s2 = (T(1), -(U + δ)*ny, (U + δ)*nx, T(1)); g = F(s1, s2, nx, ny, S)
        @test abs(-g[2]*ny + g[3]*nx) < tol*U
        # rotation invariance
        θ = T(1.1); cθ, sθ = cos(θ), sin(θ); rot(s) = (s[1], cθ*s[2] - sθ*s[3], sθ*s[2] + cθ*s[3], s[4])
        A = F(rot(l), rot(r), cθ*nx - sθ*ny, sθ*nx + cθ*ny, S); B = F(l, r, nx, ny, S)
        @test near(A, (B[1], cθ*B[2] - sθ*B[3], sθ*B[2] + cθ*B[3], B[4]))
    end
    # the half-range flux diffuses that shear jump; Roe and AUSM+-up do not (the comparison's point)
    U = T(0.15)*a; δ = T(1e-3)*U
    s1 = (T(1), -(U - δ)*ny, (U - δ)*nx, T(1)); s2 = (T(1), -(U + δ)*ny, (U + δ)*nx, T(1))
    g = FC._fvflux_efm(s1..., s2..., nx, ny, S, d)
    @test -g[2]*ny + g[3]*nx < -T(0.1)*δ*S
end

# The shear floor on the reference fluxes (LUDWIG_KINETIC_FLUX = roe_shear / roe_lowmach_shear / ausmup_shear, 17 Nov 2026)
@testset "shear floor on Roe and AUSM+-up ($T)" for T in (Float64, Float32)
    a = sound(T); d = T(DROT); tol = T == Float64 ? 1e-12 : 2e-5
    st(r, ux, uy, t) = (T(r), T(ux), T(uy), T(t))
    nx, ny, S = T(0.6), T(0.8), T(1.3)
    near(x, y) = all(isapprox.(x, y; rtol=tol, atol=tol))
    roe(l, r, cf) = FC._fvflux_roe(l..., r..., nx, ny, S, d, cf)
    aus(l, r, cf) = FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0225, cf)
    rls(l, r, cf) = (L = FC._lowmach_normal_face_states(l, r, d, nx, ny); FC._fvflux_roe(L[1]..., L[2]..., nx, ny, S, d, cf))
    l = st(1.02, 0.11a, -0.04a, 1.01); r = st(0.97, 0.13a, 0.02a, 0.99)
    # floor 0 is the unchanged flux, bit for bit
    @test roe(l, r, 0.0) === FC._fvflux_roe(l..., r..., nx, ny, S, d) && aus(l, r, 0.0) === FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0225)
    for (name, F) in (("roe_shear", roe), ("ausmup_shear", aus), ("roe_lowmach_shear", rls))
        @test all(x -> x isa T, F(l, r, 0.15))                                          # no promotion
        @test near(F(l, l, 0.15), F(l, l, 0.0))                                         # equal states: the exact flux
        flip(F) = (l, r, cf) -> (G = F == rls ? (L = FC._lowmach_normal_face_states(r, l, d, -nx, -ny);
                     FC._fvflux_roe(L[1]..., L[2]..., -nx, -ny, S, d, cf)) : F == roe ? FC._fvflux_roe(r..., l..., -nx, -ny, S, d, cf) :
                     FC._fvflux_ausmup(r..., l..., -nx, -ny, S, d, 0.0225, cf); G)
        @test near(F(l, r, 0.15), .-flip(F)(l, r, 0.15))                                # conservative
        c1 = st(1.2, 0, 0, 1/1.2); c2 = st(0.9, 0, 0, 1/0.9)                            # stationary contact: unchanged
        @test near(F(c1, c2, 0.15), F(c1, c2, 0.0))
        θ = T(1.1); cθ, sθ = cos(θ), sin(θ); rot(s) = (s[1], cθ*s[2] - sθ*s[3], sθ*s[2] + cθ*s[3], s[4])
        Fr = F == roe ? FC._fvflux_roe(rot(l)..., rot(r)..., cθ*nx - sθ*ny, sθ*nx + cθ*ny, S, d, 0.15) :
             F == aus ? FC._fvflux_ausmup(rot(l)..., rot(r)..., cθ*nx - sθ*ny, sθ*nx + cθ*ny, S, d, 0.0225, 0.15) :
             (L = FC._lowmach_normal_face_states(rot(l), rot(r), d, cθ*nx - sθ*ny, sθ*nx + cθ*ny);
              FC._fvflux_roe(L[1]..., L[2]..., cθ*nx - sθ*ny, sθ*nx + cθ*ny, S, d, 0.15))
        B = F(l, r, 0.15)
        @test near(Fr, (B[1], cθ*B[2] - sθ*B[3], sθ*B[2] + cθ*B[3], B[4]))              # rotation invariant
    end
    # a pure shear jump across a face along the flow at Mach 0.15: the floor gives -rho s dut / 2 of tangential flux,
    # s = (5/18) sqrt(6/gamma) cf |u|, and the same dissipation as lowmach_shear (39-velocity flux)
    U = T(0.15)*a; δ = T(1e-3)*U
    s1 = (T(1), -(U - δ)*ny, (U - δ)*nx, T(1)); s2 = (T(1), -(U + δ)*ny, (U + δ)*nx, T(1))
    tang(g) = -g[2]*ny + g[3]*nx
    γ = 1 + 1/(1.5 + 0.5DROT); sf = (5/18)*sqrt(6/γ)*0.15*Float64(U + δ)
    for F in (roe, aus, rls)
        @test abs(tang(F(s1, s2, 0.0))) < tol*U                                          # no floor: no shear flux
        @test isapprox(tang(F(s1, s2, 0.15)), -0.5*sf*2Float64(δ)*Float64(S); rtol=T == Float64 ? 1e-6 : 1e-3)
    end
    if T == Float64
        cx, cy, cz, w, _, _ = FC.build_d3q39_static(Float64)
        L, R = FC._lowmach_shear_face_states(s1, s2, d, nx, ny, 0.15)
        k = FC._fvflux_normal6(L..., L..., L..., R..., R..., R..., nx, ny, S, cx, cy, cz, w, DROT, Int32(0))
        @test isapprox(tang(roe(s1, s2, 0.15)), tang(k); rtol=0.005)                     # matched to lowmach_shear
        # the continuous half-range flux (EFM) on the full jump diffuses at sqrt(2/(pi gamma)) c: the lattice's 0.853 of it
        e = FC._fvflux_efm(s1..., s2..., nx, ny, S, d); k1 = FC._fvflux_normal6(s1..., s1..., s1..., s2..., s2..., s2..., nx, ny, S, cx, cy, cz, w, DROT, Int32(0))
        @test isapprox(tang(k1)/tang(e), (5/18)*sqrt(3π); rtol=1e-3)
    end
    # the entropy floor (roe_lowmach_shear_entropy): ce = 0 is the floored flux bit for bit; equal states exact; conservative;
    # on a density jump at equal pressure across a face with no normal velocity, the mass flux of the 39-velocity flux
    @test FC._fvflux_roe(l..., r..., nx, ny, S, d, 0.15, 0.0) === roe(l, r, 0.15)
    @test near(FC._fvflux_roe(l..., l..., nx, ny, S, d, 0.15, 1.0), roe(l, l, 0.0))
    @test near(FC._fvflux_roe(l..., r..., nx, ny, S, d, 0.15, 1.0), .-FC._fvflux_roe(r..., l..., -nx, -ny, S, d, 0.15, 1.0))
    e1 = (T(1.002), -U*ny, U*nx, T(1/1.002)); e2 = (T(0.998), -U*ny, U*nx, T(1/0.998))
    @test abs(roe(e1, e2, 0.15)[1]) < tol*U                                              # Roe: no density diffusion there
    if T == Float64
        cx, cy, cz, w, _, _ = FC.build_d3q39_static(Float64)
        L, R = FC._lowmach_shear_face_states(e1, e2, d, nx, ny, 0.15)
        k = FC._fvflux_normal6(L..., L..., L..., R..., R..., R..., nx, ny, S, cx, cy, cz, w, DROT, Int32(0))
        @test k[1] > 0 && isapprox(FC._fvflux_roe(e1..., e2..., nx, ny, S, d, 0.15, 1.0)[1], k[1]; rtol=0.02)
        @test isapprox(FC._fvflux_ausmup(e1..., e2..., nx, ny, S, d, 0.0225, 0.15, 1.0)[1], k[1]; rtol=0.02)   # AUSM+-up too
    end
    # AUSM+-up with the entropy floor: ce = 0 keeps ausmup_shear (to the sign of a zero); equal states exact; conservative
    @test near(FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0225, 0.15, 0.0), aus(l, r, 0.15))
    @test near(FC._fvflux_ausmup(l..., l..., nx, ny, S, d, 0.0225, 0.15, 1.0), aus(l, l, 0.0))
    @test near(FC._fvflux_ausmup(l..., r..., nx, ny, S, d, 0.0225, 0.15, 1.0), .-FC._fvflux_ausmup(r..., l..., -nx, -ny, S, d, 0.0225, 0.15, 1.0))
    @test abs(aus(e1, e2, 0.15)[1]) < tol*U                                              # plain AUSM+-up: none there
    # the normal-jump scaling keeps the tangential velocity and scales only the normal jump, by z
    L, R = FC._lowmach_normal_face_states(l, r, d, nx, ny)
    un(s) = s[2]*nx + s[3]*ny; ut(s) = -s[2]*ny + s[3]*nx
    z = min(1, sqrt(max((Float64(l[2])^2 + Float64(l[3])^2)/Float64(l[4]), (Float64(r[2])^2 + Float64(r[3])^2)/Float64(r[4]))/gcs2(Float64)))
    @test isapprox(ut(L), ut(l); atol=tol) && isapprox(ut(R), ut(r); atol=tol) && L[1] === l[1] && R[4] === r[4]
    @test isapprox((un(R) - un(L))/(un(r) - un(l)), z; rtol=T == Float64 ? 1e-9 : 1e-4)
    @test isapprox(un(L) + un(R), un(l) + un(r); atol=tol)
end

# The 39-velocity flux with its lattice scaled to the face temperature (LUDWIG_KINETIC_FLUX = lowmach_shear_tscaled)
@testset "temperature-scaled lattice flux" begin
    cx, cy, cz, w, _, _ = FC.build_d3q39_static(Float64); d = DROT; nx, ny, S = 0.6, 0.8, 1.3
    fixed(l, r) = FC._fvflux_normal6(l..., l..., l..., r..., r..., r..., nx, ny, S, cx, cy, cz, w, d, Int32(0))
    scaled(l, r) = FC._fvflux_lattice_tscaled(l..., r..., nx, ny, S, cx, cy, cz, w, d)
    efm(l, r) = FC._fvflux_efm(l..., r..., nx, ny, S, d)
    # equal states: the exact flux at any temperature (the fixed lattice only for 0.81 < T < 1.67)
    for T in (0.1, 0.3, 0.6, 2.0, 8.0, 50.0), M in (0.0, 0.8, 2.5)
        a = sound(Float64, T); s = (1.1, M*a*0.6, -M*a*0.3, T); e = efm(s, s); sc = max(abs(e[2]), abs(e[3]))
        @test maximum(abs.(scaled(s, s) .- e))/sc < 1e-13
    end
    @test maximum(abs.(fixed((1.0, 0.0, 0.0, 0.3), (1.0, 0.0, 0.0, 0.3)) .- efm((1.0, 0.0, 0.0, 0.3), (1.0, 0.0, 0.0, 0.3)))) > 0.1
    # face temperature 1: the fixed-lattice flux bit for bit
    l = (1.02, 0.11, -0.04, 1.01); r = (0.97, 0.13, 0.02, 0.99)
    @test scaled(l, r) === fixed(l, r)
    # conservative and rotation invariant, away from T = 1
    l = (1.02, 0.3, -0.1, 3.1); r = (0.95, 0.4, 0.05, 2.7)
    @test all(isapprox.(scaled(l, r), .-FC._fvflux_lattice_tscaled(r..., l..., -nx, -ny, S, cx, cy, cz, w, d); atol=1e-13))
    θ = 0.7; c, s_ = cos(θ), sin(θ); rot(q) = (q[1], c*q[2] - s_*q[3], s_*q[2] + c*q[3], q[4])
    A = FC._fvflux_lattice_tscaled(rot(l)..., rot(r)..., c*nx - s_*ny, s_*nx + c*ny, S, cx, cy, cz, w, d); B = scaled(l, r)
    @test all(isapprox.(A, (B[1], c*B[2] - s_*B[3], s_*B[2] + c*B[3], B[4]); atol=1e-13))
    # scaling all temperatures and velocities by (k, sqrt k): the flux scales as the Euler flux does (mass sqrt k,
    # momentum k, energy k^1.5); the fixed lattice does not
    k = 4.0; lk = (l[1], l[2]*sqrt(k), l[3]*sqrt(k), l[4]*k); rk = (r[1], r[2]*sqrt(k), r[3]*sqrt(k), r[4]*k)
    f1 = scaled(l, r); fk = scaled(lk, rk)
    @test all(isapprox.(fk, (f1[1]*sqrt(k), f1[2]*k, f1[3]*k, f1[4]*k^1.5); rtol=1e-12))
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
    @test parsed(LUDWIG_KINETIC_FLUX="roe")[1] == "6" && parsed(LUDWIG_KINETIC_FLUX="ausmup")[1] == "7"
    @test parsed(LUDWIG_KINETIC_FLUX="roe_shear")[1] == "8" && parsed(LUDWIG_KINETIC_FLUX="roe_lowmach_shear")[1] == "9" &&
          parsed(LUDWIG_KINETIC_FLUX="ausmup_shear")[1] == "10" && parsed(LUDWIG_KINETIC_FLUX="roe_lowmach_shear_entropy")[1] == "11" && parsed(LUDWIG_KINETIC_FLUX="ausmup_shear_entropy")[1] == "12" && parsed(LUDWIG_KINETIC_FLUX="lowmach_shear_tscaled")[1] == "13"
    ef(; env...) = run_julia(LOAD * "print(ROE_ENTROPY_FLOOR)"; env...)
    @test ef() == "1.0" && ef(LUDWIG_ROE_ENTROPY_FLOOR="0.25") == "0.25" && ef(LUDWIG_ROE_ENTROPY_FLOOR="x") == "1.0" &&
          ef(LUDWIG_ROE_ENTROPY_FLOOR="5") == "2.0"
    minf2(; env...) = run_julia(LOAD * "print(AUSM_MINF2)"; env...)
    @test minf2() == "0.0" && minf2(LUDWIG_AUSM_MINF="abc") == "0.0" && minf2(LUDWIG_AUSM_MINF="-1") == "0.0"
    @test parse(Float64, minf2(LUDWIG_AUSM_MINF="0.15")) == 0.15^2
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
v = M.KINETIC_FLUX_VARIANT
want = v == 5 ? plain(M._lowmach_shear_face_states(l, r, 2.0, nx, ny, M.KINETIC_SHEAR_FLOOR)...) :
       v == 6 ? M._fvflux_roe(l..., r..., nx, ny, S, 2.0) : v == 7 ? M._fvflux_ausmup(l..., r..., nx, ny, S, 2.0, M.AUSM_MINF2) :
       v == 8 ? M._fvflux_roe(l..., r..., nx, ny, S, 2.0, M.KINETIC_SHEAR_FLOOR) :
       v == 9 ? (L = M._lowmach_normal_face_states(l, r, 2.0, nx, ny); M._fvflux_roe(L[1]..., L[2]..., nx, ny, S, 2.0, M.KINETIC_SHEAR_FLOOR)) :
       v == 10 ? M._fvflux_ausmup(l..., r..., nx, ny, S, 2.0, M.AUSM_MINF2, M.KINETIC_SHEAR_FLOOR) :
       v == 11 ? (L = M._lowmach_normal_face_states(l, r, 2.0, nx, ny); M._fvflux_roe(L[1]..., L[2]..., nx, ny, S, 2.0, M.KINETIC_SHEAR_FLOOR, M.ROE_ENTROPY_FLOOR)) :
       v == 12 ? M._fvflux_ausmup(l..., r..., nx, ny, S, 2.0, M.AUSM_MINF2, M.KINETIC_SHEAR_FLOOR, M.ROE_ENTROPY_FLOOR) :
       v == 13 ? (L = M._lowmach_shear_face_states(l, r, 2.0, nx, ny, M.KINETIC_SHEAR_FLOOR); M._fvflux_lattice_tscaled(L[1]..., L[2]..., nx, ny, S, cx, cy, cz, w, 2.0)) : plain(l, r)
print(M.KINETIC_FLUX_VARIANT, ' ', got == want, ' ', got == plain(l, r))
"""

@testset "solver face flux dispatch" begin
    @test split(run_julia(SOLVER)) == ["0", "true", "true"]                                      # default: plain flux
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["5", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="roe")) == ["6", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="ausmup", LUDWIG_AUSM_MINF="0.3")) == ["7", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="roe_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["8", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="roe_lowmach_shear", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["9", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="ausmup_shear", LUDWIG_AUSM_MINF="0.3", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["10", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="roe_lowmach_shear_entropy", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["11", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="ausmup_shear_entropy", LUDWIG_AUSM_MINF="0.3", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["12", "true", "false"]
    @test split(run_julia(SOLVER; LUDWIG_KINETIC_FLUX="lowmach_shear_tscaled", LUDWIG_KINETIC_SHEAR_FLOOR="0.15")) == ["13", "true", "false"]
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
