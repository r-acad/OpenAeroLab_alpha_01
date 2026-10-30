using Test, KernelAbstractions

const LBM_F = Float32
const F = Float32
include(joinpath(@__DIR__, "..", "collision", "lattice.jl"))
include(joinpath(@__DIR__, "cmesh_gen.jl"))
include(joinpath(@__DIR__, "cmesh_metrics.jl"))

include(joinpath(@__DIR__, "cmesh_solver.jl"))
include(joinpath(@__DIR__, "cmesh_solver_ka.jl"))

"""Manufactured no-slip tangential shear on a straight or circular foil wall."""
function couette_fixture(; curved::Bool=false)
    ni, nj, g = 5, 4, 3
    ds, dn, radius, shear = 0.2f0, 0.1f0, 10.0f0, 0.7f0
    xn = zeros(Float32, ni + 1, nj + 1)
    yn = similar(xn)
    for j in 1:nj+1, i in 1:ni+1
        theta = (Float32(i) - 3.5f0) * ds / radius
        r = radius + Float32(j - 1) * dn
        if curved
            xn[i,j] = r * sin(theta)
            yn[i,j] = r * cos(theta) - radius
        else
            xn[i,j] = (Float32(i) - 3.5f0) * ds
            yn[i,j] = Float32(j - 1) * dn
        end
    end
    xc = zeros(Float32, ni, nj)
    yc = similar(xc)
    sx = zeros(Float32, ni, nj + 1)
    sy = similar(sx)
    wall_distance = zeros(Float32, ni, nj)
    for j in 1:nj+1, i in 1:ni
        sx[i,j] = -(yn[i+1,j] - yn[i,j])
        sy[i,j] = xn[i+1,j] - xn[i,j]
    end
    for j in 1:nj, i in 1:ni
        xc[i,j] = 0.25f0 * (xn[i,j] + xn[i+1,j] + xn[i,j+1] + xn[i+1,j+1])
        yc[i,j] = 0.25f0 * (yn[i,j] + yn[i+1,j] + yn[i,j+1] + yn[i+1,j+1])
        # Distance to the local straight wall segment used by the mesh metrics.
        face_x = 0.5f0 * (xn[i,1] + xn[i+1,1])
        face_y = 0.5f0 * (yn[i,1] + yn[i+1,1])
        face_length = hypot(sx[i,1], sy[i,1])
        wall_distance[i,j] = ((xc[i,j] - face_x) * sx[i,1] +
                              (yc[i,j] - face_y) * sy[i,1]) / face_length
    end
    u = zeros(Float32, ni + 2g, nj + 2g)
    v = zeros(Float32, size(u))
    for j in 1:nj, i in 1:ni
        if curved
            theta = atan(xc[i,j], yc[i,j] + radius)
            r = hypot(xc[i,j], yc[i,j] + radius)
            speed = shear * (r - radius)
            u[i+g,j+g] = speed * cos(theta)
            v[i+g,j+g] = -speed * sin(theta)
        else
            u[i+g,j+g] = shear * yc[i,j]
        end
    end
    for i in 1:ni
        u[i+g,g] = -u[i+g,g+1]
        v[i+g,g] = -v[i+g,g+1]
    end
    (; ni, nj, g, shear, xc, yc, sx, sy, wall_distance, u, v)
end

@testset "wake-cut state and metric share the paired cell" begin
    f = couette_fixture()
    i, partner = 1, f.ni
    for mod in (CMeshSolver, CMeshSolverKA)
        _, _, xet, yet = mod._sst_index_metrics(
            f.xc, f.yc, f.sx, f.sy, f.wall_distance,
            i, 1, f.ni, f.nj, 2, 1, Int32(1))
        @test xet == 0.5f0 * (f.xc[i,2] - f.xc[partner,1])
        @test yet == 0.5f0 * (f.yc[i,2] - f.yc[partner,1])
    end
end

@testset "CPU KA SST and EARSM kernels accept the corrected metric" begin
    x = [0.5 * (1 - cospi(k / 32)) for k in 0:32]
    y = [5 * 0.12 * (0.2969 * sqrt(q) - 0.1260q - 0.3516q^2 +
                      0.2843q^3 - 0.1036q^4) for q in x]
    foil = CMeshGen.AirfoilPts(vcat(reverse(x), x[2:end]),
                              vcat(-reverse(y), y[2:end]))
    mesh = CMeshGen.generate_cmesh(foil; n_foil=64, n_wake=16,
        n_normal=24, dn1=0.002, growth_max=1.8, R_ff=25.0,
        L_wake=25.0, smooth_iters=80, bl_freeze=6)
    met = CMeshMetrics.build_metrics(mesh)
    for i in 1:met.NI
        xxi, yxi, xet, yet = CMeshSolverKA._sst_index_metrics(
            met.xc, met.yc, met.Sjx, met.Sjy, met.wdist,
            i, 1, met.NI, met.NJ, met.NIW, met.NIF, Int32(1))
        @test isfinite(xxi * yet - xet * yxi)
        @test xxi * yet - xet * yxi > 0.0f0
    end
    cx, cy, cz, wt, _, _ = build_d3q39_static()
    s = CMeshSolverKA.build_sol_ka(CPU(), met; drot=2.0f0,
        cxv=cx, cyv=cy, czv=cz, wv=wt)
    s.wall_flux = true
    CMeshSolverKA.set_freestream_ka!(s, 0.2, 2.0)
    s.nu_lam = 1.0f-4
    CMeshSolverKA.set_sst_ka!(s)
    CMeshSolverKA.applyBC_ka!(s, Int32(2))
    fill!(s.dtf, 1.0f-6)
    s.earsm = Int32(1)
    CMeshSolverKA.sst_step_ka!(s; corrected_wall_metric=true)
    @test all(isfinite, s.tk)
    @test all(isfinite, s.tom)
    @test all(isfinite, s.nut)
    @test all(isfinite, s.exx)
    @test all(isfinite, s.exy)
    @test all(isfinite, s.eyy)
end

function recovered_shear(mod, f, i=3, j=1; corrected_wall_metric::Bool=true)
    xxi, yxi, xet, yet = mod._sst_index_metrics(
        f.xc, f.yc, f.sx, f.sy, f.wall_distance,
        i, j, f.ni, f.nj, 0, f.ni, Int32(corrected_wall_metric))
    ia, ja = i + f.g, j + f.g
    duxi = 0.5f0 * (f.u[ia+1,ja] - f.u[ia-1,ja])
    duet = 0.5f0 * (f.u[ia,ja+1] - f.u[ia,ja-1])
    dvxi = 0.5f0 * (f.v[ia+1,ja] - f.v[ia-1,ja])
    dvet = 0.5f0 * (f.v[ia,ja+1] - f.v[ia,ja-1])
    jac = xxi * yet - xet * yxi
    dudx = (duxi * yet - duet * yxi) / jac
    dudy = (-duxi * xet + duet * xxi) / jac
    dvdx = (dvxi * yet - dvet * yxi) / jac
    dvdy = (-dvxi * xet + dvet * xxi) / jac
    # At the center column, the wall tangent is x and its outward normal is y.
    # Thus t · (grad(u) n) is d(u_x)/dy, even on the circular wall.
    (; normal_shear=dudy, dudx, dvdx, dvdy)
end

@testset "SST no-slip wall metric and manufactured Couette shear" begin
    for mod in (CMeshSolver, CMeshSolverKA)
        for curved in (false, true)
            f = couette_fixture(; curved)
            wall = recovered_shear(mod, f)
            interior = recovered_shear(mod, f, 3, 2)
            legacy_wall = recovered_shear(mod, f; corrected_wall_metric=false)
            legacy_interior = recovered_shear(mod, f, 3, 2;
                                              corrected_wall_metric=false)
            @test isapprox(wall.normal_shear, f.shear; atol=0.006f0)
            @test isapprox(interior.normal_shear, f.shear; atol=0.006f0)
            @test isapprox(legacy_wall.normal_shear, 2f0 * f.shear; atol=0.02f0)
            @test legacy_interior.normal_shear == interior.normal_shear
            @test abs(wall.dudx) < 0.006f0
            @test abs(wall.dvdy) < 0.006f0
        end
    end
end
