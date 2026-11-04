using Test, KernelAbstractions

const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

@testset "line-implicit tridiagonal operator" begin
    ni,nj=3,8
    ls=fill(2.0f0,ni,nj); ln=fill(3.0f0,ni,nj)
    d0=fill(4.0f0,ni,nj); rhs=reshape(Float32.(1:ni*nj),ni,nj)
    d=similar(rhs); bp=similar(rhs)
    M._k_implicit_line_solve!(CPU())(d,bp,rhs,ls,ln,d0,Int32(ni),Int32(nj);ndrange=ni)
    KernelAbstractions.synchronize(CPU())
    for j in 1:nj,i in 1:ni
        lhs=(d0[i,j]+ls[i,j]+ln[i,j])*d[i,j]-
            (j==1 ? 0.0f0 : ls[i,j]*d[i,j-1])-
            (j==nj ? 0.0f0 : ln[i,j]*d[i,j+1])
        @test isapprox(lhs,rhs[i,j];rtol=2e-6)
    end
end

function airfoil_implicit_smoke(backend)
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),
                             vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,
        dn1=.003,growth_max=1.8,R_ff=20.0,L_wake=20.0,
        smooth_iters=60,bl_freeze=5)
    met=CMeshMetrics.build_metrics(mesh)
    cx,cy,cz,w,_,_=build_d3q39_static()
    s=M.build_sol_ka(backend,met;drot=2.0f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    s.wall_flux=true
    M.set_freestream_ka!(s,.2,2.0)
    s.nu_lam=1.0f-4
    M.set_sst_ka!(s)
    work=M.build_implicit_line_workspace(s)
    result=M.step_implicit_line_ka!(s,work,.45f0)
    @test isfinite(result.merit_before)
    @test result.accepted
    @test result.merit_after < result.merit_before
    @test all(isfinite,s.rho) && minimum(s.rho)>0
    @test all(isfinite,s.T) && minimum(s.T)>0
    @test_throws ArgumentError M.step_implicit_line_ka!(s,work,0.0f0)
    M.sst_step_ka!(s)
    @test all(isfinite,s.tk) && all(isfinite,s.tom)
end

@testset "single-phase airfoil implicit flow CPU" begin
    airfoil_implicit_smoke(CPU())
end

if get(ENV,"OPENLUDWIG_TEST_CUDA", "0")=="1"
    using CUDA
    @test CUDA.functional()
    @testset "single-phase airfoil implicit flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend())
    end
end
