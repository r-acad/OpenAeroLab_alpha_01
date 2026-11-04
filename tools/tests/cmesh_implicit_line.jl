using Test, KernelAbstractions

const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

@testset "implicit physical-state reduction" begin
    rho=ones(Float32,2,2);ux=zeros(Float32,2,2)
    uy=zeros(Float32,2,2);T=ones(Float32,2,2)
    @test M._airfoil_physical_state(rho,ux,uy,T)
    rho[1]=1.0f-5
    @test !M._airfoil_physical_state(rho,ux,uy,T)
    rho[1]=1.0f0;ux[2]=NaN32
    @test !M._airfoil_physical_state(rho,ux,uy,T)
    ux[2]=0.0f0;T[3]=Inf32
    @test !M._airfoil_physical_state(rho,ux,uy,T)
end

@testset "matrix-free localized perturbation" begin
    v=ntuple(_->zeros(Float32,100,100),4)
    v[4][1,1]=1.0f0
    h=M._airfoil_fd_step(v,1.0,10_000,1.0,0.2,1.0,1.0f0)
    @test h<=1.0f-3
    @test h>0
end

function reference_flow_merit(s)
    vr=s.vol
    r=max(Float64(s.rho_inf),1e-12)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u)
    n=length(vr)
    a=sqrt(sum(abs2,s.dc_r ./vr)/n)/r
    b=sqrt(sum(abs2,s.dc_mx./vr)/n)/(r*u)
    c=sqrt(sum(abs2,s.dc_my./vr)/n)/(r*u)
    d=sqrt(sum(abs2,s.dc_e ./vr)/n)/(r*e)
    sqrt(a*a+b*b+c*c+d*d)
end

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

@testset "streamwise implicit factor" begin
    ni,nj=9,3
    lw=fill(2.0f0,ni,nj);le=fill(3.0f0,ni,nj)
    d0=fill(4.0f0,ni,nj);rhs=reshape(Float32.(1:ni*nj),ni,nj)
    x=similar(rhs);cp=zeros(Float64,ni,nj)
    M._k_implicit_i_solve!(CPU())(x,cp,rhs,lw,le,d0,Int32(ni),Int32(nj),
        Int32(ni),Int32(1);ndrange=nj)
    KernelAbstractions.synchronize(CPU())
    for j in 1:nj,i in 1:ni
        lhs=(d0[i,j]+lw[i,j]+le[i,j])*x[i,j]-
            (i==1 ? 0.0f0 : lw[i,j]*x[i-1,j])-
            (i==ni ? 0.0f0 : le[i,j]*x[i+1,j])
        @test isapprox(lhs,rhs[i,j];rtol=2e-6)
    end
    block=Int32(3)
    xb=similar(rhs);cpb=zeros(Float32,ni,nj)
    M._k_implicit_i_solve!(CPU())(xb,cpb,rhs,lw,le,d0,Int32(ni),Int32(nj),
        block,Int32(3);ndrange=3nj)
    KernelAbstractions.synchronize(CPU())
    for j in 1:nj,i in 1:ni
        first_i=3div(i-1,3)+1;last_i=first_i+2
        lhs=(d0[i,j]+lw[i,j]+le[i,j])*xb[i,j]-
            (i==first_i ? 0.0f0 : lw[i,j]*xb[i-1,j])-
            (i==last_i ? 0.0f0 : le[i,j]*xb[i+1,j])
        @test isapprox(lhs,rhs[i,j];rtol=2e-6)
    end
end

function airfoil_implicit_smoke(backend; jfnk=false,linear_solver=:native,
                                preconditioner=:line)
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
    work=jfnk ? M.build_jfnk_airfoil_workspace(s;m=6,preconditioner) :
                M.build_implicit_line_workspace(s)
    step=jfnk ? M.step_jfnk_airfoil_ka! : M.step_implicit_line_ka!
    kw=jfnk && linear_solver==:krylov ?
        M.build_krylov_airfoil_workspace(s,work) : nothing
    result=kw===nothing ? step(s,work,.45f0) :
        step(s,work,.45f0;krylov_workspace=kw)
    @test isfinite(result.merit_before)
    @test result.accepted
    @test result.merit_after < result.merit_before
    @test isapprox(M._implicit_merit(s),reference_flow_merit(s);rtol=2e-5)
    if kw!==nothing
        @test result.linear_iterations>0 &&
              isfinite(result.linear_relative) &&
              0.0<=result.linear_relative<1.0
    end
    @test all(isfinite,s.rho) && minimum(s.rho)>0
    @test all(isfinite,s.T) && minimum(s.T)>0
    if kw===nothing
        @test_throws ArgumentError step(s,work,0.0f0)
    else
        @test_throws ArgumentError step(s,work,0.0f0;krylov_workspace=kw)
    end
    M.sst_step_ka!(s)
    @test all(isfinite,s.tk) && all(isfinite,s.tom)
    if jfnk
        ctrl=M.build_jfnk_airfoil_controller(s;m=4,linear_solver,preconditioner,
            nonmonotone_window=3)
        r=M.step_jfnk_rans_ka!(s,ctrl;max_backtracks=16)
        @test r.accepted && ctrl.iterations==1
        @test r.pseudo_cfl_next>=r.pseudo_cfl_used
        @test r.physical_trials>=1 && r.best_trial_merit<=r.merit_after
        @test length(ctrl.merit_history)==1 &&
              ctrl.merit_history[1]==r.merit_before
        @test all(isfinite,s.tk) && all(isfinite,s.tom)
    end
end

@testset "single-phase airfoil implicit flow CPU" begin
    airfoil_implicit_smoke(CPU())
end

@testset "single-phase airfoil JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true)
end

@testset "single-phase airfoil Krylov GMRES flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,linear_solver=:krylov)
end

@testset "single-phase airfoil ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:adi)
end
@testset "single-phase airfoil block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:block_adi)
end

if get(ENV,"OPENLUDWIG_TEST_CUDA", "0")=="1"
    using CUDA
    @test CUDA.functional()
    @testset "implicit physical-state reduction CUDA" begin
        good=CUDA.fill(1.0f0,4,4)
        zero_velocity=CUDA.zeros(Float32,4,4)
        bad=CUDA.fill(NaN32,4,4)
        @test M._airfoil_physical_state(good,zero_velocity,zero_velocity,good)
        @test !M._airfoil_physical_state(good,bad,zero_velocity,good)
    end
    @testset "single-phase airfoil implicit flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend())
    end
    @testset "single-phase airfoil JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true)
    end
    @testset "single-phase airfoil Krylov GMRES flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,linear_solver=:krylov)
    end
    @testset "single-phase airfoil ADI JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:adi)
    end
    @testset "single-phase airfoil block ADI JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:block_adi)
    end
end
