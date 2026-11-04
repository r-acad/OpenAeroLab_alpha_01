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

@testset "wake-cut coupled line operator" begin
    ni,nj,niw,nif=6,4,2,2
    ls=fill(2.0f0,ni,nj);ln=fill(3.0f0,ni,nj)
    d0=fill(4.0f0,ni,nj);rhs=reshape(Float32.(1:ni*nj),ni,nj)
    x=similar(rhs);bp=similar(rhs)
    M._k_implicit_cut_line_solve!(CPU())(x,bp,rhs,ls,ln,d0,
        Int32(ni),Int32(nj),Int32(niw),Int32(nif);
        ndrange=niw+nif)
    KernelAbstractions.synchronize(CPU())
    for j in 1:nj,i in 1:ni
        south=j>1 ? x[i,j-1] : (i<=niw || i>niw+nif ? x[ni+1-i,1] : 0.0f0)
        north=j<nj ? x[i,j+1] : 0.0f0
        lhs=(d0[i,j]+ls[i,j]+ln[i,j])*x[i,j]-ls[i,j]*south-ln[i,j]*north
        @test isapprox(lhs,rhs[i,j];rtol=2e-6)
    end
end

@testset "damped Newton steps reduce pseudo CFL" begin
    next=M._airfoil_next_cfl
    @test next(1f0,1f0,0.005f0,10f0,1.05f0)==1.05f0
    @test next(1f0,0.5f0,0.005f0,10f0,1.05f0)==0.5f0
    @test next(1f0,0.0000610352f0,0.005f0,10f0,1.05f0)==0.5f0
    @test next(0.005f0,0.0000610352f0,0.005f0,10f0,1.05f0)==0.005f0
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
        r=M.step_jfnk_rans_ka!(s,ctrl;max_backtracks=16,
            sst_substeps=2)
        @test r.accepted && ctrl.iterations==1
        @test r.pseudo_cfl_next>=r.pseudo_cfl_used
        @test r.sst_substeps==2
        @test r.step_kind==:implicit && r.recovery_steps==0
        @test r.physical_trials>=1 && r.best_trial_merit<=r.flow_trial_merit_after
        @test r.merit_after==M._implicit_merit(s)
        @test length(ctrl.merit_history)==1 &&
              ctrl.merit_history[1]==r.merit_before
        @test all(isfinite,s.tk) && all(isfinite,s.tom)
    end
    s
end

function airfoil_recovery_and_diagnostics(backend)
    s=airfoil_implicit_smoke(backend;jfnk=true)
    reference=deepcopy(s)
    c=M.build_jfnk_airfoil_controller(s;m=4,recovery_steps=3,recovery_limit=5)
    for _ in 1:3
        M.step_ka!(reference,.45f0);M.sst_step_ka!(reference)
    end
    M.refresh_raw_residual_ka!(reference)
    r=M._airfoil_explicit_recovery!(s,c)
    @test r.accepted && r.recovery_steps==3 && c.recovery_iterations==3
    @test c.recovery_events==1 && c.cfl==c.restart_cfl
    for name in (:rho,:ux,:uy,:T,:tk,:tom,:nut,:tnl)
        @test Array(getproperty(s,name))==Array(getproperty(reference,name))
    end
    r=M._airfoil_explicit_recovery!(s,c)
    @test r.accepted && r.recovery_steps==2 && c.recovery_iterations==5
    before=Array(s.rho)
    @test !M._airfoil_explicit_recovery!(s,c).accepted
    @test before==Array(s.rho)
    names=(:tk,:tom,:nut,:tnl,:pvort,:exx,:exy,:eyy,:dtf)
    saved=map(name->Array(getproperty(s,name)),names)
    report=M.steady_airfoil_diagnostics_ka(s)
    @test all(isfinite,report.flow_l2) && all(>=(0),report.flow_l2)
    @test isfinite(report.normalized_mass_imbalance)
    @test isfinite(report.turb_k_update_defect) && isfinite(report.turb_omega_update_defect)
    @test !report.all_six_equation_residuals_available
    for (name,copy) in zip(names,saved)
        @test Array(getproperty(s,name))==copy
    end
    @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;recovery_steps=10,recovery_limit=5)
end

function airfoil_central_difference_smoke(backend;linear_solver=:native)
    s=airfoil_implicit_smoke(backend;jfnk=true)
    w=M.build_jfnk_airfoil_workspace(s;m=6)
    kw=linear_solver==:krylov ? M.build_krylov_airfoil_workspace(s,w) : nothing
    r=M.step_jfnk_airfoil_ka!(s,w,.45f0;
        fd_scheme=:central,fd_local_cap=.01f0,max_backtracks=16,
        krylov_workspace=kw)
    @test r.accepted && isfinite(r.linear_relative)
    @test r.merit_after<r.merit_before
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,w,.45f0;
        fd_scheme=:invalid)
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,w,.45f0;
        fd_local_cap=.1f0)
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
@testset "single-phase airfoil cut-line JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:cut_line)
end
@testset "single-phase airfoil cut-block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:cut_block_adi)
end
@testset "symmetric matrix-free airfoil Jacobian CPU" begin
    airfoil_central_difference_smoke(CPU())
    airfoil_central_difference_smoke(CPU();linear_solver=:krylov)
end

@testset "bounded recovery preserves explicit equations and diagnostic probe state CPU" begin
    airfoil_recovery_and_diagnostics(CPU())
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
    @testset "single-phase airfoil cut-line JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:cut_line)
    end
    @testset "single-phase airfoil cut-block ADI JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:cut_block_adi)
    end
    @testset "symmetric matrix-free airfoil Jacobian CUDA" begin
        airfoil_central_difference_smoke(CUDA.CUDABackend())
        airfoil_central_difference_smoke(CUDA.CUDABackend();linear_solver=:krylov)
    end
    @testset "bounded recovery preserves explicit equations and diagnostic probe state CUDA" begin
        airfoil_recovery_and_diagnostics(CUDA.CUDABackend())
    end
end
