using Test, KernelAbstractions, LinearAlgebra, StaticArrays

const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

@testset "CPU Newton-Krylov inner products avoid mesh-sized temporaries" begin
    a=ntuple(k->fill(Float32(.1k),512,512),4)
    b=ntuple(k->fill(Float32(.2k),512,512),4)
    @test M._implicit_cpu_strided(a[1])
    expected=sum(sum(Float64.(a[k]).*Float64.(b[k])) for k in 1:4)
    @test isapprox(M._av_dot(a,b),expected;rtol=1e-9)
    bytes()=@allocated M._av_dot(a,b)
    bytes(); bytes()
    @test bytes()<1024
end

@testset "CPU implicit merit avoids line-search temporaries" begin
    vol=[Float32(1+.01i+.02j) for i in 1:64,j in 1:32]
    raw=ntuple(k->[Float32(.03k+.001i-.002j) for i in 1:64,j in 1:32],4)
    scales=(.5f0,.25f0,.25f0,.1f0)
    expected=sqrt(sum(sum((Float64(raw[k][i]/vol[i])*scales[k])^2
                          for k in 1:4) for i in eachindex(vol))/length(vol))
    merit=M._implicit_merit_cpu(raw,vol,scales[1],scales[2],scales[4])
    @test isapprox(merit,expected;rtol=1e-10)
    bytes()=@allocated M._implicit_merit_cpu(raw,vol,scales[1],scales[2],scales[4])
    bytes();bytes()
    @test bytes()<1024
end

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
    fd_bytes()=@allocated M._airfoil_fd_step(v,1.0,10_000,1.0,0.2,1.0,1.0f0)
    fd_bytes();fd_bytes()
    @test fd_bytes()<1024
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

@testset "C-mesh multilevel aggregation preserves scalar operator" begin
    ni,nj,niw,nif=8,4,2,4
    lw=[Float32(1+.1i+.02j) for i in 1:ni,j in 1:nj]
    le=[Float32(1.3+.03i+.04j) for i in 1:ni,j in 1:nj]
    ls=[Float32(2+.05i+.06j) for i in 1:ni,j in 1:nj]
    ln=[Float32(2.4+.07i+.01j) for i in 1:ni,j in 1:nj]
    d0=fill(1.5f0,ni,nj)
    cw=zeros(Float32,ni÷2,nj);ce=similar(cw);cs=similar(cw)
    cn=similar(cw);cd0=similar(cw)
    M._k_airfoil_mg_aggregate!(CPU())(cw,ce,cs,cn,cd0,
        lw,le,ls,ln,d0,Int32(ni÷2),Int32(nj);ndrange=(ni÷2,nj))
    xc=reshape(Float32.(1:(ni÷2)*nj),ni÷2,nj)./17f0
    xf=repeat(xc;inner=(2,1))
    rf=similar(xf);rc=similar(xc);restricted=similar(xc)
    M._k_airfoil_mg_residual!(CPU())(rf,zero(xf),xf,lw,le,ls,ln,d0,
        Int32(ni),Int32(nj),Int32(niw),Int32(nif);ndrange=(ni,nj))
    M._k_airfoil_mg_restrict!(CPU())(restricted,rf,Int32(ni÷2),Int32(nj);
        ndrange=(ni÷2,nj))
    M._k_airfoil_mg_residual!(CPU())(rc,zero(xc),xc,cw,ce,cs,cn,cd0,
        Int32(ni÷2),Int32(nj),Int32(niw÷2),Int32(nif÷2);
        ndrange=(ni÷2,nj))
    KernelAbstractions.synchronize(CPU())
    @test isapprox(restricted,rc;rtol=2e-6,atol=2e-6)
end

@testset "wall-normal aggregation preserves scalar operator" begin
    ni,nj,niw,nif=8,4,2,4
    lw=[Float32(1+.1i+.02j) for i in 1:ni,j in 1:nj]
    le=[Float32(1.3+.03i+.04j) for i in 1:ni,j in 1:nj]
    ls=[Float32(2+.05i+.06j) for i in 1:ni,j in 1:nj]
    ln=[Float32(2.4+.07i+.01j) for i in 1:ni,j in 1:nj]
    d0=fill(1.5f0,ni,nj)
    cw=zeros(Float32,ni,nj÷2);ce=similar(cw);cs=similar(cw)
    cn=similar(cw);cd0=similar(cw)
    M._k_airfoil_mg_aggregate_j!(CPU())(cw,ce,cs,cn,cd0,
        lw,le,ls,ln,d0,Int32(ni),Int32(nj÷2);ndrange=(ni,nj÷2))
    xc=reshape(Float32.(1:ni*(nj÷2)),ni,nj÷2)./17f0
    xf=repeat(xc;inner=(1,2))
    rf=similar(xf);rc=similar(xc);restricted=similar(xc)
    M._k_airfoil_mg_residual!(CPU())(rf,zero(xf),xf,lw,le,ls,ln,d0,
        Int32(ni),Int32(nj),Int32(niw),Int32(nif);ndrange=(ni,nj))
    M._k_airfoil_mg_restrict_j!(CPU())(restricted,rf,Int32(ni),Int32(nj÷2);
        ndrange=(ni,nj÷2))
    M._k_airfoil_mg_residual!(CPU())(rc,zero(xc),xc,cw,ce,cs,cn,cd0,
        Int32(ni),Int32(nj÷2),Int32(niw),Int32(nif);
        ndrange=(ni,nj÷2))
    KernelAbstractions.synchronize(CPU())
    @test isapprox(restricted,rc;rtol=2e-6,atol=2e-6)
end

@testset "wake-aware coarse factor solves scalar system" begin
    ni,nj,niw,nif=8,4,2,4
    lw=fill(1f0,ni,nj);le=fill(1.2f0,ni,nj)
    ls=fill(2f0,ni,nj);ln=fill(2.2f0,ni,nj)
    d0=fill(1.5f0,ni,nj)
    level=M._airfoil_mg_level((n,m)->zeros(Float32,n,m),
        ni,nj,niw,nif,lw,le,ls,ln,d0)
    rhs=reshape(Float32.(1:ni*nj),ni,nj)./17f0
    factor=M._airfoil_mg_coarse_factor(level)
    @test factor!==nothing
    x=reshape(Float32.(factor\vec(rhs)),ni,nj)
    residual=similar(rhs)
    M._k_airfoil_mg_residual!(CPU())(residual,rhs,x,lw,le,ls,ln,d0,
        Int32(ni),Int32(nj),Int32(niw),Int32(nif);ndrange=(ni,nj))
    KernelAbstractions.synchronize(CPU())
    @test maximum(abs,residual)<2e-6
end

function check_block_line_inverse(backend,transfer)
    ni,nj=3,5
    lower=zeros(Float32,4,4,ni,nj)
    diagonal=zeros(Float32,4,4,ni,nj)
    upper=zeros(Float32,4,4,ni,nj)
    vol=[Float32(0.8+.03i+.02j) for i in 1:ni,j in 1:nj]
    rhs=ntuple(k->[Float32(.1k+.03i-.02j) for i in 1:ni,j in 1:nj],4)
    for j in 1:nj,i in 1:ni
        for k in 1:4
            diagonal[k,k,i,j]=4f0+.1f0*j
            if j>1
                lower[k,k,i,j]=-.3f0
            end
            if j<nj
                upper[k,k,i,j]=-.25f0
            end
        end
        diagonal[1,4,i,j]=.2f0
        diagonal[2,3,i,j]=-.15f0
        diagonal[4,1,i,j]=.1f0
        j>1 && (lower[2,1,i,j]=.03f0)
        j<nj && (upper[4,2,i,j]=-.02f0)
    end
    out=ntuple(_->transfer(zeros(Float32,ni,nj)),4)
    cp=transfer(zeros(Float32,4,4,ni,nj))
    M._k_airfoil_block_line_solve!(backend)(out,cp,transfer.(rhs),
        transfer(vol),transfer(lower),transfer(diagonal),transfer(upper),
        Int32(ni),Int32(nj);ndrange=ni)
    KernelAbstractions.synchronize(backend)
    actual=Array.(out)
    for i in 1:ni
        a=zeros(Float64,4nj,4nj)
        b=zeros(Float64,4nj)
        for j in 1:nj
            rows=(4j-3):(4j)
            a[rows,rows]=diagonal[:,:,i,j]
            j>1 && (a[rows,(4j-7):(4j-4)]=lower[:,:,i,j])
            j<nj && (a[rows,(4j+1):(4j+4)]=upper[:,:,i,j])
            b[rows]=[rhs[k][i,j]*vol[i,j] for k in 1:4]
        end
        expected=a\b
        for j in 1:nj,k in 1:4
            @test isapprox(actual[k][i,j],expected[4j-4+k];rtol=2e-5,atol=1e-6)
        end
    end
end

function check_block_row_inverse(backend,transfer)
    ni,nj=5,3
    lower=zeros(Float32,4,4,ni,nj)
    diagonal=zeros(Float32,4,4,ni,nj)
    upper=zeros(Float32,4,4,ni,nj)
    vol=[Float32(0.8+.03i+.02j) for i in 1:ni,j in 1:nj]
    rhs=ntuple(k->[Float32(.1k+.03i-.02j) for i in 1:ni,j in 1:nj],4)
    for j in 1:nj,i in 1:ni
        for k in 1:4
            diagonal[k,k,i,j]=4f0+.1f0*i
            i>1 && (lower[k,k,i,j]=-.3f0)
            i<ni && (upper[k,k,i,j]=-.25f0)
        end
        diagonal[1,4,i,j]=.2f0
        diagonal[2,3,i,j]=-.15f0
        diagonal[4,1,i,j]=.1f0
        i>1 && (lower[2,1,i,j]=.03f0)
        i<ni && (upper[4,2,i,j]=-.02f0)
    end
    out=ntuple(_->transfer(zeros(Float32,ni,nj)),4)
    cp=transfer(zeros(Float32,4,4,ni,nj))
    M._k_airfoil_block_row_solve!(backend)(out,cp,transfer.(rhs),
        transfer(vol),transfer(lower),transfer(diagonal),transfer(upper),
        Int32(ni),Int32(nj),Int32(ni),Int32(1);ndrange=nj)
    KernelAbstractions.synchronize(backend)
    actual=Array.(out)
    for j in 1:nj
        a=zeros(Float64,4ni,4ni)
        b=zeros(Float64,4ni)
        for i in 1:ni
            rows=(4i-3):(4i)
            a[rows,rows]=diagonal[:,:,i,j]
            i>1 && (a[rows,(4i-7):(4i-4)]=lower[:,:,i,j])
            i<ni && (a[rows,(4i+1):(4i+4)]=upper[:,:,i,j])
            b[rows]=[rhs[k][i,j]*vol[i,j] for k in 1:4]
        end
        expected=a\b
        for i in 1:ni,k in 1:4
            @test isapprox(actual[k][i,j],expected[4i-4+k];rtol=2e-5,atol=1e-6)
        end
    end
end

@testset "coupled block line inverse CPU" begin
    check_block_line_inverse(CPU(),identity)
    check_block_row_inverse(CPU(),identity)
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
                                preconditioner=:line,pseudo_scale=1f0)
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
    if preconditioner in (:kinetic_fullvisc_block_adi,
                           :kinetic_weno_fullvisc_block_adi,
                           :kinetic_weno_fullvisc_unfloored_adi)
        s.fullvisc=Int32(1);s.recon=Int32(1)
    end
    M.set_freestream_ka!(s,.2,2.0)
    s.nu_lam=1.0f-4
    M.set_sst_ka!(s)
    work=jfnk ? M.build_jfnk_airfoil_workspace(s;m=6,preconditioner,pseudo_scale) :
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
            pseudo_scale,
            nonmonotone_window=3)
        r=M.step_jfnk_rans_ka!(s,ctrl;max_backtracks=16,
            sst_substeps=2)
        @test r.accepted && ctrl.iterations==1
        @test ctrl.min_cfl<=r.pseudo_cfl_next<=ctrl.max_cfl
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

@testset "six-state WENO kinetic face center block directional action" begin
    cx,cy,cz,w,_,_=build_d3q39_static()
    eint=2f0;drot=2f0
    qs=(M._airfoil_conserved4(1.03f0,0.13f0,-0.03f0,1.04f0,eint),
        M._airfoil_conserved4(1.05f0,0.16f0,-0.02f0,1.03f0,eint),
        M._airfoil_conserved4(1.08f0,0.19f0,-0.04f0,1.02f0,eint),
        M._airfoil_conserved4(0.97f0,0.15f0,0.03f0,0.95f0,eint),
        M._airfoil_conserved4(0.95f0,0.12f0,0.02f0,0.96f0,eint),
        M._airfoil_conserved4(0.92f0,0.10f0,0.01f0,0.97f0,eint))
    nxr=.7f0;nyr=-.4f0
    jl,jr=M._airfoil_weno_face_jacobians(qs...,nxr,nyr,
        cx,cy,cz,w,drot,eint)
    dl=SVector{4,Float32}(0.15f0,-0.09f0,0.07f0,0.12f0)
    dr=SVector{4,Float32}(-0.08f0,0.04f0,0.11f0,-0.05f0)
    h=.002f0
    fp=M._airfoil_kinetic_flux6(qs[1],qs[2],qs[3]+h*dl,
        qs[4]+h*dr,qs[5],qs[6],nxr,nyr,cx,cy,cz,w,drot,eint)
    fm=M._airfoil_kinetic_flux6(qs[1],qs[2],qs[3]-h*dl,
        qs[4]-h*dr,qs[5],qs[6],nxr,nyr,cx,cy,cz,w,drot,eint)
    actual=(fp-fm)/(2h)
    predicted=jl*dl+jr*dr
    @test all(isfinite,predicted)
    @test norm(predicted-actual)/norm(actual)<0.05
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

function airfoil_coupled_rollback(backend)
    s=airfoil_implicit_smoke(backend;jfnk=true)
    c=M.build_jfnk_airfoil_controller(s;m=4)
    names=M._AIRFOIL_COUPLED_STATE
    before=map(name->Array(getproperty(s,name)),names)
    M._airfoil_coupled_backup!(s,c)
    # A flow step has already been accepted; an invalid SST update must roll
    # back that step as well as the turbulence fields and time-step array.
    fill!(s.ux,0.125f0)
    bad!(x;kwargs...)=(fill!(x.tk,NaN32);fill!(x.dtf,42f0);nothing)
    result=M._airfoil_coupled_sst!(s,c;advance! = bad!)
    @test !result.accepted
    for (name,expected) in zip(names,before)
        @test Array(getproperty(s,name))==expected
    end
    M._airfoil_coupled_backup!(s,c)
    fill!(s.uy,0.125f0)
    throwing!(x;kwargs...)=(fill!(x.nut,NaN32);error("synthetic SST failure"))
    @test_throws ErrorException M._airfoil_coupled_sst!(s,c;advance! = throwing!)
    for (name,expected) in zip(names,before)
        @test Array(getproperty(s,name))==expected
    end
    # Exercise the real controller rejection path, not only the helper.
    c2=M.build_jfnk_airfoil_controller(s;m=4,initial_cfl=.05f0,
        min_cfl=.05f0,max_cfl=.05f0)
    calls=Ref(0)
    bad_controller!(x;kwargs...)=(calls[]+=1;fill!(x.tk,NaN32);nothing)
    r=M.step_jfnk_rans_ka!(s,c2;max_backtracks=16,
        _sst_advance! = bad_controller!)
    @test calls[]==1
    @test !r.accepted && r.step_kind==:rejected && c2.rejections==1
    @test c2.iterations==0 && r.merit_after==r.merit_before
    for (name,expected) in zip(names,before)
        @test Array(getproperty(s,name))==expected
    end
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
@testset "single-phase airfoil swept-line JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:swept_line)
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:cut_swept_line)
end
@testset "single-phase airfoil multilevel JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:multilevel)
end

@testset "two-state kinetic face block directional action" begin
    cx,cy,cz,w,_,_=build_d3q39_static()
    eint=2.0f0;drot=2.0f0
    ql=M._airfoil_conserved4(1.08f0,0.19f0,-0.04f0,1.02f0,eint)
    qr=M._airfoil_conserved4(0.97f0,0.15f0,0.03f0,0.95f0,eint)
    nxr=0.7f0;nyr=-0.4f0
    jl,jr=M._airfoil_kinetic_face_jacobians(ql,qr,nxr,nyr,
        cx,cy,cz,w,drot,eint)
    dl=SVector{4,Float32}(0.15f0,-0.09f0,0.07f0,0.12f0)
    dr=SVector{4,Float32}(-0.08f0,0.04f0,0.11f0,-0.05f0)
    h=0.002f0
    fp=M._airfoil_kinetic_flux2(ql+h*dl,qr+h*dr,nxr,nyr,cx,cy,cz,w,drot,eint)
    fm=M._airfoil_kinetic_flux2(ql-h*dl,qr-h*dr,nxr,nyr,cx,cy,cz,w,drot,eint)
    actual=(fp-fm)/(2h)
    predicted=jl*dl+jr*dr
    @test all(isfinite,predicted)
    @test norm(predicted-actual)/norm(actual)<0.02
    wall=M._airfoil_wall_pressure_jacobian(0.19f0,-0.04f0,nxr,nyr,eint)
    @test wall[1,:]==zeros(Float32,4) && wall[4,:]==zeros(Float32,4)
    @test wall[2,4]==nxr*M.CS2/eint
end

@testset "frozen-gradient full-viscous face block directional action" begin
    eint=2f0
    ql=M._airfoil_conserved4(1.08f0,0.19f0,-0.04f0,1.02f0,eint)
    qr=M._airfoil_conserved4(0.97f0,0.15f0,0.03f0,0.95f0,eint)
    gl=SVector{6,Float32}(0.7f0,-0.5f0,0.4f0,0.6f0,-0.2f0,0.3f0)
    gr=SVector{6,Float32}(0.5f0,-0.3f0,0.2f0,0.4f0,-0.1f0,0.2f0)
    dx=0.018f0;dy=0.009f0;nxr=0.02f0;nyr=0.012f0
    mu=0.004f0;ke=0.006f0
    jl,jr=M._airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,
        dx,dy,nxr,nyr,mu,ke,eint)
    dl=SVector{4,Float32}(0.15f0,-0.09f0,0.07f0,0.12f0)
    dr=SVector{4,Float32}(-0.08f0,0.04f0,0.11f0,-0.05f0)
    function flux(q1,q2)
        _,u1,v1,t1=M._airfoil_primitive4(q1,eint)
        _,u2,v2,t2=M._airfoil_primitive4(q2,eint)
        area=hypot(nxr,nyr)
        fx,fy,fe=M._viscous_face_full(gl...,gr...,
            u1,v1,t1,u2,v2,t2,dx,dy,nxr/area,nyr/area,area,mu,ke)
        SVector{4,Float32}(0f0,-fx,-fy,-fe)
    end
    h=0.001f0
    actual=(flux(ql+h*dl,qr+h*dr)-flux(ql-h*dl,qr-h*dr))/(2h)
    predicted=jl*dl+jr*dr
    @test all(isfinite,predicted)
    @test norm(predicted-actual)/norm(actual)<0.01
end

@testset "four-variable wake-cut block inverse" begin
    ni,nj,niw,nif=6,4,2,2
    lower=zeros(Float32,4,4,ni,nj)
    upper=zeros(Float32,4,4,ni,nj)
    diagonal=zeros(Float32,4,4,ni,nj)
    vol=fill(1.0f0,ni,nj)
    rhs=ntuple(k->fill(Float32(0.03k),ni,nj),4)
    for j in 1:nj,i in 1:ni
        for k in 1:4
            diagonal[k,k,i,j]=4f0+0.1f0*k
            if j>1 || i<=niw || i>niw+nif
                lower[k,k,i,j]=-0.2f0
            end
            j<nj && (upper[k,k,i,j]=-0.3f0)
        end
        diagonal[2,4,i,j]=0.1f0
        diagonal[4,1,i,j]=-0.05f0
    end
    out=ntuple(_->zeros(Float32,ni,nj),4)
    cp=similar(lower)
    M._k_airfoil_block_cut_line_solve!(CPU())(out,cp,rhs,vol,
        lower,diagonal,upper,Int32(ni),Int32(nj),Int32(niw),Int32(nif);
        ndrange=niw+nif)
    KernelAbstractions.synchronize(CPU())
    for j in 1:nj,i in 1:ni
        x=SVector{4,Float32}((out[k][i,j] for k in 1:4)...)
        south=j>1 ? SVector{4,Float32}((out[k][i,j-1] for k in 1:4)...) :
            (i<=niw || i>niw+nif ?
             SVector{4,Float32}((out[k][ni+1-i,1] for k in 1:4)...) : zero(x))
        north=j<nj ? SVector{4,Float32}((out[k][i,j+1] for k in 1:4)...) : zero(x)
        lhs=diagonal[:,:,i,j]*x+lower[:,:,i,j]*south+upper[:,:,i,j]*north
        expected=SVector{4,Float32}((rhs[k][i,j] for k in 1:4)...)
        @test isapprox(lhs,expected;rtol=2e-5,atol=2e-6)
    end
end

@testset "single-phase airfoil kinetic block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,preconditioner=:kinetic_block_adi)
    airfoil_implicit_smoke(CPU();jfnk=true,linear_solver=:krylov,
        preconditioner=:kinetic_block_adi)
end

@testset "single-phase airfoil full-viscous block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,
        preconditioner=:kinetic_fullvisc_block_adi)
end

@testset "single-phase airfoil WENO full-viscous block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,
        preconditioner=:kinetic_weno_fullvisc_block_adi)
end

@testset "single-phase airfoil unfloored block ADI JFNK flow CPU" begin
    airfoil_implicit_smoke(CPU();jfnk=true,
        preconditioner=:kinetic_block_unfloored_adi)
    s=airfoil_implicit_smoke(CPU();jfnk=true,
        preconditioner=:kinetic_weno_fullvisc_unfloored_adi,
        pseudo_scale=0.5f0)
    w=M.build_jfnk_airfoil_workspace(s;m=4,
        preconditioner=:kinetic_weno_fullvisc_unfloored_adi)
    s.recon=Int32(0)
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,w,0.45f0)
end

@testset "symmetric matrix-free airfoil Jacobian CPU" begin
    airfoil_central_difference_smoke(CPU())
    airfoil_central_difference_smoke(CPU();linear_solver=:krylov)
end

@testset "bounded recovery preserves explicit equations and diagnostic probe state CPU" begin
    airfoil_recovery_and_diagnostics(CPU())
end

@testset "invalid coupled SST steps restore the previous state CPU" begin
    airfoil_coupled_rollback(CPU())
end

if get(ENV,"OPENLUDWIG_TEST_CUDA", "0")=="1"
    using CUDA
    @test CUDA.functional()
    @testset "implicit physical-state reduction CUDA" begin
        good=CUDA.fill(1.0f0,4,4)
        zero_velocity=CUDA.zeros(Float32,4,4)
        bad=CUDA.fill(NaN32,4,4)
        @test !M._implicit_cpu_strided(good)
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
    @testset "single-phase airfoil swept-line JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:swept_line)
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:cut_swept_line)
    end
    @testset "single-phase airfoil multilevel JFNK flow CUDA" begin
        airfoil_implicit_smoke(CUDA.CUDABackend();jfnk=true,preconditioner=:multilevel)
    end
    @testset "coupled block line inverse CUDA" begin
        check_block_line_inverse(CUDA.CUDABackend(),CUDA.CuArray)
        check_block_row_inverse(CUDA.CUDABackend(),CUDA.CuArray)
    end
    @testset "symmetric matrix-free airfoil Jacobian CUDA" begin
        airfoil_central_difference_smoke(CUDA.CUDABackend())
        airfoil_central_difference_smoke(CUDA.CUDABackend();linear_solver=:krylov)
    end
    @testset "bounded recovery preserves explicit equations and diagnostic probe state CUDA" begin
        airfoil_recovery_and_diagnostics(CUDA.CUDABackend())
    end
    @testset "invalid coupled SST steps restore the previous state CUDA" begin
        airfoil_coupled_rollback(CUDA.CUDABackend())
    end
end
