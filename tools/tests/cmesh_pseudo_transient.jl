using Test,KernelAbstractions,LinearAlgebra,StaticArrays
const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

function small_case(backend=CPU())
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,
        dn1=.003,growth_max=1.8,R_ff=20.,L_wake=20.,smooth_iters=60,bl_freeze=5)
    met=CMeshMetrics.build_metrics(mesh)
    cx,cy,cz,w,_,_=build_d3q39_static()
    s=M.build_sol_ka(backend,met;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    s.wall_flux=true;M.set_freestream_ka!(s,.3,4.)
    s.nu_lam=hypot(s.uxi,s.uyi)/3f6;M.set_sst_ka!(s)
    s
end

@testset "Backward Euler merit uses the actual conservative state" begin
    r0,u0,v0,t0=1f0,.2f0,.01f0,1f0
    r,u,v,t=.9f0,.18f0,.02f0,1.02f0
    eint=2f0;d=2f0
    q(r,u,v,t)=(r,r*u,r*v,r*(eint*t+.5f0*(u*u+v*v)))
    dq=q(r,u,v,t).-q(r0,u0,v0,t0)
    residual=-d.*dq
    value=M._airfoil_pseudo_component(r,u,v,t,r0,u0,v0,t0,residual...,d,1f0,eint,1f0,5f0,.5f0)
    @test value==0
    @test sum(abs2,residual)>0 # steady and pseudo-transient merit differ
    @test M._airfoil_physical_fraction(1f0,.2f0,0f0,1f0,2f0,1f0,2f0,.2f0)≈.1f0
    @test M._airfoil_physical_fraction(1f0,.2f0,0f0,1f0,0f0,4f0,2f0,.2f0)≈.101f0
end

function check_pseudo(backend)
    s=small_case(backend)
    c=M.build_jfnk_airfoil_controller(s;m=8,initial_cfl=2f0,min_cfl=.005f0,
        max_cfl=1f3,globalization=:pseudo_transient)
    result=M.step_jfnk_rans_ka!(s,c;max_backtracks=16,sst_substeps=2)
    @test result.accepted
    @test result.globalization==:pseudo_transient
    @test result.acceptance_merit_after<result.merit_before
    @test 0<result.alpha<=result.physical_alpha<=1
    @test result.linear_relative<=.95
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
    d=M.steady_airfoil_diagnostics_ka(s)
    @test d.all_six_equation_residuals_available
    @test all(isfinite,(d.turb_k_equation_l2,d.turb_omega_equation_l2))
    @test d.turb_omega_wall_constraint_cells==2s.NIF
    @test d.turb_omega_transport_cells==length(s.vol)-2s.NIF
    @test d.turb_omega_wall_constraint_relative_linf<=1f-6
    @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;max_cfl=100f0)
    @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;globalization=:unknown)
    @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;globalization=:pseudo_transient,physical_fraction=1f0)
    work=M.build_jfnk_airfoil_workspace(s;m=8)
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,work,1f0;linear_maxiter=0)
    bounded=M.step_jfnk_airfoil_ka!(s,work,1f0;linear_maxiter=5,globalization=:pseudo_transient)
    @test bounded.linear_iterations<=5
    @test isfinite(bounded.linear_relative)
    s
end

function check_raw_sst_parity(backend)
    a=small_case();b=small_case(backend)
    for j in axes(a.tk,2),i in axes(a.tk,1)
        a.rho[i,j]=1f0+.01f0*sin(Float32(i+j))
        a.T[i,j]=1f0+.01f0*cos(Float32(i-j))
        a.tk[i,j]=a.k_inf*(1f0+.1f0*sin(Float32(i))*cos(Float32(j)))
        a.tom[i,j]=a.w_inf*(1f0+.1f0*cos(Float32(i))*sin(Float32(j)))
    end
    for name in (:rho,:ux,:uy,:T,:tk,:tom,:nut,:tnl)
        copyto!(getproperty(b,name),getproperty(a,name))
    end
    M.prepare_dt_ka!(a,.45f0);M.prepare_dt_ka!(b,.45f0)
    ea=M.sst_step_ka!(a;raw_equations=true)
    eb=M.sst_step_ka!(b;raw_equations=true)
    ir=a.g+1:a.g+a.NI;jr=a.g+1:a.g+a.NJ
    for name in (:k,:omega,:omega_wall_constraint_relative,:omega_all_transport)
        av=Array(view(getproperty(ea,name),ir,jr));bv=Array(view(getproperty(eb,name),ir,jr))
        @test all(isfinite,bv)
        @test norm(Float64.(av).-Float64.(bv))<=3e-4*norm(Float64.(av))+3e-6
    end
end

@testset "Exact omega constraint cell indices" begin
    @test !M._sst_wall_constraint_cell(12,1,12,48)
    @test M._sst_wall_constraint_cell(13,1,12,48)
    @test M._sst_wall_constraint_cell(60,2,12,48)
    @test !M._sst_wall_constraint_cell(61,2,12,48)
    @test !M._sst_wall_constraint_cell(30,3,12,48)
end

@testset "Physical pseudo-transient globalization CPU" begin
    check_pseudo(CPU())
end

@testset "Raw SST defects do not vanish at a clipped fixed point" begin
    s=small_case()
    fill!(s.rho,1f0);fill!(s.ux,s.uxi);fill!(s.uy,s.uyi);fill!(s.T,1f0)
    fill!(s.tk,1f-12);fill!(s.tom,1f6);fill!(s.nut,1f-18)
    M.prepare_dt_ka!(s,.45f0)
    eq=M.sst_step_ka!(s;raw_equations=true)
    i=s.g+20;j=s.g+10
    rawk=eq.k[i,j]
    @test rawk≈.09f0*1f-12*1f6 rtol=1e-5
    @test isfinite(eq.omega[i,j]) && eq.omega[i,j]>0
    old=copy(s.tk)
    M.sst_step_ka!(s)
    @test s.tk[i,j]==old[i,j]==1f-12
    @test rawk>0 # clipping hides this equation defect in an update-only test
    eq=M.sst_step_ka!(s;raw_equations=true)
    wi=s.g+s.NIW+10;wj=s.g+1
    @test eq.omega[wi,wj]==0
    @test eq.omega_wall_constraint_relative[wi,wj]==0
    s.tom[wi,wj]*=1.1f0
    eq=M.sst_step_ka!(s;raw_equations=true)
    @test eq.omega_wall_constraint_relative[wi,wj]≈.1f0 atol=2f-7
end

if get(ENV,"OPENLUDWIG_TEST_CUDA","0")=="1"
    using CUDA
    @test CUDA.functional()
    @testset "Physical pseudo-transient globalization CUDA" begin
        check_pseudo(CUDA.CUDABackend())
        check_raw_sst_parity(CUDA.CUDABackend())
    end
end
