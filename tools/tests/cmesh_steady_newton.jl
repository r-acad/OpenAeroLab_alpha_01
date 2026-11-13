# julia --startup-file=no --project=src tools/tests/cmesh_steady_newton.jl
# The opt-in pseudo-transient Newton-Krylov airfoil march (src/compressible/cmesh_steady_newton_ka.jl) on a small
# C-mesh: its parts, a short march, and mesh sequencing. CPU, Float64. Development checks, not validation.
using Test,KernelAbstractions,LinearAlgebra,StaticArrays
const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

function small_nodes()
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,
        dn1=.003,growth_max=1.8,R_ff=20.,L_wake=20.,smooth_iters=60,bl_freeze=5)
    (X=mesh.X,Y=mesh.Y,NIW=mesh.NIW,NIF=mesh.NIF,NI=mesh.NI,NJ=mesh.NJ)
end
"A prepared Float64 CPU solver on `nd`: thermal MUSCL, full viscous terms, SST, Mach 0.3, alpha 4 deg, Re 1e5."
function small_solver(nd;unlimited=true)
    T=Float64
    met=CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=0.,trip_lower=0.,scalar_type=T)
    cx,cy,cz,w,_,_=build_d3q39_static(T)
    s=M.build_sol_ka(CPU(),met;drot=T(2),cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type=T)
    s.wall_flux=true;s.fullvisc=Int32(1)
    M.set_airfoil_muscl_ka!(s,nd.X,nd.Y;limiter=:venkat)
    unlimited&&M.newton_unlimited_reconstruction!(s)
    M.set_freestream_ka!(s,.3,4.);s.nu_lam=hypot(s.uxi,s.uyi)/1e5
    M.set_sst_ka!(s);M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0)
    M.sst_step_ka!(s;corrected_wall_metric=true)
    s
end
interior(s,a)=Array(view(a,s.g+1:s.g+s.NI,s.g+1:s.g+s.NJ))
six(s)=(d=M.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true);[d.flow_l2...,d.turb_k_equation_l2,d.turb_omega_equation_l2])

@testset "GMRES and extrapolation on known systems" begin
    A=[4. 1 0;1 3 1;0 1 2];b=[1.,2,3]
    x,its,hist=M.newton_gmres((y,v)->(y.=A*v),(z,v)->(z.=v),b;tol=1e-12,restart=3,maxit=6)
    @test x≈A\b
    @test its<=3&&hist[end]<=1e-12
    # a linear fixed-point iteration with three modes: RRE from five iterates returns its fixed point
    G=Diagonal([.9,.5,-.3]);c=[1.,1,1];xs=[zeros(3)];for _ in 1:4;push!(xs,G*xs[end]+c);end
    @test M.newton_rre(xs,ones(3))≈(I-G)\c
end

@testset "Newton march on a small airfoil" begin
    nd=small_nodes();s=small_solver(nd)
    cold=six(s)
    m=M.newton_rans_init(s;cfl=5.,agg=(4,4))
    # one coupled iteration: the linear solve meets its goal within the Krylov budget and the step is admissible
    r=M.newton_rans_step!(s,m;blocks=:roe,delta=.3,lintol=.3,maxit=60)
    @test r.ok&&0<r.alpha<=1
    @test r.its<60&&r.lin<=.3
    @test all(isfinite,(r.cl,r.cd,r.dk,r.dw))
    out=M.newton_rans_march!(s,m;nsteps=40,cflmax=1e3,growth=1.5,every=10,quiet=true,rre=(3,8),
        blocks=:roe,delta=.3,lintol=.3,maxit=60,load_tol=1e-5,window=10)
    warm=six(s)
    @test out.steps<=40
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
    @test all(warm[1:4].<1e-3.*cold[1:4])             # the four flow residuals fall by three orders at least
    @test all(isfinite,warm)
    d=M.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true)
    @test abs(d.normalized_mass_imbalance)<1e-4
    @test d.turb_omega_wall_constraint_relative_linf<=1e-6
    @test 0.2<d.cl<0.7&&0<d.cd<0.1
    @test minimum(interior(s,s.tk))>0&&minimum(interior(s,s.tom))>0
    @test_throws ArgumentError M.newton_coarsen_nodes(nd,5)

    # sequencing: the settled state restricted by node skipping and interpolated back is a close starting guess
    ndc=M.newton_coarsen_nodes(nd,2)
    @test (ndc.NI,ndc.NJ,ndc.NIW,ndc.NIF)==(nd.NI÷2,nd.NJ÷2,nd.NIW÷2,nd.NIF÷2)
    sc=small_solver(ndc);mc=M.newton_rans_init(sc;cfl=5.,agg=(2,2))
    M.newton_rans_march!(sc,mc;nsteps=40,cflmax=1e3,every=40,quiet=true,rre=nothing,blocks=:roe,delta=.3,lintol=.3,maxit=60,load_tol=1e-5,window=10)
    sf=small_solver(nd);M.newton_prolong!(sf,sc,2)
    @test M._airfoil_physical_state(sf.rho,sf.ux,sf.uy,sf.T)
    @test minimum(interior(sf,sf.tk))>0&&minimum(interior(sf,sf.tom))>0
    clf,cdf=M.loads_ka(sf)
    @test abs(clf-d.cl)<0.5*abs(d.cl)                  # the interpolated coarse state already carries most of the lift
end

@testset "Cold run with sequencing" begin
    nd=small_nodes()
    r=M.newton_cold_run(nd,small_solver;levels=(2,1),nsteps=(40,40),cflmax=(1e3,1e3),load_tol=(1e-3,1e-4),window=(10,10),
        rre=(3,8),lintol=.3,maxit=60,quiet=true)
    @test length(r.summary["levels"])==2
    @test all(r.summary["final_six_over_cold"][1:4].<1e-3)
    @test isfinite(r.summary["final_CL"])&&isfinite(r.summary["final_CD"])
    @test r.summary["levels"][2]["cells"]==nd.NI*nd.NJ
end
