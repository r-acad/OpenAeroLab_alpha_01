# julia --startup-file=no --project=src tools/tests/cmesh_steady_newton_pair.jl
# The opt-in two-block (C+H) Newton-Krylov airfoil march (src/compressible/cmesh_steady_newton_pair_ka.jl) on a small
# marched mesh of a blunt section: its parts, a short march and mesh sequencing. CPU, Float64. Development checks,
# not validation.
using Test,KernelAbstractions,LinearAlgebra,StaticArrays
const LBM_F=Float32
include(joinpath(@__DIR__,"..","..","src","collision","lattice.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_metrics.jl"))
include(joinpath(@__DIR__,"..","..","src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA

"NACA 0012 truncated at 95 % chord (a blunt base of 1.7 % chord), unit chord; a sharp NACA 0012 with `cut = 1`."
function foil(;cut=0.95,n=40)
    yt(x)=5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    xs=[cut*.5*(1-cospi(k/n)) for k in 0:n];y=[x==1.0 ? 0.0 : yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end])./cut,vcat(-reverse(y),y[2:end])./cut)
end
small_mesh(;cut=0.95)=CMeshGen.marched_airfoil_mesh(foil(;cut);n_foil=96,n_wake=24,n_normal=32,dn1=1e-3,alpha_deg=2.0,far=20.0,wake_aspect=100.0).mesh
"A prepared Float64 CPU block: thermal MUSCL with the chord-scaled limiter, full viscous terms, SST, Mach 0.3, alpha 2 deg, Re 1e5."
function block(nd)
    T=Float64
    met=CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=0.,trip_lower=0.,scalar_type=T)
    cx,cy,cz,w,_,_=build_d3q39_static(T)
    s=M.build_sol_ka(CPU(),met;drot=T(2),cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type=T)
    s.wall_flux=true;s.fullvisc=Int32(1)
    M.set_airfoil_muscl_ka!(s,nd.X,nd.Y;limiter=:venkat_chord,smooth_coefficient=1f0)
    M.set_freestream_ka!(s,.3,2.);s.nu_lam=hypot(s.uxi,s.uyi)/1e5
    M.set_sst_ka!(s);M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0)
    M.sst_step_ka!(s;corrected_wall_metric=true)
    s
end
function pair(m)
    cn=(X=m.X,Y=m.Y,NIW=m.NIW,NIF=m.NIF,NI=m.NI,NJ=m.NJ);hn=M.newton_pair_strip_nodes(m.strip)
    M.newton_pair_build(block(cn),block(hn),cn,hn)
end

@testset "SST coefficients: the pair kernel equals the single-block kernel away from interfaces" begin
    m=small_mesh(cut=1.0);s=block((X=m.X,Y=m.Y,NIW=m.NIW,NIF=m.NIF))
    M._newton_sst_bc!(s);a=M.newton_sst_init(s);b=M.newton_sst_init(s)
    M._k_newton_sst_coeffs!(s.backend)(a.akW,a.akE,a.akS,a.akN,a.awW,a.awE,a.awS,a.awN,a.dk,a.sk,a.dw,a.sw,a.sf2,
        s.tk,s.tom,s.nut,s.rho,s.T,s.ux,s.uy,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.wdist,s.gammat,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),Int32(1);ndrange=(s.NI,s.NJ))
    M.newton_pair_sst_coeffs!(s,b,M._newton_pair_sst_geo(s))
    @test all(getfield(a,n)==getfield(b,n) for n in fieldnames(typeof(a)))
end

@testset "pair build, ghost cells and the shared interface flux" begin
    m=small_mesh();p=pair(m);c,h=p.c,p.h;g=c.g
    @test (h.NIW,h.NIF,h.NI,h.NJ)==(0,m.strip.NJ,m.strip.NJ,m.NIW)
    @test p.sign==(1.0,-1.0)&&all(>(0),p.dist)
    # ghost cells of one block are the other block's cells
    c.rho[g+1:g+c.NI,g+1:g+c.NJ].*=1 .+0.01 .*rand(c.NI,c.NJ);M.newton_pair_bc!(p)
    k=3;il=p.NW+1-k;iu=p.NW+c.NIF+k
    @test h.rho[g,k+g]==c.rho[iu+g,1+g]&&h.rho[p.NS+g+1,k+g]==c.rho[il+g,1+g]
    @test c.rho[il+g,g]==h.rho[p.NS+g,k+g]&&c.rho[iu+g,g]==h.rho[1+g,k+g]
    # one flux per interface face, so the two blocks conserve together
    M.newton_pair_residual!(p)
    for side in 1:2,k in 1:p.NW
        ic,ih=M._newton_pair_cells(p,k,side);ihf=side==1 ? p.NS+1 : 1
        @test c.Fj_r[ic,1]==p.sign[side]*h.Fi_r[ihf,k]&&c.Fj_e[ic,1]==p.sign[side]*h.Fi_e[ihf,k]
    end
    @test all(isfinite,M.newton_pair_flow_l2(p))
end

@testset "march on a small blunt section" begin
    p=pair(small_mesh());m=M.newton_pair_init(p;cfl=5.,agg=(4,4))
    cold=M.newton_pair_six(p,m)
    r=M.newton_pair_step!(p,m;delta=.3,lintol=.3,maxit=60)
    @test r.ok&&0<r.alpha<=1
    @test r.its<60&&r.lin<=.3&&r.kits<60&&r.wits<60
    @test all(isfinite,(r.cl,r.cd,r.dk,r.dw))
    out=M.newton_pair_march!(p,m;nsteps=40,cflmax=1e3,growth=1.5,every=10,quiet=true,rre=(3,8),delta=.3,lintol=.3,maxit=60,
        load_tol=1e-5,window=10,vortex_relax=.3)
    warm=M.newton_pair_six(p,m)
    @test out.steps<=40
    @test all(s->M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T),(p.c,p.h))
    @test all(warm[1:4].<1e-2 .*cold[1:4])
    cl,cd=M.newton_pair_loads(p)
    @test 0.1<cl<0.4&&0.005<cd<0.06
    @test p.h.gam==p.c.gam&&p.c.gam!=0
    # the strip's wall is the base: its drag is part of the loads
    @test M.newton_pair_loads(p)[2]≈Float64(M.loads_ka(p.c)[2])+Float64(M.loads_ka(p.h)[2])
end

@testset "cold run with mesh sequencing" begin
    m=small_mesh();nodes=(X=m.X,Y=m.Y,NIW=m.NIW,NIF=m.NIF,NI=m.NI,NJ=m.NJ,strip=m.strip)
    r=M.newton_pair_cold_run(nodes,block,block;levels=(2,1),nsteps=(30,30),cflmax=(1e3,1e3),load_tol=(1e-3,1e-4),window=(10,10),
        lintol=.3,maxit=60,quiet=true,vortex_relax=.3)
    @test length(r.summary["levels"])==2&&r.summary["levels"][1]["cells"]<r.summary["levels"][2]["cells"]
    @test all(isfinite,r.summary["final_six_residuals"])
    @test all(r.summary["final_six_over_cold"][1:4].<1e-2)
    @test 0.1<r.summary["final_CL"]<0.4
end
