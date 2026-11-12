using Test,KernelAbstractions
include(joinpath(@__DIR__,"..","..","solvers","pbcav_s1030","src","pbcav","PBCav.jl"))

@testset "SST nonlinear relaxation preserves the implicit equation" begin
    mesh=PBCav.box_mesh(4,4;periodic=(true,true))
    results=Dict{Float64,Tuple{Float64,Float64}}()
    for alpha in (1.,.5)
        cfg=PBCav.PBConfig(mu=1e-6,dt=.5,nonorth=1.,turbulence=:sst,
            sst_tu=.01,sst_mut_ratio=10.,turb_mut_base=:committed,
            turb_step_tol=1e-9,turb_step_maxit=200,turb_step_relax=alpha)
        bc=PBCav.bcset(mesh;kinds=Dict{UInt8,PBCav.BCKind}())
        st=PBCav.allocate_state(mesh);PBCav.init_state!(st,mesh,(x,y)->(1.,0.,0.))
        work=PBCav.pbwork(mesh,bc,cfg);PBCav.init_turbulence!(st,mesh,bc,cfg;work)
        kin,win=PBCav.sst_freestream(1.,.01,10.,cfg.mu)
        r=PBCav.pbcav_run!(st,mesh,bc,cfg;wall_cap=30.,max_steps=1,work)
        @test r.outcome==:REACHED
        @test r.nstep==1
        omega=first(PBCav.lev(st.rw,2))
        kinetic=first(PBCav.lev(st.rk,2))
        # No strain, gradients, walls or transport: backward Euler has these
        # two exact positive scalar roots (outer SST constants, F1=0).
        exact_w=2win/(1+sqrt(1+4cfg.dt*.0828win))
        exact_k=kin/(1+cfg.dt*.09exact_w)
        @test isapprox(omega,exact_w;rtol=1e-7,atol=1e-12)
        @test isapprox(kinetic,exact_k;rtol=1e-7,atol=1e-12)
        @test all(x->isapprox(x,exact_w;rtol=1e-7,atol=1e-12),PBCav.lev(st.rw,2))
        @test all(x->isapprox(x,exact_k;rtol=1e-7,atol=1e-12),PBCav.lev(st.rk,2))
        results[alpha]=(kinetic,omega)
    end
    @test all(isapprox.(results[1.],results[.5];rtol=1e-7,atol=1e-12))
    k,_=PBCav.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_TURB_STEP_RELAX"=>"0.5","PB_TURB_STEP_MAXIT"=>"80"))
    @test PBCav.freeair_config(k).turb_step_relax==.5
    @test PBCav.freeair_config(k).turb_step_maxit==80
    @test_throws ArgumentError PBCav.read_knobs(Dict("PB_TURB_STEP_RELAX"=>"0"))
end
