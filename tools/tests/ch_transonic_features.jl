# julia --startup-file=no --project=src tools/tests/ch_transonic_features.jl
# The options of the compressible two-block (C+H) gas/SST pair (apps/airfoil_web/src/runtime): physical time, prescribed
# trips and the viscosity laws, stepped on a small marched mesh of a blunt section. CPU. Development checks, not validation.
using Test,KernelAbstractions,SHA
const LBM_F=Float32
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"src","collision","lattice.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_gen.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_metrics.jl"))
include(joinpath(ROOT,"src","compressible","cmesh_solver_ka.jl"))
for name in ("gas_strip_prototype.jl","gas_strip_sst_prototype.jl","gas_strip_sst_steady.jl")
    include(joinpath(ROOT,"apps","airfoil_web","src","runtime",name))
end
const G=CMeshSolverKA;const P=GasStripPrototype;const S=GasStripSSTPrototype;const L=GasStripSSTSteady

"NACA 0012 truncated at 95 % chord (a blunt base of 1.7 % chord), unit chord."
function foil(;cut=0.95,n=40)
    yt(x)=5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    xs=[cut*.5*(1-cospi(k/n)) for k in 0:n];y=[yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end])./cut,vcat(-reverse(y),y[2:end])./cut)
end
mesh(;n_normal=24,dn1=2e-4)=CMeshGen.marched_airfoil_mesh(foil();n_foil=64,n_wake=16,n_normal,dn1,alpha_deg=0.,far=15.,wake_aspect=100.).mesh
const MESH=mesh()

"""The pair as the C+H worker builds it. `trips`: (upper, lower) chord stations, a station ahead of the leading edge
(-1) for a side without a trip; `nothing` keeps the metrics' default stations. `law`: :kinematic, :sutherland, :dynamic."""
function pair(;m=MESH,law=:kinematic,trips=nothing,intermittency=false,mach=.5,re=2e5,irsj=false)
    base=(m.strip.X[1,1],m.strip.Y[1,1],m.strip.X[1,end],m.strip.Y[1,end])
    kinds=fill(UInt8(0),m.NI)
    mc=trips===nothing ? CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,kinds;extra_segments=[base]) :
        CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,kinds;extra_segments=[base],trip_upper=trips[1],trip_lower=trips[2],trip_side=:wall_order)
    mh=CMeshMetrics._build_metrics(m.strip.X,m.strip.Y,m.NIW,0,fill(UInt8(4),m.strip.NI);channel_provenance=true,wall_segments=[base])
    cx,cy,cz,w,_,_=build_d3q39_static()
    build(met)=G.build_sol_ka(CPU(),met;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    c,h=build(mc),build(mh);G.GLOBAL_DT[]=0
    for s in (c,h)
        G.set_freestream_ka!(s,mach,2.;pg_correction=false)
        s.fullvisc=Int32(1);s.nu_lam=Float32(hypot(s.uxi,s.uyi)/re);s.recon=Int32(1);s.wall_flux=true
        law===:sutherland && G.set_air_sutherland_ka!(s,271.)
        law===:dynamic && G.set_air_constant_dynamic_ka!(s)
        irsj && (s.irsj=Int32(1))
    end
    S.build_coupled(P.build_pair(c,h);intermittency)
end
function march!(p,n;cfl=.1)
    pilot=L.build_steady(p)
    for _ in 1:n;L.step!(pilot,cfl;max_pseudo_dt=.01,clock_margin=.9);end
    p
end
state(p)=[Array(a) for q in (p.c,p.h) for a in (q.solver.rho,q.solver.ux,q.solver.uy,q.solver.T,q.turbulence.rk,q.turbulence.rw,q.solver.nut)]
residuals(p)=[Array(a) for q in (p.c,p.h) for a in (P.residuals(q.solver)...,q.turbulence.dck,q.turbulence.dcw,q.turbulence.sk,q.turbulence.sw,q.source_dt,q.solver.nut)]
function closed(p)
    gas=P.balance(p.flow);scalar=S.scalar_balance(p)
    gas.interface.exact && all(e->abs(e.face64_telescoping_error)<1e-7,gas.equations) && all(e->abs(e.face64_telescoping_error)<1e-7,scalar)
end
"The SST interface flux is one value for both blocks."
function scalar_interface_common(p)
    c,h=p.c.solver,p.h.solver
    all(((a,b),)->begin
        ca=Array(a)[:,1];hb=Array(b)[:,1];ht=Array(b)[:,h.NJ+1]
        all(ca[c.NIW+1-k]==-hb[k] && ca[c.NIW+c.NIF+k]==ht[k] for k in 1:c.NIW)
    end,((p.c.turbulence.Fjk,p.h.turbulence.Fjk),(p.c.turbulence.Fjw,p.h.turbulence.Fjw)))
end

# The SST transport of one block computed twice from the same prepared state, into scratch arrays: by the species kernels
# as the adapter calls them with constant kinematic viscosity, and by the adapter's own kernels with the block's law.
function species_transport(q)
    s=q.solver;a=q.turbulence;z(x)=fill!(similar(x),NaN32);ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    nut=copy(s.nut);sigk,sigw,sk,sw,rate,status=z(a.sigk),z(a.sigw),z(a.sk),z(a.sw),z(a.rate),z(q.status)
    S._k_sst_species_coeff!(s.backend)(nut,sigk,sigw,sk,sw,rate,status,s.rho,q.tem.alpha,q.pressure,s.tk,s.tom,
        s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,0f0,Int32(0),Int32(0),1f0,0f0,
        (0f0,0f0,0f0,0f0),ni,nj,g,Int32(s.NIW),Int32(s.NIF),Int32(0),Int32(0),q.ctx;ndrange=(s.NI,s.NJ))
    Fik,Fiw,Fjk,Fjw=z(a.Fik),z(a.Fiw),z(a.Fjk),z(a.Fjw)
    S._k_sst_species_flux_i!(s.backend)(Fik,Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,s.nu_lam,s.k_inf,s.w_inf,nothing,nothing,nothing,nothing,ni,nj,g,0f0,q.ctx;ndrange=(s.NI+1,s.NJ))
    S._k_sst_species_flux_j!(s.backend)(Fjk,Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,s.nu_lam,s.k_inf,s.w_inf,Int32(2),nothing,nothing,nothing,nothing,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,Int32(0),0f0,q.ctx;ndrange=(s.NI,s.NJ+1))
    dck,dcw,cap=z(a.dck),z(a.dcw),z(q.source_dt)
    S._k_sst_species_residual!(s.backend)(dck,dcw,cap,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
        a.sk,a.sw,a.rate,a.rk,a.rw,nothing,nothing,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,Int32(2),
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,Int32(0),0f0,0.25f0,q.ctx;ndrange=(s.NI,s.NJ))
    map(Array,(nut,sigk,sigw,sk,sw,rate,status,Fik,Fiw,Fjk,Fjw,dck,dcw,cap))
end
function gas_transport(q;trip=false)
    s=q.solver;a=q.turbulence;z(x)=fill!(similar(x),NaN32);ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);l=S.viscosity_law(s)
    nut=copy(s.nut);sigk,sigw,sk,sw,rate,status=z(a.sigk),z(a.sigw),z(a.sk),z(a.sw),z(a.rate),z(q.status)
    S.gas_coeff!(s.backend)(nut,sigk,sigw,sk,sw,rate,status,s.rho,s.T,s.tk,s.tom,s.ux,s.uy,s.wdist,s.gammat,
        s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,l,Int32(trip),ni,nj,g,q.ctx;ndrange=(s.NI,s.NJ))
    Fik,Fiw,Fjk,Fjw=z(a.Fik),z(a.Fiw),z(a.Fjk),z(a.Fjw)
    S.gas_flux_i!(s.backend)(Fik,Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.T,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,l,s.k_inf,s.w_inf,ni,nj,g,q.ctx;ndrange=(s.NI+1,s.NJ))
    S.gas_flux_j!(s.backend)(Fjk,Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.T,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,l,s.k_inf,s.w_inf,ni,nj,Int32(s.NIW),Int32(s.NIF),g,q.ctx;ndrange=(s.NI,s.NJ+1))
    dck,dcw,cap=z(a.dck),z(a.dcw),z(q.source_dt)
    S.gas_scalar_residual!(s.backend)(dck,dcw,cap,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
        a.sk,a.sw,a.rate,s.rho,s.T,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,l,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,q.ctx;ndrange=(s.NI,s.NJ))
    map(Array,(nut,sigk,sigw,sk,sw,rate,status,Fik,Fiw,Fjk,Fjw,dck,dcw,cap))
end
"The omega ghost values after the adapter's wall kernel rewrote them with the block's law."
function wall_omega(q)
    s=q.solver;tom=copy(s.tom)
    S.gas_wall_omega!(s.backend)(tom,s.rho,s.T,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,S.viscosity_law(s),
        Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),q.ctx.block;ndrange=(max(s.NI,s.NJ),s.g))
    Array(tom)
end
"mu(T)/mu_inf of Sutherland's law with the temperature ratio Sr = S/T_inf, in Float64."
sutherland(t,sr)=t^1.5*(1+sr)/(t+sr)

@testset "Constant kinematic viscosity without a trip is the previous path" begin
    p=march!(pair(),40)
    @test all(q->!S.variable_viscosity(q) && !q.intermittency,(p.c,p.h))
    # The adapter's own kernels with law 0 and unit intermittency reproduce the species kernels bit for bit on a
    # developed state of both blocks: they are the same discretisation.
    for q in (p.c,p.h)
        reference=species_transport(q)
        @test all(isequal.(reference,gas_transport(q)))
        @test all(isequal.(reference,gas_transport(q;trip=true))) == all(==(1f0),Array(q.solver.gammat))
        @test wall_omega(q)==Array(q.solver.tom)
    end
    @test closed(p)&&scalar_interface_common(p)
end

@testset "Viscosity laws" begin
    # the law itself
    kin,suth,dyn=pair(),pair(law=:sutherland),pair(law=:dynamic)
    sr=110.4/271.;nu=Float64(kin.c.solver.nu_lam)
    @test suth.c.solver.suth==Float32(sr)&&suth.h.solver.air_sutherland==1&&dyn.h.solver.air_sutherland==2&&dyn.c.solver.suth==0
    for (r,t) in ((1f0,1f0),(.8f0,1.2f0),(1.3f0,.85f0),(.6f0,1.6f0))
        @test S.nu_at(r,t,S.viscosity_law(kin.c.solver))===kin.c.solver.nu_lam
        @test S.nu_at(r,t,S.viscosity_law(suth.c.solver))≈nu*sutherland(Float64(t),sr)/r rtol=2e-6
        @test S.nu_at(r,t,S.viscosity_law(dyn.c.solver))≈nu/r rtol=2e-6
    end
    @test S.nu_at(1f0,1f0,S.viscosity_law(suth.c.solver))===kin.c.solver.nu_lam   # exact at the freestream state
    # On the uniform freestream every law is the constant: the six residuals, sources, clocks and eddy viscosity of
    # both blocks are those of constant kinematic viscosity bit for bit.
    @test residuals(suth)==residuals(kin)
    @test residuals(dyn)==residuals(kin)
    # A manufactured uniform state away from the freestream (rho = 0.8, T = 1.2): the molecular part of a face
    # conductance and the wall value of omega carry mu(T)/rho.
    for (p,factor) in ((suth,sutherland(1.2,sr)/.8),(dyn,1/.8))
        for q in (p.c,p.h,kin.c,kin.h);fill!(q.solver.rho,.8f0);fill!(q.solver.T,1.2f0);end
        for (q,k) in ((p.c,kin.c),(p.h,kin.h))
            s=q.solver;a=q.turbulence;g=s.g;i=s.NI÷2;j=s.NJ÷2
            args=(s.rho,s.T,s.nut,a.sigk,a.sigw,s.xc,s.yc)
            face=.8*sqrt(Float64(s.Six[i,j])^2+Float64(s.Siy[i,j])^2)/hypot(Float64(s.xc[i,j]-s.xc[i-1,j]),Float64(s.yc[i,j]-s.yc[i-1,j]))
            ci=S.gas_i_diff(args...,s.Six,s.Siy,S.viscosity_law(s),i,j,s.NI,g,q.ctx)
            ck=S.gas_i_diff(args...,s.Six,s.Siy,S.viscosity_law(k.solver),i,j,s.NI,g,q.ctx)
            @test (ci[1]-ck[1])/face≈nu*(factor-1) rtol=1e-3
            cj=S.gas_j_diff(args...,s.Sjx,s.Sjy,s.vol,S.viscosity_law(s),i,j,s.NI,s.NJ,s.NIW,s.NIF,g,q.ctx)
            cjk=S.gas_j_diff(args...,s.Sjx,s.Sjy,s.vol,S.viscosity_law(k.solver),i,j,s.NI,s.NJ,s.NIW,s.NIF,g,q.ctx)
            facej=.8*sqrt(Float64(s.Sjx[i,j])^2+Float64(s.Sjy[i,j])^2)/hypot(Float64(s.xc[i,j]-s.xc[i,j-1]),Float64(s.yc[i,j]-s.yc[i,j-1]))
            @test (cj[2]-cjk[2])/facej≈nu*(factor-1) rtol=1e-3
        end
        # wall omega: ghost + cell = 2 x 60 nu_w/(0.075 d1^2), on the foil (C block) and on the base (H strip)
        c=p.c.solver;g=c.g;i=c.NIW+c.NIF÷3;tom=wall_omega(p.c)
        d1=.5Float64(c.vol[i,1])/hypot(Float64(c.Sjx[i,1]),Float64(c.Sjy[i,1]))
        @test tom[i+g,g]+tom[i+g,g+1]≈2*60*nu*factor/(.075d1^2) rtol=1e-4
        h=p.h.solver;j=h.NJ÷2;tom=wall_omega(p.h)
        d1=.5Float64(h.vol[1,j])/hypot(Float64(h.Six[1,j]),Float64(h.Siy[1,j]))
        @test tom[g,j+g]+tom[g+1,j+g]≈2*60*nu*factor/(.075d1^2) rtol=1e-4
    end
    # each law marches; the pair stays conservative and the laws give different solutions
    runs=Dict(law=>march!(pair(;law),100) for law in (:kinematic,:sutherland,:dynamic))
    for p in values(runs)
        @test all(a->all(isfinite,a),state(p))
        @test closed(p)&&scalar_interface_common(p)
    end
    @test state(runs[:sutherland])!=state(runs[:kinematic])&&state(runs[:dynamic])!=state(runs[:sutherland])
    @test all(q->S.variable_viscosity(q),(runs[:sutherland].c,runs[:sutherland].h))
    # refusals that remain: IRS-J with a law (its smoothing coefficient has a constant viscosity), and a bare
    # Sutherland ratio without a law (a factor in the gas flux alone, with no SST counterpart)
    P.STEADY_IRSJ[]=true
    @test pair(irsj=true) isa S.CoupledPair
    @test_throws ArgumentError pair(law=:sutherland,irsj=true)
    P.STEADY_IRSJ[]=false
    bare=pair();for s in (bare.c.solver,bare.h.solver);s.suth=.4f0;end
    @test_throws ArgumentError S.residual!(bare)
end

@testset "Prescribed trips" begin
    tripped=pair(trips=(.3,.3),intermittency=true)
    c=tripped.c.solver;gamma=Array(c.gammat)
    foil_columns=c.NIW+1:c.NIW+c.NIF
    laminar=[i for i in foil_columns if gamma[i,1]==0];turbulent=[i for i in foil_columns if gamma[i,1]==1]
    @test !isempty(laminar)&&!isempty(turbulent)&&length(laminar)+length(turbulent)>=c.NIF-4   # at most a short ramp
    @test all(i->Float64(c.xc[i,1])<.3,laminar)&&all(i->Float64(c.xc[i,1])>.3,turbulent)
    @test all(==(1f0),gamma[1:c.NIW,:])&&all(==(1f0),gamma[c.NIW+c.NIF+1:end,:])&&all(==(1f0),Array(tripped.h.solver.gammat))
    @test tripped.c.intermittency&&!tripped.h.intermittency
    # Zero trip: without intermittency the field is not read (the metrics' default stations leave a ramp on the foil),
    # and with it a field of ones gives the fully turbulent result bit for bit.
    plain=march!(pair(),60)
    @test any(<(1f0),Array(plain.c.solver.gammat))
    @test state(march!(pair(trips=(-1.,-1.)),60))==state(plain)
    @test state(march!(pair(trips=(-1.,-1.),intermittency=true),60))==state(plain)
    # The same developed state in both models: behind the trip the SST sources are the fully turbulent ones bit for
    # bit; ahead of it the k source is the destruction alone, where the fully turbulent model produces k.
    S.restore!(tripped,S.backup(plain),(0f0,0f0));S.residual!(tripped);S.residual!(plain)
    sk=Array(tripped.c.turbulence.sk);full=Array(plain.c.turbulence.sk)
    @test sk[turbulent,:]==full[turbulent,:]&&Array(tripped.c.turbulence.sw)[turbulent,:]==Array(plain.c.turbulence.sw)[turbulent,:]
    @test Array(tripped.h.turbulence.sk)==Array(plain.h.turbulence.sk)
    g=c.g;rho=Array(c.rho)[g+1:g+c.NI,g+1:g+c.NJ];k=Array(c.tk)[g+1:g+c.NI,g+1:g+c.NJ];w=Array(c.tom)[g+1:g+c.NI,g+1:g+c.NJ]
    @test sk[laminar,:]==((0f0 .-.09f0 .*w.*k).*(rho.*Array(c.vol)))[laminar,:]
    @test maximum(full[laminar,:])>0&&maximum(sk[laminar,:])<0
    @test closed(tripped)&&scalar_interface_common(tripped)
    # Development (a finer wall spacing, Re 1e6): ahead of the trip the eddy viscosity of the tripped run stays at or
    # below its freestream value, where the fully turbulent run has grown it.
    fine=mesh(n_normal=32,dn1=5e-5)
    t=march!(pair(m=fine,trips=(.3,.3),intermittency=true,mach=.3,re=1e6),2000;cfl=.2)
    f=march!(pair(m=fine,mach=.3,re=1e6),2000;cfl=.2)
    s=t.c.solver;ahead=[i for i in s.NIW+1:s.NIW+s.NIF if s.gammat[i,1]==0]
    ratio(p)=maximum(Array(p.c.solver.nut)[ahead.+s.g,s.g+1:s.g+s.NJ])/s.nu_lam
    @test ratio(t)<=.1001                    # the freestream ratio mu_t/mu is 0.1
    @test ratio(f)>1.5ratio(t)
    @test closed(t)
end

@testset "Physical time" begin
    for options in ((;),(law=:sutherland,trips=(.3,.3),intermittency=true))
        p=pair(;options...)
        @test p.elapsed==0&&p.accepted_steps==0
        # The boundary flux of every stage with the SSP-RK3 weights closes the inventory change of the step.
        weights=(1/6,1/6,2/3);boundary=zeros(6)
        observe=(q,stage,phase)->begin
            phase===:residual || return
            gas=P.balance(q.flow);scalar=S.scalar_balance(q)
            for e in 1:4;boundary[e]+=weights[stage]*gas.equations[e].external_flux;end
            for e in 1:2;boundary[4+e]+=weights[stage]*(scalar[e].boundary-scalar[e].source);end
        end
        total=0.;mass=0.;energy=0.;tke=0.
        for n in 1:30
            boundary.=0;before=S.inventory(p)
            r=S.step!(p,.1;stage_observer=observe);total+=r.dt
            after=S.inventory(p)
            mass=max(mass,abs(after[1]-before[1]+r.dt*boundary[1])/before[1])
            energy=max(energy,abs(after[4]-before[4]+r.dt*boundary[4])/before[4])
            tke=max(tke,abs(after[5]-before[5]+r.dt*boundary[5]-r.weighted_k_floor)/before[5])
            # one clock: the same timestep in every cell of both blocks
            @test all(==(Float32(r.dt)),Array(p.c.solver.dtf))&&all(==(Float32(r.dt)),Array(p.h.solver.dtf))
            @test p.c.solver.last_global_dt==p.h.solver.last_global_dt==Float32(r.dt)
        end
        @test total>0&&p.elapsed==total&&p.flow.elapsed==total&&p.accepted_steps==30&&p.flow.iterations==30
        @test mass<1e-10&&energy<1e-10        # Float64 inventories of the Float32 state: round-off
        @test tke<1e-6                        # the SST trial is formed in Float32
        @test closed(p)&&scalar_interface_common(p)
        # a cap lands the clock on a requested time
        cap=S.step!(p,.1).dt/3;before=p.elapsed
        r=S.step!(p,.1;max_dt=cap)
        @test r.dt<=cap&&r.dt>.999cap&&p.elapsed==before+r.dt
        @test all(a->all(isfinite,a),state(p))
    end
    # IRS-J is a steady option: the physical step refuses it and leaves the clock where it was
    P.STEADY_IRSJ[]=true
    smoothed=pair(irsj=true)
    @test_throws ArgumentError S.step!(smoothed,.1)
    @test smoothed.elapsed==0
    P.STEADY_IRSJ[]=false
end
