using Test,LinearAlgebra,KernelAbstractions
include(joinpath(@__DIR__,"..","..","solvers","pbcav_s1030","src","pbcav","PBCav.jl"))
const P=PBCav

function corner_bc(m,U;outlet=P.BC_PRESSURE_OUTLET)
    P.bcset(m;kinds=Dict(tag=>tag==P.BT_E ? outlet : P.BC_FARFIELD for tag in unique(m.btag)),
        values=Dict(tag=>(U[1],U[2],0.) for tag in unique(m.btag)))
end
function fixed_viscosity!(w,cfg)
    fill!(w.muf,cfg.mu*2);fill!(w.mub,cfg.mu*2)
end

@testset "Far-field outflow convects the owner; incoming and generic Dirichlet conditions retain their meaning" begin
    m=P.box_mesh(3,2);u=fill(.6,m.nc);v=fill(-.4,m.nc);z=zeros(m.nc)
    ub=fill(.9,m.nb);vb=fill(.2,m.nb);Fb=fill(.25,m.nb)
    function terms(kind,phi,turbulence;convection=:linear_upwind_unlimited)
        cfg=P.PBConfig(;mu=.001,dt=.1,nonorth=1.,turbulence,convection)
        bc=P.bcset(m;kinds=Dict(tag=>kind for tag in unique(m.btag)))
        d=zeros(m.nb);x=similar(d);y=similar(d)
        P.momentum_bface_terms!(d,x,y,m,bc,cfg,u,v,fill(phi,m.nb),ub,vb,z,z,z,z;mub=fill(cfg.mu,m.nb))
        (;d,x,y)
    end
    for turbulence in (:laminar,:sst)
        far=terms(P.BC_FARFIELD,.25,turbulence)
        upwind=terms(P.BC_FARFIELD,.25,turbulence;convection=:upwind)
        @test far.d==upwind.d && far.x==upwind.x && far.y==upwind.y
        @test far.x.-far.d.*u[m.bown]≈(.001 .*m.ba).*(ub.-u[m.bown])
        @test far.y.-far.d.*v[m.bown]≈(.001 .*m.ba).*(vb.-v[m.bown])
        incoming=terms(P.BC_FARFIELD,-.25,turbulence)
        inlet=terms(P.BC_INLET,-.25,turbulence)
        @test incoming.d==inlet.d && incoming.x==inlet.x && incoming.y==inlet.y
        for kind in (P.BC_INLET,P.BC_MOVING_WALL,P.BC_NOSLIP)
            prescribed=terms(kind,.25,turbulence)
            @test prescribed.d==far.d
            @test prescribed.x≈far.x.-.25 .*(ub.-u[m.bown])
            @test prescribed.y≈far.y.-.25 .*(vb.-v[m.bown])
        end
        for phi in (-.25,.25)
            outlet=terms(P.BC_PRESSURE_OUTLET,phi,turbulence)
            @test outlet.x.-outlet.d.*u[m.bown]≈zeros(m.nb) atol=1e-15
            @test outlet.y.-outlet.d.*v[m.bown]≈zeros(m.nb) atol=1e-15
        end
    end
end

@testset "Uniform flow and frozen coupled boundary Jacobian at the far/outlet corner" begin
    m=P.box_mesh(8,8;walls=(false,false,false,false))
    for turbulence in (:laminar,:sst),U in ((1.,.2),(1.,-.2))
        bc=corner_bc(m,U)
        cfg=P.PBConfig(;mu=1e-4,dt=.1,nonorth=1.,turbulence,coupling=:coupled,cpl_newton=false,cpl_visc_nonorth=:jacobian)
        st=P.allocate_state(m);P.init_state!(st,m,(x,y)->(U[1],U[2],0.));st.dt=cfg.dt
        w=P.pbwork(m,bc,cfg);turbulence===:sst&&fixed_viscosity!(w,cfg)
        @test P.assemble_momentum!(w,m,st,bc,cfg,st.dt)<1e-11
        cw=P.CoupledWork(m,w,cfg);P.cpl_freeze!(cw,w,m,st,bc,cfg,st.dt,true)
        @test maximum(abs,cw.R)<1e-11
        # Nonuniform iterate at fixed convecting flux and frozen viscosity.
        P.lev(st.ux,1).+=.01 .*sin.(3m.xc.+2m.yc)
        P.lev(st.uy,1).+=.01 .*cos.(2m.xc.-3m.yc)
        P.lev(st.p,1).+=.002 .*sin.(m.xc.+m.yc)
        function residual(s)
            ww=P.pbwork(m,bc,cfg);turbulence===:sst&&fixed_viscosity!(ww,cfg)
            P.assemble_momentum!(ww,m,s,bc,cfg,s.dt)
            cc=P.CoupledWork(m,ww,cfg);P.cpl_freeze!(cc,ww,m,s,bc,cfg,s.dt,true)
            copy(cc.R),ww,cc
        end
        R,w,cw=residual(st)
        direction=sin.(collect(1:3m.nc).*1.2345);J=similar(direction)
        P.cpl_apply!(J,direction,cw,w,m,bc,cfg)
        h=1e-6;plus=deepcopy(st);minus=deepcopy(st)
        for (index,field) in enumerate((:ux,:uy,:p))
            delta=P.cfield(direction,index,m.nc)
            P.lev(getfield(plus,field),1).+=h.*delta
            P.lev(getfield(minus,field),1).-=h.*delta
        end
        Rp=first(residual(plus));Rm=first(residual(minus));fd=(Rm.-Rp)./(2h)
        @test norm(fd.-J)/max(norm(J),1.)<2e-7
        # Fixed analytical far-field flux has no increment; pressure outlet does.
        @test all(iszero,cw.Fbl[bc.kind.==UInt8(P.BC_FARFIELD)])
        @test any(!iszero,cw.Fbl[bc.kind.==UInt8(P.BC_PRESSURE_OUTLET)])
    end
end

function march_perturbation(;turbulence=:laminar,periodic=false,outlet=P.BC_PRESSURE_OUTLET,amplitude=-.05,steps=20,dt=.1)
    m=P.box_mesh(8,8;periodic=(periodic,false),walls=(false,false,false,false))
    U=(1.,.2)
    bc=periodic ? P.bcset(m;kinds=Dict(tag=>P.BC_FARFIELD for tag in unique(m.btag)),
        values=Dict(tag=>(U[1],U[2],0.) for tag in unique(m.btag))) : corner_bc(m,U;outlet)
    cfg=P.PBConfig(;mu=1e-5,dt,nonorth=1.,turbulence,coupling=:coupled,n_outer_max=40,
        turb_mut_base=:committed,turb_step_relax=.5,sst_tu=.001,sst_mut_ratio=1.)
    st=P.allocate_state(m)
    # An outgoing shear pulse isolates the missing outflow damping. With a right
    # pressure outlet the localized corner pulse additionally tests pressure coupling.
    pulse(x,y)=y>.875 ? (periodic||x>.75 ? amplitude : 0.) : 0.
    P.init_state!(st,m,(x,y)->(U[1]+pulse(x,y),U[2],0.));st.dt=cfg.dt
    for L in 1:P.NTIME
        P._launch!(P._k_start_bflux!,CPU(),m.nb,P.lev(st.Fb,L),bc.kind,bc.ub,bc.vb,m.bSx,m.bSy)
    end
    w=P.pbwork(m,bc,cfg)
    turbulence===:sst&&P.init_turbulence!(st,m,bc,cfg;work=w)
    initial=sqrt(sum(abs2,P.lev(st.ux,2).-U[1])+sum(abs2,P.lev(st.uy,2).-U[2]))
    errors=Float64[]
    r=P.pbcav_run!(st,m,bc,cfg;wall_cap=60.,max_steps=steps,work=w,
        on_step=(s,rec,ledger)->push!(errors,sqrt(sum(abs2,P.lev(s.ux,2).-U[1])+sum(abs2,P.lev(s.uy,2).-U[2]))))
    (;r,initial,errors)
end

@testset "Free-stream pressure outlet uses known incoming flow without changing generic pressure outlets" begin
    m=P.box_mesh(4,4;walls=(false,false,false,false));kind=UInt8(P.BC_FREESTREAM_OUTLET)
    @test P.is_pressure_bc(kind) && !P.is_velocity_bc(kind) && P.is_inflow_bc(kind)
    u=fill(.4,m.nc);v=fill(-.1,m.nc);ub=fill(.9,m.nb);vb=fill(.2,m.nb);z=zeros(m.nc)
    bc=P.bcset(m;kinds=Dict(tag=>P.BC_FREESTREAM_OUTLET for tag in unique(m.btag)))
    for phi in (-.25,.25)
        Fb=fill(phi,m.nb);uf=zeros(m.nb);vf=similar(uf)
        P._launch!(P._k_bvel!,CPU(),m.nb,uf,vf,u,v,bc.kind,ub,vb,m.bown,m.bSx,m.bSy,m.bSmag,Fb)
        @test uf==(phi<0 ? ub : u[m.bown])
        @test vf==(phi<0 ? vb : v[m.bown])
        for turbulence in (:laminar,:sst)
            cfg=P.PBConfig(;mu=.001,dt=.1,nonorth=1.,turbulence)
            d=zeros(m.nb);x=similar(d);y=similar(d)
            P.momentum_bface_terms!(d,x,y,m,bc,cfg,u,v,Fb,uf,vf,z,z,z,z;mub=fill(cfg.mu,m.nb))
            expected=phi<0 ? -phi.*(ub.-u[m.bown]) : zeros(m.nb)
            @test x.-d.*u[m.bown]≈expected
            expected=phi<0 ? -phi.*(vb.-v[m.bown]) : zeros(m.nb)
            @test y.-d.*v[m.bown]≈expected
        end
        kb=zeros(m.nb);wb=similar(kb);mub=similar(kb)
        P._launch!(P._k_turb_bvals!,CPU(),m.nb,kb,wb,fill(.3,m.nc),fill(4.,m.nc),bc.kind,m.bown,zeros(m.nb),.05,2.,Fb)
        @test kb==fill(phi<0 ? .05 : .3,m.nb)
        @test wb==fill(phi<0 ? 2. : 4.,m.nb)
        P._launch!(P._k_mueff_bface!,CPU(),m.nb,mub,fill(.0005,m.nc),bc.kind,m.bown,.001,.003,Fb)
        @test mub==fill(phi<0 ? .004 : .0015,m.nb)
    end
    fs=P.FreeStream(;alpha=12.);G=.6;P.set_farfield!(bc,m,fs,G)
    Ux,Uy=P.uinf(fs)
    for b in (1,m.nb)
        du,dv=P.point_vortex_velocity(m.bxf[b],m.byf[b],G,fs.xref,fs.yref)
        @test bc.ub[b]≈Ux+du && bc.vb[b]≈Uy+dv
        @test bc.pb[b]≈fs.p_inf-fs.rho*(Ux*du+Uy*dv)-.5fs.rho*(du^2+dv^2)
    end
end

@testset "Incoming pressure-flux Newton derivative on a frozen backflow branch" begin
    m=P.box_mesh(8,8;walls=(false,false,false,false));U=(-1.,.2)
    bc=corner_bc(m,U;outlet=P.BC_FREESTREAM_OUTLET)
    for turbulence in (:laminar,:sst)
        cfg=P.PBConfig(;mu=1e-4,dt=.1,nonorth=1.,turbulence,cpl_newton=true)
        st=P.allocate_state(m);P.init_state!(st,m,(x,y)->(U[1]+.05sin(3x+2y),U[2]+.02cos(x-y),0.));st.dt=cfg.dt
        for L in 1:P.NTIME
            P._launch!(P._k_start_bflux!,CPU(),m.nb,P.lev(st.Fb,L),bc.kind,bc.ub,bc.vb,m.bSx,m.bSy)
        end
        function residual(s)
            w=P.pbwork(m,bc,cfg);turbulence===:sst&&fixed_viscosity!(w,cfg)
            P.assemble_momentum!(w,m,s,bc,cfg,s.dt)
            cw=P.CoupledWork(m,w,cfg);P.cpl_freeze!(cw,w,m,s,bc,cfg,s.dt,true)
            copy(cw.R),w,cw
        end
        R,w,cw=residual(st);east=bc.kind.==UInt8(P.BC_FREESTREAM_OUTLET)
        @test all(<(0),cw.Fbconv[east])
        @test maximum(abs,bc.ub[east].-P.lev(st.ux,1)[m.bown[east]])>.001
        direction=zeros(3m.nc);P.cfield(direction,3,m.nc).=sin.(collect(1:m.nc).*1.234)
        J=similar(direction);P.cpl_apply!(J,direction,cw,w,m,bc,cfg)
        @test maximum(abs,cw.Fbl[east])>1e-6
        h=1e-6;plus=deepcopy(st);minus=deepcopy(st)
        P.lev(plus.p,1).+=h.*P.cfield(direction,3,m.nc)
        P.lev(minus.p,1).-=h.*P.cfield(direction,3,m.nc)
        fd=(first(residual(minus)).-first(residual(plus)))./(2h)
        @test norm(fd.-J)/max(norm(J),1.)<2e-7
    end
end

@testset "Mixed corner backflow recovers; native checkpoint preserves the new boundary kind" begin
    for turbulence in (:laminar,:sst)
        result=march_perturbation(;turbulence,outlet=P.BC_FREESTREAM_OUTLET,amplitude=-1.2,steps=30,dt=.05)
        @test result.r.outcome==:REACHED
        @test length(result.errors)==30
        @test last(result.errors)<.6result.initial
    end
    private=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
    scratch=joinpath(private,"TEMP_DELETEABLE","farfield_restart_tests");mkpath(scratch)
    mktempdir(scratch) do dir
        m=P.box_mesh(4,4;walls=(false,false,false,false));U=(-1.,.2);bc=corner_bc(m,U;outlet=P.BC_FREESTREAM_OUTLET)
        st=P.allocate_state(m);P.init_state!(st,m,(x,y)->(U[1],U[2],0.));st.dt=.02
        k,_=P.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_RE"=>"10000","PB_DT"=>"0.02","PB_VORTEX"=>"none"))
        checkpoint=joinpath(dir,"new_boundary.pbrestart");control=Dict{String,Any}();ledger=P.PBLedger()
        P.write_freeair_checkpoint(checkpoint,m,st,bc,k;control,ledger)
        restored=P.allocate_state(m);otherbc=corner_bc(m,U;outlet=P.BC_FREESTREAM_OUTLET)
        meta=P.read_freeair_checkpoint!(checkpoint,m,restored,otherbc,k;control,ledger)
        @test P.state_sha(restored)==P.state_sha(st)
        @test otherbc.kind==bc.kind && otherbc.ub==bc.ub && otherbc.vb==bc.vb && otherbc.pb==bc.pb
        @test count(P.is_pressure_bc,otherbc.kind)==otherbc.npdir
        cfg=P.freeair_config(k);r=P.pbcav_run!(restored,m,otherbc,cfg;wall_cap=30.,max_steps=1,control,ledger)
        @test r.outcome==:REACHED && restored.nstep==1
        @test maximum(abs,P.lev(restored.ux,2).-U[1])<1e-10
        wrongbc=corner_bc(m,U);before=P.state_sha(restored)
        @test_throws ArgumentError P.read_freeair_checkpoint!(checkpoint,m,restored,wrongbc,k)
        @test P.state_sha(restored)==before
    end
end

@testset "Outgoing shear and mixed far-field/pressure-outlet corner perturbations leave the domain" begin
    for turbulence in (:laminar,:sst),periodic in (true,false)
        result=march_perturbation(;turbulence,periodic)
        @test result.r.outcome==:REACHED
        @test length(result.errors)==20
        @test all(isfinite,result.errors)
        @test last(result.errors)<.6result.initial
        @test maximum(result.errors)<1.1result.initial
        @test isempty(result.r.ledger.rejections)
    end
end

@testset "Off-level assembled wall shear selects the matching pressure-backflow branch" begin
    m=P.box_mesh(2,2;walls=(false,false,false,true))
    kinds=Dict(P.BT_W=>P.BC_FARFIELD,P.BT_E=>P.BC_FREESTREAM_OUTLET,P.BT_S=>P.BC_FARFIELD,P.BT_N=>P.BC_NOSLIP)
    values=Dict(P.BT_W=>(.4,.2,0.),P.BT_E=>(.9,2.,0.),P.BT_S=>(.4,.2,0.),P.BT_N=>(0.,0.,0.))
    bc=P.bcset(m;kinds,values)
    cfg=P.PBConfig(;mu=.001,dt=.1,nonorth=1.,turbulence=:sst)
    st=P.allocate_state(m);P.init_state!(st,m,(x,y)->(.4,.2,0.))
    east=bc.kind.==UInt8(P.BC_FREESTREAM_OUTLET);north=m.btag.==P.BT_N
    for L in 1:P.NTIME;P.lev(st.Fb,L)[north].=0.;end
    P.lev(st.Fb,1)[east].=.25
    P.lev(st.Fb,3)[east].=-.25
    # Matching-level reference: put L3's flux into L1 without changing L3's
    # velocity, then use the same independent assembled-wall diagnostic.
    reference=deepcopy(st);copyto!(P.lev(reference.Fb,1),P.lev(reference.Fb,3))
    actual=P._wall_flux_assembled(st,m,bc,cfg,3,nothing)
    expected=P._wall_flux_assembled(reference,m,bc,cfg,3,nothing)
    wronglevel=P._wall_flux_assembled(st,m,bc,cfg,1,nothing)
    @test maximum(abs,expected[1][north].-wronglevel[1][north])>1e-6
    @test actual[1][north]≈expected[1][north] atol=1e-14
    @test actual[2][north]≈expected[2][north] atol=1e-14
end

@testset "Parity diagnostics count actual exterior momentum and boundary Newton terms" begin
    m=P.box_mesh(2,2;walls=(false,false,false,false));east=m.btag.==P.BT_E
    eowners=zeros(m.nc);for b in findall(east);eowners[m.bown[b]]+=1;end
    cfg=P.PBConfig(;mu=.001,dt=.1,nonorth=1.)
    st=P.allocate_state(m);P.init_state!(st,m,(x,y)->(0.,0.,0.));st.dt=cfg.dt
    P.lev(st.Fb,1)[east].=-.25
    makebc(kind)=P.bcset(m;kinds=Dict(tag=>kind for tag in unique(m.btag)),values=Dict(tag=>(.9,.2,0.) for tag in unique(m.btag)))
    external=makebc(P.BC_FREESTREAM_OUTLET);generic=makebc(P.BC_PRESSURE_OUTLET)
    for (bc,ex,ey) in ((external,.45,.10),(generic,0.,0.))
        w=P.pbwork(m,bc,cfg);P.assemble_momentum!(w,m,st,bc,cfg,st.dt)
        tx,ty,_=P._mom_terms(m,w,st,bc,cfg)
        # Only the eastern incoming face contributes. Each component's scale
        # includes |assembled RHS| and the actual exterior boundary RHS term.
        @test tx≈ex.*eowners atol=1e-14
        @test ty≈ey.*eowners atol=1e-14
    end
    far=makebc(P.BC_FARFIELD);P.lev(st.Fb,1)[east].=.25
    scales=[]
    for convection in (:upwind,:linear_upwind_unlimited)
        cc=P.PBConfig(;mu=.001,dt=.1,nonorth=1.,convection)
        w=P.pbwork(m,far,cc);P.assemble_momentum!(w,m,st,far,cc,st.dt)
        push!(scales,P._mom_terms(m,w,st,far,cc))
    end
    @test scales[1]==scales[2] # no fictitious deferred outflow term
    P.lev(st.Fb,1)[east].=-.25
    w=P.pbwork(m,external,cfg);cw=P.CoupledWork(m,w,cfg);w.cpl=cw
    cw.Fbconv[east].=-.25;cw.ebox[east].=.9;cw.eboy[east].=.2
    Tb=zeros(m.nb);Tb[east].=.5
    nx,ny=P._newton_terms_a1(m,w,st,external,cfg,zeros(m.nf),Tb)
    @test nx≈1.35 .*eowners atol=1e-14
    @test ny≈.30 .*eowners atol=1e-14
    gx,gy=P._newton_terms_a1(m,w,st,generic,cfg,zeros(m.nf),Tb)
    @test all(iszero,gx)&&all(iszero,gy)
    cw.Fbconv[east].=.25
    ox,oy=P._newton_terms_a1(m,w,st,external,cfg,zeros(m.nf),Tb)
    @test all(iszero,ox)&&all(iszero,oy)
    # K12 exposes the new buffers to CPU/device parity checks, rather than
    # silently omitting the actual boundary Newton calculation.
    P.assemble_momentum!(w,m,st,external,cfg,st.dt)
    P.cpl_freeze!(cw,w,m,st,external,cfg,st.dt,true)
    op=last(only(filter(kv->startswith(first(kv),"K12 "),P._parity_ops(cfg))))
    outputs=op(w,st,m,external,cfg,st.dt,(ref=true,k12=:v3))
    names=Set(x.out for x in outputs)
    @test all(name->name in names,("ebox","eboy","dFb0"))
    @test only(filter(x->x.out=="dFb0",outputs)).cls==:R3
end

# Opt in only when the shared test GPU is available. This compares the actual
# new boundary kernels and frozen coupled operator, without running a GPU march.
if get(ENV,"OPENLUDWIG_TEST_CUDA","0")=="1"
    using CUDA
    @testset "New free-stream boundary residual and Jacobian agree between CPU and CUDA" begin
        @test CUDA.functional()
        CUDA.allowscalar(false)
        hostmesh=P.box_mesh(8,8;walls=(false,false,false,false),distort=.03)
        hostbc=corner_bc(hostmesh,(.8,.2);outlet=P.BC_FREESTREAM_OUTLET)
        hoststate=P.allocate_state(hostmesh)
        P.init_state!(hoststate,hostmesh,(x,y)->(.1+.9cos(2pi*y),.2+.03sin(2pi*x),.01sin(x+2y)))
        hoststate.dt=.1
        for L in 1:P.NTIME
            P._launch!(P._k_start_bflux!,CPU(),hostmesh.nb,P.lev(hoststate.Fb,L),hostbc.kind,hostbc.ub,hostbc.vb,hostmesh.bSx,hostmesh.bSy)
        end
        function boundary_operator(backend,turbulence)
            m=P.to_backend(backend,hostmesh);bc=P.to_backend(backend,hostbc);st=P.to_backend(backend,deepcopy(hoststate))
            cfg=P.PBConfig(;mu=1e-4,dt=.1,nonorth=1.,backend,turbulence,cpl_newton=true)
            w=P.pbwork(m,bc,cfg);turbulence===:sst&&fixed_viscosity!(w,cfg)
            P.assemble_momentum!(w,m,st,bc,cfg,st.dt)
            cw=P.CoupledWork(m,w,cfg);P.cpl_freeze!(cw,w,m,st,bc,cfg,st.dt,true)
            direction=P._dev(backend,sin.(collect(1:3m.nc).*1.2345));J=similar(direction)
            P.cpl_apply!(J,direction,cw,w,m,bc,cfg)
            data=Dict{String,Vector{Float64}}("R"=>Array(cw.R),"J"=>Array(J),"Fbconv"=>Array(cw.Fbconv),
                "ebox"=>Array(cw.ebox),"eboy"=>Array(cw.eboy),"dFb0"=>Array(cw.dFb0))
            for name in (:ubf,:vbf,:gux,:guy,:gvx,:gvy,:bfx,:bfy,:bdg,:Fs,:Fbs)
                data[string(name)]=Array(getfield(w,name))
            end
            turbulence===:sst&&(data["muf"]=Array(w.muf);data["mub"]=Array(w.mub))
            data
        end
        east=hostbc.kind.==UInt8(P.BC_FREESTREAM_OUTLET)
        for turbulence in (:laminar,:sst)
            cpu=boundary_operator(CPU(),turbulence);gpu=boundary_operator(CUDA.CUDABackend(),turbulence)
            @test any(<(-1e-3),cpu["Fbconv"][east])&&any(>(1e-3),cpu["Fbconv"][east])
            @test Set(keys(cpu))==Set(keys(gpu))
            for name in keys(cpu)
                # Float64 operation/FMA roundoff budget, many orders below the
                # march's stopping criteria; no convergence tolerance is changed.
                error=maximum(abs,gpu[name].-cpu[name])/max(1.,maximum(abs,cpu[name]))
                @test isfinite(error)&&error<=2048eps(Float64)
            end
        end
    end
end
