using Test,KernelAbstractions
include(joinpath(@__DIR__,"..","..","solvers","pbcav_s1030","src","pbcav","PBCav.jl"))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","pbcav_native_output_tests");mkpath(SCRATCH)

@testset "Portable native fields and checked continuation" begin
    m=PBCav.box_mesh(4,4)
    cfg=PBCav.PBConfig(mu=1e-3,dt=.02,nonorth=1.)
    bc=PBCav.bcset(m;kinds=Dict(t=>PBCav.BC_NOSLIP for t in unique(m.btag)))
    st=PBCav.allocate_state(m)
    PBCav.init_state!(st,m,(x,y)->(.4+x,.1-y,x-y))
    st.t=.25;st.dt=.01;st.dt_prev=.008;st.nstep=12
    k,_=PBCav.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_RE"=>"1000","PB_VORTEX"=>"point"))
    @test k["PB_START_DT"]==.005
    startup,_=PBCav.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_START_DT"=>"0.001"))
    @test startup["PB_START_DT"]==.001
    @test_throws ArgumentError PBCav.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_START_DT"=>"-1"))
    @test_throws ArgumentError PBCav.read_knobs(Dict("PB_DOMAIN"=>"tunnel","PB_START_DT"=>"0.005"))
    control=Dict{String,Any}("streak"=>3,"nrej"=>0,"nstall"=>0,"tally"=>Dict{String,Int}())
    vortex=PBCav.PointVortex(G=.7,hist=[(10,.7)])
    ledger=PBCav.PBLedger();ledger.nsteps=12;ledger.rejections[:linear_failed]=2
    mktempdir(SCRATCH) do dir
        path=joinpath(dir,"state.pbrestart")
        PBCav.write_freeair_checkpoint(path,m,st,bc,k;vortex,control,ledger)
        target=PBCav.allocate_state(m);restored_control=Dict{String,Any}();restored_ledger=PBCav.PBLedger()
        meta=PBCav.read_freeair_checkpoint!(path,m,target,bc,k;control=restored_control,ledger=restored_ledger)
        @test PBCav.state_sha(target)==PBCav.state_sha(st)
        @test restored_control==control
        @test restored_ledger.nsteps==12
        @test restored_ledger.rejections[:linear_failed]==2
        @test meta["circulation"]==.7
        @test only(meta["vortex_history"])==(10,.7)
        changed=copy(k);changed["PB_RE"]=2000.
        before=PBCav.state_sha(target)
        @test_throws ArgumentError PBCav.read_freeair_checkpoint!(path,m,target,bc,changed)
        @test PBCav.state_sha(target)==before
        # Krylov workspace and iteration budgets remain strict restart numerics.
        # Refusal must precede any destination state/BC/controller/ledger mutation.
        for (name,value) in (("PB_CPL_RESTART",60),("PB_CPL_MAXIT",900))
            changed=merge(copy(k),Dict(name=>value))
            untouched=deepcopy(target);otherbc=deepcopy(bc);otherbc.ub .= .456
            othercontrol=Dict{String,Any}("sentinel"=>77);otherledger=PBCav.PBLedger();otherledger.nsteps=77
            before_state=PBCav.state_sha(untouched);before_bc=(copy(otherbc.kind),copy(otherbc.ub),copy(otherbc.vb),copy(otherbc.pb))
            @test_throws ArgumentError PBCav.read_freeair_checkpoint!(path,m,untouched,otherbc,changed;control=othercontrol,ledger=otherledger)
            @test PBCav.state_sha(untouched)==before_state
            @test (otherbc.kind,otherbc.ub,otherbc.vb,otherbc.pb)==before_bc
            @test othercontrol==Dict("sentinel"=>77)
            @test otherledger.nsteps==77
        end
        budget=copy(k);budget["PB_MAX_STEPS"]=100
        @test PBCav.read_freeair_checkpoint!(path,m,target,bc,budget)["nstep"]==12
        @test_throws ArgumentError PBCav.write_freeair_checkpoint(path,m,st,bc,k)
        old_history=joinpath(dir,"loads_history.csv");write(old_history,"retained earlier history\n")
        reuse=merge(copy(k),Dict("PB_RESTART"=>path,"PB_OUT_DIR"=>dir))
        refused=try PBCav.freeair_drive!(m,reuse);nothing catch err;err end
        @test refused isa ArgumentError
        @test occursin("new PB_OUT_DIR",sprint(showerror,refused))
        @test read(old_history,String)=="retained earlier history\n"
        # Each malformed checkpoint must leave every destination untouched,
        # including BCs, time levels, continuation controls and ledger counters.
        header,arrays=PBCav.PBRestartIO.read_pbrestart(path)
        mutations=(
            # Older a58 data must not be relabeled or admitted after promotion.
            (meta,a)->(meta["source_sha256"]="a58ae590af2ca9ed4ab0f18b7c6c680fc61ea14670465581a1a99a189c788913"),
            (meta,a)->(meta["nstep"]=12.5),
            (meta,a)->(meta["dt"]=-1.),
            (meta,a)->(meta["t"]="bad time"),
            (meta,a)->(a["rho"][1]=-1.),
            (meta,a)->(a["rho"][1]=2.),
            (meta,a)->(a["q"][1]=.01),
            (meta,a)->(a["p"][1]=NaN),
            (meta,a)->(a["mut"][1]=-.01),
            (meta,a)->(a["bc_kind"][1]=0xff),
            (meta,a)->(meta["control"]["streak"]=1.5),
            (meta,a)->(meta["control"]["tally"]=Dict("unknown_reason"=>1)),
            (meta,a)->(meta["ledger"]["nsteps"]=12.5),
            (meta,a)->(meta["ledger"]["mass_error"]=NaN),
            (meta,a)->(meta["ledger"]["unknown_counter"]=1),
            (meta,a)->(meta["vortex_values"]=[NaN]),
            (meta,a)->(meta["vortex_steps"]=[20]),
            (meta,a)->(meta["vortex_values"]=Float64[]),
        )
        for (index,mutate!) in enumerate(mutations)
            metadata=deepcopy(header["meta"]);data=deepcopy(arrays);mutate!(metadata,data)
            malformed=joinpath(dir,"invalid_$index.pbrestart")
            PBCav.PBRestartIO.write_pbrestart(malformed,metadata,collect(pairs(data)))
            untouched=PBCav.allocate_state(m)
            PBCav.init_state!(untouched,m,(x,y)->(.2,.3,.4))
            otherbc=PBCav.bcset(m;kinds=Dict(t=>PBCav.BC_NOSLIP for t in unique(m.btag)))
            otherbc.ub .= .123
            before_state=PBCav.state_sha(untouched);before_ub=copy(otherbc.ub)
            othercontrol=Dict{String,Any}("sentinel"=>99)
            otherledger=PBCav.PBLedger();otherledger.nsteps=99
            @test_throws ArgumentError PBCav.read_freeair_checkpoint!(malformed,m,untouched,otherbc,k;control=othercontrol,ledger=otherledger)
            @test PBCav.state_sha(untouched)==before_state
            @test otherbc.ub==before_ub
            @test othercontrol==Dict("sentinel"=>99)
            @test otherledger.nsteps==99
        end
        vtu=joinpath(dir,"field.vtu")
        PBCav.lev(st.ux,1) .+= 100. # The exporter must use committed level 2.
        PBCav.write_freeair_vtu(vtu,m,st,cfg,PBCav.FreeStream())
        text=read(vtu,String)
        @test occursin("NumberOfCells=\"16\"",text)
        for name in ("velocity_u","velocity_v","Cp","eddy_viscosity_ratio")
            @test occursin("Name=\"$name\"",text)
        end
        exported_u=parse.(Float64,split(only(match(r"Name=\"velocity_u\"[^>]*>([^<]*)",text).captures)))
        @test exported_u==collect(PBCav.lev(st.ux,2))
        @test_throws ArgumentError PBCav.write_freeair_vtu(vtu,m,st,cfg,PBCav.FreeStream())
        sst=deepcopy(st);sst.rk .= .001;sst.rw .= 1.;sst.mut .= .0001
        sst_knobs=merge(copy(k),Dict("PB_TURBULENCE"=>:sst))
        sst_path=joinpath(dir,"sst.pbrestart")
        PBCav.write_freeair_checkpoint(sst_path,m,sst,bc,sst_knobs;vortex,control,ledger)
        sst_target=PBCav.allocate_state(m)
        PBCav.read_freeair_checkpoint!(sst_path,m,sst_target,bc,sst_knobs)
        @test PBCav.state_sha(sst_target)==PBCav.state_sha(sst)
        sst_header,sst_arrays=PBCav.PBRestartIO.read_pbrestart(sst_path)
        sst_arrays["rw"][1,1,3]=0. # Previous time levels must also be admissible.
        sst_bad=joinpath(dir,"sst_invalid.pbrestart")
        PBCav.PBRestartIO.write_pbrestart(sst_bad,sst_header["meta"],collect(pairs(sst_arrays)))
        before_sst=PBCav.state_sha(sst_target)
        @test_throws ArgumentError PBCav.read_freeair_checkpoint!(sst_bad,m,sst_target,bc,sst_knobs)
        @test PBCav.state_sha(sst_target)==before_sst
    end
end

@testset "Native provenance remains bound to the loaded solver" begin
    loaded=PBCav.freeair_source_signature()
    # Simulate disk content changing while a long run is in progress. A final
    # checkpoint must continue to identify the code actually loaded by the run.
    @eval PBCav _freeair_disk_source_signature()="changed source on disk"
    @test PBCav.freeair_source_signature()==loaded
end

@testset "Uninterrupted and split native continuation preserve timestep regrowth and vortex history" begin
    m=PBCav.box_mesh(4,4)
    kinds=Dict(t=>PBCav.BC_NOSLIP for t in unique(m.btag));kinds[PBCav.BT_N]=PBCav.BC_MOVING_WALL
    makebc()=PBCav.bcset(m;kinds,values=Dict(PBCav.BT_N=>(.1,0.,0.)))
    cfg=PBCav.PBConfig(mu=.1,dt=.02,nonorth=1.,regrow_after=3,regrow_factor=2.,n_outer_max=30)
    function initial()
        s=PBCav.allocate_state(m);PBCav.init_state!(s,m,(x,y)->(0.,0.,0.));s.dt=.0025;s
    end
    fs=PBCav.FreeStream()
    knobs,_=PBCav.read_knobs(Dict("PB_DOMAIN"=>"freeair","PB_RE"=>"10","PB_VORTEX"=>"point","PB_VORTEX_EVERY"=>"2"))
    whole=initial();wholebc=makebc();wholecontrol=Dict{String,Any}();wholeledger=PBCav.PBLedger()
    wholevortex=PBCav.PointVortex(G=.5,every=2)
    whole_run=PBCav.freeair_run!(whole,m,wholebc,cfg,fs;wall_cap=60.,max_steps=6,vortex=wholevortex,control=wholecontrol,ledger=wholeledger)
    @test whole_run.outcome==:REACHED
    first=initial();firstbc=makebc();firstcontrol=Dict{String,Any}();firstledger=PBCav.PBLedger()
    firstvortex=PBCav.PointVortex(G=.5,every=2)
    first_run=PBCav.freeair_run!(first,m,firstbc,cfg,fs;wall_cap=60.,max_steps=2,vortex=firstvortex,control=firstcontrol,ledger=firstledger)
    @test first_run.outcome==:REACHED
    @test firstcontrol["streak"]==2
    mktempdir(SCRATCH) do dir
        checkpoint=joinpath(dir,"split.pbrestart")
        PBCav.write_freeair_checkpoint(checkpoint,m,first,firstbc,knobs;vortex=firstvortex,control=firstcontrol,ledger=firstledger)
        resumed=PBCav.allocate_state(m);resumedbc=makebc();resumedcontrol=Dict{String,Any}();resumedledger=PBCav.PBLedger()
        meta=PBCav.read_freeair_checkpoint!(checkpoint,m,resumed,resumedbc,knobs;control=resumedcontrol,ledger=resumedledger)
        resumedvortex=PBCav.PointVortex(G=meta["circulation"],every=2,hist=meta["vortex_history"])
        second_run=PBCav.freeair_run!(resumed,m,resumedbc,cfg,fs;wall_cap=60.,max_steps=4,vortex=resumedvortex,control=resumedcontrol,ledger=resumedledger)
        @test second_run.outcome==:REACHED
        # A restart rebuilds PBWork and the multigrid hierarchy. It preserves the
        # continuation state exactly; fresh factorizations may change rounding
        # far below the solve limits, so bound each field at Float64 roundoff.
        differences=Dict(string(f)=>maximum(abs,getfield(whole,f).-getfield(resumed,f)) for f in PBCav.STATE_FIELDS)
        println("Native split maximum absolute field differences: ",differences)
        for f in PBCav.STATE_FIELDS
            @test differences[string(f)]<=1024eps(Float64)*max(1.,maximum(abs,getfield(whole,f)))
        end
        @test (whole.t,whole.dt,whole.dt_prev,whole.nstep)==(resumed.t,resumed.dt,resumed.dt_prev,resumed.nstep)
        @test wholecontrol==resumedcontrol
        @test wholevortex.hist==resumedvortex.hist
        @test wholevortex.G==resumedvortex.G
        @test wholebc.ub==resumedbc.ub && wholebc.vb==resumedbc.vb && wholebc.pb==resumedbc.pb
        for f in filter(!=(:last_rejection),fieldnames(PBCav.PBLedger))
            a=getfield(wholeledger,f);b=getfield(resumedledger,f)
            @test a isa AbstractFloat ? isapprox(a,b;rtol=1024eps(Float64),atol=1024eps(Float64)) : a==b
        end
        function wall_loads(s,bc)
            work=PBCav.pbwork(m,bc,cfg)
            loads=PBCav.LoadsWork(m;walls=1:m.nb)
            PBCav.airfoil_loads!(loads,s,m,bc,work,cfg,fs)
        end
        full_loads=wall_loads(whole,wholebc);split_loads=wall_loads(resumed,resumedbc)
        @test abs(full_loads.CD)>1e-6
        for f in (:CL,:CD,:CM)
            @test isapprox(getfield(full_loads,f),getfield(split_loads,f);rtol=1024eps(Float64),atol=1024eps(Float64))
        end
    end
end
