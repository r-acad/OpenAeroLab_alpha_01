include(joinpath(@__DIR__,"..","..","src","cmesh","test_sst_wall_metric.jl"))

const SST_UPDATED_FIELDS=(:tk,:tom,:nut,:tnl,:exx,:exy,:eyy)

function metric_default_case(mod,T;earsm=false)
    ka=isdefined(mod,:build_sol_ka)
    f=couette_fixture(;scalar_type=T)
    met=CMeshMetrics.build_metrics(Float64.(f.X),Float64.(f.Y),0,f.ni;scalar_type=T)
    cx,cy,cz,w,_,_=build_d3q39_static(T)
    s=ka ? mod.build_sol_ka(CPU(),met;cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type=T) :
        mod.build_sol(CPU(),met;cxv=cx,cyv=cy,czv=cz,wv=w)
    s.wall_flux=true;s.nu_lam=T(1e-4)
    if ka
        mod.set_freestream_ka!(s,.2,0.);mod.set_sst_ka!(s)
    else
        mod.set_freestream!(s,.2,0.);mod.set_sst!(s)
    end
    copyto!(s.ux,f.u);copyto!(s.uy,f.v)
    fill!(s.tk,T(1e-3));fill!(s.tom,one(T));fill!(s.nut,T(1e-4));fill!(s.tnl,T(1e-4))
    fill!(s.gammat,one(T));fill!(s.dtf,T(1e-4));s.earsm=Int32(earsm)
    if ka
        mod.applyBC_ka!(s,Int32(2))
    else
        mod.applyBC!(s;wall=:noslip)
    end
    s
end

function advance_metric!(mod,s,setting)
    if isdefined(mod,:build_sol_ka)
        setting===nothing ? mod.sst_step_ka!(s) : mod.sst_step_ka!(s;corrected_wall_metric=setting)
    else
        setting===nothing ? mod.sst_step!(s) : mod.sst_step!(s;corrected_wall_metric=setting)
    end
end

@testset "External SST and EARSM defaults equal explicit corrected metrics" begin
    for (mod,T) in ((CMeshSolver,Float32),(CMeshSolverKA,Float32),(CMeshSolverKA,Float64)),earsm in (false,true)
        current=metric_default_case(mod,T;earsm);explicit=metric_default_case(mod,T;earsm)
        legacy=metric_default_case(mod,T;earsm)
        @test mod._sst_default_wall_metric(current)
        advance_metric!(mod,current,nothing);advance_metric!(mod,explicit,true);advance_metric!(mod,legacy,false)
        @test all(isequal(getproperty(current,n),getproperty(explicit,n)) for n in SST_UPDATED_FIELDS)
        @test all(all(isfinite,getproperty(current,n)) for n in SST_UPDATED_FIELDS)
        @test any(!isequal(getproperty(current,n),getproperty(legacy,n)) for n in SST_UPDATED_FIELDS)
    end
end

@testset "Unrelated channel, mixture and cavitation defaults stay legacy" begin
    for T in (Float32,Float64)
        for (field,value) in ((:cav,Int32(1)),(:ff_mix,Int32(1)),(:emix,T(1)),(:keps,T(1)),(:outer_mode,Int32(1)),(:outlet_mode,Int32(1)))
            s=metric_default_case(CMeshSolverKA,T);setproperty!(s,field,value)
            @test !CMeshSolverKA._sst_default_wall_metric(s)
        end
        a=metric_default_case(CMeshSolverKA,T);b=metric_default_case(CMeshSolverKA,T)
        a.cav=b.cav=Int32(1)
        advance_metric!(CMeshSolverKA,a,nothing);advance_metric!(CMeshSolverKA,b,false)
        @test all(isequal(getproperty(a,n),getproperty(b,n)) for n in SST_UPDATED_FIELDS)
    end
    s=metric_default_case(CMeshSolver,Float32);s.outer_mode=Int32(1)
    @test !CMeshSolver._sst_default_wall_metric(s)
end

@testset "Default raw diagnostics use the same metric and preserve the state" begin
    for T in (Float32,Float64)
        s=metric_default_case(CMeshSolverKA,T)
        names=CMeshSolverKA._AIRFOIL_COUPLED_STATE
        saved=map(n->copy(getproperty(s,n)),names);clock=s.last_global_dt
        current=CMeshSolverKA.steady_airfoil_diagnostics_ka(s)
        corrected=CMeshSolverKA.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true)
        legacy=CMeshSolverKA.steady_airfoil_diagnostics_ka(s;corrected_wall_metric=false)
        @test isequal(current,corrected)
        @test current.turb_k_equation_l2!=legacy.turb_k_equation_l2
        @test all(isequal(a,getproperty(s,n)) for (a,n) in zip(saved,names))&&s.last_global_dt==clock
    end
end

@testset "RANS coupling and explicit recovery resolve the same metric default" begin
    M=CMeshSolverKA
    for recovery in (false,true)
        a=metric_default_case(M,Float64);b=metric_default_case(M,Float64)
        ca=M.build_jfnk_airfoil_controller(a;m=2,recovery_steps=recovery ? 1 : 0,recovery_limit=1)
        cb=M.build_jfnk_airfoil_controller(b;m=2,recovery_steps=recovery ? 1 : 0,recovery_limit=1)
        M._airfoil_coupled_backup!(a,ca);M._airfoil_coupled_backup!(b,cb)
        ra=recovery ? M._airfoil_explicit_recovery!(a,ca) : M._airfoil_coupled_sst!(a,ca)
        rb=recovery ? M._airfoil_explicit_recovery!(b,cb;corrected_wall_metric=true) :
            M._airfoil_coupled_sst!(b,cb;corrected_wall_metric=true)
        @test ra.accepted&&rb.accepted
        @test all(isequal(getproperty(a,n),getproperty(b,n)) for n in M._AIRFOIL_COUPLED_STATE)
    end
end
