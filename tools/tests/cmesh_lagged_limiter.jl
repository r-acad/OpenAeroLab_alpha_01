include(joinpath(@__DIR__,"cmesh_thermal_precision.jl"))

function lagged_fixture(;limiter=:venkat,scalar_type=Float64,backend=CPU())
    ni,nj=6,4
    X=[i/ni+.15*(j/nj)^2 for i in 0:ni,j in 0:nj]
    Y=[(j/nj)^2+.05*i/ni for i in 0:ni,j in 0:nj]
    met=CMeshMetrics.build_metrics(X,Y,0,ni;scalar_type)
    cx,cy,cz,qw,_,_=build_d3q39_static(scalar_type)
    s=M.build_sol_ka(backend,met;cxv=cx,cyv=cy,czv=cz,wv=qw,scalar_type)
    M.set_freestream_ka!(s,.3,4.);s.wall_flux=true;s.fullvisc=Int32(1);s.nu_lam=1e-5
    mesh=(;X,Y);affine_state!(s,mesh)
    M.set_airfoil_muscl_ka!(s,X,Y;limiter)
    M.applyBC_ka!(s,Int32(2));s,mesh
end

function prepare_lagged!(s,w)
    for (dst,src) in zip((w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0),(s.rho,s.ux,s.uy,s.T));copyto!(dst,src);end
    M._raw_residual_ka!(s,Int32(2),Int32(1));M._av_copy!(w.base_residual,M._airfoil_residual(s))
    @. w.line.d0=.01*s.vol
    M._airfoil_refresh_lagged_limiter!(s,w.lagged_limiter,w.line)
    w
end

@testset "Lagged component-limiter model is explicit and scope checked" begin
    f32,_=lagged_fixture(;scalar_type=Float32)
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(f32;lagged_component_limiter=true)
    s,mesh=lagged_fixture()
    M.clear_airfoil_muscl_ka!(s)
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(s;lagged_component_limiter=true)
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y)
    s.cav=Int32(4)
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(s;lagged_component_limiter=true)
    s.cav=Int32(0);s.wall_flux=false
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(s;lagged_component_limiter=true)
    s.wall_flux=true
    current=M.build_jfnk_airfoil_workspace(s)
    @test current.lagged_limiter===nothing
    @test M.airfoil_jacobian_contract(current).jacobian_model===:current
    w=M.build_jfnk_airfoil_workspace(s;lagged_component_limiter=true)
    r=w.lagged_limiter
    @test sizeof(r.component_factors)+sum(sizeof,r.reference_residual)+sizeof(r.saved_dt)==9sizeof(Float64)*length(s.vol)
    @test !r.ready&&r.refreshes==0
    @test M.airfoil_jacobian_contract(w).physical_residual_model===:current
    @test_throws ArgumentError M._airfoil_jacobian_vec!(w.work,w.delta,s,w)
    prepare_lagged!(s,w)
    @test r.ready&&r.refreshes==1
    @test all(isequal(a,b) for (a,b) in zip(r.reference_residual,w.base_residual))
    other,_=lagged_fixture()
    saved=map(copy,(other.rho,other.ux,other.uy,other.T))
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(other,w,1f0)
    @test all(isequal(a,b) for (a,b) in zip(saved,(other.rho,other.ux,other.uy,other.T)))
    @test !r.ready
    prepare_lagged!(s,w)
    @test_throws ArgumentError M._airfoil_jacobian_vec!(w.work,w.delta,other,w)
    @test all(isequal(a,b) for (a,b) in zip(saved,(other.rho,other.ux,other.uy,other.T)))
end

@testset "Lagged R1 independent affine trace and dense/two-sided derivatives" begin
    for limiter in (:barth,:venkat)
        s,mesh=lagged_fixture(;limiter)
        w=M.build_jfnk_airfoil_workspace(s;m=8,lagged_component_limiter=true)
        prepare_lagged!(s,w);r=w.lagged_limiter;mw=M._AIRFOIL_MUSCL[s]
        saved=map(copy,(s.rho,s.ux,s.uy,s.T));n=length(s.vol);h=1e-6;i=3;j=3
        face_values=Vector{Float64}[]
        for sign in (-1,1)
            copyto!(s.rho,w.line.rho0)
            for jj in 1:s.NJ,ii in 1:s.NI
                s.rho[ii+s.g,jj+s.g]+=sign*h*(.01+.002*s.xc[ii,jj]-.001*s.yc[ii,jj])
            end
            M._airfoil_muscl_prepare!(s,mw;fixed_component_factors=r.component_factors)
            push!(face_values,[mw.states[i,j,1,f] for f in 1:4])
        end
        copyto!(s.rho,w.line.rho0)
        for f in 1:4
            exact=.01+.002*s.xc[i,j]-.001*s.yc[i,j]+r.component_factors[i,j,1]*(
                .002*mw.geometry.offsets[i,j,1,f]-.001*mw.geometry.offsets[i,j,2,f])
            @test abs((face_values[2][f]-face_values[1][f])/(2h)-exact)<1e-8
        end
        dense=zeros(4n,4n);unit=ntuple(_->zeros(s.NI,s.NJ),4)
        plus=map(similar,unit);minus=map(similar,unit)
        for col in 1:4n
            k=div(col-1,n)+1;cell=mod(col-1,n)+1;unit[k][cell]=1.
            for sign in (-1,1)
                M._k_implicit_line_trial!(CPU())(s.rho,s.ux,s.uy,s.T,
                    w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0,unit...,s.eint,-sign*h,
                    Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
                M._airfoil_lagged_residual_ka!(s,r)
                M._av_copy!(sign==1 ? plus : minus,M._airfoil_residual(s))
            end
            for q in 1:4
                @views dense[(q-1)*n+1:q*n,col].=vec((plus[q]-minus[q])/(2h)./s.vol+w.line.d0.*unit[q]./s.vol)
            end
            unit[k][cell]=0.
        end
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),saved);copyto!(dst,src);end
        v=ntuple(k->reshape([.01sin(.17*(q+(k-1)*n)) for q in 1:n],s.NI,s.NJ),4)
        reference=dense*vcat(vec.(v)...)
        for scheme in (:forward,:central),scale in (1f0,.1f0)
            M._airfoil_jacobian_vec!(w.work,v,s,w;fd_scheme=scheme,fd_scale=scale)
            @test norm(vcat(vec.(w.work)...)-reference)/norm(reference)<2e-5
        end
        @test all(isequal(a,b) for (a,b) in zip(saved,(s.rho,s.ux,s.uy,s.T)))
        @test r.ready
        @test_throws ArgumentError M._airfoil_muscl_prepare!(s,mw;fixed_component_factors=zeros(Float32,s.NI,s.NJ,4))
        @test_throws ArgumentError M._airfoil_muscl_prepare!(s,mw;fixed_component_factors=zeros(Float64,s.NI,s.NJ,3))
    end
end

@testset "Lagged R1 refresh, physical R0 merit and error rollback" begin
    for linear_solver in (:native,:krylov)
        s,_=lagged_fixture()
        controller=M.build_jfnk_airfoil_controller(s;m=8,linear_solver,
            initial_cfl=1f0,max_cfl=2f0,globalization=:pseudo_transient,lagged_component_limiter=true)
        w=controller.workspace;r=w.lagged_limiter
        saved=map(copy,(s.rho,s.ux,s.uy,s.T,s.dtf));clock=s.last_global_dt
        @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=0)
        @test all(isequal(a,b) for (a,b) in zip(saved,(s.rho,s.ux,s.uy,s.T,s.dtf)))
        @test s.last_global_dt==clock&&!r.ready
        count=r.refreshes
        result=M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=16,
            krylov_workspace=controller.krylov_workspace)
        @test result.jacobian_model===:lagged_component_limiter
        @test result.physical_residual_model===result.physical_merit_model===:current
        @test result.jacobian_refreshes==count+1&&!r.ready
        @test isfinite(result.linear_relative)&&result.linear_iterations<=16
        @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
        if result.accepted
            @test result.acceptance_merit_after<result.merit_before
            @test result.merit_after<2result.merit_before
        end
        # Ending a step invalidates the old linearization, even after rejection.
        current=map(copy,(s.rho,s.ux,s.uy,s.T))
        @test_throws ArgumentError M._airfoil_jacobian_vec!(w.work,w.delta,s,w)
        @test !r.ready
        @test all(isequal(a,b) for (a,b) in zip(current,(s.rho,s.ux,s.uy,s.T)))
    end
end

@testset "Physical R0 ignores lagged cache and invalid storage preserves state" begin
    s,_=lagged_fixture()
    w=M.build_jfnk_airfoil_workspace(s;lagged_component_limiter=true)
    prepare_lagged!(s,w);r=w.lagged_limiter
    M._raw_residual_ka!(s,Int32(2),Int32(1));physical=map(copy,M._airfoil_residual(s))
    factors=copy(r.component_factors);fill!(r.component_factors,0.)
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    @test all(isequal(a,b) for (a,b) in zip(physical,M._airfoil_residual(s)))
    copyto!(r.component_factors,factors)
    for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),(w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0));copyto!(dst,src);end
    saved=map(copy,(s.rho,s.ux,s.uy,s.T,s.dtf));clock=s.last_global_dt
    for k in 1:4;fill!(w.delta[k],.01k);end
    good_factors=r.component_factors
    r.component_factors=zeros(Float64,s.NI,s.NJ,3)
    @test_throws ArgumentError M._airfoil_jacobian_vec!(w.work,w.delta,s,w)
    @test all(isequal(a,b) for (a,b) in zip(saved,(s.rho,s.ux,s.uy,s.T,s.dtf)))
    @test s.last_global_dt==clock&&!r.ready
    @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,w,1f0)
    @test all(isequal(a,b) for (a,b) in zip(saved,(s.rho,s.ux,s.uy,s.T,s.dtf)))
    r.component_factors=good_factors
    prepare_lagged!(s,w)
    @test r.ready
end


# No public injection keyword is required.
struct ThrowingFlowTransactionKrylov end
const flow_transaction_precondition_calls=Ref(0)
function M._airfoil_krylov_gmres!(s::M.KSol,w::M.JFNKAirfoilWorkspace,
        ::ThrowingFlowTransactionKrylov;kwargs...)
    M._airfoil_line_precondition!(w.delta,w.rhs,s,w)
    flow_transaction_precondition_calls[]+=1
    fill!(s.rho,2.);fill!(s.ux,.7);fill!(s.uy,-.3);fill!(s.T,1.2)
    fill!(s.dtf,.123);s.last_global_dt=.456
    error("injected flow transaction failure")
end
flow_record(s)=(;fields=map(n->Array(getproperty(s,n)),M._AIRFOIL_COUPLED_STATE),clock=s.last_global_dt)
flow_same(s,b)=all(isequal(Array(a),x) for (a,x) in zip(
    map(n->getproperty(s,n),M._AIRFOIL_COUPLED_STATE),b.fields))&&isequal(s.last_global_dt,b.clock)

@testset "Flow exceptions restore current and R1 transactions" begin
    for (lagged,pc,T) in ((false,:line,Float32),(false,:line,Float64),(false,:kinetic_muscl_ls_fullvisc_block_adi,Float64),(true,:line,Float64))
        s,_=lagged_fixture(;scalar_type=T)
        w=M.build_jfnk_airfoil_workspace(s;m=2,preconditioner=pc,
            block_scalar_type=Float64,lagged_component_limiter=lagged)
        fill!(s.dtf,.00234);s.last_global_dt=.00123
        before=flow_record(s);calls=flow_transaction_precondition_calls[]
        caught=try
            M.step_jfnk_airfoil_ka!(s,w,.45f0;krylov_workspace=ThrowingFlowTransactionKrylov())
            nothing
        catch err
            err
        end
        @test caught isa ErrorException
        @test occursin("injected flow transaction failure",sprint(showerror,caught))
        @test flow_transaction_precondition_calls[]==calls+1
        @test flow_same(s,before)
        lagged&&(@test !w.lagged_limiter.ready)
        M.clear_airfoil_muscl_ka!(s)
    end
end

# Build malformed SAME-TYPE storage without mutating the valid fixture.
replace_field(x,name,value)=typeof(x)((f===name ? value : getfield(x,f) for f in fieldnames(typeof(x)))...)
@testset "Wrong backup shapes reject before copying or inner dispatch" begin
    s,_=lagged_fixture();w=M.build_jfnk_airfoil_workspace(s;m=2)
    before=flow_record(s);calls=flow_transaction_precondition_calls[]
    for field in (:rho0,:ux0,:uy0,:T0,:dr)
        old=getproperty(w.line,field)
        badline=replace_field(w.line,field,similar(old,1,1))
        bad=replace_field(w,:line,badline)
        @test_throws ArgumentError M.step_jfnk_airfoil_ka!(s,bad,.45f0;
            krylov_workspace=ThrowingFlowTransactionKrylov())
        @test flow_same(s,before)
        @test flow_transaction_precondition_calls[]==calls
    end
    M.clear_airfoil_muscl_ka!(s)
end


@testset "Flow transaction preserves successful current and R1 results" begin
    for lagged in (false,true)
        s,_=lagged_fixture();control,_=lagged_fixture()
        w=M.build_jfnk_airfoil_workspace(s;m=6,lagged_component_limiter=lagged)
        wc=M.build_jfnk_airfoil_workspace(control;m=6,lagged_component_limiter=lagged)
        expected=M._step_jfnk_airfoil_impl!(control,wc,.45f0)
        actual=M.step_jfnk_airfoil_ka!(s,w,.45f0)
        @test all(isequal(getproperty(actual,k),getproperty(expected,k)) for k in keys(expected))
        @test flow_same(s,flow_record(control))
        if lagged
            @test actual.jacobian_model===:lagged_component_limiter
            @test !actual.frozen_base_ready&&!w.lagged_limiter.ready
        else
            @test keys(actual)==keys(expected)
        end
        M.clear_airfoil_muscl_ka!(s);M.clear_airfoil_muscl_ka!(control)
    end
end
