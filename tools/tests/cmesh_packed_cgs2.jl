using Test,KernelAbstractions,StaticArrays,LinearAlgebra,Random,SHA,Serialization
BLAS.set_num_threads(1)
const REPO=normpath(joinpath(@__DIR__,"..",".."));const LBM_F=Float32
include(joinpath(REPO,"src","collision","lattice.jl"));include(joinpath(REPO,"src","cmesh","cmesh_gen.jl"))
include(joinpath(REPO,"src","cmesh","cmesh_metrics.jl"));include(joinpath(REPO,"src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA;const P=M.AirfoilPackedCGS2

function cgs2_fixture(;scalar_type=Float64,backend=CPU(),sst=true)
    ni,nj=6,4;X=[i/ni+.15*(j/nj)^2 for i in 0:ni,j in 0:nj]
    Y=[(j/nj)^2+.05i/ni for i in 0:ni,j in 0:nj]
    met=CMeshMetrics.build_metrics(X,Y,0,ni;scalar_type,trip_upper=0.,trip_lower=0.)
    cx,cy,cz,qw,_,_=build_d3q39_static(scalar_type)
    s=M.build_sol_ka(backend,met;cxv=cx,cyv=cy,czv=cz,wv=qw,scalar_type)
    M.set_freestream_ka!(s,.3,4.);s.wall_flux=true;s.fullvisc=Int32(1);s.nu_lam=1e-5
    sst&&M.set_sst_ka!(s)
    arrays=map(Array,(s.rho,s.ux,s.uy,s.T))
    for j in 1:nj,i in 1:ni
        ia=i+s.g;ja=j+s.g;x=met.xc[i,j];y=met.yc[i,j]
        arrays[1][ia,ja]=1+.01x+.02y;arrays[2][ia,ja]=.2+.03x-.01y
        arrays[3][ia,ja]=.03-.02x+.04y;arrays[4][ia,ja]=1+.02x-.01y
    end
    for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),arrays);copyto!(dst,src);end
    M.set_airfoil_muscl_ka!(s,X,Y;limiter=:venkat);M.applyBC_ka!(s,Int32(2))
    if sst
        M.prepare_dt_ka!(s,.45f0);M.sst_step_ka!(s;corrected_wall_metric=true)
    end
    s,(;X,Y)
end
function cgs2_controller(s;m=8,kwargs...)
    M.build_jfnk_airfoil_controller(s;m,linear_solver=:native_cgs2,
        initial_cfl=.5f0,max_cfl=2f0,max_attempts=1,globalization=:pseudo_transient,
        preconditioner=:kinetic_muscl_ls_fullvisc_block_adi,block_scalar_type=Float64,
        lagged_component_limiter=true,kwargs...)
end
record(s)=(;fields=map(n->Array(getproperty(s,n)),M._AIRFOIL_COUPLED_STATE),clock=s.last_global_dt)
same(s,a)=all(isequal(Array(getproperty(s,n)),x) for(n,x) in zip(M._AIRFOIL_COUPLED_STATE,a.fields))&&isequal(s.last_global_dt,a.clock)
flat(a)=vcat((vec(Array(x)) for x in a)...)
function scatter!(out,x)
    n=length(out[1]);for k in 1:4;copyto!(out[k],reshape(view(x,(k-1)*n+1:k*n),size(out[k])));end;out
end

@testset "Packed CGS2 independent dense QR, Gram and restarted true residual" begin
    rng=MersenneTwister(4711);n=12
    A=Matrix(Diagonal(1 .+range(0.,1.;length=n)))+.12triu(randn(rng,n,n),1)
    b=randn(rng,n);p=1 .+.2rand(rng,n);expected=qr(A)\b
    apply!(out,x)=scatter!(out,A*flat(x));pre!(out,x)=scatter!(out,flat(x)./p)
    for m in (2,12)
        admission=P.memory_admission(3,1,m;available_bytes=8*1024^3,operator_bound_bytes=0)
        w=P.PackedWorkspace(CPU(),Ref(0),3,1,m;admission)
        rhs=ntuple(_->zeros(3,1),4);scatter!(rhs,b)
        r=P.solve!(w,rhs,apply!,pre!;method=:cgs2,rtol=1e-10,maxiter=120)
        @test r.true_relative<=1e-10
        @test norm(flat(w.delta)-expected)/norm(expected)<3e-10
        @test isapprox(r.true_relative,norm(b-A*flat(w.delta))/norm(b);atol=1e-14)
        @test !w.busy[]
        m==2&&(@test length(r.cycles)>1&&r.iterations>m)
        m==12&&(@test P.gram_error(w,r.iterations)<2e-12)
    end
    admission=P.memory_admission(3,1,8;available_bytes=8*1024^3,operator_bound_bytes=0)
    w=P.PackedWorkspace(CPU(),Ref(0),3,1,8;admission)
    Q=Matrix(qr(randn(rng,12,6)).Q)[:,1:6];coeff=randn(rng,5)
    for j in 1:5;scatter!(P.basis_view(w,j),Q[:,j]);end
    scatter!(P.basis_view(w,6),Q[:,1:5]*coeff+1e-9Q[:,6])
    h=P._cgs2!(w,P.basis_view(w,6),5)
    @test norm(h[1:5]+h[6:10]-coeff)<1e-12
    @test abs(sqrt(h[11])-1e-9)<1e-14
    @test norm(transpose(Q[:,1:5])*flat(P.basis_view(w,6)))<1e-10
    zero=ntuple(_->zeros(3,1),4);never!(y,x)=error("zero RHS must not invoke operator")
    @test P.cycle!(w,zero,never!,never!;method=:cgs2).iterations==0
    bad=ntuple(_->ones(3,1),4);bad[1][1]=Inf
    @test_throws ArgumentError P.cycle!(w,bad,apply!,pre!;method=:cgs2)
    @test !w.busy[]&&all(iszero,flat(w.delta))
end

@testset "Optional factory owns requested basis and refuses unsupported scope" begin
    s,mesh=cgs2_fixture();c=cgs2_controller(s;m=8);kw=c.krylov_workspace
    @test kw isa M.PackedCGS2AirfoilWorkspace
    @test c.workspace.m==2&&length(c.workspace.basis)==3&&length(c.workspace.preconditioned)==2
    @test kw.packed.m==8&&size(kw.packed.V)==(24,4,9)&&size(kw.packed.Z)==(24,4,8)
    @test kw.s===s&&kw.w===c.workspace&&kw.packed.owner===s
    @test P.basis_view(kw.packed,1)===P.basis_view(kw.packed,1)
    contract=M.airfoil_linear_solver_contract(c)
    @test contract.linear_solver=="native_cgs2"&&contract.requested_basis==8&&contract.operator_basis==2
    @test contract.state_scalar_type==contract.block_scalar_type=="Float64"
    @test contract.source_sha256==bytes2hex(open(sha256,joinpath(REPO,"src","compressible",contract.source_file)))
    io=IOBuffer();serialize(io,contract);seekstart(io);@test deserialize(io)==contract
    @test contract.packed_admission.estimated_total_bytes<=contract.packed_admission.admission_budget_bytes
    @test contract.existing_backend_payload_bytes>0
    a=zeros(5,4);b=zeros(3,2)
    aliases=(a,view(a,:,1:3),reshape(a,20),a)
    @test M._airfoil_cgs2_backend_bytes(aliases,CPU())==sizeof(a)
    @test M._airfoil_cgs2_backend_bytes((aliases,b),CPU())==sizeof(a)+sizeof(b)
    @test occursin("loaded",contract.source_identity_scope)
    @test_throws ArgumentError cgs2_controller(s;available_bytes=1)
    @test_throws ArgumentError cgs2_controller(s;available_bytes=true)
    @test_throws ArgumentError cgs2_controller(s;recovery_steps=1)
    @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;linear_solver=:native_cgs2)
    for kwargs in ((;block_scalar_type=Float32,lagged_component_limiter=true,
                    preconditioner=:kinetic_muscl_ls_fullvisc_block_adi),
                   (;block_scalar_type=Float64,lagged_component_limiter=false,
                    preconditioner=:kinetic_muscl_ls_fullvisc_block_adi),
                   (;block_scalar_type=Float64,lagged_component_limiter=true,preconditioner=:line))
        @test_throws ArgumentError M.build_jfnk_airfoil_controller(s;linear_solver=:native_cgs2,kwargs...)
    end
    f32,_=cgs2_fixture(;scalar_type=Float32)
    @test_throws ArgumentError cgs2_controller(f32)
    s.fullvisc=Int32(0);@test_throws ArgumentError cgs2_controller(s);s.fullvisc=Int32(1)
    M.clear_airfoil_muscl_ka!(s);@test_throws ArgumentError cgs2_controller(s)
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    default=M.build_jfnk_airfoil_controller(s;m=8)
    explicit=M.build_jfnk_airfoil_controller(s;m=8,linear_solver=:native)
    library=M.build_jfnk_airfoil_controller(s;m=8,linear_solver=:krylov)
    @test default.workspace.m==explicit.workspace.m==library.workspace.m==8
    @test default.krylov_workspace===explicit.krylov_workspace===nothing
    @test library.krylov_workspace isa M.KrylovAirfoilWorkspace
    @test M._airfoil_linear_residual_limit(nothing,:steady,.95,.1)==Inf
    @test M._airfoil_linear_residual_limit(nothing,:pseudo_transient,.95,.1)==.95
    @test M._airfoil_linear_residual_limit(kw,:steady,.95,.1)==.1
    @test M._airfoil_linear_residual_limit(kw,:pseudo_transient,.95,.1)==.1
    wrong,_=cgs2_fixture();saved=record(wrong)
    @test_throws ArgumentError M._airfoil_krylov_gmres!(wrong,c.workspace,kw)
    @test same(wrong,saved)&&!kw.packed.busy[]
    kw.busy[]=true
    @test_throws ArgumentError M._airfoil_krylov_gmres!(s,c.workspace,kw)
    kw.busy[]=false
    for x in c.workspace.delta;fill!(x,3.);end
    for x in kw.packed.delta;fill!(x,4.);end
    for x in c.workspace.rhs;fill!(x,Inf);end
    @test_throws ArgumentError M._airfoil_krylov_gmres!(s,c.workspace,kw)
    @test all(iszero,flat(c.workspace.delta))&&all(iszero,flat(kw.packed.delta))&&!kw.busy[]
    include(joinpath(REPO,"apps","airfoil_web","src","backends","rans_acceleration.jl"))
    receipt=RANSAcceleration.numerical_source_receipt()
    @test receipt["src/compressible/cmesh_steady_packed_cgs2_ka.jl"]==contract.source_sha256
end

@testset "Accepted full physical and SST controller step" begin
    s,_=cgs2_fixture();c=cgs2_controller(s);calls=Ref(0)
    advance!(s;kwargs...)=(calls[]+=1;M.sst_step_ka!(s;kwargs...))
    result=M.step_jfnk_rans_ka!(s,c;linear_tol=.1,linear_maxiter=32,sst_substeps=1,
        corrected_wall_metric=true,_sst_advance! =advance!)
    @test result.accepted&&calls[]==1
    @test result.linear_relative<=.1
    @test result.jacobian_model===:lagged_component_limiter&&result.physical_residual_model===:current
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)&&M._airfoil_valid_sst_state(s)
    @test !c.workspace.lagged_limiter.ready&&!c.krylov_workspace.packed.busy[]
end

@testset "Requested forcing miss rejects before physical/SST with full rollback" begin
    s,_=cgs2_fixture();c=cgs2_controller(s);saved=record(s);calls=Ref(0)
    never_sst!(s;kwargs...)=(calls[]+=1;error("SST must not run after a forcing miss"))
    result=M.step_jfnk_rans_ka!(s,c;linear_tol=1e-12,linear_maxiter=1,sst_substeps=1,
        corrected_wall_metric=true,_sst_advance! =never_sst!)
    @test !result.accepted&&calls[]==0
    @test 1e-12<result.linear_relative<.95
    @test result.physical_trials==0
    @test same(s,saved)&&!c.workspace.lagged_limiter.ready&&!c.krylov_workspace.packed.busy[]
end

@testset "SST invalidity/exception restore all13 fields and clock" begin
    for throws in (false,true)
        s,_=cgs2_fixture();c=cgs2_controller(s);saved=record(s);calls=Ref(0)
        function corrupt!(s;kwargs...)
            calls[]+=1
            for name in M._AIRFOIL_COUPLED_STATE;fill!(getproperty(s,name),-13.);end
            s.last_global_dt=17.
            throws&&error("Injected post-flow SST exception")
            nothing
        end
        if throws
            @test_throws ErrorException M.step_jfnk_rans_ka!(s,c;linear_tol=.1,linear_maxiter=32,sst_substeps=1,
                corrected_wall_metric=true,_sst_advance! =corrupt!)
        else
            result=M.step_jfnk_rans_ka!(s,c;linear_tol=.1,linear_maxiter=32,sst_substeps=1,
                corrected_wall_metric=true,_sst_advance! =corrupt!)
            @test !result.accepted
        end
        @test calls[]==1&&same(s,saved)
        @test !c.workspace.lagged_limiter.ready&&!c.krylov_workspace.packed.busy[]
    end
end
