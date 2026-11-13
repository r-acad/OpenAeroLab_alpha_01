# Internal packed arithmetic for the explicitly selected optional controller.
module AirfoilPackedCGS2
using KernelAbstractions, LinearAlgebra
const KA=KernelAbstractions
const TREE_WG=256
const TREE_BLOCK=2048

# The supported public API is the controller/adapter below. These arithmetic
# helpers remain internal; their callbacks are owned by that serial adapter.

function memory_admission(ni::Int,nj::Int,m::Int;available_bytes::Integer,
                          operator_bound_bytes::Integer,limit_bytes::Integer=2*1024^3)
    ni>0&&nj>0&&2<=m<=96||throw(ArgumentError("Invalid packed dimensions/restart"))
    operator_bound_bytes>=0&&available_bytes>=0||throw(ArgumentError("Invalid memory observation"))
    n=Base.checked_mul(ni,nj);n<=typemax(Int32)||throw(ArgumentError("Packed cell index exceeds Int32"));nb=cld(n,TREE_BLOCK)
    # V/Z plus six four-component vectors. Small host QR storage is counted too.
    payload=8*(4n*(2m+1)+24n+nb*m+nb+2m+2+4(m+1)*m+4m+1)+256*4*(2m+1)
    bound=operator_bound_bytes+cld(5payload,4)
    budget=min(Int(limit_bytes),fld(Int(available_bytes),2))
    bound<=budget||throw(ArgumentError("Packed Krylov total storage refused: $bound > $budget bytes"))
    (;packed_payload_bytes=payload,reserve_factor=1.25,operator_bound_bytes,
      estimated_total_bytes=bound,admission_budget_bytes=budget,restart=m,ni,nj,cells=n,
      scope="Includes the small PUBLIC m2 operator and private V/Z/scratch; excludes already allocated flow state only if operator bound explicitly excludes it")
end

struct PackedWorkspace{B,A,V,P,S,O,BV}
    backend::B
    owner::O
    ni::Int
    nj::Int
    n::Int
    m::Int
    V::A
    Z::A
    Vviews::BV
    Zviews::BV
    delta::V
    residual::V
    original_rhs::V
    full_rhs::V
    total_delta::V
    work::V
    part::P
    normpart::S
    scalars::S
    busy::Base.RefValue{Bool}
end

function PackedWorkspace(backend,owner,ni::Int,nj::Int,m::Int;admission)
    ni>0&&nj>0&&2<=m<=96||throw(ArgumentError("Invalid packed dimensions"))
    admission.restart==m&&admission.ni==ni&&admission.nj==nj||throw(ArgumentError("Admission/dimension mismatch"))
    n=ni*nj;nb=cld(n,TREE_BLOCK)
    vec()=ntuple(_->KA.zeros(backend,Float64,ni,nj),4)
    v=KA.zeros(backend,Float64,n,4,m+1);z=KA.zeros(backend,Float64,n,4,m)
    views(a,count)=[ntuple(k->reshape(view(a,:,k,j),ni,nj),4) for j in 1:count]
    PackedWorkspace(backend,owner,ni,nj,n,m,
        v,z,views(v,m+1),views(z,m),
        vec(),vec(),vec(),vec(),vec(),vec(),KA.zeros(backend,Float64,nb,m),
        KA.zeros(backend,Float64,nb),KA.zeros(backend,Float64,2m+2),Ref(false))
end

basis_view(w::PackedWorkspace,j::Int)=w.Vviews[j]
z_view(w::PackedWorkspace,j::Int)=w.Zviews[j]
copy4!(a,b)=(foreach(k->copyto!(a[k],b[k]),1:4);a)
fill4!(a,x)=(foreach(k->fill!(a[k],Float64(x)),1:4);a)
axpy4!(a,x,b)=(foreach(k->(@. a[k]=a[k]+Float64(x)*b[k]),1:4);a)
scale4!(a,x)=(foreach(k->(@. a[k]=Float64(x)*a[k]),1:4);a)
function dot4(a,b)
    if a[1] isa StridedArray&&b[1] isa StridedArray&&KA.get_backend(a[1]) isa KA.CPU
        total=0.0
        @inbounds @simd for i in eachindex(a[1],b[1])
            total+=Float64(a[1][i])*b[1][i]+Float64(a[2][i])*b[2][i]+
                   Float64(a[3][i])*b[3][i]+Float64(a[4][i])*b[4][i]
        end
        return total
    end
    Float64(sum(a[1].*b[1].+a[2].*b[2].+a[3].*b[3].+a[4].*b[4]))
end
default_ops()=(;copy! =copy4!,fill! =fill4!,axpy! =axpy4!,scale! =scale4!,dot=dot4)

function _check_vector(w,a)
    length(a)==4&&all(x->eltype(x)===Float64&&size(x)==(w.ni,w.nj)&&
        KA.get_backend(x)==w.backend,a)||throw(ArgumentError("Packed prototype requires matching Float64/backend/shape"))
end
function _measure(f,w,profile,times,key)
    if !profile;return f();end
    KA.synchronize(w.backend);start=time_ns()
    answer=f();KA.synchronize(w.backend)
    times[key]=get(times,key,0.)+(time_ns()-start)*1e-9
    answer
end

@kernel function _mdot!(part,@Const(V),@Const(w1),@Const(w2),@Const(w3),@Const(w4),n::Int32,nb::Int32)
    group=@index(Group,Linear);tid=@index(Local,Linear)
    sh=@localmem Float64 (256,)
    block=(group-1)%nb+1;column=(group-1)÷nb+1
    acc=0.0;base=(block-1)*2048
    @inbounds for k in 0:7
        idx=base+k*256+tid
        if idx<=n
            acc+=V[idx,1,column]*w1[idx]+V[idx,2,column]*w2[idx]+
                 V[idx,3,column]*w3[idx]+V[idx,4,column]*w4[idx]
        end
    end
    sh[tid]=acc;@synchronize
    for stage in 1:8
        width=256>>stage
        @inbounds tid<=width&&(sh[tid]+=sh[tid+width])
        @synchronize
    end
    # KA CPU splits execution at barriers. Recompute group-derived locals after
    # the final barrier, as in NativePBCav's checked _k_mdot_tree! implementation.
    @inbounds if tid==1
        block=(group-1)%nb+1;column=(group-1)÷nb+1
        part[block,column]=sh[1]
    end
end

@kernel function _sum_columns!(sc,@Const(part),nb::Int32,offset::Int32)
    column=@index(Group,Linear);tid=@index(Local,Linear)
    sh=@localmem Float64 (256,)
    acc=0.0;k=tid
    @inbounds while k<=nb;acc+=part[k,column];k+=256;end
    sh[tid]=acc;@synchronize
    for stage in 1:8
        width=256>>stage
        @inbounds tid<=width&&(sh[tid]+=sh[tid+width])
        @synchronize
    end
    @inbounds tid==1&&(sc[offset+column]=sh[1])
end

@kernel function _update!(w1,w2,w3,w4,@Const(V),@Const(sc),j::Int32,offset::Int32,n::Int32)
    idx=@index(Global,Linear)
    @inbounds if idx<=n
        a=0.0;b=0.0;c=0.0;d=0.0
        for column in Int32(1):j
            h=sc[offset+column]
            a+=V[idx,1,column]*h;b+=V[idx,2,column]*h
            c+=V[idx,3,column]*h;d+=V[idx,4,column]*h
        end
        w1[idx]-=a;w2[idx]-=b;w3[idx]-=c;w4[idx]-=d
    end
end

@kernel function _norm_partial!(part,@Const(w1),@Const(w2),@Const(w3),@Const(w4),n::Int32)
    block=@index(Group,Linear);tid=@index(Local,Linear)
    sh=@localmem Float64 (256,)
    acc=0.0;base=(block-1)*2048
    @inbounds for k in 0:7
        idx=base+k*256+tid
        if idx<=n;acc+=w1[idx]^2+w2[idx]^2+w3[idx]^2+w4[idx]^2;end
    end
    sh[tid]=acc;@synchronize
    for stage in 1:8
        width=256>>stage
        @inbounds tid<=width&&(sh[tid]+=sh[tid+width])
        @synchronize
    end
    @inbounds tid==1&&(part[block]=sh[1])
end
@kernel function _norm_sum!(sc,@Const(part),nb::Int32,slot::Int32)
    tid=@index(Local,Linear);sh=@localmem Float64 (256,)
    acc=0.0;k=tid
    @inbounds while k<=nb;acc+=part[k];k+=256;end
    sh[tid]=acc;@synchronize
    for stage in 1:8
        width=256>>stage
        @inbounds tid<=width&&(sh[tid]+=sh[tid+width])
        @synchronize
    end
    @inbounds tid==1&&(sc[slot]=sh[1])
end
@kernel function _normalize!(w1,w2,w3,w4,@Const(sc),slot::Int32,n::Int32)
    idx=@index(Global,Linear)
    @inbounds if idx<=n
        squared=sc[slot]
        if isfinite(squared)&&squared>0
            inverse=inv(sqrt(squared))
            w1[idx]*=inverse;w2[idx]*=inverse;w3[idx]*=inverse;w4[idx]*=inverse
        end
    end
end

function _cgs2!(w,v,j;profile=false,times=Dict{Symbol,Float64}())
    be=w.backend;nb=cld(w.n,TREE_BLOCK);n=Int32(w.n)
    for offset in (0,j)
        _measure(w,profile,times,:batched_dot_reductions) do
            _mdot!(be,TREE_WG)(w.part,w.V,v...,n,Int32(nb);ndrange=nb*j*TREE_WG)
            _sum_columns!(be,TREE_WG)(w.scalars,w.part,Int32(nb),Int32(offset);ndrange=j*TREE_WG)
        end
        _measure(w,profile,times,:projection_updates) do
            _update!(be)(v...,w.V,w.scalars,Int32(j),Int32(offset),n;ndrange=w.n)
        end
    end
    _measure(w,profile,times,:norm_reduction) do
        _norm_partial!(be,TREE_WG)(w.normpart,v...,n;ndrange=nb*TREE_WG)
        _norm_sum!(be,TREE_WG)(w.scalars,w.normpart,Int32(nb),Int32(2j+1);ndrange=TREE_WG)
    end
    _measure(w,profile,times,:normalization) do
        _normalize!(be)(v...,w.scalars,Int32(2j+1),n;ndrange=w.n)
    end
    KA.synchronize(be)
    hs=_measure(w,profile,times,:host_coefficient_read) do;Array(view(w.scalars,1:2j+1));end
    all(isfinite,hs)&&hs[end]>=0||throw(ArgumentError("Nonfinite/negative CGS2 projection or norm"))
    hs
end

"One explicit private cycle; timing is instrumented only when profile=true."
function cycle!(w::PackedWorkspace,rhs,apply!,precondition!;method=:mgs,rtol=.1,
                maxiter=w.m,owner=w.owner,ops=default_ops(),profile=false)
    owner===w.owner||throw(ArgumentError("Packed workspace owner mismatch"))
    _check_vector(w,rhs)
    method in (:mgs,:cgs2)&&0<rtol<1&&1<=maxiter<=w.m||throw(ArgumentError("Invalid cycle request"))
    w.busy[]&&throw(ArgumentError("Packed workspace is already in use"))
    w.busy[]=true;times=Dict{Symbol,Float64}();complete=false
    try
        KA.synchronize(w.backend);started=time_ns();ops.fill!(w.delta,0.)
        ops.copy!(w.original_rhs,rhs);beta=sqrt(max(0.,ops.dot(rhs,rhs)))
        isfinite(beta)||throw(ArgumentError("Nonfinite RHS norm"))
        beta==0&&(complete=true;return (;iterations=0,qr_relative=0.,true_relative=0.,cycle_seconds=0.,timings=times,method))
        ops.copy!(basis_view(w,1),rhs);ops.scale!(basis_view(w,1),1/beta)
        H=zeros(maxiter+1,maxiter);rawH=zeros(maxiter+1,maxiter);g=zeros(maxiter+1);g[1]=beta
        cs=zeros(maxiter);sn=zeros(maxiter);rel=Inf;last=0;coefficients=Float64[]
        for j in 1:maxiter
            last=j;v=basis_view(w,j+1);z=z_view(w,j)
            _measure(w,profile,times,:preconditioner) do;precondition!(z,basis_view(w,j));end
            _measure(w,profile,times,:jacobian_action) do;apply!(v,z);end
            _measure(w,profile,times,:orthogonalization_and_host_coefficients) do
                if method===:mgs
                    for i in 1:j
                        H[i,j]=_measure(w,profile,times,:dot_reduction_and_host_scalar) do;ops.dot(v,basis_view(w,i));end
                        _measure(w,profile,times,:projection_updates) do;ops.axpy!(v,-H[i,j],basis_view(w,i));end
                    end
                    H[j+1,j]=_measure(w,profile,times,:norm_reduction_and_host_scalar) do;sqrt(max(0.,ops.dot(v,v)));end
                    isfinite(H[j+1,j])||throw(ArgumentError("Nonfinite MGS Arnoldi vector"))
                    H[j+1,j]>0&&_measure(w,profile,times,:normalization) do;ops.scale!(v,1/H[j+1,j]);end
                else
                    hs=_cgs2!(w,v,j;profile,times)
                    for i in 1:j;H[i,j]=hs[i]+hs[j+i];end
                    H[j+1,j]=sqrt(hs[2j+1])
                end
            end
            rawH[1:j+1,j].=view(H,1:j+1,j)
            qstart=time_ns()
            for i in 1:j-1
                top=cs[i]*H[i,j]+sn[i]*H[i+1,j]
                H[i+1,j]=-sn[i]*H[i,j]+cs[i]*H[i+1,j];H[i,j]=top
            end
            den=hypot(H[j,j],H[j+1,j]);isfinite(den)&&den>0||throw(ArgumentError("Singular Arnoldi column"))
            cs[j]=H[j,j]/den;sn[j]=H[j+1,j]/den;H[j,j]=den;H[j+1,j]=0.
            g[j+1]=-sn[j]*g[j];g[j]=cs[j]*g[j];rel=abs(g[j+1])/beta
            times[:host_QR]=get(times,:host_QR,0.)+(time_ns()-qstart)*1e-9
            if rel<=rtol||j==maxiter
                coefficients=UpperTriangular(view(H,1:j,1:j))\view(g,1:j)
                _measure(w,profile,times,:correction_assembly) do
                    for k in 1:j;ops.axpy!(w.delta,coefficients[k],z_view(w,k));end
                end
                break
            end
        end
        KA.synchronize(w.backend);cycle_seconds=(time_ns()-started)*1e-9
        _measure(w,profile,times,:independent_true_action) do;apply!(w.work,w.delta);end
        ops.copy!(w.residual,w.original_rhs);ops.axpy!(w.residual,-1.,w.work)
        true_relative=sqrt(max(0.,ops.dot(w.residual,w.residual)))/beta
        isfinite(true_relative)||throw(ArgumentError("Nonfinite true residual"))
        complete=true
        (;iterations=last,qr_relative=rel,true_relative,cycle_seconds,timings=times,
          method,coefficients,H=copy(H),hessenberg=copy(rawH),
          scope="Cycle time excludes independent true-action audit; not a physical convergence certificate")
    finally
        !complete&&ops.fill!(w.delta,0.)
        w.busy[]=false
    end
end

"Bounded restarted private solve; always recomputes true ORIGINAL-RHS error."
function solve!(w::PackedWorkspace,rhs,apply!,precondition!;method=:mgs,rtol=.1,
                maxiter=120,owner=w.owner,ops=default_ops(),profile=false)
    owner===w.owner||throw(ArgumentError("Packed workspace owner mismatch"))
    w.busy[]&&throw(ArgumentError("Packed workspace is already in use"))
    maxiter>=1&&maxiter<=10000&&0<rtol<1&&method in (:mgs,:cgs2)||throw(ArgumentError("Invalid restarted solve request"))
    _check_vector(w,rhs);ops.copy!(w.full_rhs,rhs);ops.copy!(w.residual,rhs)
    ops.fill!(w.total_delta,0.);beta=sqrt(max(0.,ops.dot(rhs,rhs)))
    isfinite(beta)||throw(ArgumentError("Nonfinite RHS norm"))
    beta==0&&(ops.fill!(w.delta,0.);return (;iterations=0,true_relative=0.,cycles=Any[],method))
    history=Any[];total=0;relative=1.
    try
        while total<maxiter&&relative>rtol
            current=sqrt(max(0.,ops.dot(w.residual,w.residual)))
            tolerance=min(.999,rtol*beta/current)
            result=cycle!(w,w.residual,apply!,precondition!;method,rtol=tolerance,
                maxiter=min(w.m,maxiter-total),owner,ops,profile)
            total+=result.iterations;ops.axpy!(w.total_delta,1.,w.delta)
            apply!(w.work,w.total_delta);ops.copy!(w.residual,w.full_rhs);ops.axpy!(w.residual,-1.,w.work)
            relative=sqrt(max(0.,ops.dot(w.residual,w.residual)))/beta
            isfinite(relative)||throw(ArgumentError("Nonfinite restarted true residual"))
            push!(history,merge(result,(;total_iterations=total,total_true_relative=relative)))
        end
        ops.copy!(w.delta,w.total_delta)
        (;iterations=total,true_relative=relative,cycles=history,method)
    catch
        ops.fill!(w.delta,0.);rethrow()
    end
end

function gram_error(w::PackedWorkspace,k::Int;ops=default_ops())
    1<=k<=w.m+1||throw(ArgumentError("Invalid basis audit count"))
    maximum(abs(ops.dot(basis_view(w,i),basis_view(w,j))-(i==j)) for i in 1:k,j in 1:k)
end
end # module

using SHA
const _AIRFOIL_PACKED_CGS2_SOURCE_SHA256=bytes2hex(open(sha256,@__FILE__))
export PackedCGS2AirfoilWorkspace,airfoil_cgs2_memory_guard,airfoil_linear_solver_contract

struct PackedCGS2AirfoilWorkspace{S,W,P,R}
    s::S
    w::W
    packed::P
    receipt::R
    busy::Base.RefValue{Bool}
end

function _airfoil_cgs2_validate(s,m,preconditioner,block_scalar_type,lagged_component_limiter)
    2<=m<=96||throw(ArgumentError("CGS2 restart length must be2:96"))
    block_scalar_type===Float64||throw(ArgumentError("Native CGS2 initially requires Float64 blocks"))
    lagged_component_limiter||throw(ArgumentError("Native CGS2 requires explicit lagged component-limiterR1"))
    preconditioner===:kinetic_muscl_ls_fullvisc_block_adi||throw(ArgumentError("Native CGS2 initially requires the MUSCL LS full-viscous block preconditioner"))
    s.fullvisc==1||throw(ArgumentError("Native CGS2 requires fullvisc=1"))
    _airfoil_lagged_limiter_admit(s)
    nothing
end

# Count actual state/MUSCL backend arrays, including aliases only once. This
# requires no CUDA dependency and does not inspect or parse source at runtime.
function _airfoil_cgs2_backend_bytes(roots,backend)
    intervals=Tuple{UInt,UInt}[];seen=Base.IdSet()
    function visit(x)
        x===nothing||x isa Type||x isa Function||x isa Module||x isa AbstractString||x isa Symbol||x isa Number ? (return) : nothing
        x in seen&&return;push!(seen,x)
        if x isa SubArray||x isa Base.ReshapedArray
            visit(parent(x))
        elseif x isa AbstractArray
            if eltype(x)<:Number
                if !isempty(x)&&KA.get_backend(x)==backend
                    lo=UInt(pointer(x));push!(intervals,(lo,lo+sizeof(eltype(x))*length(x)))
                end
            else;foreach(visit,x);end
        elseif x isa Tuple||x isa NamedTuple
            foreach(visit,x)
        elseif x isa AbstractDict
            foreach(visit,values(x))
        elseif isstructtype(typeof(x))&&!isbitstype(typeof(x))&&parentmodule(typeof(x))===@__MODULE__
            foreach(name->visit(getfield(x,name)),fieldnames(typeof(x)))
        end
    end
    visit(roots);sort!(intervals);total=0;lo=UInt(0);hi=UInt(0)
    for (a,b) in intervals
        if a>hi;total+=Int(hi-lo);lo=a;hi=b;else;hi=max(hi,b);end
    end
    total+Int(hi-lo)
end
_airfoil_cgs2_existing_bytes(s)=_airfoil_cgs2_backend_bytes((s,_AIRFOIL_MUSCL[s]),s.backend)

"""Admit the optional Float64 CGS2 state/operator/controller/packed storage.

GPU callers must supply a fresh `available_bytes` observation. The2GiB and
half-free numeric-storage bound includes existing state/MUSCL and32MiB of
transient reserve; it is not a measured CUDA allocator/pool peak guarantee.
"""
function airfoil_cgs2_memory_guard(s::KSol,m::Int;
    preconditioner::Symbol=:kinetic_muscl_ls_fullvisc_block_adi,
    block_scalar_type::Type{<:AbstractFloat}=Float64,
    lagged_component_limiter::Bool=true,available_bytes=nothing)
    _airfoil_cgs2_validate(s,m,preconditioner,block_scalar_type,lagged_component_limiter)
    cpu=s.backend isa KA.CPU
    !cpu&&available_bytes===nothing&&throw(ArgumentError("Native CGS2 GPU construction requires fresh available_bytes"))
    available=available_bytes===nothing ? Sys.free_memory() : available_bytes
    available isa Integer&&!(available isa Bool)&&available>=0||throw(ArgumentError("available_bytes must be a nonnegative integer observation"))
    n=BigInt(s.NI)*s.NJ;ng=(BigInt(s.NI)+2s.g)*(BigInt(s.NJ)+2s.g);sb=8
    # Existing publicm2 V/Z/scratch,line arrays and13 coupled rollback fields;
    # R3 block storage, R1's9 fields and R3 LS's4 factors are counted explicitly.
    operator=4sb*n*(2*2+8)+8sb*n+4sb*ng+13sb*ng+
        160sb*n+8sb*n+32sb*(s.NI+s.NJ)+9sb*n+4sb*n
    operator_bound=cld(5operator,4)
    existing=_airfoil_cgs2_existing_bytes(s)
    transient=32*1024^2
    packed_admission=AirfoilPackedCGS2.memory_admission(s.NI,s.NJ,m;
        available_bytes=available,operator_bound_bytes=Int(operator_bound+cld(5BigInt(existing),4)+transient))
    (;packed_admission,linear_solver="native_cgs2",requested_basis=m,operator_basis=2,
      state_scalar_type="Float64",block_scalar_type="Float64",existing_backend_payload_bytes=existing,
      operator_and_controller_bound_bytes=Int(operator_bound),transient_reserve_bytes=transient,
      scope="All live numeric state/MUSCL/operator/controller/packed storage with reserve; excludes a claimed allocator/pool peak")
end

function _build_packed_cgs2_airfoil_workspace(s,w,m,receipt)
    w.m==2&&w.lagged_limiter!==nothing||throw(ArgumentError("CGS2 requires its admittedm2/R1 operator"))
    packed=AirfoilPackedCGS2.PackedWorkspace(s.backend,s,s.NI,s.NJ,m;admission=receipt.packed_admission)
    source_sha256=_AIRFOIL_PACKED_CGS2_SOURCE_SHA256
    PackedCGS2AirfoilWorkspace(s,w,packed,merge(receipt,(;source_sha256,source_file=basename(@__FILE__),
        source_identity_scope="CGS2 source bytes captured when this implementation was loaded")),Ref(false))
end

_airfoil_linear_residual_limit(::PackedCGS2AirfoilWorkspace,globalization,configured,requested)=min(configured,requested)

function _airfoil_krylov_gmres!(s::KSol,w::JFNKAirfoilWorkspace,kw::PackedCGS2AirfoilWorkspace;
    linear_tol::Float64=.1,linear_cycles::Int=1,fd_scale::Float32=1f0,
    fd_scheme::Symbol=:forward,fd_local_cap::Float32=1f-3,linear_maxiter::Union{Nothing,Int}=nothing)
    kw.s===s&&kw.w===w&&kw.packed.owner===s||throw(ArgumentError("CGS2 workspace owner/operator mismatch"))
    1<=linear_cycles<=4||throw(ArgumentError("linear_cycles must be1:4"))
    maxiter=linear_maxiter===nothing ? kw.packed.m*linear_cycles : linear_maxiter
    1<=maxiter<=10000||throw(ArgumentError("linear_maxiter must be1:10000"))
    kw.busy[]&&throw(ArgumentError("CGS2 adapter is already in use"))
    kw.busy[]=true
    apply!(out,input)=_airfoil_jacobian_vec!(out,input,s,w;fd_scale,fd_scheme,fd_local_cap)
    precondition!(out,input)=_airfoil_line_precondition!(out,input,s,w)
    ops=(;copy! =_av_copy!,fill! =(a,x)->_av_fill!(a,Float32(x)),axpy! =_av_axpy!,scale! =_av_scale!,dot=_av_dot)
    try
        result=AirfoilPackedCGS2.solve!(kw.packed,w.rhs,apply!,precondition!;
            method=:cgs2,rtol=linear_tol,maxiter,owner=s,ops)
        _av_copy!(w.delta,kw.packed.delta)
        return result.iterations,result.true_relative
    catch
        _av_fill!(w.delta,0f0);AirfoilPackedCGS2.fill4!(kw.packed.delta,0.);rethrow()
    finally
        kw.busy[]=false
    end
end

"Portable solver/storage/source metadata; rebuild workspaces after state restore."
function airfoil_linear_solver_contract(c::JFNKAirfoilController)
    kw=c.krylov_workspace
    kw isa PackedCGS2AirfoilWorkspace&&return kw.receipt
    (;linear_solver=kw===nothing ? "native" : "krylov",requested_basis=c.workspace.m,
      operator_basis=c.workspace.m,state_scalar_type=string(eltype(c.workspace.rhs[1])))
end
