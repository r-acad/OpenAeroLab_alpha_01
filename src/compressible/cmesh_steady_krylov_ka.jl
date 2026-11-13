# Krylov.jl adapter for the opt-in steady airfoil JFNK prototype. The
# nonlinear residual and frozen wall-normal line inverse stay on the chosen
# KA backend; only the four flow fields are packed for Krylov's vector API.
using Krylov
export KrylovAirfoilWorkspace, build_krylov_airfoil_workspace

mutable struct AirfoilJacobianOperator{S,W}
    s::S
    w::W
    fd_scale::Float32
    fd_scheme::Symbol
    fd_local_cap::Float32
end

struct AirfoilLineInverse{S,W}
    s::S
    w::W
end

Base.size(op::Union{AirfoilJacobianOperator,AirfoilLineInverse}) =
    (4length(op.s.vol),4length(op.s.vol))
Base.eltype(op::Union{AirfoilJacobianOperator,AirfoilLineInverse}) = eltype(op.s.rho)

function _airfoil_unpack!(fields, packed, ni::Int, nj::Int)
    data=reshape(packed,ni,nj,4)
    for k in 1:4
        copyto!(fields[k],view(data,:,:,k))
    end
    fields
end

function _airfoil_pack!(packed, fields, ni::Int, nj::Int)
    data=reshape(packed,ni,nj,4)
    for k in 1:4
        copyto!(view(data,:,:,k),fields[k])
    end
    packed
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::AirfoilJacobianOperator,
                            x::AbstractVector{F}) where {F<:AbstractFloat}
    s=op.s;w=op.w
    _airfoil_unpack!(w.rhs,x,s.NI,s.NJ)
    _airfoil_jacobian_vec!(w.work,w.rhs,s,w;
        fd_scale=op.fd_scale,fd_scheme=op.fd_scheme,
        fd_local_cap=op.fd_local_cap)
    _airfoil_pack!(y,w.work,s.NI,s.NJ)
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::AirfoilLineInverse,
                            x::AbstractVector{F}) where {F<:AbstractFloat}
    s=op.s;w=op.w
    _airfoil_unpack!(w.rhs,x,s.NI,s.NJ)
    _airfoil_line_precondition!(w.delta,w.rhs,s,w)
    _airfoil_pack!(y,w.delta,s.NI,s.NJ)
end

struct KrylovAirfoilWorkspace{S,W,B,K,O,P}
    s::S
    w::W
    rhs::B
    residual::B
    solver::K
    jacobian::O
    line_inverse::P
end

function build_krylov_airfoil_workspace(s::KSol,w::JFNKAirfoilWorkspace)
    size(w.rhs[1])==size(s.dtf) ||
        throw(ArgumentError("JFNK workspace does not match solver"))
    n=4length(s.vol)
    rhs=similar(s.dtf,eltype(s.dtf),(n,))
    residual=similar(rhs)
    jacobian=AirfoilJacobianOperator(s,w,1.0f0,:forward,1.0f-3)
    line_inverse=AirfoilLineInverse(s,w)
    solver=Krylov.GmresWorkspace(jacobian,rhs;memory=w.m)
    KrylovAirfoilWorkspace(s,w,rhs,residual,solver,jacobian,line_inverse)
end

function _airfoil_krylov_gmres!(s::KSol,w::JFNKAirfoilWorkspace,
                                kw::KrylovAirfoilWorkspace;
                                linear_tol::Float64=0.1,
                                linear_cycles::Int=1,
                                fd_scale::Float32=1.0f0,
                                fd_scheme::Symbol=:forward,
                                fd_local_cap::Float32=1.0f-3,
                                linear_maxiter::Union{Nothing,Int}=nothing)
    kw.s===s && kw.w===w ||
        throw(ArgumentError("Krylov workspace does not match solver and JFNK workspace"))
    0<linear_tol<1 || throw(ArgumentError("linear_tol must be in (0,1)"))
    1<=linear_cycles<=4 || throw(ArgumentError("linear_cycles must be 1:4"))
    maxiter=linear_maxiter===nothing ? w.m*linear_cycles : linear_maxiter
    1<=maxiter<=10000 || throw(ArgumentError("linear_maxiter must be in 1:10000"))
    kw.jacobian.fd_scale=fd_scale
    kw.jacobian.fd_scheme=fd_scheme
    kw.jacobian.fd_local_cap=fd_local_cap
    _airfoil_pack!(kw.rhs,w.rhs,s.NI,s.NJ)
    beta=norm(kw.rhs)
    beta==0 && return (0,0.0)
    Krylov.gmres!(kw.solver,kw.jacobian,kw.rhs;
        N=kw.line_inverse,restart=true,itmax=maxiter,
        atol=zero(eltype(kw.rhs)),rtol=eltype(kw.rhs)(linear_tol))
    # Report the true residual of the matrix-free system, including after
    # restart or an iteration limit. Krylov's estimate alone can be optimistic.
    mul!(kw.residual,kw.jacobian,kw.solver.x)
    @. kw.residual=kw.rhs-kw.residual
    rel=norm(kw.residual)/beta
    isfinite(rel) || throw(ArgumentError("nonfinite Krylov linear residual"))
    _airfoil_unpack!(w.delta,kw.solver.x,s.NI,s.NJ)
    kw.solver.stats.niter,rel
end
