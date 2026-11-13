# Matrix-free pseudo-transient Newton-Krylov for the steady A+/C airfoil flow.
# Experimental: uses the exact nonlinear residual but a finite-difference
# Jacobian and optional scalar or flux-consistent block right
# preconditioner. SST is segregated. The ADI variants have not yet shown a
# qualified time-to-convergence gain on the NACA 0012 study mesh.
using LinearAlgebra
include("cmesh_steady_multilevel_ka.jl")
include("cmesh_steady_block_line_ka.jl")
include("cmesh_steady_kinetic_blocks_ka.jl")
include("cmesh_steady_lagged_limiter_ka.jl")
export JFNKAirfoilWorkspace, build_jfnk_airfoil_workspace, step_jfnk_airfoil_ka!
export JFNKAirfoilController, build_jfnk_airfoil_controller, step_jfnk_rans_ka!
export airfoil_jacobian_contract

struct JFNKAirfoilWorkspace{L,V,A,S,M,K,J}
    line::L
    adi::A
    mg::M
    kinetic::K
    lagged_limiter::J
    coupled_cut::Bool
    swept_line::Bool
    sweep_old::S
    basis::Vector{V}
    preconditioned::Vector{V}
    rhs::V
    delta::V
    work::V
    base_residual::V
    fd_minus::V
    full_rhs::V
    total_delta::V
    m::Int
end

# Kept separate so the convergence diagnostics can evolve without changing
# the nonlinear acceptance test or the solver's physical equations.
include("cmesh_steady_diagnostics_ka.jl")
include("cmesh_steady_globalization_ka.jl")

"""Allocate an experimental steady flow workspace.

`lagged_component_limiter=true` selects a declared Float64 thermal-MUSCL R1
for Jv only. It owns nine additional scalar fields per cell: four component
factors, four R1 baseline residuals and one error-rollback dt field. The low
level factory retains caller-managed memory admission; large GPU studies
must preflight restart, precision and all optional storage before allocation.
Current physical R0, nonlinear merits and scientific stopping rules are not
changed. The default uses the current limiter in every Jv sample.
"""
function build_jfnk_airfoil_workspace(s::KSol;m::Int=8,
                                      preconditioner::Symbol=:line,
                                      pseudo_scale::Float32=1f0,
                                      block_scalar_type::Type{<:AbstractFloat}=Float32,
                                      lagged_component_limiter::Bool=false)
    lagged_component_limiter&&_airfoil_lagged_limiter_admit(s)
    block_scalar_type in (Float32,Float64)||throw(ArgumentError("Block preconditioner precision must be Float32 or Float64"))
    2<=m<=96 || throw(ArgumentError("JFNK restart length must be 2:96"))
    preconditioner in (:line,:cut_line,:adi,:block_adi,:cut_block_adi,
                       :swept_line,:cut_swept_line,:multilevel,
                       :kinetic_block_adi,:kinetic_fullvisc_block_adi,
                       :kinetic_weno_fullvisc_block_adi,
                       :kinetic_muscl_fullvisc_block_adi,
                       :kinetic_muscl_ls_fullvisc_block_adi,
                       :kinetic_block_unfloored_adi,
                       :kinetic_weno_fullvisc_unfloored_adi) ||
        throw(ArgumentError("unknown airfoil preconditioner"))
    (pseudo_scale==1f0 || preconditioner in
        (:kinetic_block_unfloored_adi,:kinetic_weno_fullvisc_unfloored_adi)) ||
        throw(ArgumentError("pseudo_scale requires an unfloored block preconditioner"))
    vec()=ntuple(_->similar(s.dtf),4)
    v=vec()
    swept_line=preconditioner in (:swept_line,:cut_swept_line)
    block_length=preconditioner==:adi ? s.NI : min(s.NI,64)
    adi=preconditioner in (:adi,:block_adi,:cut_block_adi,
                           :swept_line,:cut_swept_line,:multilevel) ?
        (lamW=similar(s.dtf),lamE=similar(s.dtf),
         cp=similar(s.dtf,preconditioner==:adi ? Float64 : Float32),
         d0_eff=swept_line ? similar(s.dtf) : nothing,
         block_length=Int32(block_length),
         blocks_per_row=Int32(cld(s.NI,block_length))) : nothing
    coupled_cut=preconditioner in (:cut_line,:cut_block_adi,:cut_swept_line)
    line=build_implicit_line_workspace(s)
    mg=preconditioner==:multilevel ? build_airfoil_mg_workspace(s,line,adi) : nothing
    kinetic=preconditioner in (:kinetic_block_adi,:kinetic_fullvisc_block_adi,
                               :kinetic_weno_fullvisc_block_adi,
                               :kinetic_muscl_fullvisc_block_adi,
                               :kinetic_muscl_ls_fullvisc_block_adi,
                               :kinetic_block_unfloored_adi,
                               :kinetic_weno_fullvisc_unfloored_adi) ?
        build_airfoil_kinetic_block_workspace(s;block_length=min(s.NI,64),
            fullvisc_block=preconditioner in (:kinetic_fullvisc_block_adi,
                :kinetic_weno_fullvisc_block_adi,
                :kinetic_muscl_fullvisc_block_adi,
                :kinetic_muscl_ls_fullvisc_block_adi,
                :kinetic_weno_fullvisc_unfloored_adi),
            weno_block=preconditioner in (:kinetic_weno_fullvisc_block_adi,
                :kinetic_weno_fullvisc_unfloored_adi),
            muscl_block=preconditioner in (:kinetic_muscl_fullvisc_block_adi,:kinetic_muscl_ls_fullvisc_block_adi),
            muscl_ls_block=preconditioner==:kinetic_muscl_ls_fullvisc_block_adi,
            unfloored=preconditioner in (:kinetic_block_unfloored_adi,
                :kinetic_weno_fullvisc_unfloored_adi),
            pseudo_scale,block_scalar_type) : nothing
    lagged=lagged_component_limiter ? _build_airfoil_lagged_limiter_workspace(s) : nothing
    JFNKAirfoilWorkspace(line,adi,mg,kinetic,lagged,coupled_cut,
        swept_line,swept_line ? vec() : nothing,
        [vec() for _ in 1:m+1],[vec() for _ in 1:m],v,
        vec(),vec(),vec(),vec(),vec(),vec(),m)
end

_airfoil_residual(s::KSol)=(s.dc_r,s.dc_mx,s.dc_my,s.dc_e)

function _av_copy!(dst,src)
    for k in 1:4; copyto!(dst[k],src[k]); end
    dst
end
function _av_fill!(dst,a::Float32)
    for k in 1:4; fill!(dst[k],a); end
    dst
end
function _av_axpy!(dst,a::Float64,src)
    af=eltype(dst[1])(a)
    for k in 1:4; @. dst[k]=dst[k]+af*src[k]; end
    dst
end
function _av_scale!(dst,a::Float64)
    af=eltype(dst[1])(a)
    for k in 1:4; @. dst[k]=af*dst[k]; end
    dst
end
function _av_dot(a,b)
    # Fused CPU traversal avoids a mesh-sized broadcast temporary per
    # Arnoldi product. Accumulate in Float64: four separate Float32 BLAS dots
    # were faster but lost significant precision on uniform-field probes.
    # Keep the existing backend reduction for GPU arrays.
    if _implicit_cpu_strided(a[1]) && _implicit_cpu_strided(b[1])
        x1,x2,x3,x4=a;y1,y2,y3,y4=b
        total=0.0
        @inbounds @simd for i in eachindex(x1,y1,x2,y2,x3,y3,x4,y4)
            total+=Float64(x1[i])*y1[i]+Float64(x2[i])*y2[i]+
                   Float64(x3[i])*y3[i]+Float64(x4[i])*y4[i]
        end
        return total
    end
    Float64(sum(a[1].*b[1] .+ a[2].*b[2] .+
                a[3].*b[3] .+ a[4].*b[4]))
end
_av_norm(a)=sqrt(max(0.0,_av_dot(a,a)))

# Frozen scalar inverse of the pseudo-transient operator. For RHS in
# residual-per-volume units, multiply by the cell volume before the solve.
# The ADI option applies streamwise, diagonal scale, then wall-normal factors.
@kernel function _k_implicit_swept_rhs!(rhs,@Const(input),@Const(vol),
    @Const(old),@Const(lamW),@Const(lamE),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        rhs[i,j]=input[i,j]*vol[i,j]+
            (i>1 ? lamW[i,j]*old[i-1,j] : 0.0f0)+
            (i<NI ? lamE[i,j]*old[i+1,j] : 0.0f0)
    end
end

function _airfoil_line_precondition!(out,input,s::KSol,w::JFNKAirfoilWorkspace)
    w.kinetic!==nothing && return _airfoil_kinetic_precondition!(out,input,s,w.kinetic)
    line=w.line; b=s.backend; ni=Int32(s.NI);nj=Int32(s.NJ)
    for k in 1:4
        if w.mg!==nothing
            _airfoil_mg_precondition!(out[k],input[k],s.vol,w.mg,b)
            continue
        end
        if w.swept_line
            fill!(w.sweep_old[k],0.0f0)
            for sweep in 1:3
                _k_implicit_swept_rhs!(b)(w.work[k],input[k],s.vol,
                    w.sweep_old[k],w.adi.lamW,w.adi.lamE,ni,nj;
                    ndrange=(s.NI,s.NJ))
                if w.coupled_cut
                    _k_implicit_cut_line_solve!(b)(out[k],line.bp,w.work[k],
                        line.lamS,line.lamN,w.adi.d0_eff,ni,nj,
                        Int32(s.NIW),Int32(s.NIF);ndrange=s.NIF+s.NIW)
                else
                    _k_implicit_line_solve!(b)(out[k],line.bp,w.work[k],
                        line.lamS,line.lamN,w.adi.d0_eff,ni,nj;ndrange=s.NI)
                end
                sweep<3 && copyto!(w.sweep_old[k],out[k])
            end
            continue
        end
        @. w.work[k]=input[k]*s.vol
        if w.adi!==nothing
            _k_implicit_i_solve!(b)(out[k],w.adi.cp,w.work[k],
                w.adi.lamW,w.adi.lamE,line.d0,ni,nj,
                w.adi.block_length,w.adi.blocks_per_row;
                ndrange=Int(w.adi.blocks_per_row)*s.NJ)
            @. w.work[k]=out[k]*line.d0
        end
        if w.coupled_cut
            _k_implicit_cut_line_solve!(b)(out[k],line.bp,w.work[k],
                line.lamS,line.lamN,line.d0,ni,nj,
                Int32(s.NIW),Int32(s.NIF);
                ndrange=s.NIF+s.NIW)
        else
            _k_implicit_line_solve!(b)(out[k],line.bp,w.work[k],
                line.lamS,line.lamN,line.d0,ni,nj;ndrange=s.NI)
        end
    end
    out
end

# A(v) = [R(U+eps v)-R(U)]/(eps V) + d0*v/V. The state is restored
# after each residual evaluation. Float32-aware epsilon grows with the
# overall conservative state norm and shrinks with the Krylov direction norm.
function _airfoil_fd_step(v,nv::Float64,ncell::Int,r::Float64,u::Float64,
                          E::Float64,fd_scale::Float32;
                          local_cap::Float32=1.0f-3)
    state_norm=sqrt(ncell*(r*r+2*(r*u)^2+E*E))
    F=eltype(v[1])
    h=F(fd_scale*sqrt(eps(F))*(1.0+state_norm)/nv)
    # A global 2-norm step can perturb one wall cell by O(sqrt(N) eps^1/2)
    # when a preconditioned Krylov vector is localized there. Bound the
    # conservative change in every cell before sampling the nonlinear flux.
    inv_r=F(1/r)
    inv_m=F(1/(r*u))
    inv_e=F(1/E)
    max_scaled=if _implicit_cpu_strided(v[1])
        # Four cache-friendly CPU reductions avoid a mesh-sized broadcast
        # temporary at every matrix-free Jacobian action.
        max(maximum(abs,v[1])*inv_r,maximum(abs,v[2])*inv_m,
            maximum(abs,v[3])*inv_m,maximum(abs,v[4])*inv_e)
    else
        maximum(@. max(abs(v[1])*inv_r,abs(v[2])*inv_m,
                    abs(v[3])*inv_m,abs(v[4])*inv_e))
    end
    max_scaled>0 && (h=min(h,local_cap/max_scaled))
    isfinite(h) && h>0 || throw(ArgumentError("invalid matrix-free perturbation"))
    h
end

function _airfoil_jacobian_vec_impl!(out,v,s::KSol,w::JFNKAirfoilWorkspace;
                                fd_scale::Float32=1.0f0,
                                fd_scheme::Symbol=:forward,
                                fd_local_cap::Float32=1.0f-3)
    fd_scheme in (:forward,:central) ||
        throw(ArgumentError("fd_scheme must be :forward or :central"))
    isfinite(fd_local_cap) && 1.0f-4<=fd_local_cap<=0.05f0 ||
        throw(ArgumentError("fd_local_cap must be in [1e-4,0.05]"))
    reference=_airfoil_jacobian_reference(w)
    w.lagged_limiter===nothing||_airfoil_lagged_owner(s,w.lagged_limiter)
    nv=_av_norm(v)
    nv==0.0 && return _av_fill!(out,0.0f0)
    line=w.line
    ncell=length(s.vol)
    r=Float64(s.rho_inf)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    E=r*(Float64(s.eint*s.T_inf)+0.5*u*u)
    h=_airfoil_fd_step(v,nv,ncell,r,u,E,fd_scale;local_cap=fd_local_cap)
    b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g)
    physical=false
    try
        for _ in 1:16
            _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,
                line.rho0,line.ux0,line.uy0,line.T0,v...,
                s.eint,-h,ni,nj,g;ndrange=(s.NI,s.NJ))
            if _airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
                if fd_scheme==:central
                    _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,
                        line.rho0,line.ux0,line.uy0,line.T0,v...,
                        s.eint,h,ni,nj,g;ndrange=(s.NI,s.NJ))
                    physical=_airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
                else
                    physical=true
                end
                physical && break
            end
            h*=0.5f0
        end
        physical||throw(ArgumentError("no physical matrix-free perturbation after 16 halvings"))
        if fd_scheme==:central
            # The accepted state is U-hv. Retain its residual, then sample
            # U+hv; symmetric differencing reduces truncation at the larger
            # perturbations needed for a useful Float32 Jacobian signal.
            _airfoil_jacobian_residual_ka!(s,w)
            _av_copy!(w.fd_minus,_airfoil_residual(s))
            _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,
                line.rho0,line.ux0,line.uy0,line.T0,v...,
                s.eint,-h,ni,nj,g;ndrange=(s.NI,s.NJ))
        end
        _airfoil_jacobian_residual_ka!(s,w)
        current=_airfoil_residual(s)
        for k in 1:4
            if fd_scheme==:central
                @. out[k]=((current[k]-w.fd_minus[k])/(2h)+
                           line.d0*v[k])/s.vol
            else
                @. out[k]=((current[k]-reference[k])/h+
                           line.d0*v[k])/s.vol
            end
        end
    catch
        w.lagged_limiter===nothing||(w.lagged_limiter.ready=false)
        rethrow()
    finally
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),
                             (line.rho0,line.ux0,line.uy0,line.T0))
            copyto!(dst,src)
        end
    end
    out
end

function _airfoil_jacobian_vec!(out,v,s::KSol,w::JFNKAirfoilWorkspace;kwargs...)
    w.lagged_limiter===nothing&&return _airfoil_jacobian_vec_impl!(out,v,s,w;kwargs...)
    try
        _airfoil_jacobian_vec_impl!(out,v,s,w;kwargs...)
    catch
        w.lagged_limiter.ready=false
        rethrow()
    end
end

function _airfoil_fgmres!(s::KSol,w::JFNKAirfoilWorkspace;
                          linear_tol::Float64=0.1,fd_scale::Float32=1.0f0,
                          fd_scheme::Symbol=:forward,
                          fd_local_cap::Float32=1.0f-3,maxiter::Int=w.m)
    0<linear_tol<1 || throw(ArgumentError("linear_tol must be in (0,1)"))
    1<=maxiter<=w.m || throw(ArgumentError("cycle maxiter must be in 1:restart"))
    m=maxiter;V=w.basis;Z=w.preconditioned
    H=zeros(Float64,m+1,m);g=zeros(Float64,m+1)
    cs=zeros(Float64,m);sn=zeros(Float64,m)
    _av_fill!(w.delta,0.0f0)
    beta=_av_norm(w.rhs)
    beta==0 && return (0,0.0)
    _av_copy!(V[1],w.rhs);_av_scale!(V[1],1/beta);g[1]=beta
    rel=Inf
    for j in 1:m
        _airfoil_line_precondition!(Z[j],V[j],s,w)
        _airfoil_jacobian_vec!(V[j+1],Z[j],s,w;fd_scale,fd_scheme,fd_local_cap)
        for i in 1:j
            H[i,j]=_av_dot(V[j+1],V[i])
            _av_axpy!(V[j+1],-H[i,j],V[i])
        end
        H[j+1,j]=_av_norm(V[j+1])
        isfinite(H[j+1,j]) || throw(ArgumentError("nonfinite JFNK Arnoldi vector"))
        H[j+1,j]>0 && _av_scale!(V[j+1],1/H[j+1,j])
        # Incremental Givens QR avoids solving a new dense least-squares
        # problem at every Arnoldi vector, particularly for m > 24.
        for i in 1:j-1
            top=cs[i]*H[i,j]+sn[i]*H[i+1,j]
            H[i+1,j]=-sn[i]*H[i,j]+cs[i]*H[i+1,j]
            H[i,j]=top
        end
        den=hypot(H[j,j],H[j+1,j])
        den>0 || throw(ArgumentError("singular JFNK Arnoldi column"))
        cs[j]=H[j,j]/den;sn[j]=H[j+1,j]/den
        H[j,j]=den;H[j+1,j]=0.0
        g[j+1]=-sn[j]*g[j];g[j]=cs[j]*g[j]
        rel=abs(g[j+1])/beta
        if rel<=linear_tol || j==m
            y=UpperTriangular(@view H[1:j,1:j]) \ (@view g[1:j])
            for k in 1:j; _av_axpy!(w.delta,y[k],Z[k]); end
            return (j,rel)
        end
    end
    return (m,rel)
end

function _airfoil_fgmres_restarted!(s::KSol,w::JFNKAirfoilWorkspace;
                                    linear_tol::Float64=0.1,
                                    linear_cycles::Int=1,
                                    fd_scale::Float32=1.0f0,
                                    fd_scheme::Symbol=:forward,
                                    fd_local_cap::Float32=1.0f-3,
                                    linear_maxiter::Union{Nothing,Int}=nothing,
                                    restart_callback=nothing)
    1<=linear_cycles<=4 || throw(ArgumentError("linear_cycles must be 1:4"))
    maxiter=linear_maxiter===nothing ? w.m*linear_cycles : linear_maxiter
    1<=maxiter<=10000 || throw(ArgumentError("linear_maxiter must be in 1:10000"))
    _av_copy!(w.full_rhs,w.rhs)
    _av_fill!(w.total_delta,0.0f0)
    beta=_av_norm(w.full_rhs)
    beta==0 && return (0,0.0)
    total=0;rel=1.0
    while total<maxiter
        current_beta=_av_norm(w.rhs)
        current_beta<=linear_tol*beta && break
        # Every restart aims at the ORIGINAL absolute forcing goal. Solving
        # each correction to the same relative tolerance can oversolve it.
        cycle_tol=min(0.999,linear_tol*beta/current_beta)
        kits,_=_airfoil_fgmres!(s,w;linear_tol=cycle_tol,fd_scale,fd_scheme,fd_local_cap,
            maxiter=min(w.m,maxiter-total))
        total+=kits
        _av_axpy!(w.total_delta,1.0,w.delta)
        _airfoil_jacobian_vec!(w.work,w.total_delta,s,w;
            fd_scale,fd_scheme,fd_local_cap)
        for k in 1:4; @. w.rhs[k]=w.full_rhs[k]-w.work[k]; end
        rel=_av_norm(w.rhs)/beta
        isfinite(rel) || throw(ArgumentError("nonfinite restarted linear residual"))
        restart_callback===nothing || restart_callback((;iterations=total,relative=rel))
        rel<=linear_tol && break
    end
    _av_copy!(w.delta,w.total_delta)
    total,rel
end

"""One opt-in pseudo-transient Newton-Krylov flow iteration on a single-phase
free-air airfoil. Returns acceptance, line-search factor, linear iterations,
linear relative residual, and nonlinear merit. This does not advance SST."""
# Default native/Krylov admission is unchanged; optional adapters may require stricter accuracy.
_airfoil_linear_residual_limit(::Any,globalization,configured,requested)=globalization===:pseudo_transient ? configured : Inf

function _step_jfnk_airfoil_impl!(s::KSol,w::JFNKAirfoilWorkspace,cfl::Float32;
                               linear_tol::Float64=0.1,linear_cycles::Int=1,
                               max_backtracks::Int=8,fd_scale::Float32=1.0f0,
                               fd_scheme::Symbol=:forward,
                               fd_local_cap::Float32=1.0f-3,
                               merit_reference::Float64=NaN,
                               globalization::Symbol=:steady,
                               physical_fraction::Float32=0.2f0,
                               linear_residual_max::Float64=0.95,
                               linear_maxiter::Union{Nothing,Int}=nothing,
                               krylov_workspace=nothing)
    globalization in (:steady,:pseudo_transient) ||
        throw(ArgumentError("globalization must be :steady or :pseudo_transient"))
    isfinite(physical_fraction) && 0.0f0<physical_fraction<=0.5f0 ||
        throw(ArgumentError("physical_fraction must be in (0,0.5]"))
    0<linear_residual_max<1 || throw(ArgumentError("linear_residual_max must be in (0,1)"))
    _implicit_admit(s,cfl;max_cfl=globalization==:pseudo_transient ? 1.0f5 : 10.0f0)
    isfinite(fd_scale) && 0.1f0<=fd_scale<=16.0f0 ||
        throw(ArgumentError("fd_scale must be in [0.1,16]"))
    fd_scheme in (:forward,:central) ||
        throw(ArgumentError("fd_scheme must be :forward or :central"))
    isfinite(fd_local_cap) && 1.0f-4<=fd_local_cap<=0.05f0 ||
        throw(ArgumentError("fd_local_cap must be in [1e-4,0.05]"))
    1<=max_backtracks<=16 || throw(ArgumentError("max_backtracks must be 1:16"))
    size(w.rhs[1])==size(s.dtf) && size(w.line.rho0)==size(s.rho) ||
        throw(ArgumentError("JFNK workspace does not match solver"))
    if w.kinetic!==nothing
        w.kinetic.fullvisc_block && s.fullvisc!=Int32(1) &&
            throw(ArgumentError("full-viscous block preconditioner requires fullvisc=1"))
        w.kinetic.weno_block && s.recon!=Int32(1) &&
            throw(ArgumentError("WENO block preconditioner requires recon=1"))
    end
    line=w.line;b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g)
    saved=(s.rho,s.ux,s.uy,s.T)
    backup=(line.rho0,line.ux0,line.uy0,line.T0)
    for (dst,src) in zip(backup,saved);copyto!(dst,src);end
    _raw_residual_ka!(s,Int32(2),Int32(1))
    before=_implicit_merit(s)
    isfinite(before) || throw(ArgumentError("nonfinite initial flow residual"))
    isnan(merit_reference) || (isfinite(merit_reference) && merit_reference>=0) ||
        throw(ArgumentError("merit_reference must be finite and nonnegative"))
    reference=isnan(merit_reference) ? before : max(before,merit_reference)
    _av_copy!(w.base_residual,_airfoil_residual(s))
    if before==0.0
        prepare_dt_ka!(s,min(cfl,0.45f0))
        return (accepted=true,alpha=0.0f0,linear_iterations=0,
                linear_relative=0.0,merit_before=before,merit_after=before,
                best_trial_merit=before,best_trial_alpha=0.0f0,
                physical_trials=0,merit_reference=reference,globalization,
                acceptance_merit_after=before,physical_alpha=1.0f0)
    end
    _k_implicit_line_coeffs!(b)(line.lamS,line.lamN,line.d0,
        s.rho,s.ux,s.uy,s.T,s.nut,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        cfl,ni,nj,g;ndrange=(s.NI,s.NJ))
    if w.kinetic!==nothing && w.kinetic.unfloored
        _k_airfoil_kinetic_unfloored_d0!(b)(line.d0,
            s.ux,s.uy,s.T,s.Six,s.Siy,cfl,w.kinetic.pseudo_scale,
            ni,nj,g;ndrange=(s.NI,s.NJ))
    end
    w.kinetic!==nothing && _airfoil_kinetic_build!(s,w.kinetic,line.d0)
    if w.adi!==nothing
        _k_implicit_i_coeffs!(b)(w.adi.lamW,w.adi.lamE,
            s.rho,s.ux,s.uy,s.T,s.nut,s.Six,s.Siy,s.vol,
            s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
            ni,nj,g;ndrange=(s.NI,s.NJ))
        w.swept_line && (@. w.adi.d0_eff=line.d0+w.adi.lamW+w.adi.lamE)
        w.mg!==nothing && _prepare_airfoil_mg!(w.mg,b)
    end
    w.lagged_limiter===nothing||_airfoil_refresh_lagged_limiter!(s,w.lagged_limiter,line)
    for k in 1:4; @. w.rhs[k]=w.base_residual[k]/s.vol; end
    kits,rel=krylov_workspace===nothing ?
        _airfoil_fgmres_restarted!(s,w;
            linear_tol,linear_cycles,fd_scale,fd_scheme,fd_local_cap,linear_maxiter) :
        _airfoil_krylov_gmres!(s,w,krylov_workspace;
            linear_tol,linear_cycles,fd_scale,fd_scheme,fd_local_cap,linear_maxiter)
    if rel>_airfoil_linear_residual_limit(krylov_workspace,globalization,linear_residual_max,linear_tol)
        _raw_residual_ka!(s,Int32(2),Int32(1))
        return (accepted=false,alpha=0.0f0,linear_iterations=kits,
            linear_relative=rel,merit_before=before,merit_after=before,
            best_trial_merit=Inf,best_trial_alpha=0.0f0,physical_trials=0,
            merit_reference=reference,globalization,
            acceptance_merit_after=before,physical_alpha=0.0f0)
    end
    physical_alpha=globalization==:pseudo_transient ?
        _airfoil_physical_alpha(s,w,physical_fraction) : 1.0f0
    alpha=physical_alpha
    best=Inf;best_alpha=0.0f0;physical_trials=0
    for _ in 1:max_backtracks
        _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,backup...,
            w.delta...,s.eint,alpha,ni,nj,g;ndrange=(s.NI,s.NJ))
        good=_airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
        if good
            physical_trials+=1
            _raw_residual_ka!(s,Int32(2),Int32(1))
            after=_implicit_merit(s)
            if isfinite(after) && after<best
                best=after;best_alpha=alpha
            end
            # An opt-in recent-maximum reference can pass a small local
            # residual rise, while the per-step cap limits an SST-coupled
            # excursion. The default reference is the current merit.
            objective=globalization==:pseudo_transient ? _airfoil_pseudo_merit(s,w) : after
            objective_reference=globalization==:pseudo_transient ? before : reference
            steady_cap=globalization==:pseudo_transient ? 2.0 : 1.02
            if isfinite(after) && isfinite(objective) &&
               objective<objective_reference*(1.0-1e-4*Float64(alpha)) &&
               after<before*steady_cap
                prepare_dt_ka!(s,min(cfl,0.45f0))
                return (accepted=true,alpha,linear_iterations=kits,
                        linear_relative=rel,merit_before=before,merit_after=after,
                        best_trial_merit=best,best_trial_alpha=best_alpha,
                        physical_trials,merit_reference=reference,globalization,
                        acceptance_merit_after=objective,physical_alpha)
            end
        end
        alpha*=0.5f0
    end
    for (dst,src) in zip(saved,backup);copyto!(dst,src);end
    _raw_residual_ka!(s,Int32(2),Int32(1))
    return (accepted=false,alpha=0.0f0,linear_iterations=kits,
            linear_relative=rel,merit_before=before,merit_after=before,
            best_trial_merit=best,best_trial_alpha=best_alpha,
            physical_trials,merit_reference=reference,globalization,
            acceptance_merit_after=before,physical_alpha)
end

"""One physical-residual-controlled flow iteration.

`lagged_component_limiter=true` is selected when building the workspace. It
uses a declared lagged-limiter R1 only for Jv; every trial uses current R0 and
current BE/steady merits. Initially limited to Float64 thermal MUSCL airfoils.
The lagged reference is refreshed per linearization and invalidated on exit.
This flow step does not advance SST or certify scientific convergence.
"""
function step_jfnk_airfoil_ka!(s::KSol,w::JFNKAirfoilWorkspace,cfl::Float32;kwargs...)
    r=w.lagged_limiter
    r===nothing&&return _step_jfnk_airfoil_impl!(s,w,cfl;kwargs...)
    r.ready=false
    _airfoil_lagged_owner(s,r)
    for (dst,src) in zip((w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0),(s.rho,s.ux,s.uy,s.T));copyto!(dst,src);end
    copyto!(r.saved_dt,s.dtf);clock=s.last_global_dt
    try
        result=_step_jfnk_airfoil_impl!(s,w,cfl;kwargs...)
        r.ready=false
        merge(result,airfoil_jacobian_contract(w))
    catch
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),(w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0));copyto!(dst,src);end
        copyto!(s.dtf,r.saved_dt);s.last_global_dt=clock;r.ready=false
        rethrow()
    end
end

"""Adaptive pseudo-CFL controller for the opt-in steady A+/C SST march.
This is a convergence experiment. It does not advance physical time, and SST
still uses the solver's segregated, stability-capped transport step. A positive
`nonmonotone_window` uses the maximum of recent flow merits for line search;
this is experimental and does not establish a steady RANS convergence gate."""
const _AIRFOIL_COUPLED_STATE = (:rho,:ux,:uy,:T,:tk,:tom,:nut,:tnl,:pvort,:exx,:exy,:eyy,:dtf)

mutable struct JFNKAirfoilController{W,K,S}
    workspace::W
    krylov_workspace::K
    coupled_backup::S
    cfl::Float32
    min_cfl::Float32
    max_cfl::Float32
    growth::Float32
    max_attempts::Int
    iterations::Int
    rejections::Int
    nonmonotone_window::Int
    merit_history::Vector{Float64}
    recovery_steps::Int
    recovery_limit::Int
    recovery_iterations::Int
    recovery_events::Int
    restart_cfl::Float32
    globalization::Symbol
    physical_fraction::Float32
end

function build_jfnk_airfoil_controller(s::KSol;m::Int=4,
    initial_cfl::Float32=2.0f0,min_cfl::Float32=0.05f0,
    max_cfl::Float32=2.0f0,growth::Float32=1.05f0,
    max_attempts::Int=4,linear_solver::Symbol=:native,
    nonmonotone_window::Int=0,preconditioner::Symbol=:line,
    pseudo_scale::Float32=1f0,
    recovery_steps::Int=0,recovery_limit::Int=200,
    globalization::Symbol=:steady,physical_fraction::Float32=0.2f0,
    block_scalar_type::Type{<:AbstractFloat}=Float32,
    lagged_component_limiter::Bool=false,available_bytes=nothing)
    globalization in (:steady,:pseudo_transient) ||
        throw(ArgumentError("globalization must be :steady or :pseudo_transient"))
    isfinite(physical_fraction) && 0.0f0<physical_fraction<=0.5f0 ||
        throw(ArgumentError("physical_fraction must be in (0,0.5]"))
    allowed_cfl=globalization==:pseudo_transient ? 1.0f5 : 10.0f0
    isfinite(min_cfl) && isfinite(initial_cfl) && isfinite(max_cfl) &&
        0.0f0<min_cfl<=initial_cfl<=max_cfl<=allowed_cfl ||
        throw(ArgumentError("invalid JFNK pseudo-CFL bounds"))
    isfinite(growth) && 1.0f0<growth<=2.0f0 ||
        throw(ArgumentError("JFNK CFL growth must be in (1,2]"))
    1<=max_attempts<=8 || throw(ArgumentError("max_attempts must be 1:8"))
    linear_solver in (:native,:krylov,:native_cgs2) ||
        throw(ArgumentError("linear_solver must be :native, :krylov or :native_cgs2"))
    0<=nonmonotone_window<=20 ||
        throw(ArgumentError("nonmonotone_window must be 0:20"))
    0<=recovery_steps<=100 && recovery_steps<=recovery_limit<=10_000 ||
        throw(ArgumentError("recovery_steps must be 0:100 and recovery_limit must be between recovery_steps and 10000"))
    linear_solver===:native_cgs2&&recovery_steps!=0&&
        throw(ArgumentError("Native CGS2 initially requires recovery_steps=0"))
    cgs2_receipt=linear_solver===:native_cgs2 ? airfoil_cgs2_memory_guard(s,m;
    preconditioner,block_scalar_type,lagged_component_limiter,available_bytes) : nothing
operator_m=linear_solver===:native_cgs2 ? 2 : m
workspace=build_jfnk_airfoil_workspace(s;m=operator_m,preconditioner,pseudo_scale,block_scalar_type,lagged_component_limiter)
krylov_workspace=linear_solver===:native_cgs2 ?
    _build_packed_cgs2_airfoil_workspace(s,workspace,m,cgs2_receipt) : linear_solver==:krylov ?
    build_krylov_airfoil_workspace(s,workspace) : nothing
    # Reusable backups avoid allocating mesh-sized arrays on every nonlinear
    # step while allowing an invalid segregated SST update to be rolled back.
    coupled_backup=map(name->similar(getproperty(s,name)),_AIRFOIL_COUPLED_STATE)
    JFNKAirfoilController(workspace,krylov_workspace,coupled_backup,initial_cfl,
        min_cfl,max_cfl,growth,max_attempts,0,0,
        nonmonotone_window,Float64[],recovery_steps,recovery_limit,0,0,initial_cfl,
        globalization,physical_fraction)
end

function _airfoil_coupled_backup!(s::KSol,c::JFNKAirfoilController)
    for (name,dst) in zip(_AIRFOIL_COUPLED_STATE,c.coupled_backup)
        copyto!(dst,getproperty(s,name))
    end
    nothing
end

function _airfoil_coupled_restore!(s::KSol,c::JFNKAirfoilController)
    for (name,src) in zip(_AIRFOIL_COUPLED_STATE,c.coupled_backup)
        copyto!(getproperty(s,name),src)
    end
    nothing
end

function _airfoil_valid_sst_state(s::KSol)
    # Wall k ghost cells are deliberately negative for the zero-wall BC, so
    # require nonnegative turbulence only in real cells while checking all
    # ghosts for nonfinites. Stresses can have either sign.
    mapreduce((k,w,n,l,p)->isfinite(k) && isfinite(w) && isfinite(n) && isfinite(l) && isfinite(p),
              &,s.tk,s.tom,s.nut,s.tnl,s.pvort) || return false
    ir=(s.g+1):(s.g+s.NI); jr=(s.g+1):(s.g+s.NJ)
    mapreduce((k,w,n,l)->isfinite(k) && isfinite(w) && isfinite(n) && isfinite(l) &&
                           k>=0.0f0 && w>=0.0f0 && n>=0.0f0 && l>=0.0f0,
              &,view(s.tk,ir,jr),view(s.tom,ir,jr),view(s.nut,ir,jr),view(s.tnl,ir,jr)) &&
        mapreduce((x,y,z)->isfinite(x) && isfinite(y) && isfinite(z),&,s.exx,s.exy,s.eyy)
end

"""Advance SST transactionally after a proposed flow step.

The caller must save the full pre-flow state in `coupled_backup`. Invalid
turbulence or residuals reject the complete coupled step and restore that
state. Exceptions also restore it before propagation. The injected advance
function enables failure-path testing without modifying the solver kernels.
"""
function _airfoil_coupled_sst!(s::KSol,c::JFNKAirfoilController;
    corrected_wall_metric::Bool=_sst_default_wall_metric(s),nonorth_diffusion::Bool=false,
    sst_substeps::Int=1,advance! = sst_step_ka!,saved_clock=s.last_global_dt)
    accepted=false
    try
        for k in 1:sst_substeps
            k>1 && prepare_dt_ka!(s,0.45f0)
            advance!(s;corrected_wall_metric,nonorth_diffusion)
            _airfoil_physical_state(s.rho,s.ux,s.uy,s.T) && _airfoil_valid_sst_state(s) ||
                return (accepted=false,merit_after=NaN)
        end
        refresh_raw_residual_ka!(s)
        coupled_merit=_implicit_merit(s)
        isfinite(coupled_merit) || return (accepted=false,merit_after=coupled_merit)
        accepted=true
        (accepted=true,merit_after=coupled_merit)
    finally
        if !accepted
            _airfoil_coupled_restore!(s,c)
            s.last_global_dt=saved_clock
            refresh_raw_residual_ka!(s)
        end
    end
end

# A failed Newton direction is not proof that no steady solution exists. An
# explicitly requested, bounded burst of the original stable RK march can
# change the local linearization before retrying Newton. It preserves the
# original equations and is reported separately from an accepted implicit step.
# No recovery event can declare convergence. Failed recovery restores every
# evolved flow/turbulence field, including auxiliaries updated by SST.
function _airfoil_explicit_recovery!(s::KSol,c::JFNKAirfoilController;
    corrected_wall_metric::Bool=_sst_default_wall_metric(s),nonorth_diffusion::Bool=false,
    _flow_advance! = step_ka!, _sst_advance! = sst_step_ka!)
    count=min(c.recovery_steps,c.recovery_limit-c.recovery_iterations)
    count>0 || return (accepted=false,recovery_steps=0)
    names=_AIRFOIL_COUPLED_STATE
    saved=map(name->copy(getproperty(s,name)),names)
    saved_clock=s.last_global_dt
    good=false
    try
        for _ in 1:count
            _flow_advance!(s,0.45f0)
            _sst_advance!(s;corrected_wall_metric,nonorth_diffusion)
            (_airfoil_physical_state(s.rho,s.ux,s.uy,s.T) && _airfoil_valid_sst_state(s)) ||
                return (accepted=false,recovery_steps=0)
        end
        refresh_raw_residual_ka!(s)
        after=_implicit_merit(s)
        good=isfinite(after) && all(isfinite,s.tk) && all(isfinite,s.tom) &&
             all(isfinite,s.nut)
        good || return (accepted=false,recovery_steps=0)
        c.recovery_iterations+=count;c.recovery_events+=1;c.iterations+=1
        c.cfl=c.restart_cfl
        empty!(c.merit_history)
        return (accepted=true,recovery_steps=count,merit_after=after)
    finally
        if !good
            for (name,array) in zip(names,saved);copyto!(getproperty(s,name),array);end
            s.last_global_dt=saved_clock
            refresh_raw_residual_ka!(s)
        end
    end
end

@inline _airfoil_next_cfl(used::Float32,alpha::Real,
    min_cfl::Float32,max_cfl::Float32,growth::Float32) =
    alpha>=0.9f0 ? min(max_cfl,used*growth) :
                    max(min_cfl,used*max(0.5f0,alpha))

"""One accepted steady RANS pseudo-time iteration or a reported rejection.
The controller halves pseudo CFL after a rejected flow step and grows it after
acceptance. A rejected result leaves the flow and SST state unchanged."""
function step_jfnk_rans_ka!(s::KSol,c::JFNKAirfoilController;
    linear_tol::Float64=0.01,linear_cycles::Int=1,
    fd_scale::Float32=1.0f0,fd_scheme::Symbol=:forward,
    fd_local_cap::Float32=1.0f-3,corrected_wall_metric::Bool=_sst_default_wall_metric(s),
    nonorth_diffusion::Bool=false,max_backtracks::Int=8,
    sst_substeps::Int=1,linear_maxiter::Union{Nothing,Int}=nothing,
    _sst_advance! = sst_step_ka!)
    s.sst || throw(ArgumentError("JFNK RANS controller requires SST"))
    1<=sst_substeps<=100 ||
        throw(ArgumentError("sst_substeps must be 1:100"))
    result=nothing
    attempts=0
    used=c.cfl
    reference=isempty(c.merit_history) ? NaN : maximum(c.merit_history)
    saved_clock=s.last_global_dt
    _airfoil_coupled_backup!(s,c)
    for attempt in 1:c.max_attempts
        attempts=attempt
        used=c.cfl
        result=step_jfnk_airfoil_ka!(s,c.workspace,used;
            linear_tol,linear_cycles,fd_scale,fd_scheme,fd_local_cap,max_backtracks,
            linear_maxiter,
            merit_reference=reference,
            globalization=c.globalization,physical_fraction=c.physical_fraction,
            krylov_workspace=c.krylov_workspace)
        if result.accepted
            coupled=_airfoil_coupled_sst!(s,c;corrected_wall_metric,
                nonorth_diffusion,sst_substeps,saved_clock,advance! = _sst_advance!)
            if !coupled.accepted
                result=merge(result,(accepted=false,alpha=0.0f0,
                    merit_after=result.merit_before,))
                c.rejections+=1
                c.cfl=max(c.min_cfl,used*0.5f0)
                used<=c.min_cfl && break
                continue
            end
            # SST changes the viscosity used by the flow equations. Report
            # the residual of the state actually returned to the caller,
            # retaining the pre-SST trial merit as acceptance evidence.
            coupled_merit=coupled.merit_after
            if c.nonmonotone_window>0
                push!(c.merit_history,result.merit_before)
                length(c.merit_history)>c.nonmonotone_window &&
                    popfirst!(c.merit_history)
            end
            c.iterations+=1
            # A tiny accepted line-search fraction is a failed pseudo-time
            # prediction, not evidence that a larger step will work. Shrink
            # the proposed CFL until the Newton direction accepts useful
            # fractions again; grow only after a near-full step.
            c.cfl=_airfoil_next_cfl(used,result.alpha,c.min_cfl,c.max_cfl,c.growth)
            return merge(result,(attempts=attempt,pseudo_cfl_used=used,
                                 pseudo_cfl_next=c.cfl,sst_substeps,
                                 merit_after=coupled_merit,
                                 flow_trial_merit_after=result.merit_after,
                                 step_kind=:implicit,recovery_steps=0,
                                 recovery_total=c.recovery_iterations))
        end
        c.rejections+=1
        c.cfl=max(c.min_cfl,used*0.5f0)
        used<=c.min_cfl && break
    end
    recovered=_airfoil_explicit_recovery!(s,c;corrected_wall_metric,nonorth_diffusion)
    if recovered.accepted
        return merge(result,recovered,(attempts,pseudo_cfl_used=used,
            pseudo_cfl_next=c.cfl,sst_substeps=1,step_kind=:explicit_recovery,
            flow_trial_merit_after=result.merit_after,
            recovery_total=c.recovery_iterations))
    end
    merge(result,(attempts,pseudo_cfl_used=used,
                  pseudo_cfl_next=c.cfl,sst_substeps,step_kind=:rejected,
                  flow_trial_merit_after=result.merit_after,
                  recovery_steps=0,recovery_total=c.recovery_iterations))
end
