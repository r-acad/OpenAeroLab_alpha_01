# Matrix-free pseudo-transient Newton-Krylov for the steady A+/C airfoil flow.
# Experimental: uses the exact nonlinear residual but a finite-difference
# Jacobian and optional scalar line or approximate-factorization right
# preconditioner. SST is segregated. The ADI variants have not shown a
# time-to-convergence gain on the qualified NACA 0012 study mesh.
using LinearAlgebra
export JFNKAirfoilWorkspace, build_jfnk_airfoil_workspace, step_jfnk_airfoil_ka!
export JFNKAirfoilController, build_jfnk_airfoil_controller, step_jfnk_rans_ka!

struct JFNKAirfoilWorkspace{L,V,A}
    line::L
    adi::A
    basis::Vector{V}
    preconditioned::Vector{V}
    rhs::V
    delta::V
    work::V
    base_residual::V
    full_rhs::V
    total_delta::V
    m::Int
end

function build_jfnk_airfoil_workspace(s::KSol;m::Int=8,
                                      preconditioner::Symbol=:line)
    2<=m<=24 || throw(ArgumentError("JFNK restart length must be 2:24"))
    preconditioner in (:line,:adi,:block_adi) ||
        throw(ArgumentError("preconditioner must be :line, :adi, or :block_adi"))
    vec()=ntuple(_->similar(s.dtf),4)
    v=vec()
    block_length=preconditioner==:adi ? s.NI : min(s.NI,64)
    adi=preconditioner!=:line ?
        (lamW=similar(s.dtf),lamE=similar(s.dtf),
         cp=similar(s.dtf,preconditioner==:adi ? Float64 : Float32),
         block_length=Int32(block_length),
         blocks_per_row=Int32(cld(s.NI,block_length))) : nothing
    JFNKAirfoilWorkspace(build_implicit_line_workspace(s),adi,
        [vec() for _ in 1:m+1],[vec() for _ in 1:m],v,
        vec(),vec(),vec(),vec(),vec(),m)
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
    af=Float32(a)
    for k in 1:4; @. dst[k]=dst[k]+af*src[k]; end
    dst
end
function _av_scale!(dst,a::Float64)
    af=Float32(a)
    for k in 1:4; @. dst[k]=af*dst[k]; end
    dst
end
function _av_dot(a,b)
    Float64(sum(a[1].*b[1] .+ a[2].*b[2] .+
                a[3].*b[3] .+ a[4].*b[4]))
end
_av_norm(a)=sqrt(max(0.0,_av_dot(a,a)))

# Frozen scalar inverse of the pseudo-transient operator. For RHS in
# residual-per-volume units, multiply by the cell volume before the solve.
# The ADI option applies streamwise, diagonal scale, then wall-normal factors.
function _airfoil_line_precondition!(out,input,s::KSol,w::JFNKAirfoilWorkspace)
    line=w.line; b=s.backend; ni=Int32(s.NI);nj=Int32(s.NJ)
    for k in 1:4
        @. w.work[k]=input[k]*s.vol
        if w.adi!==nothing
            _k_implicit_i_solve!(b)(out[k],w.adi.cp,w.work[k],
                w.adi.lamW,w.adi.lamE,line.d0,ni,nj,
                w.adi.block_length,w.adi.blocks_per_row;
                ndrange=Int(w.adi.blocks_per_row)*s.NJ)
            @. w.work[k]=out[k]*line.d0
        end
        _k_implicit_line_solve!(b)(out[k],line.bp,w.work[k],
            line.lamS,line.lamN,line.d0,ni,nj;ndrange=s.NI)
    end
    out
end

# A(v) = [R(U+eps v)-R(U)]/(eps V) + d0*v/V. The state is restored
# after each residual evaluation. Float32-aware epsilon grows with the
# overall conservative state norm and shrinks with the Krylov direction norm.
function _airfoil_fd_step(v,nv::Float64,ncell::Int,r::Float64,u::Float64,
                          E::Float64,fd_scale::Float32)
    state_norm=sqrt(ncell*(r*r+2*(r*u)^2+E*E))
    h=Float32(fd_scale*sqrt(eps(Float32))*(1.0+state_norm)/nv)
    # A global 2-norm step can perturb one wall cell by O(sqrt(N) eps^1/2)
    # when a preconditioned Krylov vector is localized there. Bound the
    # conservative change in every cell before sampling the nonlinear flux.
    inv_r=Float32(1/r)
    inv_m=Float32(1/(r*u))
    inv_e=Float32(1/E)
    max_scaled=maximum(@. max(abs(v[1])*inv_r,abs(v[2])*inv_m,
                            abs(v[3])*inv_m,abs(v[4])*inv_e))
    max_scaled>0 && (h=min(h,1.0f-3/max_scaled))
    isfinite(h) && h>0 || throw(ArgumentError("invalid matrix-free perturbation"))
    h
end

function _airfoil_jacobian_vec!(out,v,s::KSol,w::JFNKAirfoilWorkspace;
                                fd_scale::Float32=1.0f0)
    nv=_av_norm(v)
    nv==0.0 && return _av_fill!(out,0.0f0)
    line=w.line
    ncell=length(s.vol)
    r=Float64(s.rho_inf)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    E=r*(Float64(s.eint*s.T_inf)+0.5*u*u)
    h=_airfoil_fd_step(v,nv,ncell,r,u,E,fd_scale)
    b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g)
    physical=false
    for _ in 1:16
        _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,
            line.rho0,line.ux0,line.uy0,line.T0,v...,
            s.eint,-h,ni,nj,g;ndrange=(s.NI,s.NJ))
        if _airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
            physical=true
            break
        end
        h*=0.5f0
    end
    if !physical
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),
                             (line.rho0,line.ux0,line.uy0,line.T0))
            copyto!(dst,src)
        end
        throw(ArgumentError("no physical matrix-free perturbation after 16 halvings"))
    end
    try
        _raw_residual_ka!(s,Int32(2),Int32(1))
        current=_airfoil_residual(s)
        for k in 1:4
            @. out[k]=((current[k]-w.base_residual[k])/h+
                       line.d0*v[k])/s.vol
        end
    finally
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),
                             (line.rho0,line.ux0,line.uy0,line.T0))
            copyto!(dst,src)
        end
    end
    out
end

function _airfoil_fgmres!(s::KSol,w::JFNKAirfoilWorkspace;
                          linear_tol::Float64=0.1,fd_scale::Float32=1.0f0)
    0<linear_tol<1 || throw(ArgumentError("linear_tol must be in (0,1)"))
    m=w.m;V=w.basis;Z=w.preconditioned
    H=zeros(Float64,m+1,m);g=zeros(Float64,m+1)
    _av_fill!(w.delta,0.0f0)
    beta=_av_norm(w.rhs)
    beta==0 && return (0,0.0)
    _av_copy!(V[1],w.rhs);_av_scale!(V[1],1/beta);g[1]=beta
    rel=Inf
    for j in 1:m
        _airfoil_line_precondition!(Z[j],V[j],s,w)
        _airfoil_jacobian_vec!(V[j+1],Z[j],s,w;fd_scale)
        for i in 1:j
            H[i,j]=_av_dot(V[j+1],V[i])
            _av_axpy!(V[j+1],-H[i,j],V[i])
        end
        H[j+1,j]=_av_norm(V[j+1])
        isfinite(H[j+1,j]) || throw(ArgumentError("nonfinite JFNK Arnoldi vector"))
        H[j+1,j]>0 && _av_scale!(V[j+1],1/H[j+1,j])
        y=H[1:j+1,1:j] \ g[1:j+1]
        rel=norm(g[1:j+1]-H[1:j+1,1:j]*y)/beta
        if rel<=linear_tol || j==m
            for k in 1:j; _av_axpy!(w.delta,y[k],Z[k]); end
            return (j,rel)
        end
    end
    return (m,rel)
end

function _airfoil_fgmres_restarted!(s::KSol,w::JFNKAirfoilWorkspace;
                                    linear_tol::Float64=0.1,
                                    linear_cycles::Int=1,
                                    fd_scale::Float32=1.0f0)
    1<=linear_cycles<=4 || throw(ArgumentError("linear_cycles must be 1:4"))
    _av_copy!(w.full_rhs,w.rhs)
    _av_fill!(w.total_delta,0.0f0)
    beta=_av_norm(w.full_rhs)
    beta==0 && return (0,0.0)
    total=0;rel=1.0
    for _ in 1:linear_cycles
        kits,_=_airfoil_fgmres!(s,w;linear_tol,fd_scale)
        total+=kits
        _av_axpy!(w.total_delta,1.0,w.delta)
        _airfoil_jacobian_vec!(w.work,w.total_delta,s,w;fd_scale)
        for k in 1:4; @. w.rhs[k]=w.full_rhs[k]-w.work[k]; end
        rel=_av_norm(w.rhs)/beta
        isfinite(rel) || throw(ArgumentError("nonfinite restarted linear residual"))
        rel<=linear_tol && break
    end
    _av_copy!(w.delta,w.total_delta)
    total,rel
end

"""One opt-in pseudo-transient Newton-Krylov flow iteration on a single-phase
free-air airfoil. Returns acceptance, line-search factor, linear iterations,
linear relative residual, and nonlinear merit. This does not advance SST."""
function step_jfnk_airfoil_ka!(s::KSol,w::JFNKAirfoilWorkspace,cfl::Float32;
                               linear_tol::Float64=0.1,linear_cycles::Int=1,
                               max_backtracks::Int=8,fd_scale::Float32=1.0f0,
                               merit_reference::Float64=NaN,
                               krylov_workspace=nothing)
    _implicit_admit(s,cfl)
    isfinite(fd_scale) && 0.1f0<=fd_scale<=16.0f0 ||
        throw(ArgumentError("fd_scale must be in [0.1,16]"))
    1<=max_backtracks<=16 || throw(ArgumentError("max_backtracks must be 1:16"))
    size(w.rhs[1])==size(s.dtf) && size(w.line.rho0)==size(s.rho) ||
        throw(ArgumentError("JFNK workspace does not match solver"))
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
                physical_trials=0,merit_reference=reference)
    end
    _k_implicit_line_coeffs!(b)(line.lamS,line.lamN,line.d0,
        s.rho,s.ux,s.uy,s.T,s.nut,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        cfl,ni,nj,g;ndrange=(s.NI,s.NJ))
    if w.adi!==nothing
        _k_implicit_i_coeffs!(b)(w.adi.lamW,w.adi.lamE,
            s.rho,s.ux,s.uy,s.T,s.nut,s.Six,s.Siy,s.vol,
            s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
            ni,nj,g;ndrange=(s.NI,s.NJ))
    end
    for k in 1:4; @. w.rhs[k]=w.base_residual[k]/s.vol; end
    kits,rel=krylov_workspace===nothing ?
        _airfoil_fgmres_restarted!(s,w;linear_tol,linear_cycles,fd_scale) :
        _airfoil_krylov_gmres!(s,w,krylov_workspace;linear_tol,linear_cycles,fd_scale)
    alpha=1.0f0
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
            if isfinite(after) &&
               after<reference*(1.0-1e-4*Float64(alpha)) &&
               after<before*1.02
                prepare_dt_ka!(s,min(cfl,0.45f0))
                return (accepted=true,alpha,linear_iterations=kits,
                        linear_relative=rel,merit_before=before,merit_after=after,
                        best_trial_merit=best,best_trial_alpha=best_alpha,
                        physical_trials,merit_reference=reference)
            end
        end
        alpha*=0.5f0
    end
    for (dst,src) in zip(saved,backup);copyto!(dst,src);end
    _raw_residual_ka!(s,Int32(2),Int32(1))
    return (accepted=false,alpha=0.0f0,linear_iterations=kits,
            linear_relative=rel,merit_before=before,merit_after=before,
            best_trial_merit=best,best_trial_alpha=best_alpha,
            physical_trials,merit_reference=reference)
end

"""Adaptive pseudo-CFL controller for the opt-in steady A+/C SST march.
This is a convergence experiment. It does not advance physical time, and SST
still uses the solver's segregated, stability-capped transport step. A positive
`nonmonotone_window` uses the maximum of recent flow merits for line search;
this is experimental and does not establish a steady RANS convergence gate."""
mutable struct JFNKAirfoilController{W,K}
    workspace::W
    krylov_workspace::K
    cfl::Float32
    min_cfl::Float32
    max_cfl::Float32
    growth::Float32
    max_attempts::Int
    iterations::Int
    rejections::Int
    nonmonotone_window::Int
    merit_history::Vector{Float64}
end

function build_jfnk_airfoil_controller(s::KSol;m::Int=4,
    initial_cfl::Float32=2.0f0,min_cfl::Float32=0.05f0,
    max_cfl::Float32=2.0f0,growth::Float32=1.05f0,
    max_attempts::Int=4,linear_solver::Symbol=:native,
    nonmonotone_window::Int=0,preconditioner::Symbol=:line)
    isfinite(min_cfl) && isfinite(initial_cfl) && isfinite(max_cfl) &&
        0.0f0<min_cfl<=initial_cfl<=max_cfl<=10.0f0 ||
        throw(ArgumentError("invalid JFNK pseudo-CFL bounds"))
    isfinite(growth) && 1.0f0<growth<=2.0f0 ||
        throw(ArgumentError("JFNK CFL growth must be in (1,2]"))
    1<=max_attempts<=8 || throw(ArgumentError("max_attempts must be 1:8"))
    linear_solver in (:native,:krylov) ||
        throw(ArgumentError("linear_solver must be :native or :krylov"))
    0<=nonmonotone_window<=20 ||
        throw(ArgumentError("nonmonotone_window must be 0:20"))
    workspace=build_jfnk_airfoil_workspace(s;m,preconditioner)
    krylov_workspace=linear_solver==:krylov ?
        build_krylov_airfoil_workspace(s,workspace) : nothing
    JFNKAirfoilController(workspace,krylov_workspace,initial_cfl,
        min_cfl,max_cfl,growth,max_attempts,0,0,
        nonmonotone_window,Float64[])
end

"""One accepted steady RANS pseudo-time iteration or a reported rejection.
The controller halves pseudo CFL after a rejected flow step and grows it after
acceptance. A rejected result leaves the flow and SST state unchanged."""
function step_jfnk_rans_ka!(s::KSol,c::JFNKAirfoilController;
    linear_tol::Float64=0.01,linear_cycles::Int=1,
    fd_scale::Float32=1.0f0,corrected_wall_metric::Bool=false,
    nonorth_diffusion::Bool=false,max_backtracks::Int=8)
    s.sst || throw(ArgumentError("JFNK RANS controller requires SST"))
    result=nothing
    attempts=0
    used=c.cfl
    reference=isempty(c.merit_history) ? NaN : maximum(c.merit_history)
    for attempt in 1:c.max_attempts
        attempts=attempt
        used=c.cfl
        result=step_jfnk_airfoil_ka!(s,c.workspace,used;
            linear_tol,linear_cycles,fd_scale,max_backtracks,
            merit_reference=reference,
            krylov_workspace=c.krylov_workspace)
        if result.accepted
            sst_step_ka!(s;corrected_wall_metric,nonorth_diffusion)
            if c.nonmonotone_window>0
                push!(c.merit_history,result.merit_before)
                length(c.merit_history)>c.nonmonotone_window &&
                    popfirst!(c.merit_history)
            end
            c.iterations+=1
            c.cfl=min(c.max_cfl,used*c.growth)
            return merge(result,(attempts=attempt,pseudo_cfl_used=used,
                                 pseudo_cfl_next=c.cfl))
        end
        c.rejections+=1
        c.cfl=used*0.5f0
        c.cfl<c.min_cfl && break
    end
    merge(result,(attempts,pseudo_cfl_used=used,
                  pseudo_cfl_next=c.cfl))
end
