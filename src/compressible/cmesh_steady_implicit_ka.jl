#=
Wall-normal line-implicit pseudo-time iteration for single-phase, free-air C meshes.

This is an experimental approximate Jacobian, not a Newton solve. The four
conserved flow equations use their original finite-volume residual. A scalar
spectral bound forms the tridiagonal wall-normal operator, applied separately
to each conserved increment. The streamwise direction remains explicit.
Backtracking rejects nonphysical or residual-increasing updates. SST remains
segregated and must be advanced by the caller with its own safe clock.
=#
export ImplicitLineWorkspace, build_implicit_line_workspace, step_implicit_line_ka!

struct ImplicitLineWorkspace{A,G}
    lamS::A
    lamN::A
    d0::A
    bp::A
    dr::A
    dmx::A
    dmy::A
    dE::A
    rho0::G
    ux0::G
    uy0::G
    T0::G
end

function build_implicit_line_workspace(s::KSol)
    z() = similar(s.dtf)
    ImplicitLineWorkspace(z(),z(),z(),z(),z(),z(),z(),z(),
                          similar(s.rho),similar(s.ux),similar(s.uy),similar(s.T))
end

@kernel function _k_implicit_line_coeffs!(lamS,lamN,d0,
    @Const(rho),@Const(ux),@Const(uy),@Const(T),@Const(nut),
    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
    nu::Float32,rho_inf::Float32,suth::Float32,air_sutherland::Int32,
    sst_on::Int32,cfl::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j = @index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g; ja=j+g
        u=ux[ia,ja]; v=uy[ia,ja]
        c=3.0f0*sqrt(max(T[ia,ja],1.0f-3))
        siw=hypot(Six[i,j],Siy[i,j]); sie=hypot(Six[i+1,j],Siy[i+1,j])
        sw=hypot(Sjx[i,j],Sjy[i,j]); sn=hypot(Sjx[i,j+1],Sjy[i,j+1])
        li=0.5f0*(abs(u*Six[i,j]+v*Siy[i,j])+
                    abs(u*Six[i+1,j]+v*Siy[i+1,j])+c*(siw+sie))
        nv=_air_nu_local(rho[ia,ja],T[ia,ja],nu,rho_inf,suth,air_sutherland)+
           (sst_on==Int32(1) ? nut[ia,ja] : 0.0f0)
        # V/dtau is limited by the streamwise convective and viscous clocks.
        vi=5.0f0*nv*max(siw,sie)^2/max(vol[i,j],1.0f-30)
        d0[i,j]=max(li/cfl,vi,1.0f-30)
        lamS[i,j]=abs(u*Sjx[i,j]+v*Sjy[i,j])+c*sw+
                  2.0f0*nv*sw^2/max(vol[i,j],1.0f-30)
        lamN[i,j]=abs(u*Sjx[i,j+1]+v*Sjy[i,j+1])+c*sn+
                  2.0f0*nv*sn^2/max(vol[i,j],1.0f-30)
    end
end

# Thomas solve of (V/dtau + J_j) d = R. Boundary faces enter the diagonal;
# the wake cut and wall are supplied by the existing nonlinear residual.
@kernel function _k_implicit_line_solve!(d,bp,@Const(rhs),@Const(lamS),
                                          @Const(lamN),@Const(d0),NI::Int32,NJ::Int32)
    i=@index(Global,Linear)
    @inbounds if i<=NI
        for j in 1:NJ
            lo=j==1 ? 0.0f0 : lamS[i,j]
            hi=j==NJ ? 0.0f0 : lamN[i,j]
            den=d0[i,j]+lamS[i,j]+lamN[i,j]
            if j>1
                den-=lo*bp[i,j-1]
            end
            bp[i,j]=hi/den
            d[i,j]=(rhs[i,j]+(j==1 ? 0.0f0 : lo*d[i,j-1]))/den
        end
        j=NJ-1
        next=d[i,NJ]
        while j>=1
            next=d[i,j]+bp[i,j]*next
            d[i,j]=next
            j-=1
        end
    end
end

@kernel function _k_implicit_line_trial!(rho,ux,uy,T,@Const(rho0),
    @Const(ux0),@Const(uy0),@Const(T0),@Const(dr),@Const(dmx),
    @Const(dmy),@Const(dE),eint::Float32,alpha::Float32,
    NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g; ja=j+g
        r0=rho0[ia,ja]; u0=ux0[ia,ja]; v0=uy0[ia,ja]
        E0=r0*(eint*T0[ia,ja]+0.5f0*(u0*u0+v0*v0))
        r=r0-alpha*dr[i,j]
        mx=r0*u0-alpha*dmx[i,j]
        my=r0*v0-alpha*dmy[i,j]
        E=E0-alpha*dE[i,j]
        if isfinite(r) && r>1.0f-4 && isfinite(mx) && isfinite(my) && isfinite(E)
            u=mx/r; v=my/r
            temp=(E/r-0.5f0*(u*u+v*v))/eint
            rho[ia,ja]=r; ux[ia,ja]=u; uy[ia,ja]=v; T[ia,ja]=temp
        else
            rho[ia,ja]=NaN32; ux[ia,ja]=NaN32
            uy[ia,ja]=NaN32; T[ia,ja]=NaN32
        end
    end
end

function _implicit_merit(s::KSol)
    vr=s.vol
    r=max(Float64(s.rho_inf),1e-12)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u)
    n=length(vr)
    inv_r=Float32(1/r)
    inv_m=Float32(1/(r*u))
    inv_e=Float32(1/(r*e))
    # One fused GPU reduction replaces four volume-division temporaries and
    # four synchronizing reductions in every nonlinear line-search trial.
    total=sum(@. ((s.dc_r/vr)*inv_r)^2 +
                 ((s.dc_mx/vr)*inv_m)^2 +
                 ((s.dc_my/vr)*inv_m)^2 +
                 ((s.dc_e/vr)*inv_e)^2)
    sqrt(Float64(total)/n)
end

function _airfoil_physical_state(rho,ux,uy,T)
    mapreduce((r,u,v,t)->isfinite(r) && isfinite(u) && isfinite(v) &&
                          isfinite(t) && r>1.0f-4 && t>1.0f-3,
              &,rho,ux,uy,T)
end

function _implicit_admit(s::KSol,cfl::Float32)
    isfinite(cfl) && 0.0f0<cfl<=10.0f0 ||
        throw(ArgumentError("implicit pseudo CFL must be in (0,10]"))
    GLOBAL_DT[]==0 || throw(ArgumentError("implicit steady step requires local pseudo-time"))
    s.cav==0 && s.outer_mode==0 && s.outlet_mode==0 && s.ff_mix==0 ||
        throw(ArgumentError("implicit line step supports single-phase free-air C meshes only"))
    s.sfd==0 && s.irsj==0 && s.irs_ac==0 && s.acou==0 &&
        s.sponge_sigma==0.0f0 ||
        throw(ArgumentError("implicit line step requires filters and residual smoothing off"))
    s.emix==0.0f0 || throw(ArgumentError("implicit line step requires single-phase air energy"))
    s.wall_flux || throw(ArgumentError("implicit line step requires analytic airfoil wall flux"))
    nothing
end

"""Try one steady flow iteration. Returns `(accepted, alpha, merit_before, merit_after)`.
An unaccepted trial restores the exact primitive state. The caller controls
SST separately and must use `prepare_dt_ka!` with a stable SST clock."""
function step_implicit_line_ka!(s::KSol,w::ImplicitLineWorkspace,cfl::Float32;
                                max_backtracks::Int=8)
    _implicit_admit(s,cfl)
    1<=max_backtracks<=16 || throw(ArgumentError("max_backtracks must be 1:16"))
    size(w.dr)==size(s.dtf) && size(w.rho0)==size(s.rho) ||
        throw(ArgumentError("implicit workspace does not match solver"))
    b=s.backend; ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g)
    saved=(s.rho,s.ux,s.uy,s.T)
    backup=(w.rho0,w.ux0,w.uy0,w.T0)
    for (dst,src) in zip(backup,saved); copyto!(dst,src); end
    _raw_residual_ka!(s,Int32(2),Int32(1))
    before=_implicit_merit(s)
    isfinite(before) || throw(ArgumentError("nonfinite initial flow residual"))
    if before==0.0
        prepare_dt_ka!(s,min(cfl,0.45f0))
        return (accepted=true,alpha=0.0f0,merit_before=before,merit_after=before)
    end
    _k_implicit_line_coeffs!(b)(w.lamS,w.lamN,w.d0,s.rho,s.ux,s.uy,s.T,
        s.nut,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,s.rho_inf,s.suth,
        s.air_sutherland,Int32(s.sst),cfl,ni,nj,g;ndrange=(s.NI,s.NJ))
    for (dst,src) in ((w.dr,s.dc_r),(w.dmx,s.dc_mx),
                      (w.dmy,s.dc_my),(w.dE,s.dc_e))
        _k_implicit_line_solve!(b)(dst,w.bp,src,w.lamS,w.lamN,w.d0,ni,nj;
                                   ndrange=s.NI)
    end
    alpha=1.0f0
    for _ in 1:max_backtracks
        _k_implicit_line_trial!(b)(s.rho,s.ux,s.uy,s.T,backup...,
            w.dr,w.dmx,w.dmy,w.dE,s.eint,alpha,ni,nj,g;
            ndrange=(s.NI,s.NJ))
        # Check physical state before running the flux kernels.
        good=_airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
        if good
            _raw_residual_ka!(s,Int32(2),Int32(1))
            after=_implicit_merit(s)
            if isfinite(after) && after<before*(1.0-1e-4*Float64(alpha))
                # SST's explicit transport still needs its own stability clock.
                prepare_dt_ka!(s,min(cfl,0.45f0))
                return (accepted=true,alpha,merit_before=before,merit_after=after)
            end
        end
        alpha*=0.5f0
    end
    for (dst,src) in zip(saved,backup); copyto!(dst,src); end
    _raw_residual_ka!(s,Int32(2),Int32(1))
    return (accepted=false,alpha=0.0f0,merit_before=before,merit_after=before)
end
