# Included in CMeshFreeSurfaceProjection. Matrix-free pressure projection on
# the selected KA device; CUDA uses native sparse matrix/vector products.
using KernelAbstractions
const PKA=KernelAbstractions
export ProjectionKA, build_projection_ka, projection_step_ka!, projection_host

mutable struct ProjectionKA
    backend
    host::ProjectionState
    mass
    momentum
    mass_trial
    momentum_trial
    delta
    qi
    qj
    qi_trial
    qj_trial
    Fi
    Fj
    rate
    status
    geometry
    B
    Bt
    B2
    inverse_mass
    diagonal
    rhs
    x
    accepted_pressure
    r
    z
    direction
    ap
    temporary
    correction
    elapsed::Float64
    steps::Int
    ledger::Vector{Float64}
    pressure_residual::Float64
    trial_pressure_residual::Float64
    pressure_iterations::Int
    gravity_cap::Float64
    preconditioner
    preconditioner_name::String
    gauge::Int
end

@kernel function _k_p_gather!(out,@Const(input),@Const(permutation),n)
    k=@index(Global,Linear)
    if k<=n;@inbounds out[k]=input[permutation[k]];end
end
@kernel function _k_p_scatter!(out,@Const(input),@Const(permutation),n)
    k=@index(Global,Linear)
    if k<=n;@inbounds out[permutation[k]]=input[k];end
end

@kernel function _k_p_rate!(rate,@Const(qi),@Const(qj),@Const(vol),ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        im=i==1 ? ni : i-1
        @inbounds rate[i,j]=(max(qi[i,j],0.)+max(-qi[im,j],0.)+max(qj[i,j+1],0.)+max(-qj[i,j],0.))/vol[i,j]
    end
end
@kernel function _k_p_i!(flux,@Const(mass),@Const(momentum),@Const(qi),ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        ip=i==ni ? 1 : i+1;q=@inbounds qi[i,j];donor=q>=0 ? i : ip
        for k in 1:3;@inbounds flux[i,j,k]=q*mass[donor,j,k];end
        for k in 1:2;@inbounds flux[i,j,k+3]=q*momentum[donor,j,k];end
    end
end
@kernel function _k_p_j!(flux,@Const(mass),@Const(momentum),@Const(qj),@Const(Y),densities,speed,seed,h,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj+1
        q=@inbounds qj[i,j]
        if j==1
            for k in 1:5;@inbounds flux[i,j,k]=0.;end
        elseif j==nj+1 && q<0
            water=CMeshFreeSurfaceAirfoil._wet_edge_fraction(@inbounds(Y[i,j]),@inbounds(Y[i+1,j]),h)
            ml=water*(1-seed)*densities[1];mv=water*seed*densities[2];ma=(1-water)*densities[3]
            incoming=(ml,mv,ma,(ml+mv+ma)*speed,0.)
            for k in 1:5;@inbounds flux[i,j,k]=q*incoming[k];end
        else
            donor=j==nj+1 ? nj : q>=0 ? j-1 : j
            for k in 1:3;@inbounds flux[i,j,k]=q*mass[i,donor,k];end
            for k in 1:2;@inbounds flux[i,j,k+3]=q*momentum[i,donor,k];end
        end
    end
end
@kernel function _k_p_transport!(mt,pt,delta,status,@Const(mass),@Const(momentum),@Const(fi),@Const(fj),@Const(vol),@Const(rho0),gravity,dt,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        im=i==1 ? ni : i-1;volume=@inbounds vol[i,j]
        rho=@inbounds(mass[i,j,1]+mass[i,j,2]+mass[i,j,3]);rhot=0.;good=true
        for k in 1:3
            value=@inbounds(mass[i,j,k]-dt*(fi[i,j,k]-fi[im,j,k]+fj[i,j+1,k]-fj[i,j,k])/volume)
            @inbounds mt[i,j,k]=value;rhot+=value;good &= isfinite(value) && value>=0
        end
        for k in 1:2
            source=k==2 ? -gravity*(rho-@inbounds(rho0[i,j])) : 0.
            value=@inbounds(momentum[i,j,k]-dt*(fi[i,j,k+3]-fi[im,j,k+3]+fj[i,j+1,k+3]-fj[i,j,k+3])/volume+dt*source)
            @inbounds pt[i,j,k]=value
            @inbounds delta[i,j,k]=value/rhot-momentum[i,j,k]/rho
            good &= isfinite(value)
        end
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end
@kernel function _k_p_qi!(qt,@Const(q),@Const(delta),@Const(sx),@Const(sy),ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        ip=i==ni ? 1 : i+1
        @inbounds qt[i,j]=q[i,j]+.5(delta[i,j,1]+delta[ip,j,1])*sx[i,j]+.5(delta[i,j,2]+delta[ip,j,2])*sy[i,j]
    end
end
@kernel function _k_p_qj!(qt,@Const(q),@Const(delta),@Const(sx),@Const(sy),@Const(kind),ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj+1
        value=@inbounds q[i,j]
        if j==nj+1
            @inbounds(kind[i])==2 && (value+=@inbounds(delta[i,nj,1]*sx[i,j]+delta[i,nj,2]*sy[i,j]))
        elseif j>1
            value+=@inbounds(.5(delta[i,j-1,1]+delta[i,j,1])*sx[i,j]+.5(delta[i,j-1,2]+delta[i,j,2])*sy[i,j])
        end
        @inbounds qt[i,j]=value
    end
end
@kernel function _k_p_rhs!(rhs,inverse_mass,@Const(qi),@Const(qj),@Const(mt),@Const(vol),dt,gauge,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        im=i==1 ? ni : i-1;n=ni*nj;idx=i+(j-1)*ni
        if idx>gauge;@inbounds rhs[idx-gauge]=-(qi[i,j]-qi[im,j]+qj[i,j+1]-qj[i,j])/dt;end
        rho=@inbounds(mt[i,j,1]+mt[i,j,2]+mt[i,j,3]);v=@inbounds vol[i,j]
        @inbounds inverse_mass[idx]=1/(rho*v)
        @inbounds inverse_mass[idx+n]=1/(rho*v)
    end
end
@kernel function _k_p_commit!(pt,status,delta,@Const(mt),@Const(correction),densities,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        idx=i+(j-1)*ni;n=ni*nj;rho=@inbounds(mt[i,j,1]+mt[i,j,2]+mt[i,j,3]);closure=0.
        for k in 1:3;closure+=@inbounds(mt[i,j,k])/densities[k];end
        good=abs(closure-1)<1e-8
        for k in 1:2
            change=@inbounds correction[idx+(k-1)*n]
            @inbounds delta[i,j,k]=change
            value=@inbounds(pt[i,j,k])+rho*change
            @inbounds pt[i,j,k]=value
            good &= isfinite(value)
        end
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

function build_projection_ka(backend,s::ProjectionState;cuda=false,sparse_device=nothing,triangular_solve=nothing,preconditioner=:initial_cholesky)
    dev(a)=begin buffer=PKA.allocate(backend,eltype(a),size(a));copyto!(buffer,a);buffer;end
    ni,nj=s.grid.NI,s.grid.NJ;n=ni*nj
    z(n...)=dev(zeros(Float64,n...));flags(n...)=dev(zeros(UInt8,n...))
    gauge=any(==(UInt8(2)),s.boundary_kind) ? 0 : 1
    B=gauge==0 ? s.kinematic : s.kinematic[2:end,:];np=n-gauge
    B2=copy(B);B2.nzval.=B2.nzval.^2
    cuda && sparse_device===nothing && throw(ArgumentError("CUDA sparse constructor callback required"))
    matrices=cuda ? (sparse_device(B),sparse_device(sparse(transpose(B))),sparse_device(B2)) : (B,sparse(transpose(B)),B2)
    names=(:volume,:rho_reference,:six,:siy,:sjx,:sjy,:Y,:kind)
    hostgeo=(s.grid.volume,s.rho_reference,s.grid.six,s.grid.siy,s.grid.sjx,s.grid.sjy,s.grid.Y,s.boundary_kind)
    geo=NamedTuple{names}(Tuple(dev(v) for v in hostgeo))
    altitude=minimum(s.grid.volume[i,j]/max(hypot(s.grid.six[i,j],s.grid.siy[i,j]),hypot(s.grid.sjx[i,j],s.grid.sjy[i,j])) for i in 1:ni,j in 1:nj)
    preconditioner in (:initial_cholesky,:diagonal) || throw(ArgumentError("invalid projection preconditioner"))
    prec=nothing;precname="diagonal"
    if preconditioner===:initial_cholesky
        cuda && triangular_solve===nothing && throw(ArgumentError("native CUDA triangular solve callback required"))
        rho=_rho(s);inverse_mass=vcat(vec(1 ./(rho.*s.grid.volume)),vec(1 ./(rho.*s.grid.volume)))
        factor=cholesky(Symmetric(B*spdiagm(0=>inverse_mass)*transpose(B)))
        L=sparse(factor.L)
        prec=(L=cuda ? sparse_device(L) : L,permutation=dev(factor.p),rp=z(np),y=z(np),zp=z(np),solve=triangular_solve)
        precname="initial-density Cholesky; applied on selected device as a preconditioner for the current-density operator"
    end
    ProjectionKA(backend,deepcopy(s),dev(s.mass),dev(s.momentum),z(ni,nj,3),z(ni,nj,2),z(ni,nj,2),dev(s.qi),dev(s.qj),z(ni,nj),z(ni,nj+1),
        z(ni,nj,5),z(ni,nj+1,5),z(ni,nj),flags(ni,nj),geo,matrices...,z(2n),z(np),z(np),z(np),dev(vec(s.pressure)[gauge+1:end]),z(np),z(np),z(np),z(np),z(2n),z(2n),
        s.elapsed,s.steps,copy(s.boundary_mass_integral),s.pressure_residual,0.,0,sqrt(altitude/max(s.grid.gravity,1e-30)),prec,precname,gauge)
end

function _p_apply!(c,out,x)
    mul!(c.temporary,c.Bt,x)
    c.temporary .*=c.inverse_mass
    mul!(out,c.B,c.temporary)
    out
end
function _p_precondition!(c)
    prec=c.preconditioner
    if prec===nothing
        c.z.=c.r./c.diagonal
    else
        n=length(c.r)
        _k_p_gather!(c.backend)(prec.rp,c.r,prec.permutation,n;ndrange=n);PKA.synchronize(c.backend)
        if prec.solve===nothing
            copyto!(prec.y,LowerTriangular(prec.L)\prec.rp)
            copyto!(prec.zp,UpperTriangular(transpose(prec.L))\prec.y)
        else
            prec.solve('N',prec.L,prec.rp,prec.y);prec.solve('T',prec.L,prec.y,prec.zp)
        end
        _k_p_scatter!(c.backend)(c.z,prec.zp,prec.permutation,n;ndrange=n);PKA.synchronize(c.backend)
    end
    nothing
end
function _p_cg!(c;rtol=1e-9,atol=1e-10,max_iterations=4000)
    rhsmax=maximum(abs,c.rhs);tol=max(atol,rtol*rhsmax)
    if rhsmax<=tol;fill!(c.x,0.);c.trial_pressure_residual=rhsmax;c.pressure_iterations=0;return;end
    copyto!(c.x,c.accepted_pressure)
    _p_apply!(c,c.ap,c.x);c.r.=c.rhs.-c.ap
    initial=maximum(abs,c.r)
    if initial<=tol;c.trial_pressure_residual=initial;c.pressure_iterations=0;return;end
    mul!(c.diagonal,c.B2,c.inverse_mass)
    minimum(c.diagonal)>0 || error("projection pressure diagonal is not positive")
    _p_precondition!(c);copyto!(c.direction,c.z)
    rz=Float64(dot(c.r,c.z));residual=rhsmax
    for iteration in 1:max_iterations
        _p_apply!(c,c.ap,c.direction)
        denominator=Float64(dot(c.direction,c.ap))
        denominator>0 && isfinite(denominator) || error("projection pressure CG lost positive definiteness")
        alpha=rz/denominator
        c.x .+=alpha.*c.direction;c.r .-=alpha.*c.ap
        if true
            residual=maximum(abs,c.r)
            if residual<=tol
                _p_apply!(c,c.ap,c.x);c.ap .-=c.rhs
                actual=maximum(abs,c.ap)
                actual<=4tol || error("projection pressure recursive residual disagrees with actual operator")
                c.trial_pressure_residual=actual;c.pressure_iterations=iteration;return
            end
        end
        _p_precondition!(c);next_rz=Float64(dot(c.r,c.z))
        beta=next_rz/rz;c.direction.=c.z.+beta.*c.direction;rz=next_rz
    end
    error("projection pressure CG not converged after $max_iterations iterations; residual=$residual; state uncommitted")
end

function projection_step_ka!(c::ProjectionKA;dt_cap=Inf,cfl=.35,pressure_rtol=1e-9,pressure_atol=1e-10,pressure_max_iterations=4000)
    0<cfl<=.5 && dt_cap>0 || throw(ArgumentError("invalid GPU projection clock settings"))
    g=c.host.grid;b=c.backend;geo=c.geometry;ni,nj=g.NI,g.NJ
    _k_p_rate!(b)(c.rate,c.qi,c.qj,geo.volume,ni,nj;ndrange=(ni,nj));PKA.synchronize(b)
    rate=maximum(c.rate);dt=min(Float64(dt_cap),rate>0 ? cfl/rate : Inf,sqrt(cfl)*c.gravity_cap)
    dt>64eps(max(c.elapsed,dt)) || throw(ArgumentError("projection step is below physical-clock resolution; finish the duration within roundoff"))
    _k_p_i!(b)(c.Fi,c.mass,c.momentum,c.qi,ni,nj;ndrange=(ni,nj))
    _k_p_j!(b)(c.Fj,c.mass,c.momentum,c.qj,geo.Y,c.host.phase_density,g.speed,g.vapor_seed,g.water_level,ni,nj;ndrange=(ni,nj+1))
    _k_p_transport!(b)(c.mass_trial,c.momentum_trial,c.delta,c.status,c.mass,c.momentum,c.Fi,c.Fj,geo.volume,geo.rho_reference,g.gravity,dt,ni,nj;ndrange=(ni,nj));PKA.synchronize(b)
    maximum(c.status)==0 || error("projection phase transport failed admission; previous state retained")
    _k_p_qi!(b)(c.qi_trial,c.qi,c.delta,geo.six,geo.siy,ni,nj;ndrange=(ni,nj))
    _k_p_qj!(b)(c.qj_trial,c.qj,c.delta,geo.sjx,geo.sjy,geo.kind,ni,nj;ndrange=(ni,nj+1))
    _k_p_rhs!(b)(c.rhs,c.inverse_mass,c.qi_trial,c.qj_trial,c.mass_trial,geo.volume,dt,c.gauge,ni,nj;ndrange=(ni,nj));PKA.synchronize(b)
    _p_cg!(c;rtol=pressure_rtol,atol=pressure_atol,max_iterations=pressure_max_iterations)
    mul!(c.temporary,c.Bt,c.x);c.correction.=dt.*c.inverse_mass.*c.temporary
    _k_p_commit!(b)(c.momentum_trial,c.status,c.delta,c.mass_trial,c.correction,c.host.phase_density,ni,nj;ndrange=(ni,nj));PKA.synchronize(b)
    maximum(c.status)==0 || error("projection phase-volume/momentum admission failed; previous state retained")
    ledger_increment=dt.*[Float64(sum(view(c.Fj,:,nj+1,k))) for k in 1:3]
    # q_star plus the same adjoint cell-velocity correction used by momentum.
    _k_p_qi!(b)(c.qi_trial,c.qi_trial,c.delta,geo.six,geo.siy,ni,nj;ndrange=(ni,nj))
    _k_p_qj!(b)(c.qj_trial,c.qj_trial,c.delta,geo.sjx,geo.sjy,geo.kind,ni,nj;ndrange=(ni,nj+1));PKA.synchronize(b)
    c.mass,c.mass_trial=c.mass_trial,c.mass;c.momentum,c.momentum_trial=c.momentum_trial,c.momentum
    c.qi,c.qi_trial=c.qi_trial,c.qi;c.qj,c.qj_trial=c.qj_trial,c.qj
    c.accepted_pressure,c.x=c.x,c.accepted_pressure;c.pressure_residual=c.trial_pressure_residual
    c.ledger .+=ledger_increment
    c.steps+=1;c.elapsed+=dt
    (;dt,pressure_residual=c.pressure_residual,pressure_iterations=c.pressure_iterations,
        clock="physical time; incompressible constituent equations on selected device")
end

function projection_host(c::ProjectionKA)
    s=deepcopy(c.host);s.mass=Array(c.mass);s.momentum=Array(c.momentum)
    pressure=c.gauge==0 ? Array(c.accepted_pressure) : vcat(0.,Array(c.accepted_pressure))
    s.pressure=reshape(pressure,s.grid.NI,s.grid.NJ);s.qi=Array(c.qi);s.qj=Array(c.qj)
    s.elapsed=c.elapsed;s.steps=c.steps;s.boundary_mass_integral=copy(c.ledger);s.pressure_residual=c.pressure_residual
    s
end
