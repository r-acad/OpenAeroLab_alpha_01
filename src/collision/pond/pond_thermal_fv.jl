"""Conservative thermal kinetic finite volumes on a Cartesian leaf-cell mesh.

One flux per face, including each coarse/fine face patch. MUSCL reconstruction
uses limited least-squares primitive gradients. SSP-RK2 uses one global clock;
an inadmissible stage rejects the entire step, without clipping conserved state.
Voxel walls are an explicit approximation, not a cut-cell or Bouzidi method.
"""
module PondThermalFV

using KernelAbstractions, LinearAlgebra
include("pond_thermal_flux.jl")
using .PonDThermalFlux
export ThermalFV, advance!, primitives, conserved_totals, residual!, memory_bytes

@inline wrapdelta(d,L,periodic) = periodic ? d-round(d/L)*L : d
@inline function displacement(g,i,j)
    (wrapdelta(g.x[j]-g.x[i],g.extent[1],g.periodic[1]),
     wrapdelta(g.y[j]-g.y[i],g.extent[2],g.periodic[2]),
     wrapdelta(g.z[j]-g.z[i],g.extent[3],g.periodic[3]))
end
@inline function face_displacement(g,i,f)
    (wrapdelta(g.fx[f]-g.x[i],g.extent[1],g.periodic[1]),
     wrapdelta(g.fy[f]-g.y[i],g.extent[2],g.periodic[2]),
     wrapdelta(g.fz[f]-g.z[i],g.extent[3],g.periodic[3]))
end
@inline prim(P,i) = (P[i,1],P[i,2],P[i,3],P[i,4],P[i,5])
@inline function ghost(q,axis,bc)
    # Far-field gradient extrapolates; its flux uses characteristic data below.
    bc==3 && return (q[1],-q[2],-q[3],-q[4],q[5])
    bc==2 && return (q[1],axis==1 ? -q[2] : q[2],axis==2 ? -q[3] : q[3],axis==3 ? -q[4] : q[4],q[5])
    q
end
@inline function characteristic(q,far,axis,sgn,gamma)
    un=q[axis+1]*sgn; a=sqrt(gamma*q[5]/q[1])
    un>=a && return q
    un<=-a && return far
    af=sqrt(gamma*far[5]/far[1]); uf=far[axis+1]*sgn
    rp=un+2a/(gamma-1); rm=uf-2af/(gamma-1)
    ub=(rp+rm)/2; ab=(gamma-1)*(rp-rm)/4
    ab>0 || return far
    source=ub>=0 ? q : far
    entropy=source[5]/source[1]^gamma
    rb=(ab*ab/(gamma*entropy))^(1/(gamma-1)); pb=entropy*rb^gamma
    (rb,axis==1 ? sgn*ub : source[2],axis==2 ? sgn*ub : source[3],axis==3 ? sgn*ub : source[4],pb)
end

function geometry(mesh,::Type{T},dev::D,periodic::P) where {T,D,P}
    n=length(mesh.x)
    host=(x=T.(mesh.x),y=T.(mesh.y),z=T.(mesh.z),volume=T.(mesh.volume),
        owner=mesh.owner,neighbor=mesh.neighbor,axis=mesh.axis,sign=mesh.sign,bc=mesh.boundary,
        area=T.(mesh.area),fx=T.(mesh.face_x),fy=T.(mesh.face_y),fz=T.(mesh.face_z),
        offsets=mesh.cell_face_offsets,faces=mesh.cell_faces,
        extent=T.(mesh.extent),periodic=Tuple(periodic))
    inverse=zeros(T,n,6)
    for i in 1:n
        a=0.0;b=0.0;c=0.0;d=0.0;e=0.0;f=0.0
        for k in host.offsets[i]:host.offsets[i+1]-1
            face=abs(host.faces[k]);j=host.owner[face]==i ? host.neighbor[face] : host.owner[face]
            delta=j>0 ? displacement(host,i,j) : 2 .* face_displacement(host,i,face)
            dx,dy,dz=Float64.(delta)
            distance2=dx*dx+dy*dy+dz*dz;distance2>0 || continue
            weight=Float64(host.area[face])/distance2
            a+=weight*dx*dx;b+=weight*dx*dy;c+=weight*dx*dz
            d+=weight*dy*dy;e+=weight*dy*dz;f+=weight*dz*dz
        end
        # The Cartesian neighbour fit is normally SPD. Its six cofactors avoid
        # allocating a matrix and performing a tiny SVD for every leaf cell.
        aa=d*f-e*e;ab=c*e-b*f;ac=b*e-c*d
        dd=a*f-c*c;de=b*c-a*e;ff=a*d-b*b
        determinant=a*aa+b*ab+c*ac;scale=max(a,d,f)
        if scale>0 && determinant>1e-12*scale^3
            invdet=1/determinant
            inverse[i,1]=aa*invdet;inverse[i,2]=ab*invdet;inverse[i,3]=ac*invdet
            inverse[i,4]=dd*invdet;inverse[i,5]=de*invdet;inverse[i,6]=ff*invdet
        else
            # Degenerate 1D/2D periodic fixtures deliberately have a null space.
            mi=pinv([a b c;b d e;c e f];rtol=1e-12)
            inverse[i,1]=mi[1,1];inverse[i,2]=mi[1,2];inverse[i,3]=mi[1,3]
            inverse[i,4]=mi[2,2];inverse[i,5]=mi[2,3];inverse[i,6]=mi[3,3]
        end
    end
    merge(map(v->v isa AbstractArray ? dev(v) : v,host),(inverse=dev(inverse),))
end

mutable struct ThermalFV{G,A,F,T,B}
    geometry::G
    q::A
    stage::A
    next::A
    primitive::A
    gradient::F
    limiter::A
    flux::A
    dt_cells::B
    invalid::B
    gamma::T
    freestream::NTuple{5,T}
    mu::T
    prandtl::T
    cfl::T
    second_order::Bool
    time::Float64
    steps::Int
    rejected_steps::Int
    last_dt::Float64
end

"Lengths remain physical; normalize rho and velocity by chosen reference scales."
function ThermalFV(mesh;backend=CPU(),dev=copy,T=Float32,gamma=1.4,
                   freestream=(1.,.8,0.,0.,1/1.4),initial=nothing,
                   mu=0.,prandtl=.72,cfl=.25,second_order=true,periodic=(false,false,false))
    1<gamma<=5/3 || throw(ArgumentError("gamma must lie in (1,5/3]"))
    0<cfl<=.4 || throw(ArgumentError("thermal FV CFL must lie in (0,0.4]"))
    isfinite(mu)&&mu>=0 && isfinite(prandtl)&&prandtl>0 || throw(ArgumentError("invalid viscosity/Prandtl number"))
    far=T.(freestream);n=length(mesh.x);nf=length(mesh.owner)
    all(isfinite,far)&&far[1]>0&&far[5]>0 || throw(ArgumentError("inadmissible freestream primitive state"))
    n>0 && nf>0 || throw(ArgumentError("empty finite-volume mesh"))
    q=zeros(T,n,5)
    for i in 1:n
        p=initial===nothing ? far : T.(initial(i,mesh.x[i],mesh.y[i],mesh.z[i]))
        all(isfinite,p)&&p[1]>0&&p[5]>0 || throw(ArgumentError("inadmissible initial primitive state at cell $i"))
        q[i,:].=primitive_to_conserved(p,T(gamma))
    end
    g=geometry(mesh,T,dev,periodic)
    ThermalFV(g,dev(q),dev(zeros(T,n,5)),dev(zeros(T,n,5)),dev(zeros(T,n,5)),
        dev(zeros(T,n,5,3)),dev(zeros(T,n,5)),dev(zeros(T,nf,5)),dev(zeros(T,n)),dev(zeros(T,n)),
        T(gamma),far,T(mu),T(prandtl),T(cfl),second_order,0.,0,0,0.)
end

@kernel function _primitive!(P,@Const(Q),gamma)
    i=@index(Global,Linear)
    @inbounds begin
        p=conserved_to_primitive((Q[i,1],Q[i,2],Q[i,3],Q[i,4],Q[i,5]),gamma)
        for m in 1:5;P[i,m]=p[m];end
    end
end

@kernel function _gradients!(D,L,@Const(P),g,second_order)
    i=@index(Global,Linear)
    @inbounds for m in 1:5
        b1=zero(eltype(P));b2=b1;b3=b1
        pc=P[i,m];lo=pc;hi=pc
        for k in g.offsets[i]:g.offsets[i+1]-1
            f=abs(g.faces[k]);j=g.owner[f]==i ? g.neighbor[f] : g.owner[f]
            delta=j>0 ? displacement(g,i,j) : 2 .* face_displacement(g,i,f)
            v=j>0 ? P[j,m] : ghost(prim(P,i),g.axis[f],g.bc[f])[m]
            d2=delta[1]^2+delta[2]^2+delta[3]^2
            if d2>0
                w=g.area[f]/d2;diff=v-pc
                b1+=w*delta[1]*diff;b2+=w*delta[2]*diff;b3+=w*delta[3]*diff
            end
            lo=min(lo,v);hi=max(hi,v)
        end
        d1=g.inverse[i,1]*b1+g.inverse[i,2]*b2+g.inverse[i,3]*b3
        d2=g.inverse[i,2]*b1+g.inverse[i,4]*b2+g.inverse[i,5]*b3
        d3=g.inverse[i,3]*b1+g.inverse[i,5]*b2+g.inverse[i,6]*b3
        alpha=second_order ? one(pc) : zero(pc)
        for k in g.offsets[i]:g.offsets[i+1]-1
            f=abs(g.faces[k]);d=face_displacement(g,i,f)
            v=d1*d[1]+d2*d[2]+d3*d[3]
            if v>0;alpha=min(alpha,max(zero(pc),(hi-pc)/v))
            elseif v<0;alpha=min(alpha,max(zero(pc),(lo-pc)/v));end
        end
        # Physical viscous gradients must survive first-order advection and
        # extrema. Store the advective limiter separately from the raw LS fit.
        D[i,m,1]=d1;D[i,m,2]=d2;D[i,m,3]=d3;L[i,m]=alpha
    end
end

@inline function reconstruct(P,D,L,g,i,f)
    d=face_displacement(g,i,f)
    q=ntuple(m->P[i,m]+L[i,m]*(D[i,m,1]*d[1]+D[i,m,2]*d[2]+D[i,m,3]*d[3]),5)
    q[1]>0 && q[5]>0 ? q : prim(P,i)
end

@inline function corrected_face_gradient(P,D,g,i,j,m,axis)
    d=displacement(g,i,j)
    raw=ntuple(a->(D[i,m,a]+D[j,m,a])/2,3)
    if d[axis]==0
        return raw
    end
    correction=(P[j,m]-P[i,m]-raw[1]*d[1]-raw[2]*d[2]-raw[3]*d[3])/d[axis]
    ntuple(a->raw[a]+(a==axis ? correction : zero(correction)),3)
end

@inline function corrected_temperature_gradient(P,D,g,i,j,axis)
    ti=P[i,5]/P[i,1];tj=P[j,5]/P[j,1]
    raw=ntuple(a->((D[i,5,a]-ti*D[i,1,a])/P[i,1]+(D[j,5,a]-tj*D[j,1,a])/P[j,1])/2,3)
    d=displacement(g,i,j)
    if d[axis]==0
        return raw
    end
    correction=(tj-ti-raw[1]*d[1]-raw[2]*d[2]-raw[3]*d[3])/d[axis]
    ntuple(a->raw[a]+(a==axis ? correction : zero(correction)),3)
end

@kernel function _fluxes!(F,@Const(P),@Const(D),@Const(L),g,gamma,far,mu,prandtl)
    f=@index(Global,Linear)
    @inbounds begin
        i=g.owner[f];j=g.neighbor[f];axis=g.axis[f];sgn=g.sign[f];bc=g.bc[f]
        left=reconstruct(P,D,L,g,i,f)
        if j==0 && (bc==2 || bc==3)
            # Impermeable, adiabatic stationary voxel face. Its pressure comes
            # from reflected kinetic states, including wall-normal acoustic response.
            right=ghost(left,axis,Int8(2))
            raw=kinetic_flux(left,right,axis,sgn,gamma)
            wallp=raw[axis+1]*sgn
            F[f,1]=zero(wallp);F[f,5]=zero(wallp)
            un=P[i,axis+1]*sgn
            d=face_displacement(g,i,f);distance=abs(d[axis])
            for a in 1:3
                visc=bc==3 ? mu/distance*(P[i,a+1]+(a==axis ? un*sgn/3 : zero(un))) : zero(un)
                F[f,a+1]=g.area[f]*((a==axis ? sgn*wallp : zero(wallp))+visc)
            end
        else
            right=j>0 ? reconstruct(P,D,L,g,j,f) : characteristic(left,far,axis,sgn,gamma)
            fl=kinetic_flux(left,right,axis,sgn,gamma)
            for m in 1:5;F[f,m]=g.area[f]*fl[m];end
            if mu>0 && j>0
                # Full Newtonian Stokes stress and Fourier heat flux; laminar,
                # constant dynamic viscosity. No SGS/turbulent claim is made.
                # Enforce each cell-to-cell difference in the normal derivative.
                # An uncorrected mean of central gradients has a checkerboard
                # null mode and ignores physical diffusion at extrema.
                grad=(corrected_face_gradient(P,D,g,i,j,2,axis),
                      corrected_face_gradient(P,D,g,i,j,3,axis),
                      corrected_face_gradient(P,D,g,i,j,4,axis))
                divergence=grad[1][1]+grad[2][2]+grad[3][3]
                work=zero(mu)
                for a in 1:3
                    tau=mu*(grad[a][axis]+grad[axis][a]-
                        (a==axis ? 2divergence/3 : zero(mu)))*sgn
                    F[f,a+1]-=g.area[f]*tau
                    work+=tau*(left[a+1]+right[a+1])/2
                end
                dtheta=corrected_temperature_gradient(P,D,g,i,j,axis)
                heat=mu*gamma/(gamma-1)/prandtl*dtheta[axis]*sgn
                F[f,5]-=g.area[f]*(work+heat)
            end
        end
    end
end

@kernel function _timesteps!(dt,@Const(P),g,gamma,mu,prandtl,cfl)
    i=@index(Global,Linear)
    @inbounds begin
        a=sqrt(gamma*P[i,5]/P[i,1]);rate=zero(a)
        for k in g.offsets[i]:g.offsets[i+1]-1
            f=abs(g.faces[k]);j=g.owner[f]==i ? g.neighbor[f] : g.owner[f]
            wave=abs(P[i,g.axis[f]+1])+a
            if j>0;wave=max(wave,abs(P[j,g.axis[f]+1])+sqrt(gamma*P[j,5]/P[j,1]));end
            d=face_displacement(g,i,f);dist=max(abs(d[g.axis[f]]),eps(a))
            rate+=g.area[f]*(wave+2mu/P[i,1]*max(oftype(a,4/3),gamma/prandtl)/dist)
        end
        dt[i]=cfl*g.volume[i]/rate
    end
end

@kernel function _update!(out,invalid,@Const(base),@Const(stage),@Const(F),g,dt,gamma,blend)
    i=@index(Global,Linear)
    @inbounds begin
        for m in 1:5
            divergence=zero(eltype(out))
            for k in g.offsets[i]:g.offsets[i+1]-1
                sf=g.faces[k];divergence+=(sf>0 ? one(dt) : -one(dt))*F[abs(sf),m]
            end
            out[i,m]=blend*base[i,m]+(1-blend)*(stage[i,m]-dt/g.volume[i]*divergence)
        end
        u=(out[i,1],out[i,2],out[i,3],out[i,4],out[i,5])
        invalid[i]=admissible(u,gamma,oftype(dt,1e-10),oftype(dt,1e-10)) ? zero(dt) : one(dt)
    end
end

function residual!(s,Q=s.q)
    backend=get_backend(Q);n=size(Q,1)
    _primitive!(backend)(s.primitive,Q,s.gamma;ndrange=n)
    _gradients!(backend)(s.gradient,s.limiter,s.primitive,s.geometry,s.second_order;ndrange=n)
    _fluxes!(backend)(s.flux,s.primitive,s.gradient,s.limiter,s.geometry,s.gamma,s.freestream,s.mu,s.prandtl;ndrange=size(s.flux,1))
    s.flux
end

function advance!(s;dt_limit=Inf,max_retries=16)
    backend=get_backend(s.q);n=size(s.q,1)
    residual!(s)
    _timesteps!(backend)(s.dt_cells,s.primitive,s.geometry,s.gamma,s.mu,s.prandtl,s.cfl;ndrange=n)
    KernelAbstractions.synchronize(backend)
    dt=eltype(s.q)(min(Float64(minimum(s.dt_cells)),dt_limit))
    isfinite(dt)&&dt>0 || error("invalid thermal FV timestep $dt")
    for attempt in 0:max_retries
        attempt>0 && residual!(s)
        _update!(backend)(s.stage,s.invalid,s.q,s.q,s.flux,s.geometry,dt,s.gamma,zero(dt);ndrange=n)
        KernelAbstractions.synchronize(backend)
        valid=maximum(s.invalid)==0
        if valid
            residual!(s,s.stage)
            _update!(backend)(s.next,s.invalid,s.q,s.stage,s.flux,s.geometry,dt,s.gamma,oftype(dt,.5);ndrange=n)
            KernelAbstractions.synchronize(backend)
            valid=maximum(s.invalid)==0
        end
        if valid
            s.q,s.next=s.next,s.q;s.steps+=1;s.time+=dt;s.last_dt=Float64(dt)
            return Float64(dt)
        end
        s.rejected_steps+=1;dt/=2
    end
    error("Thermal FV step rejected after $max_retries halvings; original conserved state retained")
end

function primitives(s)
    _primitive!(get_backend(s.q))(s.primitive,s.q,s.gamma;ndrange=size(s.q,1))
    KernelAbstractions.synchronize(get_backend(s.q));Array(s.primitive)
end
conserved_totals(s) = vec(sum(Float64.(Array(s.q)).*Float64.(Array(s.geometry.volume));dims=1))
memory_bytes(s)=sum(sizeof(getfield(s,k)) for k in (:q,:stage,:next,:primitive,:gradient,:limiter,:flux,:dt_cells,:invalid))+
    sum(v isa AbstractArray ? sizeof(v) : 0 for v in values(s.geometry))
end
