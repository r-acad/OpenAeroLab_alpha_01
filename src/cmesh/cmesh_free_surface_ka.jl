# Isolated KernelAbstractions Cartesian counterpart of the checked CPU
# five-state reference. No production airfoil solver imports this file.
using KernelAbstractions
const FSKA=KernelAbstractions
export CartesianKA, build_cartesian_ka, step_cartesian_ka!

@inline function _fsq(U,m)
    good,p,c=pressure_state(U,m)
    rho=total_density(U)
    (good,p,c,U[2]/rho,U[3]/rho)
end
@inline function _fsflux(U,p,u,v,nx,ny)
    un=u*nx+v*ny
    (U[1]*un,U[2]*un+p*nx,U[3]*un+p*ny,U[4]*un,U[5]*un)
end

# Device-safe transcription of the admitted HLLC/HLL face contract. Numeric
# failures return a flag to the owner instead of throwing from a device thread.
@inline function _fshllc(L,R,nx,ny,m)
    gl,pl,cl,ul,vl=_fsq(L,m);gr,pr,cr,ur,vr=_fsq(R,m)
    empty=(0.,0.,0.,0.,0.)
    gl && gr || return (false,empty)
    fl=_fsflux(L,pl,ul,vl,nx,ny);fr=_fsflux(R,pr,ur,vr,nx,ny)
    unl=ul*nx+vl*ny;unr=ur*nx+vr*ny
    sl=min(unl-cl,unr-cr);sr=max(unl+cl,unr+cr)
    sl>=0 && return (true,fl)
    sr<=0 && return (true,fr)
    rhoL=total_density(L);rhoR=total_density(R)
    denom=rhoL*(sl-unl)-rhoR*(sr-unr)
    nonsingular=isfinite(denom) && abs(denom)>eps(Float64)*max(1.,abs(rhoL*(sl-unl)),abs(rhoR*(sr-unr)))
    if nonsingular
        contact=(pr-pl+rhoL*unl*(sl-unl)-rhoR*unr*(sr-unr))/denom
        if isfinite(contact) && sl<contact<sr
            starL=_free_surface_star(L,unl,sl,contact,nx,ny)
            starR=_free_surface_star(R,unr,sr,contact,nx,ny)
            if pressure_state(starL,m)[1] && pressure_state(starR,m)[1]
                if contact>=0
                    return (true,(starL[1]*contact,fl[2]+sl*(starL[2]-L[2]),fl[3]+sl*(starL[3]-L[3]),starL[4]*contact,starL[5]*contact))
                else
                    return (true,(starR[1]*contact,fr[2]+sr*(starR[2]-R[2]),fr[3]+sr*(starR[3]-R[3]),starR[4]*contact,starR[5]*contact))
                end
            end
        end
    end
    (true,ntuple(k->(sr*fl[k]-sl*fr[k]+sl*sr*(R[k]-L[k]))/(sr-sl),Val(5)))
end

@inline function _fsthornber(L,R,nx,ny,m)
    gl,_,cl,ul,vl=_fsq(L,m);gr,_,cr,ur,vr=_fsq(R,m)
    gl && gr || return (false,(0.,0.,0.,0.,0.))
    z=min(max(hypot(ul,vl)/cl,hypot(ur,vr)/cr),1.)
    (z==1 || (ul==ur && vl==vr)) && return _fshllc(L,R,nx,ny,m)
    um=.5(ul+ur);vm=.5(vl+vr);du=.5z*(ul-ur);dv=.5z*(vl-vr)
    rl=total_density(L);rr=total_density(R)
    left=(L[1],rl*(um+du),rl*(vm+dv),L[4],L[5])
    right=(R[1],rr*(um-du),rr*(vm-dv),R[4],R[5])
    _fshllc(left,right,nx,ny,m)
end
@inline _fsface(L,R,nx,ny,m,low_mach)=low_mach ? _fsthornber(L,R,nx,ny,m) : _fshllc(L,R,nx,ny,m)

@inline _fsenthalpy(p,yl,yv,ya,m)=yl*m.c_l2*log1p((p-m.p_ref)/(m.c_l2*m.rho_l_ref))+
    (yv*m.rt_vapor+ya*m.rt_air)*log(p/m.p_ref)
@inline function _fsdensity(p,yl,yv,ya,m)
    rl=m.rho_l_ref+(p-m.p_ref)/m.c_l2
    inv(yl/rl+(yv*m.rt_vapor+ya*m.rt_air)/p)
end

@inline function _fsreconstruct(U,zcell,zface,g,m)
    good,p,_,u,v=_fsq(U,m)
    good || return (false,U)
    rho0=total_density(U);yl=U[1]/rho0;yv=U[4]/rho0;ya=U[5]/rho0;dz=zface-zcell
    target=_fsenthalpy(p,yl,yv,ya,m)-g*dz
    low=max(floatmin(Float64),p*1e-15);high=p
    _fsenthalpy(low,yl,yv,ya,m)<=target || return (false,U)
    for _ in 1:64
        _fsenthalpy(high,yl,yv,ya,m)>=target && break
        high*=2
    end
    _fsenthalpy(high,yl,yv,ya,m)>=target && isfinite(high) || return (false,U)
    pf=p
    for _ in 1:64
        residual=_fsenthalpy(pf,yl,yv,ya,m)-target
        abs(residual)<=2e-10*max(1.,abs(target)) && break
        if residual>0;high=pf;else;low=pf;end
        next=pf-residual*_fsdensity(pf,yl,yv,ya,m)
        pf=isfinite(next) && low<next<high ? next : .5(low+high)
    end
    abs(_fsenthalpy(pf,yl,yv,ya,m)-target)<=1e-8*max(1.,abs(target)) || return (false,U)
    rho=_fsdensity(pf,yl,yv,ya,m)
    state=(rho*yl,rho*u,rho*v,rho*yv,rho*ya)
    (pressure_state(state,m)[1],state)
end

@kernel function _k_fsrate!(rate,status,@Const(U),m,dx,dy,nx,ny)
    i,j=@index(Global,NTuple)
    if i<=nx && j<=ny
        q=ntuple(k->@inbounds(U[i,j,k]),Val(5))
        good,_,c,u,v=_fsq(q,m)
        @inbounds rate[i,j]=good ? (abs(u)+c)/dx+(abs(v)+c)/dy : Inf
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_fsflux_x!(flux,status,@Const(U),m,boundary,low_mach,nx,ny)
    i,j=@index(Global,NTuple)
    if i<=nx+1 && j<=ny
        good=true;f=(0.,0.,0.,0.,0.)
        if boundary==0 && (i==1 || i==nx+1)
            cell=i==1 ? 1 : nx
            state=ntuple(k->@inbounds(U[cell,j,k]),Val(5));good,p,_=pressure_state(state,m)
            f=(0.,p,0.,0.,0.)
        else
            il=i==1 ? (boundary==1 ? nx : 1) : i-1
            ir=i==nx+1 ? (boundary==1 ? 1 : nx) : i
            L=ntuple(k->@inbounds(U[il,j,k]),Val(5));R=ntuple(k->@inbounds(U[ir,j,k]),Val(5))
            good,f=_fsface(L,R,1.,0.,m,low_mach)
        end
        for k in 1:5;@inbounds flux[i,j,k]=f[k];end
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_fsflux_y!(flux,status,@Const(U),m,boundary,low_mach,dy,ymin,g,nx,ny)
    i,j=@index(Global,NTuple)
    if i<=nx && j<=ny+1
        zface=ymin+(j-1)*dy;good=true;f=(0.,0.,0.,0.,0.)
        if boundary==0 && (j==1 || j==ny+1)
            cell=j==1 ? 1 : ny;zcell=ymin+(cell-.5)*dy
            state=ntuple(k->@inbounds(U[i,cell,k]),Val(5))
            if g>0;good,state=_fsreconstruct(state,zcell,zface,g,m);end
            admitted,p,_=pressure_state(state,m);good &= admitted
            f=(0.,0.,p,0.,0.)
        else
            jl=j==1 ? (boundary==1 ? ny : 1) : j-1
            jr=j==ny+1 ? (boundary==1 ? 1 : ny) : j
            L=ntuple(k->@inbounds(U[i,jl,k]),Val(5));R=ntuple(k->@inbounds(U[i,jr,k]),Val(5))
            if g>0
                gl,L=_fsreconstruct(L,ymin+(jl-.5)*dy,zface,g,m)
                gr,R=_fsreconstruct(R,ymin+(jr-.5)*dy,zface,g,m)
                good=gl && gr
            end
            admitted,f=_fsface(L,R,0.,1.,m,low_mach);good &= admitted
        end
        for k in 1:5;@inbounds flux[i,j,k]=f[k];end
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_fssource!(source,status,@Const(U),m,dy,ymin,g,nx,ny)
    i,j=@index(Global,NTuple)
    if i<=nx && j<=ny
        sy=0.;good=true
        if g>0
            z=ymin+(j-.5)*dy;cell=ntuple(k->@inbounds(U[i,j,k]),Val(5))
            gl,lower=_fsreconstruct(cell,z,z-.5dy,g,m);gr,upper=_fsreconstruct(cell,z,z+.5dy,g,m)
            apl,pl,_=pressure_state(lower,m);apu,pu,_=pressure_state(upper,m)
            good=gl && gr && apl && apu
            sy=(pu-pl)/dy
        end
        @inbounds source[i,j]=sy
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_fsgate!(Fx,source_x,@Const(U),m,face,gate_bottom,dx,dy,ymin,ny)
    j=@index(Global,Linear)
    if j<=ny
        bottom=ymin+(j-1)*dy
        opening=clamp((gate_bottom-bottom)/dy,0.,1.);closed=1-opening
        for k in 1:5;@inbounds Fx[face,j,k]*=opening;end
        L=ntuple(k->@inbounds(U[face-1,j,k]),Val(5));R=ntuple(k->@inbounds(U[face,j,k]),Val(5))
        _,pl,_=pressure_state(L,m);_,pr,_=pressure_state(R,m)
        @inbounds source_x[face-1,j]=-closed*pl/dx
        @inbounds source_x[face,j]=closed*pr/dx
    end
end

@kernel function _k_fstrial!(trial,status,@Const(U),@Const(Fx),@Const(Fy),@Const(source),@Const(source_x),m,dx,dy,dt,nx,ny)
    i,j=@index(Global,NTuple)
    if i<=nx && j<=ny
        state=ntuple(Val(5)) do k
            @inbounds U[i,j,k]-dt*((Fx[i+1,j,k]-Fx[i,j,k])/dx+(Fy[i,j+1,k]-Fy[i,j,k])/dy-
                (k==3 ? source[i,j] : k==2 ? source_x[i,j] : 0.))
        end
        for k in 1:5;@inbounds trial[i,j,k]=state[k];end
        @inbounds status[i,j]=pressure_state(state,m)[1] ? UInt8(0) : UInt8(1)
    end
end

mutable struct CartesianKA{B,A,R,S,M}
    backend::B
    U::A
    trial::A
    Fx::A
    Fy::A
    rate::R
    source::R
    source_x::R
    status::S
    status_x::S
    status_y::S
    model::M
    dx::Float64
    dy::Float64
    ymin::Float64
    gravity::Float64
    bc_x::Int
    bc_y::Int
    low_mach::Bool
    gate_face::Int
    gate_speed::Float64
    gate_initial_bottom::Float64
    steps::Int
    elapsed::Float64
end

function build_cartesian_ka(backend,U::Array{Float64,3},m::MaterialModel,dx,dy;
    ymin=0.,gravity=9.80665,bc_x=:wall,bc_y=:wall,
    gate_face::Int=0,gate_speed::Real=1.5,gate_initial_bottom::Real=0.,flux::Symbol=:hllc)
    nx,ny,nv=size(U);nv==5 && nx>=2 && ny>=2 || throw(ArgumentError("invalid Cartesian KA state shape"))
    dx>0 && dy>0 && gravity>=0 || throw(ArgumentError("invalid grid/gravity"))
    bc_x in (:wall,:periodic,:outflow) && bc_y in (:wall,:periodic,:outflow) || throw(ArgumentError("invalid boundary"))
    gravity>0 && bc_y===:periodic && throw(ArgumentError("vertical periodicity requires gravity=0"))
    (gate_face==0 || 2<=gate_face<=nx) && gate_speed>=0 || throw(ArgumentError("invalid internal gate"))
    flux in (:hllc,:thornber) || throw(ArgumentError("Cartesian KA flux must be hllc or thornber"))
    dev(a)=begin buffer=FSKA.allocate(backend,eltype(a),size(a));copyto!(buffer,a);buffer;end
    z(n...)=dev(zeros(Float64,n...));b(n...)=dev(zeros(UInt8,n...))
    CartesianKA(backend,dev(U),z(nx,ny,5),z(nx+1,ny,5),z(nx,ny+1,5),z(nx,ny),z(nx,ny),z(nx,ny),
        b(nx,ny),b(nx+1,ny),b(nx,ny+1),m,Float64(dx),Float64(dy),Float64(ymin),Float64(gravity),
        bc_x===:wall ? 0 : bc_x===:periodic ? 1 : 2,bc_y===:wall ? 0 : bc_y===:periodic ? 1 : 2,
        flux===:thornber,gate_face,Float64(gate_speed),Float64(gate_initial_bottom),0,0.)
end

function step_cartesian_ka!(s::CartesianKA;cfl=.4,dt_cap=Inf,max_retries=16)
    0<cfl<=.5 && dt_cap>0 || throw(ArgumentError("invalid Cartesian KA clock"))
    nx,ny,_=size(s.U);b=s.backend;m=s.model
    _k_fsrate!(b)(s.rate,s.status,s.U,m,s.dx,s.dy,nx,ny;ndrange=(nx,ny))
    dt=min(Float64(dt_cap),cfl/maximum(s.rate))
    isfinite(dt) && dt>0 || error("Cartesian KA base state rejected")
    _k_fsflux_x!(b)(s.Fx,s.status_x,s.U,m,s.bc_x,s.low_mach,nx,ny;ndrange=(nx+1,ny))
    _k_fsflux_y!(b)(s.Fy,s.status_y,s.U,m,s.bc_y,s.low_mach,s.dy,s.ymin,s.gravity,nx,ny;ndrange=(nx,ny+1))
    _k_fssource!(b)(s.source,s.status,s.U,m,s.dy,s.ymin,s.gravity,nx,ny;ndrange=(nx,ny))
    fill!(s.source_x,0.)
    if s.gate_face>0
        _k_fsgate!(b)(s.Fx,s.source_x,s.U,m,s.gate_face,s.gate_initial_bottom+s.gate_speed*s.elapsed,
            s.dx,s.dy,s.ymin,ny;ndrange=ny)
    end
    maximum(s.status_x)==0 && maximum(s.status_y)==0 && maximum(s.status)==0 || error("Cartesian KA face/gravity reconstruction rejected")
    for attempt in 0:max_retries
        _k_fstrial!(b)(s.trial,s.status,s.U,s.Fx,s.Fy,s.source,s.source_x,m,s.dx,s.dy,dt,nx,ny;ndrange=(nx,ny))
        if maximum(s.status)==0
            copyto!(s.U,s.trial);s.steps+=1;s.elapsed+=dt
            return (dt=dt,retries=attempt,clock="physical time",equations=5)
        end
        dt*=.5
    end
    error("Cartesian KA exhausted retries; original state retained")
end
