# Included explicitly in CMeshFreeSurfaceAirfoil. Same conservative face and
# physical-clock contract as the scalar airfoil operator, CPU/CUDA via KA.
using KernelAbstractions
const AFKA=KernelAbstractions
export AirfoilKA, build_airfoil_ka, airfoil_step_ka!

@inline function _af_face(U,p0,pf,m)
    good,p,c,u,v=CMeshFreeSurface._fsq(U,m)
    good || return (false,U)
    pressure=p*(pf/p0);rho0=total_density(U)
    yl,yv,ya=U[1]/rho0,U[4]/rho0,U[5]/rho0
    rho=CMeshFreeSurface._fsdensity(pressure,yl,yv,ya,m)
    result=(rho*yl,rho*u,rho*v,rho*yv,rho*ya)
    (pressure_state(result,m)[1],result)
end

@kernel function _k_af_i!(F,rate,status,@Const(U),@Const(pc),@Const(pf),@Const(sx),@Const(sy),m,low_mach,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        ip=i==ni ? 1 : i+1
        L=ntuple(k->@inbounds(U[i,j,k]),Val(5));R=ntuple(k->@inbounds(U[ip,j,k]),Val(5))
        gl,L=_af_face(L,@inbounds(pc[i,j]),@inbounds(pf[i,j]),m)
        gr,R=_af_face(R,@inbounds(pc[ip,j]),@inbounds(pf[i,j]),m)
        x=@inbounds sx[i,j];y=@inbounds sy[i,j];area=hypot(x,y);nx=x/area;ny=y/area
        good,f=CMeshFreeSurface._fsface(L,R,nx,ny,m,low_mach)
        _,_,cl,ul,vl=CMeshFreeSurface._fsq(L,m);_,_,cr,ur,vr=CMeshFreeSurface._fsq(R,m)
        for k in 1:5;@inbounds F[i,j,k]=area*f[k];end
        @inbounds rate[i,j]=area*max(abs(ul*nx+vl*ny)+cl,abs(ur*nx+vr*ny)+cr)
        @inbounds status[i,j]=gl && gr && good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_af_j!(F,rate,status,@Const(U),@Const(pc),@Const(pf),@Const(sx),@Const(sy),@Const(Y),m,speed,seed,h,wall,low_mach,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj+1
        x=@inbounds sx[i,j];y=@inbounds sy[i,j];area=hypot(x,y);nx=x/area;ny=y/area
        pface=@inbounds pf[i,j]
        good=true;f=(0.,0.,0.,0.,0.);wave=0.
        if j==1 || (j==nj+1 && wall)
            cell=j==1 ? 1 : nj
            R=ntuple(k->@inbounds(U[i,cell,k]),Val(5))
            good,R=_af_face(R,@inbounds(pc[i,cell]),pface,m)
            admitted,p,c,u,v=CMeshFreeSurface._fsq(R,m);good &= admitted
            f=(0.,p*nx,p*ny,0.,0.);wave=hypot(u,v)+c
        elseif j==nj+1
            L=ntuple(k->@inbounds(U[i,nj,k]),Val(5))
            good,L=_af_face(L,@inbounds(pc[i,nj]),pface,m)
            admitted,p,c,u,v=CMeshFreeSurface._fsq(L,m);good &= admitted
            R=L
            if u*nx+v*ny<0
                water=_wet_edge_fraction(@inbounds(Y[i,j]),@inbounds(Y[i+1,j]),h)
                rl=m.rho_l_ref+(pface-m.p_ref)/m.c_l2;rv=pface/m.rt_vapor;ra=pface/m.rt_air
                ml=water*(1-seed)*rl;mv=water*seed*rv;ma=(1-water)*ra
                rho=ml+mv+ma;R=(ml,rho*speed,0.,mv,ma)
            end
            accepted,f=CMeshFreeSurface._fsface(L,R,nx,ny,m,low_mach);good &= accepted
            wave=hypot(u,v)+c
        else
            L=ntuple(k->@inbounds(U[i,j-1,k]),Val(5));R=ntuple(k->@inbounds(U[i,j,k]),Val(5))
            gl,L=_af_face(L,@inbounds(pc[i,j-1]),pface,m)
            gr,R=_af_face(R,@inbounds(pc[i,j]),pface,m)
            accepted,f=CMeshFreeSurface._fsface(L,R,nx,ny,m,low_mach);good=gl && gr && accepted
            _,_,cl,ul,vl=CMeshFreeSurface._fsq(L,m);_,_,cr,ur,vr=CMeshFreeSurface._fsq(R,m)
            wave=max(hypot(ul,vl)+cl,hypot(ur,vr)+cr)
        end
        for k in 1:5;@inbounds F[i,j,k]=area*f[k];end
        @inbounds rate[i,j]=area*wave
        @inbounds status[i,j]=good ? UInt8(0) : UInt8(1)
    end
end

@kernel function _k_af_residual!(res,rate,@Const(U),@Const(Fi),@Const(Fj),@Const(ri),@Const(rj),@Const(volume),@Const(rho0),@Const(sx0),@Const(sy0),gravity,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        im=i==1 ? ni : i-1
        rho=@inbounds(U[i,j,1]+U[i,j,4]+U[i,j,5]);vol=@inbounds volume[i,j]
        for k in 1:5
            source=k==2 ? @inbounds(sx0[i,j]) : k==3 ? @inbounds(sy0[i,j])-gravity*(rho-@inbounds(rho0[i,j])) : 0.
            @inbounds res[i,j,k]=-(Fi[i,j,k]-Fi[im,j,k]+Fj[i,j+1,k]-Fj[i,j,k])/vol+source
        end
        @inbounds rate[i,j]=(ri[i,j]+ri[im,j]+rj[i,j]+rj[i,j+1])/vol
    end
end

@kernel function _k_af_trial!(trial,status,@Const(U),@Const(res),m,dt,ni,nj)
    i,j=@index(Global,NTuple)
    if i<=ni && j<=nj
        state=ntuple(k->@inbounds(U[i,j,k]+dt*res[i,j,k]),Val(5))
        for k in 1:5;@inbounds trial[i,j,k]=state[k];end
        @inbounds status[i,j]=pressure_state(state,m)[1] ? UInt8(0) : UInt8(1)
    end
end

mutable struct AirfoilKA{B,A,R,S,M,G}
    backend::B
    U::A
    trial::A
    Fi::A
    Fj::A
    residual::A
    rate::R
    ri::R
    rj::R
    status::S
    si::S
    sj::S
    model::M
    geometry::G
    NI::Int
    NJ::Int
    speed::Float64
    seed::Float64
    gravity::Float64
    water_level::Float64
    wall::Bool
    low_mach::Bool
end

function build_airfoil_ka(backend,U::Array{Float64,3},g::AirfoilGrid,m::MaterialModel;outer_bc=:open,flux=:hllc)
    outer_bc in (:open,:wall) && flux in (:hllc,:thornber) || throw(ArgumentError("invalid airfoil KA boundary/flux"))
    dev(a)=begin buffer=AFKA.allocate(backend,eltype(a),size(a));copyto!(buffer,a);buffer;end
    z(n...)=dev(zeros(Float64,n...));b(n...)=dev(zeros(UInt8,n...))
    names=(:pc0,:pi0,:pj0,:six,:siy,:sjx,:sjy,:Y,:volume,:rho0,:source_x,:source_y)
    geo=NamedTuple{names}(Tuple(dev(getfield(g,k)) for k in names))
    ni,nj=g.NI,g.NJ
    AirfoilKA(backend,dev(U),z(ni,nj,5),z(ni,nj,5),z(ni,nj+1,5),z(ni,nj,5),z(ni,nj),z(ni,nj),z(ni,nj+1),
        b(ni,nj),b(ni,nj),b(ni,nj+1),m,geo,ni,nj,g.speed,g.vapor_seed,g.gravity,g.water_level,outer_bc===:wall,flux===:thornber)
end

function airfoil_step_ka!(c::AirfoilKA;dt_cap=Inf,cfl=.35,max_retries=20)
    0<cfl<=.5 && dt_cap>0 || throw(ArgumentError("invalid airfoil KA step"))
    b=c.backend;geo=c.geometry;ni,nj=c.NI,c.NJ;m=c.model
    _k_af_i!(b)(c.Fi,c.ri,c.si,c.U,geo.pc0,geo.pi0,geo.six,geo.siy,m,c.low_mach,ni,nj;ndrange=(ni,nj))
    _k_af_j!(b)(c.Fj,c.rj,c.sj,c.U,geo.pc0,geo.pj0,geo.sjx,geo.sjy,geo.Y,m,c.speed,c.seed,c.water_level,c.wall,c.low_mach,ni,nj;ndrange=(ni,nj+1))
    AFKA.synchronize(b)
    maximum(c.si)==0 && maximum(c.sj)==0 || error("nonlinear underwater face admission failed; unchanged state")
    _k_af_residual!(b)(c.residual,c.rate,c.U,c.Fi,c.Fj,c.ri,c.rj,geo.volume,geo.rho0,geo.source_x,geo.source_y,c.gravity,ni,nj;ndrange=(ni,nj))
    AFKA.synchronize(b)
    dt=min(Float64(dt_cap),cfl/maximum(c.rate))
    for attempt in 0:max_retries
        _k_af_trial!(b)(c.trial,c.status,c.U,c.residual,m,dt,ni,nj;ndrange=(ni,nj))
        AFKA.synchronize(b)
        if maximum(c.status)==0
            c.U,c.trial=c.trial,c.U
            mass_rate=[Float64(sum(view(c.Fj,:,nj+1,k))) for k in (1,4,5)]
            residual=max(maximum(abs,view(c.residual,:,:,1)),maximum(abs,view(c.residual,:,:,4)),maximum(abs,view(c.residual,:,:,5)))
            return (;dt,retries=attempt,outward_mass_rate=mass_rate,residual)
        end
        dt*=.5
    end
    error("nonlinear underwater KA step exhausted $max_retries admissibility retries; unchanged state")
end
