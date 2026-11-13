# Experimental physical-coordinate reconstruction for the thermal KSol path.
# This changes the spatial discretization, not the governing energy equation.
# Barth-Jespersen bounds / Venkatakrishnan smooth limiting are applied to
# primitive rho,u,v,T. Original Float64 nodes define local geometry; absolute
# Float32 centers must never be subtracted to build thin-layer gradients.
export build_airfoil_muscl_geometry, set_airfoil_muscl_ka!, clear_airfoil_muscl_ka!

const _AIRFOIL_MUSCL=WeakKeyDict{Any,Any}()
const _AIRFOIL_MUSCL_RECON=Int32(3) # 2 already means WENO-i / minmod-j.

function build_airfoil_muscl_geometry(X,Y,NIW::Integer,NIF::Integer;ghost::Int=3,scalar_type::Type{<:AbstractFloat}=Float32)
    scalar_type in (Float32,Float64)||throw(ArgumentError("MUSCL scalar_type must be Float32 or Float64"))
    size(X)==size(Y)||throw(ArgumentError("Airfoil node shapes differ"))
    ni,nj=size(X).-1
    ni>=2&&nj>=2&&ghost>=1&&2NIW+NIF==ni&&NIW>=0&&NIF>=0||
        throw(ArgumentError("Invalid single C/A+ reconstruction topology"))
    all(isfinite,X)&&all(isfinite,Y)||throw(ArgumentError("Nonfinite airfoil nodes"))
    cx=[.25*(Float64(X[i,j])+Float64(X[i+1,j])+Float64(X[i,j+1])+Float64(X[i+1,j+1])) for i in 1:ni,j in 1:nj]
    cy=[.25*(Float64(Y[i,j])+Float64(Y[i+1,j])+Float64(Y[i,j+1])+Float64(Y[i+1,j+1])) for i in 1:ni,j in 1:nj]
    offsets=zeros(scalar_type,ni,nj,2,4);weights=similar(offsets);displacements=similar(offsets)
    neighbors=zeros(Int32,ni,nj,2,4);area=zeros(Float64,ni,nj)
    closure64=0.;closure32=0.;moment_error=0.
    for j in 1:nj,i in 1:ni
        x=ntuple(k->Float64(X[(i,i+1,i+1,i)[k],(j,j,j+1,j+1)[k]]),4)
        y=ntuple(k->Float64(Y[(i,i+1,i+1,i)[k],(j,j,j+1,j+1)[k]]),4)
        a=((x[3]-x[1])*(y[4]-y[2])-(y[3]-y[1])*(x[4]-x[2]))/2
        a>0&&isfinite(a)||throw(ArgumentError("Nonpositive reconstruction cell area"))
        area[i,j]=a
        faces=(((x[1]+x[4])/2,(y[1]+y[4])/2),((x[2]+x[3])/2,(y[2]+y[3])/2),
               ((x[1]+x[2])/2,(y[1]+y[2])/2),((x[4]+x[3])/2,(y[4]+y[3])/2))
        # Outward W/E/S/N edge vectors. Both precision scales are reported;
        # a face-relative roundoff pass cannot certify a small volume defect.
        sx=(-(y[4]-y[1]),y[3]-y[2],y[2]-y[1],-(y[3]-y[4]))
        sy=(x[4]-x[1],-(x[3]-x[2]),-(x[2]-x[1]),x[3]-x[4])
        closure64=max(closure64,hypot(sum(sx),sum(sy))/a)
        closure32=max(closure32,hypot(Float64(sum(Float32.(sx))),Float64(sum(Float32.(sy))))/a)
        ds=NTuple{3,Float64}[]
        for f in 1:4
            ii,jj=f==1 ? (i-1,j) : f==2 ? (i+1,j) : f==3 ? (i,j-1) : (i,j+1)
            if f==3&&j==1&&(i<=NIW||i>NIW+NIF)
                ii=ni+1-i;jj=1
            end
            fx,fy=faces[f];ox=fx-cx[i,j];oy=fy-cy[i,j]
            if 1<=ii<=ni&&1<=jj<=nj
                dx=cx[ii,jj]-cx[i,j];dy=cy[ii,jj]-cy[i,j]
            else
                dx=2ox;dy=2oy
            end
            d2=dx*dx+dy*dy
            isfinite(d2)&&d2>0||throw(ArgumentError("Coincident reconstruction neighbors"))
            offsets[i,j,1,f]=ox;offsets[i,j,2,f]=oy
            displacements[i,j,1,f]=dx;displacements[i,j,2,f]=dy
            neighbors[i,j,1,f]=ii+ghost;neighbors[i,j,2,f]=jj+ghost
            push!(ds,(dx,dy,1/d2))
        end
        a11=sum(w*dx*dx for (dx,dy,w) in ds);a12=sum(w*dx*dy for (dx,dy,w) in ds)
        a22=sum(w*dy*dy for (dx,dy,w) in ds);det=a11*a22-a12*a12
        isfinite(det)&&det>0||throw(ArgumentError("Singular physical-coordinate reconstruction stencil"))
        for f in 1:4
            dx,dy,w=ds[f]
            weights[i,j,1,f]=(a22*dx-a12*dy)*w/det
            weights[i,j,2,f]=(a11*dy-a12*dx)*w/det
        end
        m11=sum(Float64(weights[i,j,1,f])*ds[f][1] for f in 1:4)
        m12=sum(Float64(weights[i,j,1,f])*ds[f][2] for f in 1:4)
        m21=sum(Float64(weights[i,j,2,f])*ds[f][1] for f in 1:4)
        m22=sum(Float64(weights[i,j,2,f])*ds[f][2] for f in 1:4)
        moment_error=max(moment_error,abs(m11-1),abs(m12),abs(m21),abs(m22-1))
    end
    all(isfinite,offsets)&&all(isfinite,weights)&&all(isfinite,displacements)||
        throw(ArgumentError("Reconstruction geometry is not representable in Float32"))
    (;offsets,weights,neighbors,displacements,area=scalar_type.(area),
      receipt=(;centers=:vertex_average,local_geometry_precision=:Float64,
        volume_scaled_metric_closure_float64=closure64,volume_scaled_metric_closure_float32=closure32,
        represented_gradient_moment_error=moment_error))
end

struct AirfoilMUSCLWorkspace{G,A,T}
    geometry::G
    states::A
    theta::T
    status::T
    limiter::Int32
    smooth_coefficient::Float32
    previous_recon::Int32
end

function set_airfoil_muscl_ka!(s::KSol,X,Y;limiter::Symbol=:barth,smooth_coefficient::Float32=.05f0)
    s.cav==0&&s.outer_mode==0&&s.outlet_mode==0&&s.ff_mix==0&&s.emix==0 ||
        throw(ArgumentError("Geometric thermal MUSCL requires single-phase free-air C/A+"))
    limiter in (:barth,:venkat)||throw(ArgumentError("Limiter must be :barth or :venkat"))
    isfinite(smooth_coefficient)&&0<smooth_coefficient<=1 || throw(ArgumentError("Invalid limiter smoothing coefficient"))
    size(X)==size(Y)==(s.NI+1,s.NJ+1)||throw(ArgumentError("MUSCL mesh does not match solver dimensions"))
    R=eltype(s.rho)
    host=build_airfoil_muscl_geometry(X,Y,s.NIW,s.NIF;ghost=s.g,scalar_type=R)
    # Ensure the original nodes belong to this solver, not just a same-sized mesh.
    xc=R.([.25*(X[i,j]+X[i+1,j]+X[i,j+1]+X[i+1,j+1]) for i in 1:s.NI,j in 1:s.NJ])
    yc=R.([.25*(Y[i,j]+Y[i+1,j]+Y[i,j+1]+Y[i+1,j+1]) for i in 1:s.NI,j in 1:s.NJ])
    xc==Array(s.xc)&&yc==Array(s.yc)||throw(ArgumentError("MUSCL original mesh centers differ from solver metrics"))
    dev(a)=_dev(s.backend,a)
    geom=(;offsets=dev(host.offsets),weights=dev(host.weights),neighbors=dev(host.neighbors),
        displacements=dev(host.displacements),area=dev(host.area),receipt=host.receipt)
    states=similar(s.rho,R,s.NI,s.NJ,7,4)
    theta=similar(s.dtf);status=similar(s.dtf)
    old=get(_AIRFOIL_MUSCL,s,nothing)
    previous=old===nothing ? s.recon : old.previous_recon
    w=AirfoilMUSCLWorkspace(geom,states,theta,status,limiter==:barth ? Int32(0) : Int32(1),smooth_coefficient,previous)
    _AIRFOIL_MUSCL[s]=w;s.recon=_AIRFOIL_MUSCL_RECON;w
end

function clear_airfoil_muscl_ka!(s::KSol)
    w=pop!(_AIRFOIL_MUSCL,s,nothing)
    w===nothing||(s.recon=w.previous_recon)
    nothing
end

@inline _airfoil_air_z(r,u,v,t,i,j)=@inbounds (r[i,j],u[i,j],v[i,j],t[i,j])
@inline _airfoil_trace4(q,i,j,f)=@inbounds (q[i,j,1,f],q[i,j,2,f],q[i,j,3,f],q[i,j,4,f])
@inline function _airfoil_ls_gradient(r,u,v,t,weights,neighbors,z,i,j,k,c)
    value=zero(z[k])
    for f in 1:4
        ii=neighbors[i,j,1,f];jj=neighbors[i,j,2,f]
        value+=weights[i,j,c,f]*(_airfoil_air_z(r,u,v,t,ii,jj)[k]-z[k])
    end
    value
end
@inline function _airfoil_limit_ratio(a,b,epsilon2,limiter)
    lo=zero(b);hi=one(b)
    b==lo&&return hi
    limiter==Int32(0)&&return clamp(a/b,lo,hi)
    clamp((a*a+2f0*a*b+epsilon2)/(a*a+a*b+2f0*b*b+epsilon2),lo,hi)
end

@kernel function _k_airfoil_muscl_states!(states,theta,status,gux,guy,gvx,gvy,gTx,gTy,
    @Const(r),@Const(u),@Const(v),@Const(t),@Const(offsets),@Const(weights),@Const(neighbors),@Const(area),
    limiter::Int32,kappa::Float32,eint,NI::Int32,NJ::Int32,g::Int32,fixed_factors)
    i,j=@index(Global,NTuple)
    CS2=_gas_cs2(eltype(r))
    @inbounds if i<=NI&&j<=NJ
        z=_airfoil_air_z(r,u,v,t,i+g,j+g)
        gx=ntuple(k->_airfoil_ls_gradient(r,u,v,t,weights,neighbors,z,i,j,k,1),Val(4))
        gy=ntuple(k->_airfoil_ls_gradient(r,u,v,t,weights,neighbors,z,i,j,k,2),Val(4))
        finite=all(isfinite,gx)&&all(isfinite,gy)
        th=fixed_factors===nothing ? ntuple(Val(4)) do k
            lo=z[k];hi=z[k]
            for f in 1:4
                ii=neighbors[i,j,1,f];jj=neighbors[i,j,2,f]
                q=_airfoil_air_z(r,u,v,t,ii,jj)[k]
                lo=min(lo,q);hi=max(hi,q)
            end
            value=finite ? one(z[k]) : zero(z[k])
            epsilon2=(kappa*sqrt(area[i,j]))^3*max(abs(z[k]),1f-3)^2
            for f in 1:4
                b=gx[k]*offsets[i,j,1,f]+gy[k]*offsets[i,j,2,f]
                a=b>0f0 ? hi-z[k] : lo-z[k]
                finite&&(value=min(value,_airfoil_limit_ratio(a,b,epsilon2,limiter)))
                # Smooth limiting can slightly relax a local extrema bound;
                # physical rho/T positivity is an independent exact cap.
                if (k==1||k==4)&&b<0f0
                    floor=k==1 ? 1f-4 : 1f-3
                    value=min(value,.999f0*(z[k]-floor)/(-b))
                end
            end
            clamp(value,zero(value),one(value))
        end : ntuple(k->fixed_factors[i,j,k],Val(4))
        gux[i,j]=finite ? gx[2] : 0f0;guy[i,j]=finite ? gy[2] : 0f0
        gvx[i,j]=finite ? gx[3] : 0f0;gvy[i,j]=finite ? gy[3] : 0f0
        gTx[i,j]=finite ? gx[4] : 0f0;gTy[i,j]=finite ? gy[4] : 0f0
        theta[i,j]=min(th[1],th[2],th[3],th[4]);status[i,j]=finite ? 0f0 : 1f0
        for f in 1:4
            dx=offsets[i,j,1,f];dy=offsets[i,j,2,f]
            q=ntuple(k->finite ? z[k]+th[k]*(gx[k]*dx+gy[k]*dy) : z[k],Val(4))
            states[i,j,1,f]=q[1];states[i,j,2,f]=q[2];states[i,j,3,f]=q[3];states[i,j,4,f]=q[4]
            states[i,j,5,f]=q[1]*CS2*q[4]
            states[i,j,6,f]=sqrt((1f0+CS2/eint)*CS2*q[4]);states[i,j,7,f]=0f0
        end
    end
end

function _airfoil_check_component_factors(s::KSol,factors;check_values::Bool=false)
    eltype(factors)===eltype(s.rho)&&size(factors)==(s.NI,s.NJ,4)&&KA.get_backend(factors)==s.backend||
        throw(ArgumentError("Lagged component factors must match the airfoil state type, shape and backend"))
    if check_values
        all(isfinite,factors)&&minimum(factors)>=0&&maximum(factors)<=1||
            throw(ArgumentError("Lagged component factors must be finite and in [0,1]"))
    end
    nothing
end

function _airfoil_muscl_prepare!(s::KSol,w::AirfoilMUSCLWorkspace;fixed_component_factors=nothing)
    _airfoil_physical_state(s.rho,s.ux,s.uy,s.T)||throw(ArgumentError("MUSCL received a nonphysical cell/ghost state"))
    fixed_component_factors===nothing||_airfoil_check_component_factors(s,fixed_component_factors)
    a=w.geometry
    _k_airfoil_muscl_states!(s.backend)(w.states,w.theta,w.status,s.gux,s.guy,s.gvx,s.gvy,s.gTx,s.gTy,
        s.rho,s.ux,s.uy,s.T,a.offsets,a.weights,a.neighbors,a.area,w.limiter,w.smooth_coefficient,s.eint,
        Int32(s.NI),Int32(s.NJ),Int32(s.g),fixed_component_factors;ndrange=(s.NI,s.NJ))
    w.states
end

@inline function _airfoil_thermal_face_flux(l,r,nx,ny,S,cxa,cya,cza,wa,drot)
    _fvflux_normal6(l[1],l[2],l[3],l[4],l[1],l[2],l[3],l[4],l[1],l[2],l[3],l[4],
        r[1],r[2],r[3],r[4],r[1],r[2],r[3],r[4],r[1],r[2],r[3],r[4],
        nx,ny,S,cxa,cya,cza,wa,drot,Int32(0))
end

@kernel function _k_airfoil_shared_wake_flux!(Fr,Fx,Fy,Fe,NI::Int32,NIW::Int32)
    i=@index(Global,Linear)
    @inbounds if i<=NIW
        ip=NI+Int32(1)-i
        Fr[ip,1]=-Fr[i,1];Fx[ip,1]=-Fx[i,1];Fy[ip,1]=-Fy[i,1];Fe[ip,1]=-Fe[i,1]
    end
end
