# Isolated low-Mach equation set: incompressible constituent volumes with
# variable-density conservative momentum and a pressure projection. It is
# deliberately NOT imported by the explicit barotropic underwater worker.
module CMeshFreeSurfaceProjection
using SparseArrays, LinearAlgebra
include(joinpath(@__DIR__,"cmesh_free_surface_airfoil.jl"))
using .CMeshFreeSurfaceAirfoil
using .CMeshFreeSurfaceAirfoil.CMeshFreeSurface
export ProjectionState, build_projection, projection_step!, projection_mass,
       projection_volume_error, projection_divergence, projection_loads

mutable struct ProjectionState
    grid::AirfoilGrid
    mass::Array{Float64,3}
    momentum::Array{Float64,3}
    pressure::Matrix{Float64}
    qi::Matrix{Float64}
    qj::Matrix{Float64}
    rho_reference::Matrix{Float64}
    phase_density::NTuple{3,Float64}
    boundary_kind::Vector{UInt8} # 0 slip, 1 prescribed inlet, 2 pressure outlet
    kinematic::SparseMatrixCSC{Float64,Int}
    elapsed::Float64
    steps::Int
    boundary_mass_integral::Vector{Float64}
    pressure_residual::Float64
end

projection_mass(s::ProjectionState)=[sum(s.mass[:,:,k].*s.grid.volume) for k in 1:3]
projection_volume_error(s::ProjectionState)=maximum(abs,sum(s.mass[:,:,k]./s.phase_density[k] for k in 1:3).-1)
@inline _rho(s)=dropdims(sum(s.mass;dims=3);dims=3)
@inline _velocity(s)=s.momentum./reshape(_rho(s),s.grid.NI,s.grid.NJ,1)

function _conductance(s,rho)
    g=s.grid;ni,nj=g.NI,g.NJ;ci=zeros(ni,nj);cj=zeros(ni,nj+1)
    for j in 1:nj,i in 1:ni
        ip=mod1(i+1,ni);sx,sy=g.six[i,j],g.siy[i,j]
        d=sx*(g.xc[ip,j]-g.xc[i,j])+sy*(g.yc[ip,j]-g.yc[i,j])
        d>0 || error("projection requires positive centroid-normal distances")
        ci[i,j]=(sx*sx+sy*sy)/d*2/(rho[i,j]+rho[ip,j])
    end
    for j in 2:nj,i in 1:ni
        sx,sy=g.sjx[i,j],g.sjy[i,j]
        d=sx*(g.xc[i,j]-g.xc[i,j-1])+sy*(g.yc[i,j]-g.yc[i,j-1])
        d>0 || error("projection requires positive radial centroid-normal distances")
        cj[i,j]=(sx*sx+sy*sy)/d*2/(rho[i,j]+rho[i,j-1])
    end
    for i in 1:ni
        if s.boundary_kind[i]==2
            sx,sy=g.sjx[i,nj+1],g.sjy[i,nj+1]
            x=.5(g.X[i,end]+g.X[i+1,end]);y=.5(g.Y[i,end]+g.Y[i+1,end])
            d=sx*(x-g.xc[i,nj])+sy*(y-g.yc[i,nj])
            d>0 || error("invalid projection outlet distance")
            cj[i,end]=(sx*sx+sy*sy)/d/rho[i,nj]
        end
    end
    ci,cj
end

function _pressure_matrix(s,ci,cj)
    g=s.grid;ni,nj=g.NI,g.NJ;n=ni*nj
    rows=Int[];cols=Int[];vals=Float64[];diagonal=zeros(n)
    function edge(a,b,c)
        diagonal[a]+=c;diagonal[b]+=c
        push!(rows,a,b);push!(cols,b,a);push!(vals,-c,-c)
    end
    for j in 1:nj,i in 1:ni
        a=i+(j-1)*ni;b=mod1(i+1,ni)+(j-1)*ni
        edge(a,b,ci[i,j])
        j<nj && edge(a,a+ni,cj[i,j+1])
        j==nj && (diagonal[a]+=cj[i,end])
    end
    for a in 1:n;push!(rows,a);push!(cols,a);push!(vals,diagonal[a]);end
    sparse(rows,cols,vals,n,n)
end

function projection_divergence(s::ProjectionState;qi=s.qi,qj=s.qj)
    g=s.grid;div=zeros(g.NI,g.NJ)
    for j in 1:g.NJ,i in 1:g.NI
        div[i,j]=(qi[i,j]-qi[mod1(i-1,g.NI),j]+qj[i,j+1]-qj[i,j])/g.volume[i,j]
    end
    div
end

"Integrated divergence of interpolated cell velocities; gradient is its adjoint."
function _kinematic_matrix(s)
    g=s.grid;ni,nj=g.NI,g.NJ;n=ni*nj;rows=Int[];cols=Int[];values=Float64[]
    function edge(a,b,sx,sy)
        for (offset,component) in ((0,sx),(n,sy))
            push!(rows,a,a,b,b);push!(cols,a+offset,b+offset,a+offset,b+offset)
            push!(values,.5component,.5component,-.5component,-.5component)
        end
    end
    for j in 1:nj,i in 1:ni
        a=i+(j-1)*ni;b=mod1(i+1,ni)+(j-1)*ni
        edge(a,b,g.six[i,j],g.siy[i,j])
        j<nj && edge(a,a+ni,g.sjx[i,j+1],g.sjy[i,j+1])
        if j==nj && s.boundary_kind[i]==2
            push!(rows,a,a);push!(cols,a,a+n);push!(values,g.sjx[i,end],g.sjy[i,end])
        end
    end
    sparse(rows,cols,values,n,2n)
end

function _project_faces!(s,qi,qj,rho,dt)
    g=s.grid;B=s.kinematic;n=g.NI*g.NJ
    inverse_mass=vcat(vec(1 ./(rho.*g.volume)),vec(1 ./(rho.*g.volume)))
    A=B*spdiagm(0=>inverse_mass)*transpose(B)
    rhs=-vec(projection_divergence(s;qi,qj).*g.volume)./dt
    pressure=if any(==(UInt8(2)),s.boundary_kind)
        cholesky(Symmetric(A))\rhs
    else
        # Closed domain: eliminate one gauge degree of freedom exactly.
        v=zeros(length(rhs));v[2:end]=cholesky(Symmetric(A[2:end,2:end]))\rhs[2:end];v
    end
    residual=maximum(abs,A*pressure-rhs)
    p=reshape(pressure,g.NI,g.NJ)
    correction=reshape(dt.*inverse_mass.*(transpose(B)*pressure),g.NI,g.NJ,2)
    for j in 1:g.NJ,i in 1:g.NI
        ip=mod1(i+1,g.NI)
        qi[i,j]+=.5(correction[i,j,1]+correction[ip,j,1])*g.six[i,j]+.5(correction[i,j,2]+correction[ip,j,2])*g.siy[i,j]
    end
    for j in 2:g.NJ,i in 1:g.NI
        qj[i,j]+=.5(correction[i,j-1,1]+correction[i,j,1])*g.sjx[i,j]+.5(correction[i,j-1,2]+correction[i,j,2])*g.sjy[i,j]
    end
    for i in 1:g.NI
        s.boundary_kind[i]==2 && (qj[i,end]+=correction[i,end,1]*g.sjx[i,end]+correction[i,end,2]*g.sjy[i,end])
    end
    (;pressure=p,residual)
end

function _pressure_gradient(s,p)
    g=s.grid;n=g.NI*g.NJ
    force=transpose(s.kinematic)*vec(p)
    (-reshape(force[1:n],g.NI,g.NJ)./g.volume,-reshape(force[n+1:end],g.NI,g.NJ)./g.volume)
end

function build_projection(g::AirfoilGrid,m::MaterialModel;outer_bc=:open,initialization=:source_potential,speed=g.speed)
    outer_bc in (:open,:wall) || throw(ArgumentError("invalid projection boundary"))
    densities=(m.rho_l_ref,m.p_vapor_sat/m.rt_vapor,m.p_ref/m.rt_air)
    mass=zeros(g.NI,g.NJ,3)
    for j in 1:g.NJ,i in 1:g.NI
        water=g.beta_water0[i,j]
        mass[i,j,1]=water*(1-g.vapor_seed)*densities[1]
        mass[i,j,2]=water*g.vapor_seed*densities[2]
        mass[i,j,3]=(1-water)*densities[3]
    end
    rho=dropdims(sum(mass;dims=3);dims=3)
    velocity=zeros(g.NI,g.NJ,2)
    if initialization===:source_potential
        initial,_=source_panel_initial_state(g,m)
        for j in 1:g.NJ,i in 1:g.NI
            q=primitive(ntuple(k->initial[i,j,k],5),m);velocity[i,j,1]=q.u;velocity[i,j,2]=q.v
        end
    elseif initialization===:rest
        # Zero is useful for the hydrostatic/body admission.
    elseif initialization===:uniform
        velocity[:,:,1].=speed
    else
        throw(ArgumentError("unknown projection initialization"))
    end
    momentum=velocity.*reshape(rho,g.NI,g.NJ,1)
    kind=zeros(UInt8,g.NI);ymin=minimum(g.Y[:,end])
    if outer_bc===:open
        for i in 1:g.NI
            y=.5(g.Y[i,end]+g.Y[i+1,end]);sx,sy=g.sjx[i,end],g.sjy[i,end]
            kind[i]=abs(y-ymin)<1e-9g.chord ? 0 : sx/hypot(sx,sy)<-.5 ? 1 : 2
        end
    end
    s=ProjectionState(g,mass,momentum,zeros(g.NI,g.NJ),zeros(g.NI,g.NJ),zeros(g.NI,g.NJ+1),copy(rho),densities,kind,spzeros(g.NI*g.NJ,2g.NI*g.NJ),0.,0,zeros(3),0.)
    s.kinematic=_kinematic_matrix(s)
    for j in 1:g.NJ,i in 1:g.NI
        ip=mod1(i+1,g.NI)
        s.qi[i,j]=.5(velocity[i,j,1]+velocity[ip,j,1])*g.six[i,j]+.5(velocity[i,j,2]+velocity[ip,j,2])*g.siy[i,j]
    end
    for j in 2:g.NJ,i in 1:g.NI
        s.qj[i,j]=.5(velocity[i,j-1,1]+velocity[i,j,1])*g.sjx[i,j]+.5(velocity[i,j-1,2]+velocity[i,j,2])*g.sjy[i,j]
    end
    for i in 1:g.NI
        if kind[i]==1;s.qj[i,end]=speed*g.sjx[i,end]
        elseif kind[i]==2;s.qj[i,end]=velocity[i,end,1]*g.sjx[i,end]+velocity[i,end,2]*g.sjy[i,end];end
    end
    solved=_project_faces!(s,s.qi,s.qj,rho,1.);p=solved.pressure
    gx,gy=_pressure_gradient(s,p)
    s.momentum[:,:,1].-=gx;s.momentum[:,:,2].-=gy
    s.pressure_residual=solved.residual
    s
end

function _transport(s,dt)
    g=s.grid;ni,nj=g.NI,g.NJ;velocity=_velocity(s)
    state=cat(s.mass,s.momentum;dims=3);fi=zeros(ni,nj,5);fj=zeros(ni,nj+1,5)
    boundary=zeros(3)
    for j in 1:nj,i in 1:ni
        ip=mod1(i+1,ni);donor=s.qi[i,j]>=0 ? i : ip
        for k in 1:5;fi[i,j,k]=s.qi[i,j]*state[donor,j,k];end
    end
    for j in 2:nj,i in 1:ni
        donor=s.qj[i,j]>=0 ? j-1 : j
        for k in 1:5;fj[i,j,k]=s.qj[i,j]*state[i,donor,k];end
    end
    for i in 1:ni
        q=s.qj[i,end]
        if q>=0
            for k in 1:5;fj[i,end,k]=q*state[i,nj,k];end
        else
            water=CMeshFreeSurfaceAirfoil._wet_edge_fraction(g.Y[i,end],g.Y[i+1,end],g.water_level)
            ml=water*(1-g.vapor_seed)*s.phase_density[1];mv=water*g.vapor_seed*s.phase_density[2];ma=(1-water)*s.phase_density[3]
            incoming=(ml,mv,ma,(ml+mv+ma)*g.speed,0.)
            for k in 1:5;fj[i,end,k]=q*incoming[k];end
        end
        for k in 1:3;boundary[k]+=fj[i,end,k];end
    end
    candidate=similar(state)
    for j in 1:nj,i in 1:ni,k in 1:5
        source=k==5 ? -g.gravity*(sum(s.mass[i,j,:])-s.rho_reference[i,j]) : 0.
        candidate[i,j,k]=state[i,j,k]-dt*(fi[i,j,k]-fi[mod1(i-1,ni),j,k]+fj[i,j+1,k]-fj[i,j,k])/g.volume[i,j]+dt*source
    end
    candidate,boundary
end

function _stable_dt(s;cfl=.35)
    g=s.grid;rate=0.
    for j in 1:g.NJ,i in 1:g.NI
        out=max(s.qi[i,j],0.)+max(-s.qi[mod1(i-1,g.NI),j],0.)+max(s.qj[i,j+1],0.)+max(-s.qj[i,j],0.)
        rate=max(rate,out/g.volume[i,j])
    end
    # Gravity wave cap is based on the actual minimum cell altitude, without
    # replacing the physical water acoustic speed in a compressible equation.
    altitude=minimum(g.volume[i,j]/max(hypot(g.six[i,j],g.siy[i,j]),hypot(g.sjx[i,j],g.sjy[i,j])) for i in 1:g.NI,j in 1:g.NJ)
    min(rate>0 ? cfl/rate : Inf,sqrt(cfl*altitude/max(g.gravity,1e-30)))
end

function projection_step!(s::ProjectionState;dt_cap=Inf,cfl=.35)
    0<cfl<=.5 && dt_cap>0 || throw(ArgumentError("invalid projection step settings"))
    dt=min(Float64(dt_cap),_stable_dt(s;cfl));g=s.grid
    dt>64eps(max(s.elapsed,dt)) || throw(ArgumentError("projection step is below physical-clock resolution; finish duration within roundoff"))
    oldvel=_velocity(s);candidate,boundary=_transport(s,dt)
    all(isfinite,candidate) && minimum(view(candidate,:,:,1:3))>=0 || error("projection transport refused nonpositive phase mass; previous state retained")
    rho=dropdims(sum(candidate[:,:,1:3];dims=3);dims=3)
    velocity=candidate[:,:,4:5]./reshape(rho,g.NI,g.NJ,1);delta=velocity-oldvel
    qi=copy(s.qi);qj=copy(s.qj)
    for j in 1:g.NJ,i in 1:g.NI
        ip=mod1(i+1,g.NI)
        qi[i,j]+=.5(delta[i,j,1]+delta[ip,j,1])*g.six[i,j]+.5(delta[i,j,2]+delta[ip,j,2])*g.siy[i,j]
    end
    for j in 2:g.NJ,i in 1:g.NI
        qj[i,j]+=.5(delta[i,j-1,1]+delta[i,j,1])*g.sjx[i,j]+.5(delta[i,j-1,2]+delta[i,j,2])*g.sjy[i,j]
    end
    for i in 1:g.NI
        s.boundary_kind[i]==2 && (qj[i,end]+=delta[i,end,1]*g.sjx[i,end]+delta[i,end,2]*g.sjy[i,end])
    end
    solved=_project_faces!(s,qi,qj,rho,dt);p=solved.pressure;gx,gy=_pressure_gradient(s,p)
    candidate[:,:,4].-=dt.*gx;candidate[:,:,5].-=dt.*gy
    all(isfinite,candidate) || error("projection momentum update refused nonfinite state")
    closure=sum(candidate[:,:,k]./s.phase_density[k] for k in 1:3)
    maximum(abs,closure.-1)<1e-8 || error("projection phase-volume closure failed; previous state retained")
    s.mass=copy(candidate[:,:,1:3]);s.momentum=copy(candidate[:,:,4:5]);s.pressure=p;s.qi=qi;s.qj=qj
    s.pressure_residual=solved.residual
    s.elapsed+=dt;s.steps+=1;s.boundary_mass_integral .+=dt.*boundary
    (;dt,pressure_residual=s.pressure_residual,divergence_linf=maximum(abs,projection_divergence(s)),
        volume_closure_linf=projection_volume_error(s),clock="physical time; incompressible constituent equations")
end

function projection_loads(s::ProjectionState)
    g=s.grid;fx=fy=cm=0.;q=.5s.phase_density[1]*g.speed^2
    for i in 1:g.NI
        p=s.pressure[i,1];dfx=-p*g.sjx[i,1];dfy=-p*g.sjy[i,1]
        fx+=dfx;fy+=dfy
        x=.5(g.X[i,1]+g.X[i+1,1]);y=.5(g.Y[i,1]+g.Y[i+1,1])
        cm+=(x-g.moment_x)*dfy-(y-g.moment_y)*dfx
    end
    d=max(q*g.chord,eps())
    (;cl=fy/d,cd=fx/d,cm=cm/(d*g.chord),force_x=fx,force_y=fy)
end

end
