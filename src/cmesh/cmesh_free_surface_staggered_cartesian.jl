# Isolated CPU admission prototype. Orthogonal MAC faces only; not wired into
# the airfoil app, curved body meshes, CUDA, viscous or phase-change solvers.
module StaggeredConstituentCartesian
using LinearAlgebra, SparseArrays
export MACGrid, MACState, make_state, face_velocity, dual_mass, partial_mass,
       kinetic_energy, divergence, project_faces, stable_dt, step!, pressure_force

struct MACGrid
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    periodic::Bool
    faces::Vector{NTuple{3,Int}}
    D::SparseMatrixCSC{Float64,Int}
end

function MACGrid(nx::Int,ny::Int;lx=1.,ly=1.,periodic=false)
    nx>=4 && ny>=4 && isfinite(lx) && isfinite(ly) && lx>0 && ly>0 || throw(ArgumentError("nx,ny>=4 and finite positive lengths required"))
    dx,dy=Float64(lx)/nx,Float64(ly)/ny
    faces=NTuple{3,Int}[];rows=Int[];cols=Int[];values=Float64[]
    cell(i,j)=i+(j-1)*nx
    for j in 1:ny,i in (periodic ? (1:nx) : (2:nx))
        push!(faces,(1,i,j));f=length(faces);im=mod1(i-1,nx)
        push!(rows,cell(im,j),cell(i,j));push!(cols,f,f);push!(values,dy,-dy)
    end
    for j in (periodic ? (1:ny) : (2:ny)),i in 1:nx
        push!(faces,(2,i,j));f=length(faces);jm=mod1(j-1,ny)
        push!(rows,cell(i,jm),cell(i,j));push!(cols,f,f);push!(values,dx,-dx)
    end
    MACGrid(nx,ny,dx,dy,periodic,faces,sparse(rows,cols,values,nx*ny,length(faces)))
end

mutable struct MACState
    grid::MACGrid
    mass::Array{Float64,3} # constituent partial densities [kg/m3]
    ux::Matrix{Float64}   # normal x-face velocity [m/s]
    vy::Matrix{Float64}   # normal y-face velocity [m/s]
    pressure::Matrix{Float64} # dynamic pressure [Pa], one gauge fixed
    phase_density::NTuple{3,Float64}
    reference_pressure::Matrix{Float64}
    gravity::Float64
    elapsed::Float64
    steps::Int
    boundary_mass_integral::Vector{Float64}
    pressure_impulse::Vector{Float64}
    diagnostics::Dict{String,Float64}
end

function _copy_periodic!(u,v,g)
    if g.periodic;u[end,:].=u[1,:];v[:,end].=v[:,1];end
    nothing
end

function make_state(g::MACGrid,beta;phase_density=(998.2,.017288949984560802,1.2041183163746156),
    ux=zeros(g.nx+1,g.ny),vy=zeros(g.nx,g.ny+1),gravity=0.,reference_density=nothing)
    size(beta)==(g.nx,g.ny,3) || throw(ArgumentError("three cell constituent volume fractions required"))
    all(isfinite,beta) && minimum(beta)>=0 && maximum(abs,sum(beta;dims=3).-1)<1e-13 || throw(ArgumentError("finite positive closed phase volumes required"))
    size(ux)==(g.nx+1,g.ny) && size(vy)==(g.nx,g.ny+1) || throw(ArgumentError("wrong staggered face sizes"))
    all(x->isfinite(x)&&x>0,phase_density) && isfinite(gravity) && gravity>=0 || throw(ArgumentError("finite positive material densities/nonnegative gravity required"))
    all(isfinite,ux) && all(isfinite,vy) || throw(ArgumentError("finite staggered velocities required"))
    gravity>0 && g.periodic && throw(ArgumentError("gravity requires closed vertical boundaries in this prototype"))
    mass=beta.*reshape(collect(phase_density),1,1,3);u=Float64.(ux);v=Float64.(vy)
    if g.periodic
        _copy_periodic!(u,v,g)
    else
        maximum(abs,u[[1,end],:])==0 && maximum(abs,v[:,[1,end]])==0 || throw(ArgumentError("closed slip-normal velocities must be zero"))
    end
    rho=dropdims(sum(mass;dims=3);dims=3)
    ref=reference_density===nothing ? copy(rho) : Float64.(reference_density)
    size(ref)==size(rho) || throw(ArgumentError("wrong reference density size"))
    all(x->isfinite(x)&&x>0,ref) || throw(ArgumentError("finite positive reference density required"))
    gravity>0 && maximum(abs,ref.-repeat(ref[1:1,:],g.nx,1))>1e-10 &&
        throw(ArgumentError("gravity reference must be a flat density column; not an arbitrary frozen flow"))
    p0=fill(101325.,g.nx,g.ny)
    for j in g.ny-1:-1:1,i in 1:g.nx
        p0[i,j]=p0[i,j+1]+gravity*g.dy*.5(ref[i,j]+ref[i,j+1])
    end
    MACState(g,mass,u,v,zeros(g.nx,g.ny),Float64.(phase_density),p0,Float64(gravity),0.,0,zeros(3),zeros(2),Dict{String,Float64}())
end

face_velocity(s::MACState)=[kind==1 ? s.ux[i,j] : s.vy[i,j] for (kind,i,j) in s.grid.faces]
function _face_arrays(g,w)
    u=zeros(g.nx+1,g.ny);v=zeros(g.nx,g.ny+1)
    for (f,(kind,i,j)) in enumerate(g.faces)
        if kind==1;u[i,j]=w[f];else;v[i,j]=w[f];end
    end
    _copy_periodic!(u,v,g);u,v
end
function dual_mass(g,mass;constituent=0)
    rho=constituent==0 ? dropdims(sum(mass;dims=3);dims=3) : mass[:,:,constituent]
    V=g.dx*g.dy
    [kind==1 ? .5V*(rho[mod1(i-1,g.nx),j]+rho[i,j]) : .5V*(rho[i,mod1(j-1,g.ny)]+rho[i,j]) for (kind,i,j) in g.faces]
end
dual_mass(s::MACState)=dual_mass(s.grid,s.mass)
partial_mass(s::MACState)=[sum(s.mass[:,:,k])*s.grid.dx*s.grid.dy for k in 1:3]
kinetic_energy(s::MACState)=.5sum(dual_mass(s).*face_velocity(s).^2)
divergence(s::MACState)=reshape(s.grid.D*face_velocity(s),s.grid.nx,s.grid.ny)./(s.grid.dx*s.grid.dy)
"Net pressure force on stored fluid dual momenta; not an airfoil body load."
function pressure_force(g,p)
    force=transpose(g.D)*vec(p)
    [sum(force[f] for (f,face) in enumerate(g.faces) if face[1]==k) for k in 1:2]
end

function _cg(A,b;rtol=1e-11,atol=1e-12,max_iterations=1000)
    x=zeros(length(b));r=copy(b);target=max(atol,rtol*norm(b,Inf))
    norm(r,Inf)<=target && return x
    z=r./diag(A);d=copy(z);rz=dot(r,z)
    for _ in 1:max_iterations
        ad=A*d;den=dot(d,ad);den>0 || error("staggered CG curvature failed; previous state retained")
        alpha=rz/den;x.+=alpha.*d;r.-=alpha.*ad
        norm(r,Inf)<=target && return x
        z.=r./diag(A);newrz=dot(r,z);d.=z.+(newrz/rz).*d;rz=newrz
    end
    error("staggered pressure CG did not converge; previous state retained")
end

function project_faces(g,mass,wstar,dt;solver=:direct,max_iterations=1000,rtol=1e-11,atol=1e-12)
    dt>0 && isfinite(dt) || throw(ArgumentError("finite positive physical timestep required"))
    M=dual_mass(g,mass);all(isfinite,M) && minimum(M)>0 || error("nonfinite/nonpositive dual material mass")
    all(isfinite,wstar) || error("nonfinite face predictor")
    A=g.D*spdiagm(0=>1 ./M)*transpose(g.D)
    rhs=-(g.D*wstar)./dt
    p=zeros(g.nx*g.ny)
    if solver===:direct
        p[2:end]=cholesky(Symmetric(A[2:end,2:end]))\rhs[2:end]
    elseif solver===:cg
        p[2:end]=_cg(A[2:end,2:end],rhs[2:end];max_iterations,rtol,atol)
    else
        throw(ArgumentError("pressure solver must be direct or cg"))
    end
    correction=dt.*(transpose(g.D)*p)./M
    projected=wstar.+correction
    all(isfinite,projected) && all(isfinite,p) || error("nonfinite projection; previous state retained")
    residual=norm(A*p-rhs,Inf)
    residual<=max(100atol,100rtol*norm(rhs,Inf)) || error("staggered pressure residual refused; previous state retained")
    (;velocity=projected,pressure=reshape(p,g.nx,g.ny),residual,
        energy_before=.5sum(M.*wstar.^2),energy_after=.5sum(M.*projected.^2),
        correction_energy=.5sum(M.*correction.^2),pressure_force=pressure_force(g,reshape(p,g.nx,g.ny)))
end

function _phase_fluxes(s)
    g=s.grid;fx=zeros(g.nx+1,g.ny,3);fy=zeros(g.nx,g.ny+1,3)
    for j in 1:g.ny,i in (g.periodic ? (1:g.nx) : (2:g.nx))
        donor=s.ux[i,j]>=0 ? mod1(i-1,g.nx) : i
        for k in 1:3;fx[i,j,k]=g.dy*s.ux[i,j]*s.mass[donor,j,k];end
    end
    for j in (g.periodic ? (1:g.ny) : (2:g.ny)),i in 1:g.nx
        donor=s.vy[i,j]>=0 ? mod1(j-1,g.ny) : j
        for k in 1:3;fy[i,j,k]=g.dx*s.vy[i,j]*s.mass[i,donor,k];end
    end
    if g.periodic;fx[end,:,:].=fx[1,:,:];fy[:,end,:].=fy[:,1,:];end
    fx,fy
end
@inline _fx(g,F,i,j)=F[g.periodic ? mod1(i,g.nx) : i,j]
@inline _fy(g,F,i,j)=F[i,g.periodic ? mod1(j,g.ny) : j]
function _dual_fluxes(g,Fx,Fy,kind,i,j)
    if kind==1
        im=mod1(i-1,g.nx)
        E=.5(_fx(g,Fx,i,j)+_fx(g,Fx,i+1,j));W=.5(_fx(g,Fx,i-1,j)+_fx(g,Fx,i,j))
        N=.5(_fy(g,Fy,im,j+1)+_fy(g,Fy,i,j+1));S=.5(_fy(g,Fy,im,j)+_fy(g,Fy,i,j))
    else
        jm=mod1(j-1,g.ny)
        E=.5(_fx(g,Fx,i+1,jm)+_fx(g,Fx,i+1,j));W=.5(_fx(g,Fx,i,jm)+_fx(g,Fx,i,j))
        N=.5(_fy(g,Fy,i,j)+_fy(g,Fy,i,j+1));S=.5(_fy(g,Fy,i,j-1)+_fy(g,Fy,i,j))
    end
    E,W,N,S
end
function _velocity_at(s,kind,i,j)
    g=s.grid
    if g.periodic
        return kind==1 ? s.ux[mod1(i,g.nx),mod1(j,g.ny)] : s.vy[mod1(i,g.nx),mod1(j,g.ny)]
    end
    kind==1 ? s.ux[clamp(i,1,g.nx+1),clamp(j,1,g.ny)] : s.vy[clamp(i,1,g.nx),clamp(j,1,g.ny+1)]
end

function stable_dt(s;cfl=.35,phase_fluxes=_phase_fluxes(s))
    0<cfl<=.5 || throw(ArgumentError("0<cfl<=.5 required for primal and dual explicit transport"))
    g=s.grid;fx,fy=phase_fluxes;Fx=dropdims(sum(fx;dims=3);dims=3);Fy=dropdims(sum(fy;dims=3);dims=3)
    rate=0.;V=g.dx*g.dy
    for j in 1:g.ny,i in 1:g.nx
        out=g.dy*(max(s.ux[i+1,j],0)+max(-s.ux[i,j],0))+g.dx*(max(s.vy[i,j+1],0)+max(-s.vy[i,j],0))
        rate=max(rate,out/V)
    end
    M=dual_mass(s)
    for (f,(kind,i,j)) in enumerate(g.faces)
        E,W,N,S=_dual_fluxes(g,Fx,Fy,kind,i,j)
        rate=max(rate,(max(E,0)+max(-W,0)+max(N,0)+max(-S,0))/M[f])
    end
    rate>0 ? cfl/rate : Inf
end

function step!(s::MACState;dt_cap=.001,cfl=.35,pressure_solver=:direct,pressure_max_iterations=1000)
    isfinite(dt_cap) && dt_cap>0 || throw(ArgumentError("finite positive timestep cap required"))
    g=s.grid;phaseflux=_phase_fluxes(s);dt=min(Float64(dt_cap),stable_dt(s;cfl,phase_fluxes=phaseflux))
    dt>64eps(max(s.elapsed,dt)) || throw(ArgumentError("step below physical-clock resolution"))
    fx,fy=phaseflux;candidate=similar(s.mass);V=g.dx*g.dy
    for k in 1:3,j in 1:g.ny,i in 1:g.nx
        candidate[i,j,k]=s.mass[i,j,k]-dt*(fx[i+1,j,k]-fx[i,j,k]+fy[i,j+1,k]-fy[i,j,k])/V
    end
    all(isfinite,candidate) && minimum(candidate)>=0 || error("phase transport refused; previous state retained")
    closure=maximum(abs,sum(candidate[:,:,k]./s.phase_density[k] for k in 1:3).-1)
    closure<1e-10 || error("phase volume closure refused; previous state retained")
    Fx=dropdims(sum(fx;dims=3);dims=3);Fy=dropdims(sum(fy;dims=3);dims=3)
    oldM=dual_mass(s);newM=dual_mass(g,candidate);oldw=face_velocity(s);wstar=similar(oldw)
    dual_error=0.
    for k in 1:3
        before=dual_mass(g,s.mass;constituent=k);after=dual_mass(g,candidate;constituent=k)
        for (f,(kind,i,j)) in enumerate(g.faces)
            E,W,N,S=_dual_fluxes(g,view(fx,:,:,k),view(fy,:,:,k),kind,i,j)
            dual_error=max(dual_error,abs(after[f]-before[f]+dt*(E-W+N-S)))
        end
    end
    dual_error<1e-10*max(maximum(oldM),1.) || error("dual constituent balance failed; previous state retained")
    for (f,(kind,i,j)) in enumerate(g.faces)
        E,W,N,S=_dual_fluxes(g,Fx,Fy,kind,i,j);w=oldw[f]
        east=E>=0 ? w : _velocity_at(s,kind,i+1,j);west=W>=0 ? _velocity_at(s,kind,i-1,j) : w
        north=N>=0 ? w : _velocity_at(s,kind,i,j+1);south=S>=0 ? _velocity_at(s,kind,i,j-1) : w
        wstar[f]=(oldM[f]*w-dt*(E*east-W*west+N*north-S*south))/newM[f]
    end
    transported_energy=.5sum(newM.*wstar.^2)
    # Fully discrete pressure reference and gravity use the SAME face force.
    # The reference column is flat; arbitrary frozen density fields are refused.
    reference_force=transpose(g.D)*vec(s.reference_pressure)
    for (f,(kind,_,_)) in enumerate(g.faces)
        if kind==2;wstar[f]+=dt*(reference_force[f]-s.gravity*newM[f])/newM[f];end
    end
    projected=project_faces(g,candidate,wstar,dt;solver=pressure_solver,max_iterations=pressure_max_iterations)
    u,v=_face_arrays(g,projected.velocity)
    newdiv=norm(g.D*projected.velocity,Inf)/V
    newdiv<1e-7 || error("projected divergence refused; previous state retained")
    oldenergy=kinetic_energy(s)
    diagnostics=Dict("dt_seconds"=>dt,"volume_closure"=>closure,"dual_partial_mass_balance_max_kg_m"=>dual_error,
        "divergence_linf_s_inv"=>newdiv,"pressure_residual"=>projected.residual,"energy_before"=>oldenergy,
        "energy_after_transport"=>transported_energy,"energy_before_projection"=>projected.energy_before,
        "energy_after_projection"=>projected.energy_after,"projection_correction_energy"=>projected.correction_energy)
    # Commit only after transport, pressure, geometry/units and closure admission.
    s.mass=candidate;s.ux=u;s.vy=v;s.pressure=projected.pressure
    s.elapsed+=dt;s.steps+=1;s.pressure_impulse .+=dt.*projected.pressure_force;s.diagnostics=diagnostics
    (;dt,diagnostics)
end
end
