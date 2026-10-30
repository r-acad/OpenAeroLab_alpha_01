# Included in CMeshFreeSurface. Checked CPU finite-volume reference for the
# five-state interface model. It is deliberately separate from airfoil CFD.
export CartesianWorkspace, cartesian_stable_step, cartesian_step!

struct CartesianWorkspace{T<:AbstractFloat}
    trial::Array{T,3}
    flux_x::Array{T,3}
    flux_y::Array{T,3}
    source::Array{T,3}
    face_rate::Array{T,2}
end

function CartesianWorkspace(U::Array{T,3}) where T<:AbstractFloat
    nx,ny,nv=size(U)
    nx>=2 && ny>=2 && nv==5 || throw(ArgumentError("Cartesian state must have shape (nx>=2,ny>=2,5)"))
    CartesianWorkspace(similar(U),zeros(T,nx+1,ny,5),zeros(T,nx,ny+1,5),similar(U),zeros(T,nx,ny))
end

@inline _cartesian_state(U,i,j)=ntuple(k->U[i,j,k],Val(5))

function cartesian_stable_step(U,m,dx,dy;cfl=.4)
    dx>0 && dy>0 && 0<cfl<=.5 || throw(ArgumentError("positive grid spacing and CFL <= .5 required"))
    rate=0.0
    for j in axes(U,2),i in axes(U,1)
        q=primitive(_cartesian_state(U,i,j),m)
        rate=max(rate,(abs(q.u)+q.c)/dx+(abs(q.v)+q.c)/dy)
    end
    cfl/rate
end

function _cartesian_fluxes!(U,w,m,dx,dy,ymin,gravity,bc_x,bc_y,face_flux)
    nx,ny,_=size(U)
    fill!(w.face_rate,0)
    for j in 1:ny,i in 1:nx+1
        z=ymin+(j-.5)*dy
        f=if i==1 && bc_x===:wall
            q=primitive(_cartesian_state(U,1,j),m);(0.,q.p,0.,0.,0.)
        elseif i==nx+1 && bc_x===:wall
            q=primitive(_cartesian_state(U,nx,j),m);(0.,q.p,0.,0.,0.)
        else
            il=i==1 ? (bc_x===:periodic ? nx : 1) : i-1
            ir=i==nx+1 ? (bc_x===:periodic ? 1 : nx) : i
            L=_cartesian_state(U,il,j);R=_cartesian_state(U,ir,j)
            if face_flux isa AUSMUpFlux
                i>1 && (w.face_rate[i-1,j]+=ausm_face_speed(face_flux,L,R,1.,0.,m;cell_density=total_density(L))/dx)
                i<=nx && (w.face_rate[i,j]+=ausm_face_speed(face_flux,L,R,1.,0.,m;cell_density=total_density(R))/dx)
            end
            face_flux(L,R,1.,0.,m)
        end
        for k in 1:5;w.flux_x[i,j,k]=f[k];end
    end
    for j in 1:ny+1,i in 1:nx
        zface=ymin+(j-1)*dy
        f=if j==1 && bc_y===:wall
            state=_cartesian_state(U,i,1)
            face=gravity>0 ? hydrostatic_reconstruct(state,ymin+.5dy,zface,m;gravity) : state
            p=mixture_pressure(face,m);(0.,0.,p,0.,0.)
        elseif j==ny+1 && bc_y===:wall
            state=_cartesian_state(U,i,ny)
            face=gravity>0 ? hydrostatic_reconstruct(state,ymin+(ny-.5)*dy,zface,m;gravity) : state
            p=mixture_pressure(face,m);(0.,0.,p,0.,0.)
        else
            jl=j==1 ? (bc_y===:periodic ? ny : 1) : j-1
            jr=j==ny+1 ? (bc_y===:periodic ? 1 : ny) : j
            lower=_cartesian_state(U,i,jl);upper=_cartesian_state(U,i,jr)
            if gravity>0
                zl=ymin+(jl-.5)*dy;zr=ymin+(jr-.5)*dy
                lower=hydrostatic_reconstruct(lower,zl,zface,m;gravity)
                upper=hydrostatic_reconstruct(upper,zr,zface,m;gravity)
            end
            if face_flux isa AUSMUpFlux
                j>1 && (w.face_rate[i,j-1]+=ausm_face_speed(face_flux,lower,upper,0.,1.,m;cell_density=total_density(_cartesian_state(U,i,j-1)))/dy)
                j<=ny && (w.face_rate[i,j]+=ausm_face_speed(face_flux,lower,upper,0.,1.,m;cell_density=total_density(_cartesian_state(U,i,j)))/dy)
            end
            face_flux(lower,upper,0.,1.,m)
        end
        for k in 1:5;w.flux_y[i,j,k]=f[k];end
    end
    fill!(w.source,0)
    if gravity>0
        for j in 1:ny,i in 1:nx
            z=ymin+(j-.5)*dy
            source=hydrostatic_cell_source(_cartesian_state(U,i,j),z,z-.5dy,z+.5dy,m;gravity)
            for k in 1:5;w.source[i,j,k]=source[k];end
        end
    end
    nothing
end

"""One checked, conservative Cartesian interface step on a real physical clock.

Both noncondensable air and vapour partial masses use the carrier face flux.
Hydrostatic reconstruction and the matching gravity quadrature are active when
gravity > 0. This first-order inviscid reference admits overturned interfaces,
but has no surface tension, phase-change source, viscosity, turbulence or body
wall treatment. It does not qualify an underwater airfoil calculation.

Failed phase/EOS admission halves the step and retries the unchanged base state;
no density, pressure or fraction is clipped. Walls impose exactly zero mass
and constituent flux. Periodic vertical boundaries require gravity = 0.
"""
function cartesian_step!(U::Array{T,3},w::CartesianWorkspace{T},m::MaterialModel,
    dx::Real,dy::Real;dt_cap::Real=Inf,cfl::Real=.4,gravity::Real=9.80665,
    ymin::Real=0.0,bc_x::Symbol=:wall,bc_y::Symbol=:wall,max_retries::Int=16,
    flux_modifier! = nothing,face_flux=hllc_flux) where T
    size(U)==size(w.trial) || throw(ArgumentError("Cartesian workspace shape mismatch"))
    bc_x in (:wall,:periodic,:outflow) && bc_y in (:wall,:periodic,:outflow) ||
        throw(ArgumentError("Cartesian boundary must be wall, periodic or outflow"))
    isfinite(gravity) && gravity>=0 && dt_cap>0 && max_retries>=0 || throw(ArgumentError("invalid time step/gravity settings"))
    gravity>0 && bc_y===:periodic && throw(ArgumentError("vertical periodicity is incompatible with a hydrostatic column"))
    dt=min(Float64(dt_cap),cartesian_stable_step(U,m,dx,dy;cfl))
    _cartesian_fluxes!(U,w,m,dx,dy,ymin,gravity,bc_x,bc_y,face_flux)
    flux_modifier! === nothing || flux_modifier!(U,w,m,dx,dy,ymin)
    if face_flux isa AUSMUpFlux
        # Actual pressure-diffusion clock and donor drain use the final gate/
        # wall overrides. The unchanged acoustic cap remains an extra guard.
        for j in axes(U,2),i in axes(U,1)
            outgoing=sum(max(w.flux_x[i+1,j,k],0.)+max(-w.flux_x[i,j,k],0.) for k in (1,4,5))/dx+
                     sum(max(w.flux_y[i,j+1,k],0.)+max(-w.flux_y[i,j,k],0.) for k in (1,4,5))/dy
            w.face_rate[i,j]+=outgoing/total_density(_cartesian_state(U,i,j))
        end
        dt=min(dt,cfl/maximum(w.face_rate))
    end
    nx,ny,_=size(U)
    rejected=nothing
    for attempt in 0:max_retries
        valid=true
        for j in 1:ny,i in 1:nx
            candidate=ntuple(Val(5)) do k
                U[i,j,k]-dt*((w.flux_x[i+1,j,k]-w.flux_x[i,j,k])/dx+
                    (w.flux_y[i,j+1,k]-w.flux_y[i,j,k])/dy-w.source[i,j,k])
            end
            good=pressure_state(candidate,m)[1]
            good || (rejected=(i=i,j=j,state=candidate,pressure=pressure_state(candidate,m)))
            valid &= good
            for k in 1:5;w.trial[i,j,k]=candidate[k];end
        end
        if valid
            copyto!(U,w.trial)
            return (dt=dt,retries=attempt,clock="physical time",flux=face_flux isa AUSMUpFlux ? "experimental AUSM+-up with actual-face clock" : "HLLC with HLL admissibility fallback",
                gravity=Float64(gravity),equations=5)
        end
        dt*=.5
    end
    error("Cartesian free-surface step exhausted $max_retries retries; original state retained; rejected=$rejected")
end
