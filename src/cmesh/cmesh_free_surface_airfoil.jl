# Independent development backend: nonlinear air/water interface around a
# resolved, impermeable airfoil. Not a viscous cavitation/stall model.
module CMeshFreeSurfaceAirfoil

include(joinpath(@__DIR__,"cmesh_free_surface.jl"))
Base.include(CMeshFreeSurface,joinpath(@__DIR__,"cmesh_free_surface_thornber.jl"))
include(joinpath(@__DIR__,"cmesh_gen.jl"))
using .CMeshFreeSurface
using .CMeshGen
export AirfoilGrid, AirfoilWorkspace, build_airfoil_grid, initial_state,
       airfoil_step!, airfoil_loads, reference_pressure, material_mass
export body_chord_coordinates

struct AirfoilGrid
    NI::Int
    NJ::Int
    X::Matrix{Float64}
    Y::Matrix{Float64}
    xc::Matrix{Float64}
    yc::Matrix{Float64}
    volume::Matrix{Float64}
    six::Matrix{Float64}
    siy::Matrix{Float64}
    sjx::Matrix{Float64}
    sjy::Matrix{Float64}
    pi0::Matrix{Float64}
    pj0::Matrix{Float64}
    pc0::Matrix{Float64}
    rho0::Matrix{Float64}
    beta_water0::Matrix{Float64}
    source_x::Matrix{Float64}
    source_y::Matrix{Float64}
    chord::Float64
    gravity::Float64
    water_level::Float64
    alpha::Float64
    speed::Float64
    moment_x::Float64
    moment_y::Float64
    vapor_seed::Float64
    body_surface_cells::Int
    min_corner_sine::Float64
end

struct AirfoilWorkspace
    trial::Array{Float64,3}
    fi::Array{Float64,3}
    fj::Array{Float64,3}
    rate::Matrix{Float64}
    residual::Array{Float64,3}
    boundary_ledger::Vector{Float64}
end
AirfoilWorkspace(g::AirfoilGrid)=AirfoilWorkspace(zeros(g.NI,g.NJ,5),
    zeros(g.NI,g.NJ,5),zeros(g.NI,g.NJ+1,5),zeros(g.NI,g.NJ),
    zeros(g.NI,g.NJ,5),zeros(5))

"Unrotate physical SI coordinates into dimensionless airfoil coordinates."
@inline function body_chord_coordinates(g::AirfoilGrid,x::Real,y::Real)
    ca,sa=cos(g.alpha),sin(g.alpha)
    dx=(x-g.moment_x)/g.chord;dy=(y-g.moment_y)/g.chord
    (.25+ca*dx-sa*dy,sa*dx+ca*dy)
end

"Exact pure-water/air hydrostatic reference joined at the atmospheric interface."
function reference_pressure(z,m::MaterialModel;water_level=0.,gravity=9.80665)
    hydrostatic_pressure(z,m.p_ref,water_level,z<water_level ? :water : :air,m;gravity)
end

# Eight-point Gauss integration with an explicit split at the material contact.
# One common face average is used by both neighboring cells.
const _GX=(-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
            .1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363)
const _GW=(.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
           .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763)
function _pressure_average(za,zb,m,h,g)
    abs(zb-za)<1e-14 && return reference_pressure((za+zb)/2,m;water_level=h,gravity=g)
    cuts=sort!([0.,1.,(h-za)/(zb-za)])
    filter!(t->0<=t<=1,cuts)
    value=0.
    for k in 1:length(cuts)-1
        a,b=cuts[k],cuts[k+1];b>a || continue
        for q in eachindex(_GX)
            t=(a+b)/2+(b-a)/2*_GX[q]
            value+=(b-a)/2*_GW[q]*reference_pressure(za+t*(zb-za),m;water_level=h,gravity=g)
        end
    end
    value
end

"Exact polygon area fraction below a horizontal water level."
function _water_fraction(xx,yy,h)
    vertices=[(Float64(xx[k]),Float64(yy[k])) for k in eachindex(xx)]
    original=abs(sum(vertices[k][1]*vertices[mod1(k+1,length(vertices))][2]-vertices[mod1(k+1,length(vertices))][1]*vertices[k][2] for k in eachindex(vertices)))/2
    clipped=Tuple{Float64,Float64}[]
    for k in eachindex(vertices)
        A=vertices[k];B=vertices[mod1(k+1,length(vertices))];insideA=A[2]<=h;insideB=B[2]<=h
        insideA && push!(clipped,A)
        if insideA!=insideB
            t=(h-A[2])/(B[2]-A[2]);push!(clipped,(A[1]+t*(B[1]-A[1]),h))
        end
    end
    length(clipped)<3 && return 0.
    wet=abs(sum(clipped[k][1]*clipped[mod1(k+1,length(clipped))][2]-clipped[mod1(k+1,length(clipped))][1]*clipped[k][2] for k in eachindex(clipped)))/2
    fraction=wet/original
    -1e-12<=fraction<=1+1e-12 || error("invalid geometric water-volume fraction")
    # Only geometric roundoff is bounded here; no evolved phase is clipped.
    min(1.,max(0.,fraction))
end
@inline function _wet_edge_fraction(za,zb,h)
    za<=h && zb<=h && return 1.
    za>=h && zb>=h && return 0.
    t=(h-za)/(zb-za)
    za<h ? t : 1-t
end

function build_airfoil_grid(;code::Int=12,geometry::String="",n_foil::Int=96,n_normal::Int=24,
    chord_m::Real=1.,submergence_chords::Real=1.,water_depth_chords::Real=4.,
    air_height_chords::Real=2.,upstream_chords::Real=3.,downstream_chords::Real=5.,
    alpha_deg::Real=0.,speed_m_s::Real=3.,gravity::Real=9.80665,
    radial_growth::Real=1.22,initial_vapor_fraction::Real=0.,model::MaterialModel=MaterialModel())
    n_foil>=32 && iseven(n_foil) && n_normal>=8 || throw(ArgumentError("even n_foil>=32 and n_normal>=8 required"))
    chord_m>0 && submergence_chords>0 && water_depth_chords>submergence_chords+.3 ||
        throw(ArgumentError("positive chord/depth and a floor at least .3 chords below the body required"))
    air_height_chords>.1 && upstream_chords>1 && downstream_chords>1 && radial_growth>=1 ||
        throw(ArgumentError("invalid underwater domain dimensions/radial growth"))
    0<=gravity && 0<=speed_m_s && abs(alpha_deg)<=45 || throw(ArgumentError("invalid gravity, speed or angle"))
    0<=initial_vapor_fraction<=.01 || throw(ArgumentError("initial vapor fraction must lie in [0,.01]"))
    af=isempty(geometry) ? naca4_points(code;n=n_foil+1) : read_airfoil_dat(geometry)
    if !isempty(geometry)
        conditioned=condition_airfoil(af)
        dense=conditioned.airfoil;ids=round.(Int,range(1,length(dense.x);length=n_foil+1))
        af=AirfoilPts(dense.x[ids],dense.y[ids])
    end
    bx,by=copy(af.x),copy(af.y)
    gap=hypot(bx[end]-bx[1],by[end]-by[1])
    # The native finite base is a genuine wall, never a collapsed wake cut.
    if gap>1e-10
        push!(bx,(bx[end]+bx[1])/2);push!(by,(by[end]+by[1])/2)
    else
        pop!(bx);pop!(by)
    end
    ni=length(bx);nj=n_normal;c=Float64(chord_m);h=0.;depth=Float64(submergence_chords)*c
    a=-deg2rad(Float64(alpha_deg));ca,sa=cos(a),sin(a)
    for i in eachindex(bx)
        xx,yy=bx[i]-.5,by[i]
        # Rotate body while the far-field stream remains horizontal.
        bx[i]=c*(.5+ca*xx-sa*yy);by[i]=c*(sa*xx+ca*yy)-depth
    end
    center=(.5c,-depth)
    xmin=-Float64(upstream_chords)*c;xmax=(1+Float64(downstream_chords))*c
    ymin=-Float64(water_depth_chords)*c;ymax=Float64(air_height_chords)*c
    minimum(by)>ymin && maximum(by)<h || throw(ArgumentError("airfoil intersects the floor or initial water surface"))
    X=zeros(ni+1,nj+1);Y=similar(X)
    r=Float64(radial_growth)
    for i in 1:ni
        dx,dy=bx[i]-center[1],by[i]-center[2]
        tx=dx>0 ? (xmax-center[1])/dx : dx<0 ? (xmin-center[1])/dx : Inf
        ty=dy>0 ? (ymax-center[2])/dy : dy<0 ? (ymin-center[2])/dy : Inf
        scale=min(tx,ty)
        scale>1 || error("airfoil is not inside the underwater exterior box")
        for j in 1:nj+1
            t=r==1 ? (j-1)/nj : (r^(j-1)-1)/(r^nj-1)
            X[i,j]=bx[i]+t*(scale-1)*dx;Y[i,j]=by[i]+t*(scale-1)*dy
        end
    end
    X[end,:].=X[1,:];Y[end,:].=Y[1,:]
    xc=zeros(ni,nj);yc=similar(xc);vol=similar(xc)
    six=similar(xc);siy=similar(xc);sjx=zeros(ni,nj+1);sjy=similar(sjx)
    p0=similar(xc);rho0=similar(xc);beta0=similar(xc);pi0=similar(xc);pj0=similar(sjx)
    sx0=similar(xc);sy0=similar(xc);minsin=1.
    for j in 1:nj,i in 1:ni
        xx=(X[i,j],X[i+1,j],X[i+1,j+1],X[i,j+1]);yy=(Y[i,j],Y[i+1,j],Y[i+1,j+1],Y[i,j+1])
        twice=sum(xx[k]*yy[mod1(k+1,4)]-xx[mod1(k+1,4)]*yy[k] for k in 1:4)
        twice>0 || error("folded/non-star-shaped airfoil cell ($i,$j); use a body-fitted C+H grid")
        vol[i,j]=twice/2
        xc[i,j]=sum((xx[k]+xx[mod1(k+1,4)])*(xx[k]*yy[mod1(k+1,4)]-xx[mod1(k+1,4)]*yy[k]) for k in 1:4)/(3twice)
        yc[i,j]=sum((yy[k]+yy[mod1(k+1,4)])*(xx[k]*yy[mod1(k+1,4)]-xx[mod1(k+1,4)]*yy[k]) for k in 1:4)/(3twice)
        for k in 1:4
            km=mod1(k-1,4);kp=mod1(k+1,4)
            ux,uy=xx[k]-xx[km],yy[k]-yy[km];vx,vy=xx[kp]-xx[k],yy[kp]-yy[k]
            sine=(ux*vy-uy*vx)/(hypot(ux,uy)*hypot(vx,vy))
            sine>0 || error("nonconvex underwater airfoil cell ($i,$j)")
            minsin=min(minsin,sine)
        end
        six[i,j]=Y[i+1,j+1]-Y[i+1,j];siy[i,j]=X[i+1,j]-X[i+1,j+1]
        pi0[i,j]=_pressure_average(Y[i+1,j],Y[i+1,j+1],model,h,gravity)
        p0[i,j]=reference_pressure(yc[i,j],model;water_level=h,gravity)
        densities=phase_densities(p0[i,j],model)
        beta0[i,j]=_water_fraction(xx,yy,h)
        rho0[i,j]=beta0[i,j]*((1-initial_vapor_fraction)*densities.liquid+initial_vapor_fraction*densities.vapor)+(1-beta0[i,j])*densities.air
    end
    for j in 1:nj+1,i in 1:ni
        sjx[i,j]=Y[i,j]-Y[i+1,j];sjy[i,j]=X[i+1,j]-X[i,j]
        pj0[i,j]=_pressure_average(Y[i,j],Y[i+1,j],model,h,gravity)
    end
    for j in 1:nj,i in 1:ni
        im=mod1(i-1,ni)
        sx0[i,j]=(pi0[i,j]*six[i,j]-pi0[im,j]*six[im,j]+pj0[i,j+1]*sjx[i,j+1]-pj0[i,j]*sjx[i,j])/vol[i,j]
        sy0[i,j]=(pi0[i,j]*siy[i,j]-pi0[im,j]*siy[im,j]+pj0[i,j+1]*sjy[i,j+1]-pj0[i,j]*sjy[i,j])/vol[i,j]
    end
    AirfoilGrid(ni,nj,X,Y,xc,yc,vol,six,siy,sjx,sjy,pi0,pj0,p0,rho0,beta0,sx0,sy0,c,
        Float64(gravity),h,-a,Float64(speed_m_s),c*(.5-.25ca),-depth-.25c*sa,Float64(initial_vapor_fraction),n_foil,minsin)
end

function initial_state(g::AirfoilGrid,m::MaterialModel;speed_m_s::Real=g.speed,interface_shift::Real=0.)
    U=zeros(g.NI,g.NJ,5)
    for j in 1:g.NJ,i in 1:g.NI
        z=g.yc[i,j];p=reference_pressure(z,m;water_level=g.water_level+interface_shift,gravity=g.gravity)
        xx=(g.X[i,j],g.X[i+1,j],g.X[i+1,j+1],g.X[i,j+1]);yy=(g.Y[i,j],g.Y[i+1,j],g.Y[i+1,j+1],g.Y[i,j+1])
        water=interface_shift==0 ? g.beta_water0[i,j] : _water_fraction(xx,yy,g.water_level+interface_shift)
        state=conservative(p,speed_m_s,0.,water*g.vapor_seed,1-water,m)
        for k in 1:5;U[i,j,k]=state[k];end
    end
    U
end

@inline _state(U,i,j)=ntuple(k->U[i,j,k],Val(5))
@inline function _face_state(U,p_cell_ref,p_face_ref,m)
    q=primitive(U,m);p=q.p*(p_face_ref/p_cell_ref)
    p>0 || throw(DomainError(p,"nonpositive reconstructed airfoil face pressure"))
    # Composition is transported, not reset to the initial interface.
    rho=mixture_density(p,q.mass_fractions,m)
    (rho*q.mass_fractions.liquid,rho*q.u,rho*q.v,rho*q.mass_fractions.vapor,rho*q.mass_fractions.air)
end

function _fluxes!(U,w,g,m;outer_bc=:open,face_flux=hllc_flux)
    fill!(w.rate,0);fill!(w.boundary_ledger,0)
    for j in 1:g.NJ,i in 1:g.NI
        ip=mod1(i+1,g.NI);sx,sy=g.six[i,j],g.siy[i,j];area=hypot(sx,sy)
        L=_face_state(_state(U,i,j),g.pc0[i,j],g.pi0[i,j],m)
        R=_face_state(_state(U,ip,j),g.pc0[ip,j],g.pi0[i,j],m)
        f=face_flux(L,R,sx/area,sy/area,m)
        for k in 1:5;w.fi[i,j,k]=area*f[k];end
        qL,qR=primitive(L,m),primitive(R,m)
        rate=area*max(abs(qL.u*sx/area+qL.v*sy/area)+qL.c,abs(qR.u*sx/area+qR.v*sy/area)+qR.c)
        w.rate[i,j]+=rate/g.volume[i,j];w.rate[ip,j]+=rate/g.volume[ip,j]
    end
    for j in 1:g.NJ+1,i in 1:g.NI
        sx,sy=g.sjx[i,j],g.sjy[i,j];area=hypot(sx,sy);nx,ny=sx/area,sy/area
        if j==1
            R=_face_state(_state(U,i,1),g.pc0[i,1],g.pj0[i,1],m);qR=primitive(R,m)
            f=(0.,qR.p*nx,qR.p*ny,0.,0.)
            w.rate[i,1]+=area*(hypot(qR.u,qR.v)+qR.c)/g.volume[i,1]
        elseif j==g.NJ+1
            L=_face_state(_state(U,i,g.NJ),g.pc0[i,g.NJ],g.pj0[i,j],m);qL=primitive(L,m)
            if outer_bc===:wall
                f=(0.,qL.p*nx,qL.p*ny,0.,0.)
            else
                # Incoming characteristics use the hydrostatic reservoir;
                # outgoing extrapolation is not a nonreflecting wave boundary.
                z=(g.Y[i,j]+g.Y[i+1,j])/2
                water=_wet_edge_fraction(g.Y[i,j],g.Y[i+1,j],g.water_level)
                R=qL.u*nx+qL.v*ny<0 ? conservative(g.pj0[i,j],g.speed,0.,water*g.vapor_seed,1-water,m) : L
                f=face_flux(L,R,nx,ny,m)
            end
            w.rate[i,g.NJ]+=area*(hypot(qL.u,qL.v)+qL.c)/g.volume[i,g.NJ]
            for k in 1:5;w.boundary_ledger[k]+=area*f[k];end
        else
            L=_face_state(_state(U,i,j-1),g.pc0[i,j-1],g.pj0[i,j],m)
            R=_face_state(_state(U,i,j),g.pc0[i,j],g.pj0[i,j],m)
            f=face_flux(L,R,nx,ny,m);qL,qR=primitive(L,m),primitive(R,m)
            rate=area*max(hypot(qL.u,qL.v)+qL.c,hypot(qR.u,qR.v)+qR.c)
            w.rate[i,j-1]+=rate/g.volume[i,j-1];w.rate[i,j]+=rate/g.volume[i,j]
        end
        for k in 1:5;w.fj[i,j,k]=area*f[k];end
    end
    for j in 1:g.NJ,i in 1:g.NI
        im=mod1(i-1,g.NI);rho=total_density(_state(U,i,j))
        for k in 1:5
            source=k==2 ? g.source_x[i,j] : k==3 ? g.source_y[i,j]-g.gravity*(rho-g.rho0[i,j]) : 0.
            w.residual[i,j,k]=-(w.fi[i,j,k]-w.fi[im,j,k]+w.fj[i,j+1,k]-w.fj[i,j,k])/g.volume[i,j]+source
        end
    end
    nothing
end

"Checked conservative physical-time step; a rejected state is never committed."
function airfoil_step!(U,w,g,m;dt_cap=Inf,cfl=.35,max_retries=20,outer_bc=:open,face_flux=hllc_flux)
    0<cfl<=.5 && dt_cap>0 && outer_bc in (:open,:wall) || throw(ArgumentError("invalid airfoil step settings"))
    _fluxes!(U,w,g,m;outer_bc,face_flux)
    dt=min(Float64(dt_cap),cfl/maximum(w.rate));rejected=nothing
    for attempt in 0:max_retries
        valid=true
        for j in 1:g.NJ,i in 1:g.NI
            state=ntuple(k->U[i,j,k]+dt*w.residual[i,j,k],Val(5))
            good=pressure_state(state,m)[1]
            good || (rejected=(i,j,state))
            valid &= good
            for k in 1:5;w.trial[i,j,k]=state[k];end
        end
        if valid
            copyto!(U,w.trial)
            mass_rate=[w.boundary_ledger[k] for k in (1,4,5)]
            return (;dt,retries=attempt,outward_mass_rate=mass_rate,
                residual=max(maximum(abs,view(w.residual,:,:,1)),maximum(abs,view(w.residual,:,:,4)),maximum(abs,view(w.residual,:,:,5))))
        end
        dt*=.5
    end
    error("underwater airfoil step failed admissibility after $max_retries retries; unchanged state; rejected=$rejected")
end

function airfoil_loads(U,g,m)
    fx=fy=moment=hydro_x=hydro_y=0.
    qref=.5m.rho_l_ref*g.speed^2
    for i in 1:g.NI
        state=_face_state(_state(U,i,1),g.pc0[i,1],g.pj0[i,1],m);p=mixture_pressure(state,m)
        dp=p-g.pj0[i,1];dfx=-dp*g.sjx[i,1];dfy=-dp*g.sjy[i,1]
        fx+=dfx;fy+=dfy
        x=(g.X[i,1]+g.X[i+1,1])/2;y=(g.Y[i,1]+g.Y[i+1,1])/2
        moment+=(x-g.moment_x)*dfy-(y-g.moment_y)*dfx
        hydro_x-=g.pj0[i,1]*g.sjx[i,1];hydro_y-=g.pj0[i,1]*g.sjy[i,1]
    end
    denom=max(qref*g.chord,eps())
    (;cl=fy/denom,cd=fx/denom,cm=moment/(denom*g.chord),force_x=fx,force_y=fy,
      hydrostatic_force_x=hydro_x,hydrostatic_force_y=hydro_y)
end
material_mass(U,g)=[sum(view(U,:,:,k).*g.volume) for k in (1,4,5)]

include(joinpath(@__DIR__,"cmesh_free_surface_airfoil_initialization.jl"))

end
