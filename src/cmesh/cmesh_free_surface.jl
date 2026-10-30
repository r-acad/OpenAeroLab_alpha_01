# EXPERIMENTAL THREE-CONSTITUENT CORE. This module is isolated from the
# production four-variable Species solver. It supplies conservative state,
# pressure equilibrium and numerical flux contracts for a resolved air/water
# interface, including overturned interfaces. It is not yet an admitted
# underwater airfoil backend. Gravity balancing, transport and free-surface
# benchmarks must pass before that backend is exposed.
module CMeshFreeSurface

export MaterialModel, admissible, pressure_state, mixture_pressure, mixture_density
export primitive, conservative, phase_densities, physical_flux, hll_flux
export gravity_source, hydrostatic_pressure, hydrostatic_density, phase_volume_closure
export total_density
export STATE_FORMAT, STATE_ORDER
const STATE_FORMAT="CMeshFreeSurfacePartialMass/1"
const STATE_ORDER=("liquid_partial_density","x_momentum","y_momentum","vapor_partial_density","air_partial_density")

"SI material constants for weakly compressible water and separate isothermal gases."
struct MaterialModel{T<:AbstractFloat}
    rho_l_ref::T
    p_ref::T
    c_l2::T
    rt_air::T
    rt_vapor::T
    p_vapor_sat::T
end

function MaterialModel(;rho_l_ref::Real=998.2,p_ref::Real=101325.0,
    c_l::Real=1482.0,temperature::Real=293.15,r_air::Real=287.05,
    r_vapor::Real=461.5,p_vapor_sat::Real=2339.0)
    all(x->isfinite(x)&&x>0,(rho_l_ref,p_ref,c_l,temperature,r_air,r_vapor,p_vapor_sat)) ||
        throw(ArgumentError("free-surface material parameters must be finite and positive"))
    m=MaterialModel(Float64(rho_l_ref),Float64(p_ref),Float64(c_l)^2,
        Float64(r_air)*Float64(temperature),Float64(r_vapor)*Float64(temperature),Float64(p_vapor_sat))
    m.rho_l_ref-m.p_ref/m.c_l2>0 || throw(ArgumentError("water EOS has nonpositive density intercept"))
    m
end

@inline total_density(U)=U[1]+U[4]+U[5]
@inline admissible(U)=length(U)==5 && all(isfinite,U) && U[1]>=0 && U[4]>=0 && U[5]>=0 && total_density(U)>0

"Phase densities at common absolute pressure, in liquid/vapour/air order."
@inline function phase_densities(p::Real,m::MaterialModel)
    (liquid=m.rho_l_ref+(p-m.p_ref)/m.c_l2,
     vapor=p/m.rt_vapor,air=p/m.rt_air)
end

"""Positive root of sum(m_k/rho_k(p))=1, without mixing the two gases.

State order is (m_liquid,rho*u,rho*v,m_vapour,m_air), in SI units. All
three partial densities are stored independently; total density is their sum.
The quadratic uses the stable companion
root on the dilute-gas branch. A pure-water state may require nonpositive
absolute pressure; that state is refused, rather than being clipped.
"""
@inline function pressure_state(U,m::MaterialModel)
    admissible(U) || return (false,zero(m.p_ref),zero(m.p_ref))
    rho=total_density(U);ml=U[1]
    intercept=m.rho_l_ref-m.p_ref/m.c_l2
    A=inv(m.c_l2)
    gas=U[4]*m.rt_vapor+U[5]*m.rt_air
    B=intercept-ml-gas/m.c_l2
    C=-gas*intercept
    disc=B*B-4A*C
    isfinite(disc) && disc>=0 || return (false,zero(m.p_ref),zero(m.p_ref))
    root=sqrt(disc)
    p=if gas==0
        m.p_ref+(ml-m.rho_l_ref)*m.c_l2
    elseif B>=0
        -2C/(B+root)
    else
        (-B+root)/(2A)
    end
    isfinite(p) && p>0 || return (false,p,zero(m.p_ref))
    densities=phase_densities(p,m)
    all(x->isfinite(x)&&x>0,densities) || return (false,p,zero(m.p_ref))
    compliance=ml/(m.c_l2*densities.liquid^2)+
        U[4]/(m.rt_vapor*densities.vapor^2)+U[5]/(m.rt_air*densities.air^2)
    c2=inv(rho*compliance)
    isfinite(c2) && c2>0 || return (false,p,zero(m.p_ref))
    (true,p,sqrt(c2))
end

function mixture_pressure(U,m::MaterialModel=MaterialModel())
    good,p,_=pressure_state(U,m)
    good || throw(DomainError(U,"invalid three-constituent pressure-equilibrium state"))
    p
end

function mixture_density(p::Real,Yv::Real,Ya::Real,m::MaterialModel=MaterialModel())
    isfinite(p) && p>0 && isfinite(Yv) && isfinite(Ya) && Yv>=0 && Ya>=0 && Yv+Ya<=1 ||
        throw(DomainError((p,Yv,Ya),"invalid pressure or mass fractions"))
    r=phase_densities(p,m)
    inv((1-Yv-Ya)/r.liquid+Yv/r.vapor+Ya/r.air)
end

function mixture_density(p::Real,Y::NamedTuple,m::MaterialModel=MaterialModel())
    p>0 && all(isfinite,Y) && all(>=(0),Y) && abs(sum(Y)-1)<=1e-12 ||
        throw(DomainError((p,Y),"invalid pressure or three-constituent mass fractions"))
    r=phase_densities(p,m)
    inv(Y.liquid/r.liquid+Y.vapor/r.vapor+Y.air/r.air)
end

function primitive(U,m::MaterialModel=MaterialModel())
    good,p,c=pressure_state(U,m)
    good || throw(DomainError(U,"invalid three-constituent state"))
    rho=total_density(U);ml=U[1];r=phase_densities(p,m)
    (rho=rho,p=p,u=U[2]/rho,v=U[3]/rho,c=c,
     volume_fractions=(liquid=ml/r.liquid,vapor=U[4]/r.vapor,air=U[5]/r.air),
     mass_fractions=(liquid=ml/rho,vapor=U[4]/rho,air=U[5]/rho))
end

"Conserved state from pressure, velocities and current vapour/air volume fractions."
function conservative(p::Real,u::Real,v::Real,beta_vapor::Real,beta_air::Real,
                      m::MaterialModel=MaterialModel())
    all(isfinite,(p,u,v,beta_vapor,beta_air)) && p>0 && beta_vapor>=0 && beta_air>=0 && beta_vapor+beta_air<=1 ||
        throw(DomainError((p,u,v,beta_vapor,beta_air),"invalid pressure, velocities or phase volumes"))
    r=phase_densities(p,m)
    ml=(1-beta_vapor-beta_air)*r.liquid;mv=beta_vapor*r.vapor;ma=beta_air*r.air
    rho=ml+mv+ma
    (ml,rho*u,rho*v,mv,ma)
end

phase_volume_closure(U,m::MaterialModel=MaterialModel())=sum(primitive(U,m).volume_fractions)

"Area-normalised conservative flux, with a unit face normal."
@inline function physical_flux(U,nx::Real,ny::Real,m::MaterialModel=MaterialModel())
    q=primitive(U,m);un=q.u*nx+q.v*ny
    (U[1]*un,U[2]*un+q.p*nx,U[3]*un+q.p*ny,U[4]*un,U[5]*un)
end

"Conservative HLLE reference flux; transports both gases without phase clipping."
function hll_flux(UL,UR,nx::Real,ny::Real,m::MaterialModel=MaterialModel())
    L=primitive(UL,m);R=primitive(UR,m)
    unL=L.u*nx+L.v*ny;unR=R.u*nx+R.v*ny
    sl=min(unL-L.c,unR-R.c);sr=max(unL+L.c,unR+R.c)
    FL=physical_flux(UL,nx,ny,m);FR=physical_flux(UR,nx,ny,m)
    sl>=0 && return FL
    sr<=0 && return FR
    ntuple(k->(sr*FL[k]-sl*FR[k]+sl*sr*(UR[k]-UL[k]))/(sr-sl),Val(5))
end

@inline gravity_source(U,m::MaterialModel=MaterialModel();g=(0.0,-9.80665))=
    (zero(U[1]),total_density(U)*g[1],total_density(U)*g[2],zero(U[1]),zero(U[1]))

"Exact hydrostatic pressure of a pure constituent above/below a reference point."
function hydrostatic_pressure(z::Real,p0::Real,z0::Real,phase::Symbol,m::MaterialModel=MaterialModel();gravity::Real=9.80665)
    isfinite(z) && isfinite(z0) && isfinite(p0) && p0>0 && isfinite(gravity) && gravity>=0 ||
        throw(DomainError((z,p0,z0,gravity),"invalid hydrostatic reference"))
    dz=z-z0
    p=if phase===:water || phase===:liquid
        rho0=phase_densities(p0,m).liquid
        p0+rho0*m.c_l2*expm1(-gravity*dz/m.c_l2)
    elseif phase===:air
        p0*exp(-gravity*dz/m.rt_air)
    elseif phase===:vapor
        p0*exp(-gravity*dz/m.rt_vapor)
    else
        throw(ArgumentError("phase must be water, liquid, air or vapor"))
    end
    p>0 || throw(DomainError(p,"hydrostatic column reaches nonpositive absolute pressure"))
    p
end

function hydrostatic_density(z::Real,p0::Real,z0::Real,phase::Symbol,m::MaterialModel=MaterialModel();gravity::Real=9.80665)
    p=hydrostatic_pressure(z,p0,z0,phase,m;gravity)
    r=phase_densities(p,m)
    phase in (:water,:liquid) ? r.liquid : phase===:air ? r.air : r.vapor
end

isfile(joinpath(@__DIR__,"cmesh_free_surface_hllc.jl")) && include(joinpath(@__DIR__,"cmesh_free_surface_hllc.jl"))
include(joinpath(@__DIR__,"cmesh_free_surface_ausm.jl"))
isfile(joinpath(@__DIR__,"cmesh_free_surface_hydrostatic.jl")) && include(joinpath(@__DIR__,"cmesh_free_surface_hydrostatic.jl"))
include(joinpath(@__DIR__,"cmesh_free_surface_cartesian.jl"))
include(joinpath(@__DIR__,"cmesh_free_surface_gate.jl"))
# Geometry previews can load the scalar material model from the lightweight
# web-launcher environment. Device operators remain available in the dedicated
# CFD environment, where KernelAbstractions is an explicit dependency.
if Base.find_package("KernelAbstractions")!==nothing
    include(joinpath(@__DIR__,"cmesh_free_surface_ka.jl"))
end

end # module
