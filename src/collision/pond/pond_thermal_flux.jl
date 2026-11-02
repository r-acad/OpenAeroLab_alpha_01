"""Analytic ideal-gas kinetic flux splitting for the Cartesian leaf-cell solver.

The half-space Maxwellian moments are evaluated directly, without a truncated
discrete-velocity equilibrium or population clipping. This is a KFVS flux,
not the historical PonD gauge/distribution evolution or a DUGKS collision flux.
Primitive states are `(rho,u,v,w,p)`; conserved states are `(rho,mx,my,mz,E)`.
Caller owns reconstruction, admissibility checks, and conservative time updates.

Method: Mandal & Deshpande, Computers & Fluids 23 (1994), 447-478,
https://doi.org/10.1016/0045-7930(94)90050-7 .
"""
module PonDThermalFlux

using SpecialFunctions: erfc
export kinetic_flux, euler_flux, primitive_to_conserved, conserved_to_primitive,
       admissible, max_signal

@inline function primitive_to_conserved(q::NTuple{5,T},gamma::T) where {T}
    rho,u,v,w,p=q
    (rho,rho*u,rho*v,rho*w,p/(gamma-one(T))+rho*(u*u+v*v+w*w)/T(2))
end

@inline function conserved_to_primitive(U::NTuple{5,T},gamma::T) where {T}
    rho,mx,my,mz,E=U
    u=mx/rho;v=my/rho;w=mz/rho
    (rho,u,v,w,(gamma-one(T))*(E-(mx*u+my*v+mz*w)/T(2)))
end

@inline function admissible(U::NTuple{5,T},gamma::T,rho_floor::T,p_floor::T) where {T}
    rho,mx,my,mz,E=U
    finite=isfinite(rho)&&isfinite(mx)&&isfinite(my)&&isfinite(mz)&&isfinite(E)
    finite&&rho>rho_floor&&
        (gamma-one(T))*(E-(mx*mx+my*my+mz*mz)/(T(2)*rho))>p_floor
end

@inline max_signal(q::NTuple{5,T},axis::Integer,gamma::T) where {T} =
    abs(q[axis+1])+sqrt(gamma*q[5]/q[1])

# Positive half moments for a normal velocity with mean -h <= 0 and variance
# theta. Direct formulas cancel in a far tail. There, ratios of consecutive
# positive integrals K_n(a)=integral_0^Inf t^n exp(-a*t-t^2/2)dt obey
# K_n/K_(n-1)=n/(a+K_(n+1)/K_n). Backward recurrence avoids that cancellation.
@inline function _tail_moments(h::T,theta::T) where {T}
    sigma=sqrt(theta);a=h/sigma
    A=erfc(a/sqrt(T(2)))/T(2)
    if a>T(2)
        r=zero(T);r1=zero(T);r2=zero(T);r3=zero(T)
        # 96 levels resolves the a=2 transition in Float64. Float32 needs fewer.
        depth=T===Float32 ? 40 : 96
        for k in depth:-1:1
            r=T(k)/(a+r)
            if k==3;r3=r;elseif k==2;r2=r;elseif k==1;r1=r;end
        end
        m1=sigma*A*r1
        m2=sigma*m1*r2
        m3=sigma*m2*r3
        return m1,m2,m3
    end
    B=sigma*exp(-a*a/T(2))/sqrt(T(2)*T(pi))
    (B-h*A,(h*h+theta)*A-h*B,(h*h+T(2)*theta)*B-(h*h*h+T(3)*h*theta)*A)
end

@inline function _half_moments(u::T,theta::T,positive::Bool) where {T}
    t1,t2,t3=_tail_moments(abs(u),theta)
    if positive
        u>=zero(T) ? (u+t1,u*u+theta-t2,u*(u*u+T(3)*theta)+t3) : (t1,t2,t3)
    else
        u>=zero(T) ? (-t1,t2,-t3) : (u-t1,u*u+theta-t2,u*(u*u+T(3)*theta)-t3)
    end
end

@inline function _half_flux(q::NTuple{5,T},axis::Integer,sign::Integer,gamma::T,positive::Bool) where {T}
    rho,u,v,w,p=q
    un=T(sign)*q[axis+1];theta=p/rho
    i1,i2,i3=_half_moments(un,theta,positive)
    nx=axis==1 ? T(sign) : zero(T)
    ny=axis==2 ? T(sign) : zero(T)
    nz=axis==3 ? T(sign) : zero(T)
    ut=u-un*nx;vt=v-un*ny;wt=w-un*nz
    mass=rho*i1
    internal_dof=T(2)/(gamma-one(T))-one(T)
    (mass,rho*(ut*i1+nx*i2),rho*(vt*i1+ny*i2),rho*(wt*i1+nz*i2),
        rho/T(2)*(i3+(ut*ut+vt*vt+wt*wt+internal_dof*theta)*i1))
end

"""KFVS flux oriented from left to right along `sign*e_axis`, per unit area.

Positive finite primitive states and 1 < gamma <= 5/3 are required. The
numerical kernel deliberately does not clip states: callers reject inadmissible
reconstructions/updates. Evaluate each shared (sub)face once for conservation.
"""
@inline function kinetic_flux(left::NTuple{5,T},right::NTuple{5,T},axis::Integer,
                              sign::Integer,gamma::T) where {T}
    plus=_half_flux(left,axis,sign,gamma,true)
    minus=_half_flux(right,axis,sign,gamma,false)
    ntuple(k->plus[k]+minus[k],Val(5))
end
@inline kinetic_flux(left::NTuple{5,T},right::NTuple{5,T},axis::Integer,sign::Integer=1) where {T} =
    kinetic_flux(left,right,axis,sign,T(1.4))

@inline function euler_flux(q::NTuple{5,T},axis::Integer,sign::Integer,gamma::T) where {T}
    rho,u,v,w,p=q;un=T(sign)*q[axis+1]
    E=primitive_to_conserved(q,gamma)[5]
    (rho*un,rho*u*un+(axis==1 ? T(sign)*p : zero(T)),
        rho*v*un+(axis==2 ? T(sign)*p : zero(T)),
        rho*w*un+(axis==3 ? T(sign)*p : zero(T)),(E+p)*un)
end

end
