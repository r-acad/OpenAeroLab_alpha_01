# Included inside CMeshFreeSurface. Experimental equilibrium reconstruction;
# production curvilinear free-surface integration is a separate admission gate.
export mixture_enthalpy, hydrostatic_reconstruct, hydrostatic_face_flux
export hydrostatic_cell_source

"Specific barotropic enthalpy at fixed constituent mass fractions, SI m²/s²."
function mixture_enthalpy(p::Real,Yv::Real,Ya::Real,m::MaterialModel=MaterialModel();Yl=1-Yv-Ya)
    isfinite(p) && p>0 && Yv>=0 && Ya>=0 && Yl>=0 && abs(Yl+Yv+Ya-1)<=1e-12 ||
        throw(DomainError((p,Yv,Ya),"invalid mixture hydrostatic enthalpy state"))
    water=Yl*m.c_l2*log1p((p-m.p_ref)/(m.c_l2*m.rho_l_ref))
    gas=(Yv*m.rt_vapor+Ya*m.rt_air)*log(p/m.p_ref)
    water+gas
end

"""Continue a cell state to a face along its frozen-composition equilibrium.

Integrates dp/dz=-rho*g through the exact fixed-composition barotropic
enthalpy. The safeguarded pressure inversion rejects a column reaching
nonpositive absolute pressure. It never clips partial masses or pressure.
"""
function hydrostatic_reconstruct(U,zcell::Real,zface::Real,m::MaterialModel=MaterialModel();gravity::Real=9.80665)
    all(isfinite,(zcell,zface,gravity)) && gravity>=0 ||
        throw(ArgumentError("invalid free-surface hydrostatic coordinates/gravity"))
    q=primitive(U,m);Yv=q.mass_fractions.vapor;Ya=q.mass_fractions.air;Yl=q.mass_fractions.liquid
    enthalpy(p)=mixture_enthalpy(p,Yv,Ya,m;Yl)
    target=enthalpy(q.p)-gravity*(zface-zcell)
    lo=max(floatmin(Float64),q.p*1e-15);hi=q.p
    enthalpy(lo)<=target ||
        throw(DomainError(target,"hydrostatic reconstruction reaches nonpositive absolute pressure"))
    for _ in 1:100
        enthalpy(hi)>=target && break
        hi*=2
        isfinite(hi) || throw(DomainError(hi,"hydrostatic pressure bracket overflow"))
    end
    enthalpy(hi)>=target || error("hydrostatic pressure bracket failed")
    pressure=clamp(q.p,lo,hi)
    for _ in 1:100
        residual=enthalpy(pressure)-target
        abs(residual)<=2e-10*max(1.,abs(target)) && break
        if residual>0;hi=pressure;else;lo=pressure;end
        proposed=pressure-residual*mixture_density(pressure,q.mass_fractions,m)
        pressure=isfinite(proposed) && lo<proposed<hi ? proposed : .5(lo+hi)
    end
    abs(enthalpy(pressure)-target)<=1e-8*max(1.,abs(target)) ||
        error("hydrostatic pressure inversion did not converge")
    rho=mixture_density(pressure,q.mass_fractions,m)
    (rho*Yl,rho*q.u,rho*q.v,rho*Yv,rho*Ya)
end

"Hydrostatic reconstruction followed by a contact-preserving unit-normal flux."
function hydrostatic_face_flux(left,right,zleft,zright,zface,nx,ny,m::MaterialModel=MaterialModel();gravity=9.80665)
    L=hydrostatic_reconstruct(left,zleft,zface,m;gravity)
    R=hydrostatic_reconstruct(right,zright,zface,m;gravity)
    hllc_flux(L,R,nx,ny,m)
end

"Integrated vertical gravity quadrature consistent with a cell's reconstruction."
function hydrostatic_cell_source(U,zcell,zbottom,ztop,m::MaterialModel=MaterialModel();gravity=9.80665)
    zbottom<ztop || throw(ArgumentError("cell height must be positive"))
    lower=mixture_pressure(hydrostatic_reconstruct(U,zcell,zbottom,m;gravity),m)
    upper=mixture_pressure(hydrostatic_reconstruct(U,zcell,ztop,m;gravity),m)
    (0.,0.,(upper-lower)/(ztop-zbottom),0.,0.)
end
