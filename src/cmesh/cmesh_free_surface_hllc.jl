# Included inside CMeshFreeSurface. Contact-preserving inviscid flux for its
# five conservative variables; this does not enable a production free surface.
# The public face normal is a unit vector; the caller applies the face length.
export hllc_flux

@inline function _free_surface_star(U,un,S,contact,nx,ny)
    factor=(S-un)/(S-contact)
    density=total_density(U);rho=density*factor
    (U[1]*factor,(U[2]/density+(contact-un)*nx)*rho,
          (U[3]/density+(contact-un)*ny)*rho,U[4]*factor,U[5]*factor)
end

"""Contact-preserving HLLC flux with admissibility fallback to HLL.

Wave estimates use the extremal acoustic speeds. Every material partial mass
uses the same star-state factor as total mass, retaining nonnegative fractions
when the star state is admissible. A singular or inadmissible star selects the
HLL flux; fractions are never clipped. Positivity of a complete multidimensional
step still requires a suitable timestep and a checked/retried state update.
"""
function hllc_flux(left::NTuple{5,<:Real},right::NTuple{5,<:Real},nx::Real,ny::Real,model)
    isfinite(nx) && isfinite(ny) && abs(hypot(nx,ny)-1)<=1e-10 ||
        throw(ArgumentError("free-surface HLLC requires a finite unit normal"))
    qL=primitive(left,model);qR=primitive(right,model)
    fL=physical_flux(left,nx,ny,model);fR=physical_flux(right,nx,ny,model)
    unL=qL.u*nx+qL.v*ny;unR=qR.u*nx+qR.v*ny
    SL=min(unL-qL.c,unR-qR.c);SR=max(unL+qL.c,unR+qR.c)
    SL>=0 && return fL
    SR<=0 && return fR
    denominator=qL.rho*(SL-unL)-qR.rho*(SR-unR)
    if !isfinite(denominator) || abs(denominator)<=eps(Float64)*max(1.0,abs(qL.rho*(SL-unL)),abs(qR.rho*(SR-unR)))
        return hll_flux(left,right,nx,ny,model)
    end
    contact=(qR.p-qL.p+qL.rho*unL*(SL-unL)-qR.rho*unR*(SR-unR))/denominator
    if !isfinite(contact) || !(SL<contact<SR)
        return hll_flux(left,right,nx,ny,model)
    end
    starL=_free_surface_star(left,unL,SL,contact,nx,ny)
    starR=_free_surface_star(right,unR,SR,contact,nx,ny)
    if !pressure_state(starL,model)[1] || !pressure_state(starR,model)[1]
        return hll_flux(left,right,nx,ny,model)
    end
    # Partial masses must use the same mass flux. Independently subtracting
    # their star fluxes can leave a tiny negative scalar flux at a pure phase
    # after many equilibrium steps (and a smaller dt cannot repair its sign).
    # This is the HLLC contact's passive-scalar upwind law, not phase clipping.
    if contact>=0
        fx=fL[2]+SL*(starL[2]-left[2]);fy=fL[3]+SL*(starL[3]-left[3])
        (starL[1]*contact,fx,fy,starL[4]*contact,starL[5]*contact)
    else
        fx=fR[2]+SR*(starR[2]-right[2]);fy=fR[3]+SR*(starR[3]-right[3])
        (starR[1]*contact,fx,fy,starR[4]*contact,starR[5]*contact)
    end
end
