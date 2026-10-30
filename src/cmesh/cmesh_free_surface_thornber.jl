# Optional experimental callback: manually include inside CMeshFreeSurface.
# No production route and no replacement of the default HLLC flux.
export thornber_hllc_flux

"""Thornber low-Mach velocity-jump reconstruction followed by physical HLLC.

For each Cartesian velocity component, uL'=(uL+uR)/2+z*(uL-uR)/2,
uR'=(uL+uR)/2-z*(uL-uR)/2, with z=min(max(|uL|/cL,|uR|/cR),1).
All three partial masses and the EOS pressure/sound speed stay unchanged.
The acoustic CFL remains based on the original physical sound speed.

Method source: MFC official equations §17, Thornber correction (low_Mach=2),
https://mflowcode.github.io/documentation/equations.html, accessed 2026-10-19.
This first-order barotropic extension is experimental. Contact/rest checks do
not qualify nonlinear breaking waves, shocks, or an underwater airfoil.
"""
function thornber_hllc_flux(left,right,nx,ny,model)
    qL=primitive(left,model);qR=primitive(right,model)
    z=min(max(hypot(qL.u,qL.v)/qL.c,hypot(qR.u,qR.v)/qR.c),1.)
    # Equal states and high-Mach faces take the original arithmetic exactly.
    (z==1 || (qL.u==qR.u && qL.v==qR.v)) &&
        return hllc_flux(left,right,nx,ny,model)
    um=.5(qL.u+qR.u);vm=.5(qL.v+qR.v)
    du=.5z*(qL.u-qR.u);dv=.5z*(qL.v-qR.v)
    L=(left[1],qL.rho*(um+du),qL.rho*(vm+dv),left[4],left[5])
    R=(right[1],qR.rho*(um-du),qR.rho*(vm-dv),right[4],right[5])
    hllc_flux(L,R,nx,ny,model)
end
