# Pseudo-transient globalization uses G(U)=R(U)+D*(U-U0), with the same
# frozen D=V/dtau used by JFNK. It does not change R or its stopping targets.
# See ADflow's ANK physicality/unsteady-residual line search and Yildirim2019.

@inline function _airfoil_physical_fraction(r,u,v,t,dr,de,eint,fraction)
    energy=r*(eint*t+0.5f0*(u*u+v*v))
    min(1.0f0,fraction*r/max(abs(dr),1.0f-30),
        fraction*energy/max(abs(de),1.0f-30))
end

function _airfoil_physical_alpha(s::KSol,w::JFNKAirfoilWorkspace,fraction::Float32)
    isfinite(fraction) && 0.0f0<fraction<=0.5f0 ||
        throw(ArgumentError("physical_fraction must be in (0,0.5]"))
    line=w.line;g=s.g
    if _implicit_cpu_strided(s.rho)
        alpha=1.0f0
        @inbounds for j in 1:s.NJ,i in 1:s.NI
            ia=i+g;ja=j+g
            alpha=min(alpha,_airfoil_physical_fraction(line.rho0[ia,ja],
                line.ux0[ia,ja],line.uy0[ia,ja],line.T0[ia,ja],
                w.delta[1][i,j],w.delta[4][i,j],s.eint,fraction))
        end
        return alpha
    end
    ir=g+1:g+s.NI;jr=g+1:g+s.NJ
    eint=s.eint
    mapreduce((r,u,v,t,dr,de)->_airfoil_physical_fraction(r,u,v,t,dr,de,eint,fraction),
        min,view(line.rho0,ir,jr),view(line.ux0,ir,jr),view(line.uy0,ir,jr),
        view(line.T0,ir,jr),w.delta[1],w.delta[4])
end

@inline function _airfoil_pseudo_component(r,u,v,t,r0,u0,v0,t0,
    f1,f2,f3,f4,d,vol,eint,inv_r,inv_m,inv_e)
    e=r*(eint*t+0.5f0*(u*u+v*v))
    e0=r0*(eint*t0+0.5f0*(u0*u0+v0*v0))
    a=Float64((f1+d*(r-r0))/vol*inv_r)
    b=Float64((f2+d*(r*u-r0*u0))/vol*inv_m)
    c=Float64((f3+d*(r*v-r0*v0))/vol*inv_m)
    z=Float64((f4+d*(e-e0))/vol*inv_e)
    a*a+b*b+c*c+z*z
end

function _airfoil_pseudo_merit(s::KSol,w::JFNKAirfoilWorkspace)
    r=max(Float64(s.rho_inf),1e-12)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u)
    F=eltype(s.rho);inv_r=F(1/r);inv_m=F(1/(r*u));inv_e=F(1/(r*e))
    line=w.line;g=s.g;eint=s.eint
    if _implicit_cpu_strided(s.rho)
        total=0.0
        @inbounds for j in 1:s.NJ,i in 1:s.NI
            ia=i+g;ja=j+g
            total+=_airfoil_pseudo_component(s.rho[ia,ja],s.ux[ia,ja],s.uy[ia,ja],s.T[ia,ja],
                line.rho0[ia,ja],line.ux0[ia,ja],line.uy0[ia,ja],line.T0[ia,ja],
                s.dc_r[i,j],s.dc_mx[i,j],s.dc_my[i,j],s.dc_e[i,j],line.d0[i,j],
                s.vol[i,j],eint,inv_r,inv_m,inv_e)
        end
        return sqrt(total/length(s.vol))
    end
    ir=g+1:g+s.NI;jr=g+1:g+s.NJ
    total=mapreduce((r,u,v,t,r0,u0,v0,t0,a,b,c,z,d,vol)->
        _airfoil_pseudo_component(r,u,v,t,r0,u0,v0,t0,a,b,c,z,d,vol,eint,inv_r,inv_m,inv_e),
        +,view(s.rho,ir,jr),view(s.ux,ir,jr),view(s.uy,ir,jr),view(s.T,ir,jr),
        view(line.rho0,ir,jr),view(line.ux0,ir,jr),view(line.uy0,ir,jr),view(line.T0,ir,jr),
        s.dc_r,s.dc_mx,s.dc_my,s.dc_e,line.d0,s.vol)
    sqrt(Float64(total)/length(s.vol))
end
