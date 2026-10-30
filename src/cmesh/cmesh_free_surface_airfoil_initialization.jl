# Method-of-fundamental-solutions potential initialization. This establishes
# an impermeable initial velocity, not a viscous/stall/polar surrogate.
using LinearAlgebra
export source_panel_initial_state

function source_panel_initial_state(g::AirfoilGrid,m::MaterialModel)
    n=g.NI;ns=2n;px=zeros(n);py=zeros(n);nx=zeros(n);ny=zeros(n);sx=zeros(ns);sy=zeros(ns)
    A=zeros(n+1,ns);rhs=zeros(n+1)
    for i in 1:n
        px[i]=(g.X[i,1]+g.X[i+1,1])/2;py[i]=(g.Y[i,1]+g.Y[i+1,1])/2
        ds=hypot(g.sjx[i,1],g.sjy[i,1]);nx[i]=g.sjx[i,1]/ds;ny[i]=g.sjy[i,1]/ds
        # All fundamental sources are strictly inside the solid contour.
        sx[i]=px[i]-.35ds*nx[i];sy[i]=py[i]-.35ds*ny[i]
        sx[n+i]=px[i]-.8ds*nx[i];sy[n+i]=py[i]-.8ds*ny[i]
        rhs[i]=-g.speed*nx[i]
    end
    for i in 1:n,k in 1:ns
        dx,dy=px[i]-sx[k],py[i]-sy[k];r2=dx*dx+dy*dy
        A[i,k]=(dx*nx[i]+dy*ny[i])/(2pi*r2)
    end
    # Two interior layers give enough freedom to satisfy every collocation
    # impermeability equation AND zero net source as exact linear constraints.
    A[n+1,:].=1/g.chord
    strengths=A\rhs
    residual=A[1:n,:]*strengths-rhs[1:n]
    velocity(x,y)=begin
        u=g.speed;v=0.
        for k in 1:ns
            dx,dy=x-sx[k],y-sy[k];r2=dx*dx+dy*dy
            u+=strengths[k]*dx/(2pi*r2);v+=strengths[k]*dy/(2pi*r2)
        end
        u,v
    end
    U=zeros(g.NI,g.NJ,5);minimum_pressure=Inf;peak_speed=0.
    for j in 1:g.NJ,i in 1:g.NI
        u,v=velocity(g.xc[i,j],g.yc[i,j]);water=g.beta_water0[i,j]
        rho=g.rho0[i,j];p=g.pc0[i,j]+.5rho*(g.speed^2-u*u-v*v)
        p>0 || error("potential initialization requests nonpositive pressure; use a gradual physical startup")
        state=conservative(p,u,v,water*g.vapor_seed,1-water,m)
        for k in 1:5;U[i,j,k]=state[k];end
        minimum_pressure=min(minimum_pressure,p);peak_speed=max(peak_speed,hypot(u,v))
    end
    receipt=Dict{String,Any}("method"=>"interior fundamental-source potential initialization with hydrostatic plus Bernoulli pressure",
        "wall_collocation_normal_velocity_max_m_s"=>maximum(abs,residual),
        "source_strength_sum_m2_s"=>sum(strengths),"minimum_pressure_pa"=>minimum_pressure,"peak_speed_m_s"=>peak_speed,
        "circulation"=>"not prescribed; no Kutta, viscous, wave or stall reference supplied by initialization")
    U,receipt
end
