# Compact, flux-consistent four-variable preconditioner for the steady A+/C
# airfoil. The exact nonlinear residual and matrix-free Jv remain unchanged.
# The baseline opt-in model is first-order KFVS with thin-layer viscosity.
# Additional opt-in variants differentiate center-cell WENO face dependence
# and local full-stress viscosity while keeping the production residual/Jv.

const _AIRFOIL_KI4 = SMatrix{4,4,Float32,16}(
    1f0,0f0,0f0,0f0, 0f0,1f0,0f0,0f0,
    0f0,0f0,1f0,0f0, 0f0,0f0,0f0,1f0)
const _AIRFOIL_KV4 = SMatrix{4,4,Float32,16}(
    0f0,0f0,0f0,0f0, 0f0,1f0,0f0,0f0,
    0f0,0f0,1f0,0f0, 0f0,0f0,0f0,1f0)

struct AirfoilKineticBlockWorkspace{A,V,C}
    face_i_left::A
    face_i_right::A
    face_j_left::A
    face_j_right::A
    lower_i::A
    upper_i::A
    lower_j::A
    upper_j::A
    diagonal::A
    cp::A
    row_out::V
    row_scaled::V
    block_length::Int32
    blocks_per_row::Int32
    fullvisc_block::Bool
    weno_block::Bool
    muscl_block::Bool
    muscl_theta::C
    unfloored::Bool
    pseudo_scale
end

function build_airfoil_kinetic_block_workspace(s::KSol;block_length::Int=64,
                                                fullvisc_block::Bool=false,
                                                weno_block::Bool=false,
                                                muscl_block::Bool=false,
                                                muscl_ls_block::Bool=false,
                                                unfloored::Bool=false,
                                                pseudo_scale=1f0,
                                                block_scalar_type::Type{<:AbstractFloat}=Float32)
    block_scalar_type in (Float32,Float64)||throw(ArgumentError("Block preconditioner precision must be Float32 or Float64"))
    !muscl_ls_block||muscl_block||throw(ArgumentError("MUSCL LS block requires MUSCL face states"))
    if muscl_block
        s.recon==_AIRFOIL_MUSCL_RECON&&haskey(_AIRFOIL_MUSCL,s)||
            throw(ArgumentError("MUSCL face block requires the configured original-node reconstruction"))
        !weno_block||throw(ArgumentError("MUSCL and indexed WENO block models are separate"))
    end
    1<=block_length<=s.NI || throw(ArgumentError("invalid kinetic block row length"))
    0f0<pseudo_scale<=1f0 || throw(ArgumentError("invalid block pseudo-time scale"))
    b4(ni,nj)=similar(s.dtf,block_scalar_type,4,4,ni,nj)
    v4()=ntuple(_->similar(s.dtf),4)
    AirfoilKineticBlockWorkspace(
        b4(s.NI+1,s.NJ),b4(s.NI+1,s.NJ),
        b4(s.NI,s.NJ+1),b4(s.NI,s.NJ+1),
        b4(s.NI,s.NJ),b4(s.NI,s.NJ),
        b4(s.NI,s.NJ),b4(s.NI,s.NJ),
        b4(s.NI,s.NJ),b4(s.NI,s.NJ),
        v4(),v4(),Int32(block_length),Int32(cld(s.NI,block_length)),
        fullvisc_block,weno_block,muscl_block,
        muscl_ls_block ? similar(s.dtf,eltype(s.rho),s.NI,s.NJ,4) : nothing,
        unfloored,pseudo_scale)
end

# The scalar j-line method limited pseudo-time by an explicit streamwise
# viscous clock. Both directions are in the block factorization, so this
# optional path uses only the convective/acoustic clock for V/dtau.
@kernel function _k_airfoil_kinetic_unfloored_d0!(d0,
    @Const(ux),@Const(uy),@Const(T),@Const(Six),@Const(Siy),
    cfl,pseudo_scale,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;u=ux[ia,ja];v=uy[ia,ja]
        c=3f0*sqrt(max(T[ia,ja],1f-3))
        siw=hypot(Six[i,j],Siy[i,j]);sie=hypot(Six[i+1,j],Siy[i+1,j])
        li=0.5f0*(abs(u*Six[i,j]+v*Siy[i,j])+
                  abs(u*Six[i+1,j]+v*Siy[i+1,j])+c*(siw+sie))
        d0[i,j]=max(li/cfl*pseudo_scale,1f-30)
    end
end

# Differentiate the compact correction of the production full-stress face
# flux with frozen Green-Gauss gradients and transport coefficients. Both
# momentum cross terms and the temperature conduction are kept. The full
# residual/Jv still differentiates every gradient and material property.
@inline function _airfoil_primitive_jacobian(q,eint)
    B=eltype(q)
    r,u,v,_=_airfoil_primitive4(q,eint)
    ir=1f0/r
    h=ir/eint
    hcat(SVector{3,B}(-u*ir,-v*ir,(-q[4]*ir+u*u+v*v)*h),
         SVector{3,B}(ir,0f0,-u*h),
         SVector{3,B}(0f0,ir,-v*h),
         SVector{3,B}(0f0,0f0,h))
end

@inline function _airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,
    dxc,dyc,nxr,nyr,mu,ke,eint)
    B=eltype(ql)
    _,ul,vl,tl=_airfoil_primitive4(ql,eint)
    _,ur,vr,tr=_airfoil_primitive4(qr,eint)
    area=hypot(nxr,nyr);nx=nxr/area;ny=nyr/area
    fvx,fvy,_=_viscous_face_full(gl...,gr...,
        ul,vl,tl,ur,vr,tr,dxc,dyc,nx,ny,area,mu,ke)
    invd=1f0/(dxc*dxc+dyc*dyc+1f-30)
    fac=mu*area*invd
    axu=fac*(1.3333334f0*dxc*nx+dyc*ny)
    ayu=fac*(dyc*nx-0.6666667f0*dxc*ny)
    axv=fac*(-0.6666667f0*dyc*nx+dxc*ny)
    ayv=fac*(dxc*nx+1.3333334f0*dyc*ny)
    at=ke*area*invd*(dxc*nx+dyc*ny)
    uf=0.5f0*(ul+ur);vf=0.5f0*(vl+vr)
    left=hcat(
        SVector{4,B}(0f0, axu, ayu,-0.5f0*fvx+uf*axu+vf*ayu),
        SVector{4,B}(0f0, axv, ayv,-0.5f0*fvy+uf*axv+vf*ayv),
        SVector{4,B}(0f0, 0f0, 0f0, at))
    right=hcat(
        SVector{4,B}(0f0,-axu,-ayu,-0.5f0*fvx-uf*axu-vf*ayu),
        SVector{4,B}(0f0,-axv,-ayv,-0.5f0*fvy-uf*axv-vf*ayv),
        SVector{4,B}(0f0, 0f0, 0f0,-at))
    left*_airfoil_primitive_jacobian(ql,eint),
    right*_airfoil_primitive_jacobian(qr,eint)
end

@inline function _airfoil_face_transport(rhol,rhor,tl,tr,nutl,nutr,
    nu_lam,rho_inf,suth,air_sutherland,sst_on,drot,::Type{B}=Float32) where {B<:AbstractFloat}
    rf=0.5f0*(rhol+rhor)
    nt=sst_on==Int32(1) ? 0.5f0*(nutl+nutr) : 0f0
    fmu=_mufac(0.5f0*(tl+tr),suth)
    cp=0.5f0*(5f0+drot)*_gas_cs2(B)
    if air_sutherland!=Int32(0)
        mu=rho_inf*nu_lam*fmu+rf*nt
        ke=cp*(rho_inf*nu_lam*fmu/0.71f0+rf*nt/0.9f0)
    else
        mu=rf*(nu_lam*fmu+nt)
        ke=rf*cp*(nu_lam*fmu/0.71f0+nt/0.9f0)
    end
    mu,ke
end

@inline function _airfoil_conserved4(r,u,v,t,eint,::Type{B}=Float32) where {B<:AbstractFloat}
    SVector{4,B}(r,r*u,r*v,r*(eint*t+0.5f0*(u*u+v*v)))
end

@inline function _airfoil_primitive4(q,eint)
    r=q[1];u=q[2]/r;v=q[3]/r
    t=(q[4]/r-0.5f0*(u*u+v*v))/eint
    (r,u,v,t)
end

# Repeating each side of the six-state stencil exactly suppresses the
# reconstruction slopes. This preserves the production KFVS quadrature,
# including its moving face frame, instead of substituting Euler/Roe fluxes.
@inline function _airfoil_kinetic_flux2(ql,qr,nxr,nyr,
                                       cxa,cya,cza,wa,drot,eint)
    rl,ul,vl,tl=_airfoil_primitive4(ql,eint)
    rr,ur,vr,tr=_airfoil_primitive4(qr,eint)
    area=hypot(nxr,nyr);nx=nxr/area;ny=nyr/area
    fr,fmx,fmy,fe=_fvflux_normal6(
        rl,ul,vl,tl, rl,ul,vl,tl, rl,ul,vl,tl,
        rr,ur,vr,tr, rr,ur,vr,tr, rr,ur,vr,tr,
        nx,ny,area,cxa,cya,cza,wa,drot,Int32(0))
    SVector{4,eltype(ql)}(fr,fmx,fmy,fe)
end

@inline function _airfoil_kinetic_flux6(q1,q2,q3,q4,q5,q6,nxr,nyr,
                                        cxa,cya,cza,wa,drot,eint)
    r1,u1,v1,t1=_airfoil_primitive4(q1,eint)
    r2,u2,v2,t2=_airfoil_primitive4(q2,eint)
    r3,u3,v3,t3=_airfoil_primitive4(q3,eint)
    r4,u4,v4,t4=_airfoil_primitive4(q4,eint)
    r5,u5,v5,t5=_airfoil_primitive4(q5,eint)
    r6,u6,v6,t6=_airfoil_primitive4(q6,eint)
    area=hypot(nxr,nyr);nx=nxr/area;ny=nyr/area
    fr,fmx,fmy,fe=_fvflux_normal6(
        r1,u1,v1,t1,r2,u2,v2,t2,r3,u3,v3,t3,
        r4,u4,v4,t4,r5,u5,v5,t5,r6,u6,v6,t6,
        nx,ny,area,cxa,cya,cza,wa,drot,Int32(1))
    SVector{4,eltype(q1)}(fr,fmx,fmy,fe)
end

# Center-cell blocks of the actual six-cell WENO5-Z kinetic face. Outer
# neighbor derivatives are omitted from the line factors, but their current
# values determine the reconstruction and nonlinear weights.
@inline function _airfoil_weno_face_jacobians(q1,q2,ql,qr,q5,q6,nxr,nyr,
                                               cxa,cya,cza,wa,drot,eint)
    lcols=ntuple(Val(4)) do k
        h=_airfoil_kinetic_fd_h(ql,k)
        qp=setindex(ql,ql[k]+h,k);qm=setindex(ql,ql[k]-h,k)
        (_airfoil_kinetic_flux6(q1,q2,qp,qr,q5,q6,nxr,nyr,
                               cxa,cya,cza,wa,drot,eint)-
         _airfoil_kinetic_flux6(q1,q2,qm,qr,q5,q6,nxr,nyr,
                               cxa,cya,cza,wa,drot,eint))/(2f0*h)
    end
    rcols=ntuple(Val(4)) do k
        h=_airfoil_kinetic_fd_h(qr,k)
        qp=setindex(qr,qr[k]+h,k);qm=setindex(qr,qr[k]-h,k)
        (_airfoil_kinetic_flux6(q1,q2,ql,qp,q5,q6,nxr,nyr,
                               cxa,cya,cza,wa,drot,eint)-
         _airfoil_kinetic_flux6(q1,q2,ql,qm,q5,q6,nxr,nyr,
                               cxa,cya,cza,wa,drot,eint))/(2f0*h)
    end
    hcat(lcols...),hcat(rcols...)
end

@inline function _airfoil_kinetic_fd_h(q,k)
    r=q[1]
    u=q[2]/r;v=q[3]/r
    e=q[4]-0.5f0*r*(u*u+v*v)
    scale=k==1 ? r : (k==4 ? max(e,abs(q[4])) : max(abs(q[k]),0.2f0*r))
    h=1.0f-3*max(scale,1.0f-4)
    k==1 && (h=min(h,0.1f0*r))
    k==4 && (h=min(h,0.1f0*e))
    k==2 && (h=min(h,0.1f0*e/max(abs(u),1f-3)))
    k==3 && (h=min(h,0.1f0*e/max(abs(v),1f-3)))
    h
end

@inline function _airfoil_kinetic_face_jacobians(ql,qr,nxr,nyr,
                                                cxa,cya,cza,wa,drot,eint)
    lcols=ntuple(Val(4)) do k
        h=_airfoil_kinetic_fd_h(ql,k)
        qp=setindex(ql,ql[k]+h,k);qm=setindex(ql,ql[k]-h,k)
        (_airfoil_kinetic_flux2(qp,qr,nxr,nyr,cxa,cya,cza,wa,drot,eint)-
         _airfoil_kinetic_flux2(qm,qr,nxr,nyr,cxa,cya,cza,wa,drot,eint))/(2f0*h)
    end
    rcols=ntuple(Val(4)) do k
        h=_airfoil_kinetic_fd_h(qr,k)
        qp=setindex(qr,qr[k]+h,k);qm=setindex(qr,qr[k]-h,k)
        (_airfoil_kinetic_flux2(ql,qp,nxr,nyr,cxa,cya,cza,wa,drot,eint)-
         _airfoil_kinetic_flux2(ql,qm,nxr,nyr,cxa,cya,cza,wa,drot,eint))/(2f0*h)
    end
    hcat(lcols...),hcat(rcols...)
end

# A separately selected R2 approximation for physical-coordinate MUSCL.
# Freeze each primitive reconstruction increment, not the conservative face
# value: z_face = z_cell + frozen_increment. The chain below differentiates
# that declared model exactly apart from the small central flux difference.
# Gradient/limiter derivatives and more distant stencil blocks are omitted;
# the production R0 and finite-difference Jv continue to include them.
@inline function _airfoil_muscl_trace_chain(q,z,eint)
    B=eltype(q)
    r,u,v,t=z
    energy=eint*t+0.5f0*(u*u+v*v)
    qz=hcat(SVector{4,B}(1,u,v,energy),SVector{4,B}(0,r,0,r*u),
        SVector{4,B}(0,0,r,r*v),SVector{4,B}(0,0,0,r*eint))
    p=_airfoil_primitive_jacobian(q,eint)
    zq=SMatrix{4,4,B,16}(1,p[1,1],p[2,1],p[3,1],
        0,p[1,2],p[2,2],p[3,2],0,p[1,3],p[2,3],p[3,3],
        0,p[1,4],p[2,4],p[3,4])
    qz*zq
end

@inline function _airfoil_muscl_trace_chain(q,z,eint,coeff)
    # Scale primitive rows of the inverse chain, allowing each component's
    # frozen limiter to differ. No conservative-component limiter is invented.
    B=eltype(q);r,u,v,t=z
    energy=eint*t+0.5f0*(u*u+v*v)
    qz=hcat(SVector{4,B}(1,u,v,energy),SVector{4,B}(0,r,0,r*u),
        SVector{4,B}(0,0,r,r*v),SVector{4,B}(0,0,0,r*eint))
    p=_airfoil_primitive_jacobian(q,eint)
    zq=SMatrix{4,4,B,16}(coeff[1],coeff[2]*p[1,1],coeff[3]*p[2,1],coeff[4]*p[3,1],
        0,coeff[2]*p[1,2],coeff[3]*p[2,2],coeff[4]*p[3,2],
        0,coeff[2]*p[1,3],coeff[3]*p[2,3],coeff[4]*p[3,3],
        0,coeff[2]*p[1,4],coeff[3]*p[2,4],coeff[4]*p[3,4])
    qz*zq
end

# The nearest face pair affects both reconstructed traces. Retain those
# exact LS coefficients with frozen component limiters; wider-neighbor
# contributions remain omitted from this compact line-factor R2 model.
@inline function _airfoil_muscl_ls_coefficients(theta,weights,offsets,i,j,f)
    @inbounds begin
        dx=offsets[i,j,1,f];dy=offsets[i,j,2,f]
        total=zero(dx)
        for n in 1:4
            total+=weights[i,j,1,n]*dx+weights[i,j,2,n]*dy
        end
        neighbor=weights[i,j,1,f]*dx+weights[i,j,2,f]*dy
        own=(1-theta[i,j,1]*total,1-theta[i,j,2]*total,
             1-theta[i,j,3]*total,1-theta[i,j,4]*total)
        near=(theta[i,j,1]*neighbor,theta[i,j,2]*neighbor,
              theta[i,j,3]*neighbor,theta[i,j,4]*neighbor)
        own,near
    end
end

@inline function _airfoil_muscl_ls_face_jacobians(ql,qr,zl,zr,sl,nl,sr,nr,nxr,nyr,
                                                  cxa,cya,cza,wa,drot,eint)
    B=eltype(ql)
    fl=_airfoil_conserved4(zl...,eint,B);fr=_airfoil_conserved4(zr...,eint,B)
    a,b=_airfoil_kinetic_face_jacobians(fl,fr,nxr,nyr,cxa,cya,cza,wa,drot,eint)
    a*_airfoil_muscl_trace_chain(ql,zl,eint,sl)+b*_airfoil_muscl_trace_chain(ql,zr,eint,nr),
    a*_airfoil_muscl_trace_chain(qr,zl,eint,nl)+b*_airfoil_muscl_trace_chain(qr,zr,eint,sr)
end

# Recompute, and store separately from R0, its exact component factors.
# This intentional small duplicate keeps the experimental R2 control from
# changing the physical reconstruction payload or the production residual.
@kernel function _k_airfoil_muscl_block_theta!(theta,
    @Const(r),@Const(u),@Const(v),@Const(t),@Const(offsets),@Const(weights),@Const(neighbors),@Const(area),
    limiter::Int32,kappa::Float32,NI::Int32,NJ::Int32,g::Int32,uref)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        z=_airfoil_air_z(r,u,v,t,i+g,j+g)
        gx=ntuple(k->_airfoil_ls_gradient(r,u,v,t,weights,neighbors,z,i,j,k,1),Val(4))
        gy=ntuple(k->_airfoil_ls_gradient(r,u,v,t,weights,neighbors,z,i,j,k,2),Val(4))
        finite=all(isfinite,gx)&&all(isfinite,gy)
        for k in 1:4
            lo=z[k];hi=z[k]
            for f in 1:4
                ii=neighbors[i,j,1,f];jj=neighbors[i,j,2,f]
                q=_airfoil_air_z(r,u,v,t,ii,jj)[k]
                lo=min(lo,q);hi=max(hi,q)
            end
            value=finite ? one(z[k]) : zero(z[k])
            epsilon2=_airfoil_limit_epsilon2(limiter,kappa,area[i,j],z[k],k,uref)
            for f in 1:4
                b=gx[k]*offsets[i,j,1,f]+gy[k]*offsets[i,j,2,f]
                a=b>0f0 ? hi-z[k] : lo-z[k]
                finite&&(value=min(value,_airfoil_limit_ratio(a,b,epsilon2,limiter)))
                if (k==1||k==4)&&b<0f0
                    floor=k==1 ? 1f-4 : 1f-3
                    value=min(value,.999f0*(z[k]-floor)/(-b))
                end
            end
            theta[i,j,k]=clamp(value,zero(value),one(value))
        end
    end
end

@inline function _airfoil_muscl_face_jacobians(ql,qr,zl,zr,nxr,nyr,
                                               cxa,cya,cza,wa,drot,eint)
    B=eltype(ql)
    fl=_airfoil_conserved4(zl...,eint,B)
    fr=_airfoil_conserved4(zr...,eint,B)
    a,b=_airfoil_kinetic_face_jacobians(fl,fr,nxr,nyr,cxa,cya,cza,wa,drot,eint)
    a*_airfoil_muscl_trace_chain(ql,zl,eint),b*_airfoil_muscl_trace_chain(qr,zr,eint)
end

# The production analytic wall flux has zero mass/energy convection and
# pressure traction p*S. This is its conservative-variable pressure block.
@inline function _airfoil_wall_pressure_jacobian(u,v,nxr,nyr,eint,::Type{B}=Float32) where {B<:AbstractFloat}
    gm1=_gas_cs2(B)/eint
    dp=SVector{4,B}(0.5f0*gm1*(u*u+v*v),-gm1*u,-gm1*v,gm1)
    hcat(SVector{4,B}(0f0,dp[1]*nxr,dp[1]*nyr,0f0),
         SVector{4,B}(0f0,dp[2]*nxr,dp[2]*nyr,0f0),
         SVector{4,B}(0f0,dp[3]*nxr,dp[3]*nyr,0f0),
         SVector{4,B}(0f0,dp[4]*nxr,dp[4]*nyr,0f0))
end

@kernel function _k_airfoil_kinetic_i_faces!(left,right,
    @Const(rho),@Const(ux),@Const(uy),@Const(T),
    @Const(Six),@Const(Siy),@Const(cxa),@Const(cya),@Const(cza),@Const(wa),
    @Const(xc),@Const(yc),@Const(nut),
    @Const(gux),@Const(guy),@Const(gvx),@Const(gvy),@Const(gTx),@Const(gTy),
    drot,eint,nu_lam,rho_inf,
    suth,air_sutherland::Int32,sst_on::Int32,
    fullvisc_block::Int32,weno_block::Int32,face_states,air_geometry,ls_theta,ls_weights,ls_offsets,
    NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    B=eltype(left)
    @inbounds if i<=NI+1 && j<=NJ
        ia=i+g;ja=j+g
        ql=_airfoil_conserved4(rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja],eint,B)
        qr=_airfoil_conserved4(rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],eint,B)
        if face_states!==nothing
            zl=i==1 ? _airfoil_air_z(rho,ux,uy,T,ia-1,ja) : _airfoil_trace4(face_states,i-1,j,2)
            zr=i==NI+1 ? _airfoil_air_z(rho,ux,uy,T,ia,ja) : _airfoil_trace4(face_states,i,j,1)
            if ls_theta===nothing
                a,b=_airfoil_muscl_face_jacobians(ql,qr,zl,zr,Six[i,j],Siy[i,j],
                    cxa,cya,cza,wa,drot,eint)
            else
                one4=ntuple(_->one(eint),Val(4));zero4=ntuple(_->zero(eint),Val(4))
                sl,nl=i==1 ? (one4,zero4) : _airfoil_muscl_ls_coefficients(ls_theta,ls_weights,ls_offsets,i-1,j,2)
                sr,nr=i==NI+1 ? (one4,zero4) : _airfoil_muscl_ls_coefficients(ls_theta,ls_weights,ls_offsets,i,j,1)
                a,b=_airfoil_muscl_ls_face_jacobians(ql,qr,zl,zr,sl,nl,sr,nr,Six[i,j],Siy[i,j],
                    cxa,cya,cza,wa,drot,eint)
            end
        elseif weno_block==Int32(1)
            q1=_airfoil_conserved4(rho[ia-3,ja],ux[ia-3,ja],uy[ia-3,ja],T[ia-3,ja],eint,B)
            q2=_airfoil_conserved4(rho[ia-2,ja],ux[ia-2,ja],uy[ia-2,ja],T[ia-2,ja],eint,B)
            q5=_airfoil_conserved4(rho[ia+1,ja],ux[ia+1,ja],uy[ia+1,ja],T[ia+1,ja],eint,B)
            q6=_airfoil_conserved4(rho[ia+2,ja],ux[ia+2,ja],uy[ia+2,ja],T[ia+2,ja],eint,B)
            a,b=_airfoil_weno_face_jacobians(q1,q2,ql,qr,q5,q6,
                Six[i,j],Siy[i,j],cxa,cya,cza,wa,drot,eint)
        else
            a,b=_airfoil_kinetic_face_jacobians(ql,qr,Six[i,j],Siy[i,j],
                cxa,cya,cza,wa,drot,eint)
        end
        if fullvisc_block==Int32(1) && 2<=i<=NI && nu_lam>0f0
            gl=SVector{6,B}(gux[i-1,j],guy[i-1,j],gvx[i-1,j],
                                   gvy[i-1,j],gTx[i-1,j],gTy[i-1,j])
            gr=SVector{6,B}(gux[i,j],guy[i,j],gvx[i,j],
                                   gvy[i,j],gTx[i,j],gTy[i,j])
            dxc=air_geometry===nothing ? xc[i,j]-xc[i-1,j] : -air_geometry[i,j,1,1]
            dyc=air_geometry===nothing ? yc[i,j]-yc[i-1,j] : -air_geometry[i,j,2,1]
            mu,ke=_airfoil_face_transport(rho[ia-1,ja],rho[ia,ja],
                T[ia-1,ja],T[ia,ja],nut[ia-1,ja],nut[ia,ja],
                nu_lam,rho_inf,suth,air_sutherland,sst_on,drot,B)
            va,vb=_airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,
                dxc,dyc,Six[i,j],Siy[i,j],mu,ke,eint)
            a+=va;b+=vb
        end
        _airfoil_block4_store!(left,a,i,j)
        _airfoil_block4_store!(right,b,i,j)
    end
end

@kernel function _k_airfoil_kinetic_j_faces!(left,right,
    @Const(rho),@Const(ux),@Const(uy),@Const(T),
    @Const(Sjx),@Const(Sjy),@Const(cxa),@Const(cya),@Const(cza),@Const(wa),
    @Const(xc),@Const(yc),@Const(nut),
    @Const(gux),@Const(guy),@Const(gvx),@Const(gvy),@Const(gTx),@Const(gTy),
    drot,eint,nu_lam,rho_inf,
    suth,air_sutherland::Int32,sst_on::Int32,
    fullvisc_block::Int32,weno_block::Int32,face_states,air_geometry,ls_theta,ls_weights,ls_offsets,
    NI::Int32,NJ::Int32,
    NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    B=eltype(left)
    @inbounds if i<=NI && j<=NJ+1
        ia=i+g;ja=j+g
        if j==1 && NIW<i<=NIW+NIF
            b=_airfoil_wall_pressure_jacobian(ux[ia,ja],uy[ia,ja],
                Sjx[i,j],Sjy[i,j],eint,B)
            _airfoil_block4_store!(left,zero(SMatrix{4,4,B,16}),i,j)
            _airfoil_block4_store!(right,b,i,j)
        else
            ql=_airfoil_conserved4(rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1],eint,B)
            qr=_airfoil_conserved4(rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],eint,B)
            if face_states!==nothing
                if j==1
                    partner=NI+Int32(1)-i
                    zl=_airfoil_trace4(face_states,partner,1,3)
                    zr=_airfoil_trace4(face_states,i,1,3)
                    # The left wake trace belongs to the paired physical cell.
                    ql=_airfoil_conserved4(rho[partner+g,g+1],ux[partner+g,g+1],
                        uy[partner+g,g+1],T[partner+g,g+1],eint,B)
                else
                    zl=_airfoil_trace4(face_states,i,j-1,4)
                    zr=j==NJ+1 ? _airfoil_air_z(rho,ux,uy,T,ia,ja) : _airfoil_trace4(face_states,i,j,3)
                end
                if ls_theta===nothing
                    a,b=_airfoil_muscl_face_jacobians(ql,qr,zl,zr,Sjx[i,j],Sjy[i,j],
                        cxa,cya,cza,wa,drot,eint)
                else
                    one4=ntuple(_->one(eint),Val(4));zero4=ntuple(_->zero(eint),Val(4))
                    sl,nl=j==1 ? _airfoil_muscl_ls_coefficients(ls_theta,ls_weights,ls_offsets,NI+Int32(1)-i,1,3) :
                        _airfoil_muscl_ls_coefficients(ls_theta,ls_weights,ls_offsets,i,j-1,4)
                    sr,nr=j==NJ+1 ? (one4,zero4) : _airfoil_muscl_ls_coefficients(ls_theta,ls_weights,ls_offsets,i,j,3)
                    a,b=_airfoil_muscl_ls_face_jacobians(ql,qr,zl,zr,sl,nl,sr,nr,Sjx[i,j],Sjy[i,j],
                        cxa,cya,cza,wa,drot,eint)
                end
            elseif weno_block==Int32(1)
                q1=_airfoil_conserved4(rho[ia,ja-3],ux[ia,ja-3],uy[ia,ja-3],T[ia,ja-3],eint,B)
                q2=_airfoil_conserved4(rho[ia,ja-2],ux[ia,ja-2],uy[ia,ja-2],T[ia,ja-2],eint,B)
                q5=_airfoil_conserved4(rho[ia,ja+1],ux[ia,ja+1],uy[ia,ja+1],T[ia,ja+1],eint,B)
                q6=_airfoil_conserved4(rho[ia,ja+2],ux[ia,ja+2],uy[ia,ja+2],T[ia,ja+2],eint,B)
                a,b=_airfoil_weno_face_jacobians(q1,q2,ql,qr,q5,q6,
                    Sjx[i,j],Sjy[i,j],cxa,cya,cza,wa,drot,eint)
            else
                a,b=_airfoil_kinetic_face_jacobians(ql,qr,Sjx[i,j],Sjy[i,j],
                    cxa,cya,cza,wa,drot,eint)
            end
            if fullvisc_block==Int32(1) && 2<=j<=NJ && nu_lam>0f0
                gl=SVector{6,B}(gux[i,j-1],guy[i,j-1],gvx[i,j-1],
                                       gvy[i,j-1],gTx[i,j-1],gTy[i,j-1])
                gr=SVector{6,B}(gux[i,j],guy[i,j],gvx[i,j],
                                       gvy[i,j],gTx[i,j],gTy[i,j])
                dxc=air_geometry===nothing ? xc[i,j]-xc[i,j-1] : -air_geometry[i,j,1,3]
                dyc=air_geometry===nothing ? yc[i,j]-yc[i,j-1] : -air_geometry[i,j,2,3]
                mu,ke=_airfoil_face_transport(rho[ia,ja-1],rho[ia,ja],
                    T[ia,ja-1],T[ia,ja],nut[ia,ja-1],nut[ia,ja],
                    nu_lam,rho_inf,suth,air_sutherland,sst_on,drot,B)
                va,vb=_airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,
                    dxc,dyc,Sjx[i,j],Sjy[i,j],mu,ke,eint)
                a+=va;b+=vb
            end
            _airfoil_block4_store!(left,a,i,j)
            _airfoil_block4_store!(right,b,i,j)
        end
    end
end

# A local thin-layer viscous block adds momentum and energy damping. The
# production full Green-Gauss stress is retained only in the nonlinear/Jv path.
@kernel function _k_airfoil_kinetic_coeffs!(lower_i,upper_i,lower_j,upper_j,diagonal,
    @Const(fi_l),@Const(fi_r),@Const(fj_l),@Const(fj_r),
    @Const(d0),@Const(rho),@Const(T),@Const(nut),
    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
    nu_lam,rho_inf,suth,air_sutherland::Int32,
    sst_on::Int32,fullvisc_block::Int32,
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        nu=_air_nu_local(rho[ia,ja],T[ia,ja],nu_lam,rho_inf,suth,air_sutherland)+
            (sst_on==Int32(1) ? nut[ia,ja] : 0f0)
        invv=1f0/max(vol[i,j],1f-30)
        rw=2f0*nu*(Six[i,j]^2+Siy[i,j]^2)*invv
        re=2f0*nu*(Six[i+1,j]^2+Siy[i+1,j]^2)*invv
        rs=2f0*nu*(Sjx[i,j]^2+Sjy[i,j]^2)*invv
        rn=2f0*nu*(Sjx[i,j+1]^2+Sjy[i,j+1]^2)*invv
        if fullvisc_block==Int32(1)
            rw=i==1 ? rw : 0f0
            re=i==NI ? re : 0f0
            rs=j==1 ? rs : 0f0
            rn=j==NJ ? rn : 0f0
        end
        d=d0[i,j]*_AIRFOIL_KI4+
            _airfoil_block4(fi_l,i+1,j)-_airfoil_block4(fi_r,i,j)+
            _airfoil_block4(fj_l,i,j+1)-_airfoil_block4(fj_r,i,j)+
            (rw+re+rs+rn)*_AIRFOIL_KV4
        wake=j==1 && (i<=NIW || i>NIW+NIF)
        lw=i>1 ? -_airfoil_block4(fi_l,i,j)-rw*_AIRFOIL_KV4 : zero(d)
        ue=i<NI ? _airfoil_block4(fi_r,i+1,j)-re*_AIRFOIL_KV4 : zero(d)
        ls=(j>1 || wake) ? -_airfoil_block4(fj_l,i,j)-rs*_AIRFOIL_KV4 : zero(d)
        un=j<NJ ? _airfoil_block4(fj_r,i,j+1)-rn*_AIRFOIL_KV4 : zero(d)
        _airfoil_block4_store!(lower_i,lw,i,j)
        _airfoil_block4_store!(upper_i,ue,i,j)
        _airfoil_block4_store!(lower_j,ls,i,j)
        _airfoil_block4_store!(upper_j,un,i,j)
        _airfoil_block4_store!(diagonal,d,i,j)
    end
end

@kernel function _k_airfoil_shared_wake_blocks!(left,right,NI::Int32,NIW::Int32)
    i=@index(Global,Linear)
    @inbounds if i<=NIW
        ip=NI+Int32(1)-i
        # The opposite representation swaps its two physical cells and sign.
        _airfoil_block4_store!(left,-_airfoil_block4(right,i,1),ip,1)
        _airfoil_block4_store!(right,-_airfoil_block4(left,i,1),ip,1)
    end
end

function _airfoil_kinetic_build!(s::KSol,w::AirfoilKineticBlockWorkspace,d0)
    b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g)
    fullvisc_block=Int32(w.fullvisc_block)
    weno_block=Int32(w.weno_block)
    face_states=nothing;air_geometry=nothing;ls_weights=nothing;ls_offsets=nothing
    if w.muscl_block
        mw=get(_AIRFOIL_MUSCL,s,nothing)
        s.recon==_AIRFOIL_MUSCL_RECON&&mw!==nothing||
            throw(ArgumentError("MUSCL block workspace lost its original-node reconstruction"))
        face_states=_airfoil_muscl_prepare!(s,mw)
        air_geometry=mw.geometry.displacements
        if w.muscl_theta!==nothing
            a=mw.geometry
            _k_airfoil_muscl_block_theta!(b)(w.muscl_theta,s.rho,s.ux,s.uy,s.T,
                a.offsets,a.weights,a.neighbors,a.area,mw.limiter,mw.smooth_coefficient,ni,nj,g,
                eltype(s.rho)(hypot(s.uxi,s.uyi));
                ndrange=(s.NI,s.NJ))
            ls_weights=a.weights;ls_offsets=a.offsets
        end
    end
    _k_airfoil_kinetic_i_faces!(b)(w.face_i_left,w.face_i_right,
        s.rho,s.ux,s.uy,s.T,s.Six,s.Siy,s.cxa,s.cya,s.cza,s.wa,
        s.xc,s.yc,s.nut,s.gux,s.guy,s.gvx,s.gvy,s.gTx,s.gTy,
        s.drot,s.eint,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        fullvisc_block,weno_block,face_states,air_geometry,w.muscl_theta,ls_weights,ls_offsets,
        ni,nj,g;ndrange=(s.NI+1,s.NJ))
    _k_airfoil_kinetic_j_faces!(b)(w.face_j_left,w.face_j_right,
        s.rho,s.ux,s.uy,s.T,s.Sjx,s.Sjy,s.cxa,s.cya,s.cza,s.wa,
        s.xc,s.yc,s.nut,s.gux,s.guy,s.gvx,s.gvy,s.gTx,s.gTy,
        s.drot,s.eint,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        fullvisc_block,weno_block,face_states,air_geometry,w.muscl_theta,ls_weights,ls_offsets,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g;
        ndrange=(s.NI,s.NJ+1))
    if w.muscl_block&&s.NIW>0
        _k_airfoil_shared_wake_blocks!(b)(w.face_j_left,w.face_j_right,ni,Int32(s.NIW);ndrange=s.NIW)
    end
    _k_airfoil_kinetic_coeffs!(b)(w.lower_i,w.upper_i,w.lower_j,w.upper_j,w.diagonal,
        w.face_i_left,w.face_i_right,w.face_j_left,w.face_j_right,d0,
        s.rho,s.T,s.nut,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        fullvisc_block,ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ))
    nothing
end

@inline function _airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
    if l<=NIF
        i=NIW+l;j=m
        lo=m>1 ? _airfoil_block4(lower,i,j) : zero(SMatrix{4,4,eltype(lower),16})
        hi=m<NJ ? _airfoil_block4(upper,i,j) : zero(SMatrix{4,4,eltype(lower),16})
    else
        w=l-NIF;ip=NI+Int32(1)-w
        if m<=NJ
            i=ip;j=NJ+Int32(1)-m
            lo=j<NJ ? _airfoil_block4(upper,i,j) : zero(SMatrix{4,4,eltype(lower),16})
            hi=_airfoil_block4(lower,i,j)
        else
            i=w;j=m-NJ
            lo=_airfoil_block4(lower,i,j)
            hi=j<NJ ? _airfoil_block4(upper,i,j) : zero(SMatrix{4,4,eltype(lower),16})
        end
    end
    (i,j,lo,hi)
end

@kernel function _k_airfoil_block_cut_line_solve!(out,cp,@Const(rhs),@Const(vol),
    @Const(lower),@Const(diagonal),@Const(upper),
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    l=@index(Global,Linear)
    B=eltype(diagonal)
    @inbounds if l<=NIW+NIF
        count=l<=NIF ? NJ : Int32(2)*NJ
        prev_c=zero(SMatrix{4,4,B,16})
        prev_d=zero(SVector{4,B})
        for m in Int32(1):count
            i,j,lo,hi=_airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
            d=_airfoil_block4(diagonal,i,j)-lo*prev_c
            rhs4=_airfoil_vec4(rhs,i,j,B)*vol[i,j]-lo*prev_d
            di=inv(d)
            ci=m<count ? di*hi : zero(SMatrix{4,4,B,16})
            yi=di*rhs4
            _airfoil_block4_store!(cp,ci,i,j)
            _airfoil_vec4_store!(out,yi,i,j)
            prev_c=ci;prev_d=yi
        end
        i,j,_,_=_airfoil_block_cut_cell(l,count,lower,upper,NI,NJ,NIW,NIF)
        next=_airfoil_vec4(out,i,j,B)
        for m in count-Int32(1):Int32(-1):Int32(1)
            i,j,_,_=_airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
            next=_airfoil_vec4(out,i,j,B)-_airfoil_block4(cp,i,j)*next
            _airfoil_vec4_store!(out,next,i,j)
        end
    end
end

@kernel function _k_airfoil_block_diagonal_scale!(out,@Const(x),@Const(diagonal),
                                                  @Const(vol),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    B=eltype(diagonal)
    @inbounds if i<=NI && j<=NJ
        y=_airfoil_block4(diagonal,i,j)*_airfoil_vec4(x,i,j,B)/vol[i,j]
        _airfoil_vec4_store!(out,y,i,j)
    end
end

function _airfoil_kinetic_precondition!(out,input,s::KSol,w::AirfoilKineticBlockWorkspace)
    b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ)
    _k_airfoil_block_row_solve!(b)(w.row_out,w.cp,input,s.vol,
        w.lower_i,w.diagonal,w.upper_i,ni,nj,w.block_length,w.blocks_per_row;
        ndrange=Int(w.blocks_per_row)*s.NJ)
    _k_airfoil_block_diagonal_scale!(b)(w.row_scaled,w.row_out,w.diagonal,s.vol,
        ni,nj;ndrange=(s.NI,s.NJ))
    _k_airfoil_block_cut_line_solve!(b)(out,w.cp,w.row_scaled,s.vol,
        w.lower_j,w.diagonal,w.upper_j,ni,nj,Int32(s.NIW),Int32(s.NIF);
        ndrange=s.NIW+s.NIF)
    out
end
