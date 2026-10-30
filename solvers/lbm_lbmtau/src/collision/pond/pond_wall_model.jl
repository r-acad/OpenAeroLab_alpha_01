# ==============================================================================
# Particles-on-Demand — EQUILIBRIUM WALL MODEL (compressible Van Driest WMLES)  [GPU]
# ==============================================================================
# Turns the LES (Smagorinsky subgrid) into a proper WALL-MODELLED LES: at high Re the boundary layer is
# thinner than a cell, so instead of resolving it we MODEL the wall shear stress τ_w with the log-law and
# apply it as a near-wall momentum sink. This gives the correct wall stress (hence skin friction and the
# SEPARATION point that sets bluff-body pressure drag) on a coarse near-wall grid.
#   • wall distance y + outward normal n̂ come from a SIGNED DISTANCE FIELD built from the STL triangles
#     (point-to-triangle distance, min over facets, signed by the solid mask) — works for ANY geometry.
#   • at each fluid cell in the wall band (0.5<y<y_wm) the tangential velocity u_t is transformed by the
#     Van Driest factor √(T_w/T_loc) (compressibility), the log-law u_t^+ = (1/κ)ln(y^+)+B is inverted for
#     u_τ (Newton, fixed iters), τ_w = ρ_w u_τ² with the wall density ρ_w = ρ·T/T_w (Van Driest 1956), and
#     u_t is relaxed by τ_w/(ρ y)·Δt. Reduces to the log-law when T_w=T_loc.
# Refs: Van Driest 1956; the equilibrium WMLES of Kawai–Larsson 2012 / the LUDWIG D3Q27 wall model.
# ==============================================================================

using KernelAbstractions

# squared distance from point p to triangle (a,b,c) — closest-point-on-triangle (Ericson, RTCD §5.1.5).
@inline function _pt_tri_dist2(px,py,pz, ax,ay,az, bx,by,bz, cx,cy,cz)
    abx=bx-ax;aby=by-ay;abz=bz-az; acx=cx-ax;acy=cy-ay;acz=cz-az; apx=px-ax;apy=py-ay;apz=pz-az
    d1=abx*apx+aby*apy+abz*apz; d2=acx*apx+acy*apy+acz*apz
    if d1<=0f0 && d2<=0f0; return apx*apx+apy*apy+apz*apz; end
    bpx=px-bx;bpy=py-by;bpz=pz-bz; d3=abx*bpx+aby*bpy+abz*bpz; d4=acx*bpx+acy*bpy+acz*bpz
    if d3>=0f0 && d4<=d3; return bpx*bpx+bpy*bpy+bpz*bpz; end
    vc=d1*d4-d3*d2
    if vc<=0f0 && d1>=0f0 && d3<=0f0; t=d1/(d1-d3); qx=apx-t*abx;qy=apy-t*aby;qz=apz-t*abz; return qx*qx+qy*qy+qz*qz; end
    cpx=px-cx;cpy=py-cy;cpz=pz-cz; d5=abx*cpx+aby*cpy+abz*cpz; d6=acx*cpx+acy*cpy+acz*cpz
    if d6>=0f0 && d5<=d6; return cpx*cpx+cpy*cpy+cpz*cpz; end
    vb=d5*d2-d1*d6
    if vb<=0f0 && d2>=0f0 && d6<=0f0; t=d2/(d2-d6); qx=apx-t*acx;qy=apy-t*acy;qz=apz-t*acz; return qx*qx+qy*qy+qz*qz; end
    va=d3*d6-d5*d4
    if va<=0f0 && (d4-d3)>=0f0 && (d5-d6)>=0f0; t=(d4-d3)/((d4-d3)+(d5-d6)); qx=bpx-t*(cx-bx);qy=bpy-t*(cy-by);qz=bpz-t*(cz-bz); return qx*qx+qy*qy+qz*qz; end
    den=1f0/(va+vb+vc); v=vb*den; w=vc*den; qx=apx-v*abx-w*acx;qy=apy-v*aby-w*acy;qz=apz-v*abz-w*acz
    return qx*qx+qy*qy+qz*qz
end

# signed distance field: y = +dist outside, −dist inside (sign from the cut-fraction χ). Cell (i,j,l) at
# fine coords (ox+sx·i,…). Only the near field matters — a cutoff caps the far distance for speed.
@kernel function sdf_kernel!(sdf, @Const(chi), @Const(v1),@Const(v2),@Const(v3), ntri::Int32,
        ox::Float32,oy::Float32,oz::Float32, sx::Float32,sy::Float32,sz::Float32, ymax::Float32)
    i,j,l=@index(Global,NTuple); nt=Int(ntri)
    @inbounds begin
        px=ox+sx*i; py=oy+sy*j; pz=oz+sz*l; best=ymax*ymax
        for t in 1:nt
            d2=_pt_tri_dist2(px,py,pz, v1[1,t],v1[2,t],v1[3,t], v2[1,t],v2[2,t],v2[3,t], v3[1,t],v3[2,t],v3[3,t])
            if d2<best; best=d2; end
        end
        d=sqrt(best); sdf[i,j,l] = chi[i,j,l]>0.5f0 ? -d : d
    end
end

# equilibrium Van Driest wall model (+ optional Larsson 2010 pressure-gradient correction, roadmap 1.3):
# apply the modelled wall stress as a near-wall tangential momentum sink. When larsson>0 the log-law target is
# augmented by the Larsson/Kawai NEQ term +(y/(ρ_w u_τ²))·∂p/∂x_stream (Schur-bounded |Δ|≤½·u_VD): an ADVERSE
# streamwise pressure gradient lowers τ_w (toward separation), a FAVOURABLE one raises it — the equilibrium
# log-law is exact only where ∂p/∂x≈0. p = ρ·cs²·T; ∂p/∂x_stream = ∇p·û_t (projected on the flow direction).
@kernel function wall_model_kernel!(ux,uy,uz, @Const(sdf),@Const(rho),@Const(Temp),
        nu::Float32, Tw::Float32, ywm::Float32, dt::Float32, cs2::Float32, larsson::Int32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,l=@index(Global,NTuple); nx=Int(NX);ny=Int(NY);nz=Int(NZ)
    @inbounds begin
        y=sdf[i,j,l]
        if y>0.5f0 && y<ywm && i>1 && i<nx && j>1 && j<ny && l>1 && l<nz
            gx=0.5f0*(sdf[i+1,j,l]-sdf[i-1,j,l]); gy=0.5f0*(sdf[i,j+1,l]-sdf[i,j-1,l]); gz=0.5f0*(sdf[i,j,l+1]-sdf[i,j,l-1])
            gn=sqrt(gx*gx+gy*gy+gz*gz)+1f-9; nnx=gx/gn;nny=gy/gn;nnz=gz/gn      # outward wall normal
            u=ux[i,j,l];v=uy[i,j,l];w=uz[i,j,l]; un=u*nnx+v*nny+w*nnz
            utx=u-un*nnx; uty=v-un*nny; utz=w-un*nnz; ut=sqrt(utx*utx+uty*uty+utz*utz)
            if ut>1f-5
                Tl=Temp[i,j,l]; vd=sqrt(max(Tw,1f-3)/max(Tl,1f-3)); utv=ut*vd    # Van Driest transform
                rho_w=rho[i,j,l]*Tl/max(Tw,1f-3)                                 # Van Driest wall density
                dpdx_s=0f0                                                       # Larsson streamwise ∂p/∂x
                if larsson>0
                    pxp=rho[i+1,j,l]*cs2*Temp[i+1,j,l]; pxm=rho[i-1,j,l]*cs2*Temp[i-1,j,l]
                    pyp=rho[i,j+1,l]*cs2*Temp[i,j+1,l]; pym=rho[i,j-1,l]*cs2*Temp[i,j-1,l]
                    pzp=rho[i,j,l+1]*cs2*Temp[i,j,l+1]; pzm=rho[i,j,l-1]*cs2*Temp[i,j,l-1]
                    thx=utx/ut; thy=uty/ut; thz=utz/ut                           # unit streamwise direction
                    dpdx_s=0.5f0*((pxp-pxm)*thx+(pyp-pym)*thy+(pzp-pzm)*thz)
                end
                utau=sqrt(nu*utv/y+1f-12)                                        # viscous initial guess
                for _ in 1:8
                    pgu = larsson>0 ? clamp((y*dpdx_s)/(rho_w*utau+1f-9), -0.5f0*utv, 0.5f0*utv) : 0f0
                    utve=utv-pgu                                                 # PG-corrected target (Schur-bounded)
                    yp=max(y*utau/nu, 1f-3); lg=2.4390244f0*log(yp)+5.2f0        # 1/κ=2.439 (κ=0.41), B=5.2
                    if yp<=11f0; lg=yp; end                                      # viscous sublayer u+=y+
                    fv=utve-utau*lg; df=-(lg + (yp>11f0 ? 2.4390244f0 : yp)); utau=max(utau-fv/df, 1f-8)
                end
                tauw=rho_w*utau*utau
                fr=min(tauw/(rho[i,j,l]*y)*dt/ut, 1f0)                           # fraction of u_t to remove this step
                ux[i,j,l]-=fr*utx; uy[i,j,l]-=fr*uty; uz[i,j,l]-=fr*utz
            end
        end
    end
end

# host helpers
function build_sdf(backend, dev, chi, tri, NX,NY,NZ, ox,oy,oz, sx,sy,sz; ymax=8f0)
    sdf=dev(fill(Float32(ymax),NX,NY,NZ))
    sdf_kernel!(backend)(sdf, chi, tri[1],tri[2],tri[3],Int32(tri[4]), Float32(ox),Float32(oy),Float32(oz),Float32(sx),Float32(sy),Float32(sz),Float32(ymax); ndrange=(NX,NY,NZ))
    KernelAbstractions.synchronize(backend); return sdf
end
function apply_wall_model!(backend, ux,uy,uz, sdf,rho,Temp, nu,Tw,ywm,dt, NX,NY,NZ; cs2=0.6666667f0, larsson=false)
    wall_model_kernel!(backend)(ux,uy,uz, sdf,rho,Temp, Float32(nu),Float32(Tw),Float32(ywm),Float32(dt), Float32(cs2), Int32(larsson ? 1 : 0), Int32(NX),Int32(NY),Int32(NZ); ndrange=(Int(NX),Int(NY),Int(NZ)))
    KernelAbstractions.synchronize(backend)
end
