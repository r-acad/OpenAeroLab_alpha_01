# ==============================================================================
# Particles-on-Demand — STL IMMERSED BOUNDARY (any tessellation)  [GPU]
# ==============================================================================
# Generic immersed-boundary geometry + force from a triangle mesh (STL), so ANY shape works — not just
# analytic spheres. Two pieces:
#   (1) VOXELIZE: build the solid mask χ by ray-casting (+x, parity/odd-crossing = inside) each cell centre
#       against all triangles (Möller–Trumbore). GPU kernel, robust for watertight meshes.
#   (2) FORCE: integrate the surface pressure over the TRIANGLES — for each facet (centroid, outward unit
#       normal, area) sample p=ρcs²T just outside the surface (centroid + δ·n̂) by trilinear interpolation
#       and accumulate F = −Σ p n̂ A ⇒ Cd. This is the standard mesh-based IB force, exact for any geometry.
# Uses Geometry.load_mesh (SolverMesh: triangles, normals, areas, centers). Meshes are placed by scaling a
# unit sphere (r=0.5) STL to radius R at a chosen centre; a general mesh is scaled/translated the same way.
# ==============================================================================

using KernelAbstractions

# Place an STL: scale (unit-radius-0.5 sphere ⇒ factor 2R) + translate to (cx,cy,cz) in the solver's
# FINE-coordinate frame. Returns GPU triangle-vertex arrays (for voxelisation) + host centroids/normals/areas
# (for the force integral).  `mesh` is a Geometry.SolverMesh.
function place_stl(mesh, cx, cy, cz, R, dev)
    N=length(mesh.triangles); s=2.0f0*Float32(R); off=(Float32(cx),Float32(cy),Float32(cz))
    v1=Array{Float32}(undef,3,N); v2=similar(v1); v3=similar(v1)
    cen=Array{Float32}(undef,3,N); nrm=Array{Float32}(undef,3,N); ar=Array{Float32}(undef,N)
    @inbounds for t in 1:N
        a,b,c = mesh.triangles[t]
        for d in 1:3
            v1[d,t]=Float32(a[d])*s+off[d]; v2[d,t]=Float32(b[d])*s+off[d]; v3[d,t]=Float32(c[d])*s+off[d]
            cen[d,t]=(v1[d,t]+v2[d,t]+v3[d,t])/3f0; nrm[d,t]=Float32(mesh.normals[t][d])
        end
        e1x=v2[1,t]-v1[1,t];e1y=v2[2,t]-v1[2,t];e1z=v2[3,t]-v1[3,t]
        e2x=v3[1,t]-v1[1,t];e2y=v3[2,t]-v1[2,t];e2z=v3[3,t]-v1[3,t]
        cxp=e1y*e2z-e1z*e2y;cyp=e1z*e2x-e1x*e2z;czp=e1x*e2y-e1y*e2x
        ar[t]=0.5f0*sqrt(cxp*cxp+cyp*cyp+czp*czp)
    end
    return (dev(v1),dev(v2),dev(v3), N), (cen,nrm,ar)
end

# Möller–Trumbore parity test: is a point INSIDE the mesh (odd +x-ray crossings)?  Tilted ray avoids edges.
@inline function _pt_inside(px,py,pz, v1,v2,v3, ntri)
    dx=1.0f0; dy=0.0011f0; dz=0.0007f0; cnt=0
    @inbounds for t in 1:ntri
        a1=v1[1,t];a2=v1[2,t];a3=v1[3,t]
        e1x=v2[1,t]-a1;e1y=v2[2,t]-a2;e1z=v2[3,t]-a3
        e2x=v3[1,t]-a1;e2y=v3[2,t]-a2;e2z=v3[3,t]-a3
        hx=dy*e2z-dz*e2y;hy=dz*e2x-dx*e2z;hz=dx*e2y-dy*e2x
        adet=e1x*hx+e1y*hy+e1z*hz
        if abs(adet)>1f-12
            f=1.0f0/adet; sxr=px-a1;syr=py-a2;szr=pz-a3
            u=f*(sxr*hx+syr*hy+szr*hz)
            if u>=0.0f0 && u<=1.0f0
                qx=syr*e1z-szr*e1y;qy=szr*e1x-sxr*e1z;qz=sxr*e1y-syr*e1x
                v=f*(dx*qx+dy*qy+dz*qz)
                if v>=0.0f0 && u+v<=1.0f0 && f*(e2x*qx+e2y*qy+e2z*qz)>1f-5; cnt+=1; end
            end
        end
    end
    return (cnt & 1)==1
end

# GPU voxelisation → CUT FRACTION χ∈[0,1]: super-sample ss³ sub-points per cell, χ = fraction inside the
# mesh. ss=1 ⇒ binary staircase (backward-compat); ss≥2 ⇒ sub-cell-accurate wall (χ=½ at the true surface),
# so the α-weighted penalization places the effective wall ON the surface — the sharp cut-cell flow BC.
@kernel function voxelize_kernel!(chi, @Const(v1),@Const(v2),@Const(v3), ntri::Int32, ss::Int32,
        ox::Float32,oy::Float32,oz::Float32, sx::Float32,sy::Float32,sz::Float32)
    i,j,l=@index(Global,NTuple); S=Int(ss); nt=Int(ntri)
    @inbounds begin
        cx=ox+sx*i; cy=oy+sy*j; cz=oz+sz*l; inv=1.0f0/Float32(S); cnt=0
        for a in 1:S, b in 1:S, c in 1:S
            px=cx+sx*((Float32(a)-0.5f0)*inv-0.5f0); py=cy+sy*((Float32(b)-0.5f0)*inv-0.5f0); pz=cz+sz*((Float32(c)-0.5f0)*inv-0.5f0)
            if _pt_inside(px,py,pz, v1,v2,v3, nt); cnt+=1; end
        end
        chi[i,j,l] = Float32(cnt)/Float32(S*S*S)
    end
end

# host: voxelise a grid whose cell (i,j,l) sits at fine coords (ox+sx·i,…). ss = super-sample (2 ⇒ cut fraction).
function voxelize_stl(backend, dev, tri, NX,NY,NZ, ox,oy,oz, sx,sy,sz; ss=2)
    chi=dev(zeros(Float32,NX,NY,NZ))
    voxelize_kernel!(backend)(chi, tri[1],tri[2],tri[3],Int32(tri[4]),Int32(ss), Float32(ox),Float32(oy),Float32(oz),Float32(sx),Float32(sy),Float32(sz); ndrange=(NX,NY,NZ))
    KernelAbstractions.synchronize(backend); return chi
end

# mesh-based IB force → Cd. Sample p=ρcs²T at each facet centroid + δ·n̂ (just outside), trilinear in the
# fine field ρ,T whose array index maps to fine coords by (ax+bx·i,…). Drag = −Σ p n_x A (n̂ outward).
function stl_cd(cen,nrm,ar, ρ,T, cs2, Uinf, ax,bx,ay,by,az,bz; δ=1.2)
    @inline function tl(A,x,y,z)
        nx,ny,nz=size(A); i0=clamp(floor(Int,x),1,nx-1);j0=clamp(floor(Int,y),1,ny-1);k0=clamp(floor(Int,z),1,nz-1)
        fx=x-i0;fy=y-j0;fz=z-k0
        @inbounds (A[i0,j0,k0]*(1-fx)*(1-fy)*(1-fz)+A[i0+1,j0,k0]*fx*(1-fy)*(1-fz)+A[i0,j0+1,k0]*(1-fx)*fy*(1-fz)+A[i0+1,j0+1,k0]*fx*fy*(1-fz)
                 +A[i0,j0,k0+1]*(1-fx)*(1-fy)*fz+A[i0+1,j0,k0+1]*fx*(1-fy)*fz+A[i0,j0+1,k0+1]*(1-fx)*fy*fz+A[i0+1,j0+1,k0+1]*fx*fy*fz)
    end
    Fx=0.0; Aref=0.0
    N=length(ar)
    @inbounds for t in 1:N
        nx=Float64(nrm[1,t]);ny=Float64(nrm[2,t]);nz=Float64(nrm[3,t]); A=Float64(ar[t])
        # fine-coord surface point just outside → array index
        sxp=Float64(cen[1,t])+δ*nx; syp=Float64(cen[2,t])+δ*ny; szp=Float64(cen[3,t])+δ*nz
        xi=(sxp-ax)/bx; yi=(syp-ay)/by; zi=(szp-az)/bz
        p=Float64(tl(ρ,xi,yi,zi))*Float64(cs2)*Float64(tl(T,xi,yi,zi))
        Fx += -p*nx*A
        Aref += A*abs(nx)                                   # projected frontal area = ½∮|n_x|dA
    end
    Aref*=0.5
    return Fx/(0.5*1.0*Float64(Uinf)^2*Aref), Aref
end
