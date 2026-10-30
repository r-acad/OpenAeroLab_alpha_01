# Reconstruction workspace for the existing Species stages. No independent march.
const P=@__MODULE__
const S=CMeshSolverKA
# Host construction of the geometry used by reconstructed Species faces.
# Face normals already encode each quadrilateral edge. Keep all local arithmetic
# in Float64; absolute Float32 centers need not be subtracted in a thin layer.
cross2(a,b)=a[1]*b[2]-a[2]*b[1]
add2(a,b)=(a[1]+b[1],a[2]+b[2])
sub2(a,b)=(a[1]-b[1],a[2]-b[2])
scale2(a,t)=(a[1]*t,a[2]*t)
function quad_reconstruction_geometry(bottom,top,left,right)
    B=scale2(add2(bottom,top),.5)
    C=scale2(add2(left,right),.5)
    D=scale2(add2(sub2(top,bottom),sub2(right,left)),.5)
    J=cross2(B,C);Jx=cross2(B,D);Jy=cross2(D,C)
    all(isfinite,(B...,C...,D...,J,Jx,Jy)) && J>.5*(abs(Jx)+abs(Jy)) ||
        throw(ArgumentError("quadrilateral map must have finite positive corner Jacobians"))
    shift=scale2(add2(scale2(B,Jx),scale2(C,Jy)),1/(12J))
    offsets=map(v->sub2(v,shift),
        (scale2(B,-.5),scale2(B,.5),scale2(C,-.5),scale2(C,.5)))
    (;B,C,shift,offsets,area=J)
end
function quad_reconstruction_geometry(m,i,j)
    f(x,y)=(Float64(x),Float64(y))
    quad_reconstruction_geometry(f(m.Sjy[i,j],-m.Sjx[i,j]),
        f(m.Sjy[i,j+1],-m.Sjx[i,j+1]),f(-m.Siy[i,j],m.Six[i,j]),
        f(-m.Siy[i+1,j],m.Six[i+1,j]))
end

# Geometry is built once; the existing metric arrays remain authoritative.

"""Local W/E/S/N face offsets and weighted least-squares gradient coefficients.

Both Float32 arrays have shape (NI,NJ,2,4). Each gradient component is the
sum of weights[i,j,component,face] times that neighbor's value minus the cell
value. Only available real neighbors enter the fit, including edges/corners; missing
directions have zero weights. Every cell must have a nonsingular 2D fit.
Original shared face normals and volumes remain authoritative and unmodified.
"""
function build_reconstruction_geometry(m; diffusion=nothing)
    ni,nj=m.NI,m.NJ
    ni>=2 && nj>=2 || throw(ArgumentError("at least two cells in both geometry dimensions required"))
    size(m.Six)==size(m.Siy)==(ni+1,nj) && size(m.Sjx)==size(m.Sjy)==(ni,nj+1) ||
        throw(ArgumentError("face-normal geometry shape mismatch"))
    local_offsets=Array{Float64}(undef,ni,nj,2,4)
    offsets=Array{Float32}(undef,ni,nj,2,4)
    weights=zeros(Float32,ni,nj,2,4);active=zeros(UInt8,ni,nj)
    for j in 1:nj,i in 1:ni
        q=quad_reconstruction_geometry(m,i,j)
        for f in 1:4,k in 1:2
            x=q.offsets[f][k];xf=Float32(x)
            isfinite(x) && isfinite(xf) || throw(ArgumentError("nonfinite reconstruction offset"))
            local_offsets[i,j,k,f]=x;offsets[i,j,k,f]=xf
        end
    end
    if diffusion!==nothing
        size(diffusion)==(ni,nj,3,2) && eltype(diffusion)==Float32 ||
            throw(ArgumentError("SST diffusion geometry shape/type mismatch"))
        fill!(diffusion,0f0)
        for j in 1:nj,i in 1:ni
            if i<ni
                dx=local_offsets[i,j,1,2]-local_offsets[i+1,j,1,1]
                dy=local_offsets[i,j,2,2]-local_offsets[i+1,j,2,1]
                z=_sst_diffusion_geometry(m.Six[i+1,j],m.Siy[i+1,j],dx,dy)
                for k in 1:3;diffusion[i,j,k,1]=z[k];end
            end
            if j<nj
                dx=local_offsets[i,j,1,4]-local_offsets[i,j+1,1,3]
                dy=local_offsets[i,j,2,4]-local_offsets[i,j+1,2,3]
                z=_sst_diffusion_geometry(m.Sjx[i,j+1],m.Sjy[i,j+1],dx,dy)
                for k in 1:3;diffusion[i,j,k,2]=z[k];end
            end
        end
    end
    opposite=(2,1,4,3)
    for j in 1:nj,i in 1:ni
        neighbors=((i-1,j),(i+1,j),(i,j-1),(i,j+1))
        d=ntuple(4) do f
            ii,jj=neighbors[f];o=opposite[f]
            1<=ii<=ni && 1<=jj<=nj || return (0.,0.,0.)
            dx=local_offsets[i,j,1,f]-local_offsets[ii,jj,1,o]
            dy=local_offsets[i,j,2,f]-local_offsets[ii,jj,2,o]
            d2=dx*dx+dy*dy
            isfinite(d2) && d2>0 || throw(ArgumentError("nonfinite or coincident reconstruction neighbors"))
            (dx,dy,1/d2)
        end
        a11=0.;a12=0.;a22=0.
        for (dx,dy,w) in d
            a11+=w*dx*dx;a12+=w*dx*dy;a22+=w*dy*dy
        end
        det=a11*a22-a12*a12
        all(isfinite,(a11,a12,a22,det)) && det>0 ||
            throw(ArgumentError("singular reconstruction stencil"))
        for f in 1:4
            dx,dy,w=d[f]
            w==0 && continue
            wx=((a22*dx-a12*dy)*w)/det
            wy=((a11*dy-a12*dx)*w)/det
            x,y=Float32(wx),Float32(wy)
            all(isfinite,(wx,wy,x,y)) || throw(ArgumentError("nonfinite reconstruction weight"))
            weights[i,j,1,f]=x;weights[i,j,2,f]=y
        end
        active[i,j]=0x01
    end
    (;offsets,weights,active)
end

# Same Float64 centroid displacement as the LS construction above. Only the
# represented, admitted coefficients enter the face kernels; no GPU dot-product
# sign test or subtraction of absolute Float32 centres is needed there.
function _sst_diffusion_geometry(sx,sy,dx,dy)
    sx,sy,dx,dy=Float64(sx),Float64(sy),Float64(dx),Float64(dy)
    d2=dx*dx+dy*dy;sd=sx*dx+sy*dy
    all(isfinite,(sx,sy,dx,dy,d2,sd)) && d2>0 && sd>0 ||
        throw(ArgumentError("SST diffusion requires finite positive S dot d and distance"))
    lambda=sd/d2;kx=sx-lambda*dx;ky=sy-lambda*dy
    z=(Float32(lambda),Float32(kx),Float32(ky))
    all(isfinite,(lambda,kx,ky,z...)) && z[1]>0f0 ||
        throw(ArgumentError("SST diffusion geometry is not representable in Float32"))
    z
end

struct SpeciesReconstruction{G,A4,A2,K4}
    geometry::G
    states::A4
    theta::A2
    status::A2
    sst_states::K4 # regenerated primitive k/omega faces, never checkpoint/rollback state
    sst_diffusion::K4 # static (lambda,Kx,Ky) at each cell's E/N face
    sst_gradients::K4 # raw unlimited (x,y) gradients of (k,omega)
    sst_diffusion_terms::K4 # actual face (Ck,Cw,delta_k,delta_w), i/j orientations
    # The phase-bound retreat of _k_states! (15 Oct 2026, option H; memory/project_docs/_records/2026-10-15_recon_h/DESIGN.md): 0 the
    # discrete one (one prevfloat step, then halving; the default, bitwise the path before), 1 the continuous one
    # (CMeshSpeciesContinuousPhaseRetreatV1). Fixed for the workspace's lifetime and bound in the cache configuration.
    retreat::Int32
end
function _build_reconstruction(s;retreat::Int32=Int32(0))
    retreat in (Int32(0),Int32(1)) || throw(ArgumentError("reconstruction retreat must be 0 (halving) or 1 (continuous)"))
    # One host copy of each authoritative face metric; workspace has no owner cycle.
    metrics=(;NI=s.NI,NJ=s.NJ,Six=Array(s.Six),Siy=Array(s.Siy),Sjx=Array(s.Sjx),Sjy=Array(s.Sjy))
    diffusion=s.sst ? zeros(Float32,s.NI,s.NJ,3,2) : nothing
    host=build_reconstruction_geometry(metrics;diffusion)
    dev(a)=KA.get_backend(s.rho)==KA.CPU() ? copy(a) : S._dev(s.backend,a)
    geometry=map(dev,host)
    states=fill!(similar(s.rho,s.NI,s.NJ,7,4),0f0)
    theta=fill!(similar(s.rho,s.NI,s.NJ),0f0)
    sst_states=s.sst ? fill!(similar(s.rho,s.NI,s.NJ,2,4),0f0) : nothing
    sst_diffusion=s.sst ? dev(diffusion) : nothing
    sst_gradients=s.sst ? fill!(similar(s.rho,s.NI,s.NJ,2,2),0f0) : nothing
    sst_diffusion_terms=s.sst ? fill!(similar(s.rho,s.NI+1,s.NJ+1,4,2),0f0) : nothing
    SpeciesReconstruction(geometry,states,theta,similar(theta),sst_states,
                          sst_diffusion,sst_gradients,sst_diffusion_terms,retreat)
end
@inline function _z(U,ux,uy,i,j,g)
    # Phase densities come from the authoritative Float64 conserved state.
    # Velocities remain the shared Float32 primitive cache used by the fluxes.
    @inbounds (U[i,j,1],U[i,j,4],ux[i+g,j+g],uy[i+g,j+g])
end
@inline _neighbor(i,j,f)=f==1 ? (i-1,j) : f==2 ? (i+1,j) : f==3 ? (i,j-1) : (i,j+1)
@inline function _gradient(U,ux,uy,weights,z,i,j,g,::Val{K},component,NI,NJ) where K
    v=zero(z[K])
    for f in 1:4
        ii,jj=_neighbor(i,j,f)
        if 1<=ii<=NI && 1<=jj<=NJ
            n=_z(U,ux,uy,ii,jj,g)
            @inbounds v+=weights[i,j,component,f]*(n[K]-z[K])
        end
    end
    v
end
@inline function _face(z,gx,gy,dx,dy,theta,p0,c0,T0,ep,Tmin,Tmax,single)
    # Explicit components keep the two Float64 phase and two Float32 velocity
    # paths type-stable; there is no dynamic indexing of a mixed-precision tuple.
    n=(z[1]+theta[1]*(gx[1]*dx+gy[1]*dy),
       z[2]+theta[2]*(gx[2]*dx+gy[2]*dy),
       z[3]+theta[3]*(gx[3]*dx+gy[3]*dy),
       z[4]+theta[4]*(gx[4]*dx+gy[4]*dy))
    r,q,u,v=n
    phase_good=isfinite(r) && r>0f0 && isfinite(q) && 0f0<=q<=r
    good=phase_good && isfinite(u) && isfinite(v) && isfinite(u*u+v*v)
    p=Float64(p0);c=Float64(c0);T=Float64(T0)
    if good && !(isequal(r,z[1]) && isequal(q,z[2]))
        pv,A,rl,rv,psm,lq,af=ep
        if single==Int32(1)
            good,pf32,cf32=P._phase_state_single_y(r,q,ep)
            p=Float64(pf32);c=Float64(cf32)
        else
            a=P._alpha(q/r,rl,rv)
            good,pf,cf=P._phase_state(r,a,pv,A,rl,rv,psm,af,lq)
            p=Float64(pf);c=Float64(cf)
        end
        T=clamp(p/(r*Float64(S.CS2)),Float64(Tmin),Float64(Tmax))
    end
    good=good && isfinite(p) && p>0f0 && isfinite(c) && c>0f0 && isfinite(T) && T>0f0
    # EOS arithmetic uses the precise phase state before this single face cast.
    # Unchanged phase values reuse the precise-state p/c held in Float32 caches.
    face=(Float32(r),Float32(u),Float32(v),Float32(T),Float32(p),Float32(c),Float32(q/r))
    rf,uf,vf,Tf,pf,cf,Yf=face
    good=good && isfinite(rf) && rf>0f0 && isfinite(uf) && isfinite(vf) &&
         isfinite(uf*uf+vf*vf) && isfinite(Tf) && Tf>0f0 &&
         isfinite(pf) && pf>0f0 && isfinite(cf) && cf>0f0 && isfinite(Yf) && 0f0<=Yf<=1f0
    (face,good,phase_good)
end
# Scalar arguments avoid capturing the reassigned gradient tuples in a GPU closure.
@inline function _component_limit(U,ux,uy,offsets,z,gx,gy,i,j,g,::Val{K},NI,NJ) where K
    tk=one(z[K])
    lo=z[K];hi=z[K]
    for f in 1:4
        ii,jj=_neighbor(i,j,f)
        if 1<=ii<=NI && 1<=jj<=NJ
            n=_z(U,ux,uy,ii,jj,g)
            lo=min(lo,n[K]);hi=max(hi,n[K])
        end
    end
    for f in 1:4
        d=gx*offsets[i,j,1,f]+gy*offsets[i,j,2,f]
        if d>0f0
            tk=min(tk,(hi-z[K])/d)
        elseif d<0f0
            tk=min(tk,(lo-z[K])/d)
        end
    end
    clamp(tk,zero(tk),one(tk))
end
# Continuous phase-bound retreat (option H, 15 Oct 2026; memory/project_docs/_records/2026-10-15_recon_h/DESIGN.md). The common phase theta
# t moves each face's phase densities linearly, r_f = r + t s_r and q_f = q + t s_q, with s = gx dx + gy dy formed exactly as
# _face forms it. Each phase bound _face enforces holds at t = 0 (the cell's own admitted state) and is linear in t, so its
# admissible set is an interval [0, t_c]:
#   r_f > 0       t_c = r/(-s_r)            where s_r < 0,
#   q_f >= 0      t_c = q/(-s_q)            where s_q < 0,
#   q_f <= r_f    t_c = (r - q)/(s_q - s_r) where s_q > s_r.
# Each numerator is reduced by a rounding allowance of 16 eps times the magnitudes of the terms (at t0, which bounds t), so the
# face _face then evaluates in Float64 -- with separate roundings or a fused multiply-add -- meets the bound. The slope terms
# enter the allowance as t0 (|gx dx| + |gy dy|), not t0 |s|, so a cancellation inside s (formed or contracted differently at
# the two sites) cannot defeat it (review_numerics F4). The result is the smallest t_c over the four faces and three bounds, at
# most t0 (the theta it is given), and at least 0. It is a continuous function of the state wherever 0 < q < r; near q = r the
# allowance (at least 32 eps r) is a band of width about 1e-12 in 1 - Y where t* falls below the exact bound, to 0 at
# 1 - Y ~ 1e-14 (DESIGN.md section 6). A non-finite result gives 0, the constant (admitted) state.
@inline function _phase_bound_theta(z,gx,gy,offsets,i,j,t0)
    t=t0;e=16*eps(typeof(t0))
    @inbounds for f in 1:4
        dx=offsets[i,j,1,f];dy=offsets[i,j,2,f]
        sr=gx[1]*dx+gy[1]*dy;sq=gx[2]*dx+gy[2]*dy
        ar=t0*(abs(gx[1]*dx)+abs(gy[1]*dy));aq=t0*(abs(gx[2]*dx)+abs(gy[2]*dy))
        if sr<zero(sr)
            t=min(t,(z[1]-e*(z[1]+ar))/(-sr))
        end
        if sq<zero(sq)
            t=min(t,(z[2]-e*(z[2]+aq))/(-sq))
        end
        d=sq-sr
        if d>zero(d)
            t=min(t,((z[1]-z[2])-e*(z[1]+z[2]+ar+aq))/d)
        end
    end
    isfinite(t) && t>zero(t) ? t : zero(t)
end
# `retreat` (15 Oct 2026): 0 the discrete phase retreat (one prevfloat step, then the coarse halving; bitwise the path before),
# 1 the continuous one: the same prevfloat step first, then, if a phase bound still fails, _phase_bound_theta before the halving.
# So the option differs from 0 only in cells where 0 would halve, and is never below 0's first step (review_numerics F3). The
# halving stays the fallback for a face the phase bounds do not cover, the EOS admission, and a face still failing after it.
@kernel function _k_states!(states,theta,status,geometry,@Const(U),@Const(ux),@Const(uy),
                            @Const(T),@Const(p),@Const(am),ep,Tmin::Float32,Tmax::Float32,single::Int32,
                            NI::Int32,NJ::Int32,g::Int32,retreat::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        z=_z(U,ux,uy,i,j,g)
        gx=(0.0,0.0,0f0,0f0);gy=gx;th=gx
        if geometry.active[i,j]!=0x00
            gx=(_gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(1),1,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(2),1,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(3),1,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(4),1,NI,NJ))
            gy=(_gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(1),2,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(2),2,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(3),2,NI,NJ),
                _gradient(U,ux,uy,geometry.weights,z,i,j,g,Val(4),2,NI,NJ))
            # Phase densities share a limiter to retain their affine contact
            # relation. Each velocity has its own limiter: roundoff in an
            # otherwise constant velocity must not control phase diffusion.
            limits=(
                _component_limit(U,ux,uy,geometry.offsets,z,gx[1],gy[1],i,j,g,Val(1),NI,NJ),
                _component_limit(U,ux,uy,geometry.offsets,z,gx[2],gy[2],i,j,g,Val(2),NI,NJ),
                _component_limit(U,ux,uy,geometry.offsets,z,gx[3],gy[3],i,j,g,Val(3),NI,NJ),
                _component_limit(U,ux,uy,geometry.offsets,z,gx[4],gy[4],i,j,g,Val(4),NI,NJ))
            phase=min(limits[1],limits[2])
            th=(phase,phase,limits[3],limits[4])
        end
        good=false
        # Bound/EOS reduction is shared across all components and cell faces.
        # Last attempt is the already admitted constant state, never a floor.
        attempt=0;retreated=false;closed=false
        while attempt<=8
            attempt==8 && (th=(0.0,0.0,0f0,0f0))
            good=true;phase_bad=false
            for f in 1:4
                face,ok,phase_good=_face(z,gx,gy,geometry.offsets[i,j,1,f],geometry.offsets[i,j,2,f],th,
                    p[i+g,j+g],am[i+g,j+g],T[i+g,j+g],ep,Tmin,Tmax,single)
                good=good && ok
                phase_bad=phase_bad || !phase_good
                for k in 1:7;states[i,j,k,f]=face[k];end
            end
            good && break
            # A rounded active phase bound can cross zero under contraction.
            # Try one inward common-phase step before the existing coarse half.
            # Every face is checked again; velocity limiters remain unchanged.
            if phase_bad && !retreated && th[1]>0.0
                phase=prevfloat(th[1])
                th=(phase,phase,th[3],th[4])
                retreated=true
                continue
            end
            # retreat 1 (option H): once, where the prevfloat step left a phase bound failing, the largest common phase theta
            # meeting every face's phase bounds, in closed form, below the theta just tried; velocity limiters unchanged.
            if retreat==Int32(1) && phase_bad && retreated && !closed && th[1]>0.0
                closed=true
                phase=_phase_bound_theta(z,gx,gy,geometry.offsets,i,j,th[1])
                if phase<th[1]
                    th=(phase,phase,th[3],th[4])
                    continue
                end
            end
            th=(.5f0*th[1],.5f0*th[2],.5f0*th[3],.5f0*th[4])
            attempt+=1
        end
        theta[i,j]=th[1];status[i,j]=good ? 0f0 : 1f0
    end
end
@inline _state(states,i,j,f)=@inbounds ntuple(k->states[i,j,k,f],Val(7))
@inline function _rate(l,r,rho,sx,sy,minf2)
    P._species_face_speed(l[1],l[2],l[3],l[6],l[5],r[1],r[2],r[3],r[6],r[5],rho,sx,sy,minf2)
end
@kernel function _k_cap!(cap,@Const(states),@Const(rho),@Const(ux),@Const(uy),@Const(am),@Const(p),
                        @Const(Fi),@Const(Fj),@Const(vol),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),
                        cfl,minf2,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        west=i>1 ? _rate(_state(states,i-1,j,2),_state(states,i,j,1),r,Six[i,j],Siy[i,j],minf2) :
            P._species_face_rate(rho,ux,uy,am,p,ia-1,ja,ia,ja,r,Six[i,j],Siy[i,j],minf2)
        east=i<NI ? _rate(_state(states,i,j,2),_state(states,i+1,j,1),r,Six[i+1,j],Siy[i+1,j],minf2) :
            P._species_face_rate(rho,ux,uy,am,p,ia,ja,ia+1,ja,r,Six[i+1,j],Siy[i+1,j],minf2)
        south=j>1 ? _rate(_state(states,i,j-1,4),_state(states,i,j,3),r,Sjx[i,j],Sjy[i,j],minf2) :
            P._species_face_rate(rho,ux,uy,am,p,ia,ja-1,ia,ja,r,Sjx[i,j],Sjy[i,j],minf2)
        north=j<NJ ? _rate(_state(states,i,j,4),_state(states,i,j+1,3),r,Sjx[i,j+1],Sjy[i,j+1],minf2) :
            P._species_face_rate(rho,ux,uy,am,p,ia,ja,ia,ja+1,r,Sjx[i,j+1],Sjy[i,j+1],minf2)
        outward=max(Fi[i+1,j],0f0)+max(-Fi[i,j],0f0)+max(Fj[i,j+1],0f0)+max(-Fj[i,j],0f0)
        rate=west+east+south+north+outward/r
        dt=cfl*vol[i,j]/rate
        cap[i,j]=isfinite(rate) && rate>0f0 && isfinite(dt) && dt>0f0 ? dt : 0f0
    end
end
function _reconstruct!(c,r;slot=nothing)
    s=c.solver;t=c.tem;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    # These are widened values of the already Float32 material references, not
    # a different physical calibration. Geometry and published face arrays stay Float32.
    # r.status is written in both modes (tests and diagnostics read it after a step);
    # a deferred owner (slot === :batched) reduces it in its next batch.
    _k_states!(s.backend)(r.states,r.theta,r.status,r.geometry,c.conserved.U,s.ux,s.uy,s.T,c.pressure,s.am,
        Float64.(s.outlet_eos),t.Tmin,t.Tmax,Int32(c.eos_single),ni,nj,g,r.retreat;ndrange=(s.NI,s.NJ))
    slot===nothing && (maximum(r.status)==0 || throw(ArgumentError("reconstruction rejected a cell/face state")))
    nothing
end
function _reconstructed_face_cap!(c,r,cfl;slot=nothing)
    s=c.solver
    a2=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2,1f-6))
    target=slot===nothing ? c.source_dt : slot     # deferred: the clock slot's bank slice
    _k_cap!(s.backend)(target,r.states,s.rho,s.ux,s.uy,s.am,c.pressure,s.Fi_r,s.Fj_r,s.vol,
        s.Six,s.Siy,s.Sjx,s.Sjy,cfl,minf2,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    cap=minimum(c.source_dt)
    isfinite(cap) && cap>0f0 || throw(ArgumentError("reconstructed face clock rejected"))
    cap
end

# The reconstructed residual is `_stage_residual!` (cmesh_cav_species.jl) since the rebuild of 22 Sep 2026: the
# projection fills the ghosts and projects the pressure; the residual reconstructs the faces on that state.
