"""Isolated conservative-interface gas C+H operator.

Uses the existing four-equation gas flux/energy kernels, with explicit paired
ghosts and one authoritative interface flux. No production default is changed.
SST, SFD, wall functions, mixtures and pseudo-time stepping are refused here.
The optional transparent boundary is a constant-state consistency test only.
"""
module GasStripPrototype
using KernelAbstractions
import ..CMeshSolverKA
const G=CMeshSolverKA
export GasPair, build_pair, exchange!, residual!, interface_audit, balance,
       wall_force, step!, inventory

# Opt-in j-line implicit residual smoothing in steady local pseudo-time only (21 Oct 2026). The C+H steady
# worker sets this when IRS-J is requested; both physical-time step! functions refuse a block with irsj set.
const STEADY_IRSJ=Ref(false)

mutable struct GasPair{S,C}
    c::S
    h::S
    ctx_c::C
    ctx_h::C
    wallmode::Int32
    elapsed::Float64
    iterations::Int
end

"Cheap invariants checked before every operator/stage; geometry is admitted at build."
function admit_settings(c,h,wallmode;allow_sst::Bool=false)
    wallmode in (0,2) || throw(ArgumentError("Prototype supports no-slip (2) or transparent consistency test (0)"))
    c.NI==2c.NIW+c.NIF && h.NI==c.NIW && c.NIW>=c.g && h.NIW==h.NI && h.NIF==0 && h.NJ>=c.g && c.NJ>=c.g && c.g==h.g || throw(ArgumentError("Invalid C+H layout or fewer wake stations than ghost layers"))
    c.backend==h.backend && typeof(c)==typeof(h) || throw(ArgumentError("Block backend/storage mismatch"))
    c.outer_mode==0 && h.outer_mode==2 || throw(ArgumentError("Require free-air C block and real strip H carrier"))
    for s in (c,h)
        s.cav==0 && s.ff_mix==0 && s.outlet_mode==0 && s.emix==0 || throw(ArgumentError("Gas prototype requires the pure thermal gas energy closure; no mixture/pressure outlet"))
        (allow_sst ? s.sst : !s.sst) && s.sfd==0 && (s.irsj==0 || (STEADY_IRSJ[] && s.irsj==1)) && s.irs_ac==0 && s.acou==0 && s.sponge_sigma==0 && s.gam==0 || throw(ArgumentError("Paired SST/filter/pseudo-time coupling is not admitted by this entry point"))
        s.wallfun==0 && s.earsm==0 && s.characteristic_i_outlet==0 && s.air_sutherland==0 || throw(ArgumentError("Wall functions/EARSM/characteristic i outlet/dynamic Sutherland are not admitted"))
        s.recon in (0,1) && s.fullvisc in (0,1) && isfinite(s.nu_lam) && s.nu_lam>=0 && isfinite(s.eint) && s.eint>0 || throw(ArgumentError("Invalid gas reconstruction/stress/energy settings"))
        all(isfinite,(s.drot,s.eint_v,s.latent,s.rho_inf,s.uxi,s.uyi,s.T_inf,s.suth)) && s.drot>=0 && s.rho_inf>0 && s.T_inf>0 || throw(ArgumentError("Nonfinite/nonphysical freestream or energy parameter"))
        s.eint==0.5f0*(3.0f0+s.drot)*G.CS2 || throw(ArgumentError("Thermal energy closure differs from native gas flux heat capacity"))
    end
    for key in (:drot,:eint,:eint_v,:latent,:rho_inf,:uxi,:uyi,:T_inf,:nu_lam,:fullvisc,:recon,:suth,:air_sutherland)
        isequal(getfield(c,key),getfield(h,key)) || throw(ArgumentError("Block physics mismatch: $key"))
    end
    nothing
end

function tables(c,h)
    nw,nf,ns=c.NIW,c.NIF,h.NJ
    x,y,v,sx,sy=map(Array,(c.xc,c.yc,c.vol,c.Sjx,c.Sjy))
    X,Y,V,SX,SY=map(Array,(h.xc,h.yc,h.vol,h.Sjx,h.Sjy))
    cc=zeros(Float32,nw,2,2);ch=similar(cc);hc=zeros(Float32,nw,2);hh=similar(hc)
    height(a,b,d)=a/sqrt(b*b+d*d)
    for k in 1:nw
        il=nw+1-k;iu=nw+nf+k
        cc[k,1,1]=X[k,1];cc[k,2,1]=Y[k,1];cc[k,1,2]=X[k,ns];cc[k,2,2]=Y[k,ns]
        ch[k,1,1]=x[il,1];ch[k,2,1]=y[il,1];ch[k,1,2]=x[iu,1];ch[k,2,2]=y[iu,1]
        hc[k,1]=height(V[k,1],SX[k,1],SY[k,1]);hc[k,2]=height(V[k,ns],SX[k,end],SY[k,end])
        hh[k,1]=height(v[il,1],sx[il,1],sy[il,1]);hh[k,2]=height(v[iu,1],sx[iu,1],sy[iu,1])
        SX[k,1]==-sx[il,1] && SY[k,1]==-sy[il,1] || error("Nonconforming lower C+H face $k")
        SX[k,end]===sx[iu,1] && SY[k,end]===sy[iu,1] || error("Nonconforming upper C+H face $k")
    end
    dev(a)=copyto!(similar(c.rho,Float32,size(a)...),a)
    map(dev,(cc,ch,hc,hh))
end

function build_pair(c::G.KSol,h::G.KSol;wallmode::Integer=2)
    admit_settings(c,h,wallmode)
    for s in (c,h)
        all(isfinite,s.vol) && minimum(s.vol)>0 || throw(ArgumentError("Nonpositive/nonfinite cell volume"))
    end
    cc,ch,hc,hh=tables(c,h)
    ctx(s,block,nb,hnb)=(block=Int32(block),iwall=Int32(block==1 && wallmode==2),sides=Int32(block==1 ? 2 : 3),wallmode=Int32(wallmode),NS=Int32(h.NJ),c_nb=nb,h_nb=hnb,vol=s.vol,NIW=Int32(s.NIW),NIF=Int32(s.NIF))
    GasPair(c,h,ctx(c,0,cc,hc),ctx(h,1,ch,hh),Int32(wallmode),0.0,0)
end

@kernel function column_bc!(r,u,v,t,NI::Int32,NJ::Int32,g::Int32,wall::Int32)
    j,k=@index(Global,NTuple)
    @inbounds if j<=NJ+2g && k<=g
        il=g+1-k;ir=NI+g+k;src=g+k;out=NI+g
        r[il,j]=r[src,j];t[il,j]=t[src,j]
        u[il,j]=wall==2 ? -u[src,j] : u[src,j]
        v[il,j]=wall==2 ? -v[src,j] : v[src,j]
        r[ir,j]=r[out,j];u[ir,j]=u[out,j];v[ir,j]=v[out,j];t[ir,j]=t[out,j]
    end
end

@inline function copycell!(rd,ud,vd,td,i,j,rs,us,vs,ts,ii,jj,reflect)
    @inbounds begin
        rd[i,j]=rs[ii,jj];td[i,j]=ts[ii,jj]
        ud[i,j]=reflect ? -us[ii,jj] : us[ii,jj]
        vd[i,j]=reflect ? -vs[ii,jj] : vs[ii,jj]
    end
end

@kernel function exchange_kernel!(rc,uc,vc,tc,rh,uh,vh,th,NW::Int32,NF::Int32,NS::Int32,g::Int32,wall::Int32)
    st,k=@index(Global,NTuple)
    @inbounds if st<=NW && k<=g
        s=Int32(st);q=Int32(k);il=NW+1-s+g;iu=NW+NF+s+g;is=s+g
        ghost=g+1-q;own=g+q;up=NS+g+1-q;top=NS+g+q
        copycell!(rc,uc,vc,tc,il,ghost,rh,uh,vh,th,is,own,false)
        copycell!(rc,uc,vc,tc,iu,ghost,rh,uh,vh,th,is,up,false)
        copycell!(rh,uh,vh,th,is,ghost,rc,uc,vc,tc,il,own,false)
        copycell!(rh,uh,vh,th,is,top,rc,uc,vc,tc,iu,own,false)
        if s<=g
            ic=g+1-s
            copycell!(rh,uh,vh,th,ic,ghost,rc,uc,vc,tc,il,own,wall==2)
            copycell!(rh,uh,vh,th,ic,top,rc,uc,vc,tc,iu,own,wall==2)
        end
    end
end

function exchange!(p::GasPair)
    c,h=p.c,p.h;g=Int32(c.g)
    G.applyBC_ka!(c,p.wallmode)
    column_bc!(h.backend)(h.rho,h.ux,h.uy,h.T,Int32(h.NI),Int32(h.NJ),g,p.wallmode;ndrange=(h.NJ+2h.g,h.g))
    exchange_kernel!(c.backend)(c.rho,c.ux,c.uy,c.T,h.rho,h.ux,h.uy,h.T,Int32(c.NIW),Int32(c.NIF),Int32(h.NJ),g,p.wallmode;ndrange=(c.NIW,c.g))
    nothing
end

@inline copyfaces!(::Tuple{},::Tuple{},st,il,iu,jt)=nothing
@inline function copyfaces!(a::Tuple,b::Tuple,st,il,iu,jt)
    @inbounds begin first(a)[il,1]=-first(b)[st,1];first(a)[iu,1]=first(b)[st,jt];end
    copyfaces!(Base.tail(a),Base.tail(b),st,il,iu,jt)
end
@kernel function common_flux!(a,b,NW::Int32,NF::Int32,NS::Int32)
    k=@index(Global,Linear)
    @inbounds if k<=NW
        st=Int32(k);copyfaces!(a,b,st,NW+1-st,NW+NF+st,NS+1)
    end
end

faces_i(s)=(s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e)
faces_j(s)=(s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e)
residuals(s)=(s.dc_r,s.dc_mx,s.dc_my,s.dc_e)

function residual!(p::GasPair)
    admit_settings(p.c,p.h,p.wallmode)
    exchange!(p);c,h=p.c,p.h
    G._raw_residual_body_ka!(c,p.wallmode,Int32(p.wallmode==2),false,nothing;strip_ctx=p.ctx_c,store_divergence=false)
    G._raw_residual_body_ka!(h,p.wallmode,Int32(0),false,nothing;strip_ctx=p.ctx_h,outlet_sides=Int32(2),store_divergence=false)
    common_flux!(c.backend)(faces_j(c),faces_j(h),Int32(c.NIW),Int32(c.NIF),Int32(h.NJ);ndrange=c.NIW)
    for s in (c,h)
        G._k_residual!(s.backend)(residuals(s)...,faces_i(s)...,faces_j(s)...,Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    end
    nothing
end

function interface_audit(p::GasPair)
    c,h=p.c,p.h;nw,nf,ns=c.NIW,c.NIF,h.NJ
    fc=map(Array,faces_j(c));fh=map(Array,faces_j(h))
    lower=[maximum(abs(Float64(a[nw+1-k,1])+Float64(b[k,1])) for k in 1:nw) for (a,b) in zip(fc,fh)]
    upper=[maximum(abs(Float64(a[nw+nf+k,1])-Float64(b[k,ns+1])) for k in 1:nw) for (a,b) in zip(fc,fh)]
    (;lower,upper,exact=all(iszero,vcat(lower,upper)))
end

function balance(p::GasPair)
    per=map((:mass,:momentum_x,:momentum_y,:energy)) do name
        k=findfirst(==(name),(:mass,:momentum_x,:momentum_y,:energy));div=0.;boundary=0.;stored=0.
        for s in (p.c,p.h)
            fi=Float64.(Array(faces_i(s)[k]));fj=Float64.(Array(faces_j(s)[k]))
            div+=sum(fi[2:end,:].-fi[1:end-1,:].+fj[:,2:end].-fj[:,1:end-1])
            boundary+=sum(fi[end,:])-sum(fi[1,:])+sum(fj[:,end])-sum(fj[:,1])
            stored+=sum(Float64.(Array(residuals(s)[k])))
        end
        (;equation=name,face64_divergence=div,external_flux=boundary,face64_telescoping_error=div-boundary,float32_residual_sum=stored,float32_storage_difference=stored-div)
    end
    (;equations=per,interface=interface_audit(p))
end

function wall_force(p::GasPair)
    p.wallmode==2 || throw(ArgumentError("Transparent consistency test has no physical wall load"))
    c,h=p.c,p.h;r=c.NIW+1:c.NIW+c.NIF
    x=-sum(Float64.(Array(c.Fj_mx)[r,1]))-sum(Float64.(Array(h.Fi_mx)[1,:]))
    y=-sum(Float64.(Array(c.Fj_my)[r,1]))-sum(Float64.(Array(h.Fi_my)[1,:]))
    (;x,y,base_x=-sum(Float64.(Array(h.Fi_mx)[1,:])),base_y=-sum(Float64.(Array(h.Fi_my)[1,:])))
end

function inventory(p::GasPair)
    result=zeros(Float64,4)
    for s in (p.c,p.h)
        g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ
        rho,u,v,t=map(a->Float64.(Array(a)[ir,jr]),(s.rho,s.ux,s.uy,s.T));vol=Float64.(Array(s.vol))
        fields=(rho,rho .* u,rho .* v,rho .* (Float64(s.eint) .* t .+ 0.5 .* (u .^ 2 .+ v .^ 2)))
        for k in 1:4;result[k]+=sum(fields[k].*vol);end
    end
    result
end

function snapshot!(s)
    G._k_snapshot!(s.backend)(s.rk_r,s.rk_mx,s.rk_my,s.rk_E,s.rho,s.ux,s.uy,s.T,s.eint,s.yvf,s.emix,s.eint_v,s.latent,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
end
function update!(s,w,b)
    G._k_update!(s.backend)(s.rho,s.ux,s.uy,s.T,s.rk_r,s.rk_mx,s.rk_my,s.rk_E,
        residuals(s)...,s.un_r,s.un_mx,s.un_my,s.un_E,s.um_r,s.um_mx,s.um_my,s.um_E,
        residuals(s)...,s.dtf,s.vol,s.xc,s.yc,s.eint,s.yvf,s.emix,s.eint_v,s.latent,
        w,b,0.0f0,Int32(0),Int32(1),0.0f0,s.sponge_j0f,s.gam,s.xv,s.yv,s.m2,s.beta,s.ca,s.sa,
        s.uxi,s.uyi,s.rho_inf,s.T_inf,Int32(0),s.rho_inf,s.uxi,s.uyi,s.T_inf,Int32(0),
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
end

"One shared physical SSPRK3 clock; laminar four-equation prototype only."
function step!(p::GasPair,cfl::Real;max_dt::Real=Inf,stage_observer=nothing)
    isfinite(cfl) && 0<cfl<=.5 && max_dt>0 || throw(ArgumentError("Invalid physical step limit"))
    p.c.irsj==0 && p.h.irsj==0 || throw(ArgumentError("IRS-J is a steady pseudo-time option; the physical step refuses it"))
    admit_settings(p.c,p.h,p.wallmode)
    c,h=p.c,p.h;backup=map(s->map(copy,(s.rho,s.ux,s.uy,s.T)),(c,h));previous_dt=(c.last_global_dt,h.last_global_dt)
    try
        for s in (c,h);G.prepare_dt_ka!(s,Float32(cfl));end
        dt=min(Float64(minimum(c.dtf)),Float64(minimum(h.dtf)),Float64(max_dt))
        dt32=Float32(dt);Float64(dt32)>dt && (dt32=prevfloat(dt32))
        isfinite(dt32) && dt32>0 || error("Physical timestep cannot advance")
        for s in (c,h);fill!(s.dtf,dt32);snapshot!(s);end
        for (stage,(w,b)) in enumerate(((1.0f0,0.0f0),(.25f0,.75f0),(2.0f0/3.0f0,1.0f0-2.0f0/3.0f0)))
            residual!(p)
            stage_observer===nothing || stage_observer(p,stage)
            for s in (c,h);update!(s,w,b);end
            for s in (c,h)
                all(isfinite,s.rho) && all(isfinite,s.ux) && all(isfinite,s.uy) && all(isfinite,s.T) || error("Nonfinite gas stage")
                ir=s.g+1:s.g+s.NI;jr=s.g+1:s.g+s.NJ
                minimum(s.rho)>1f-4 && minimum(s.T)>1f-3 && maximum(hypot.(@view(s.ux[ir,jr]),@view(s.uy[ir,jr])))<3.49f0 || error("Gas safeguard activated; trial refused")
            end
        end
        residual!(p)
        for s in (c,h);s.last_global_dt=dt32;end
        p.elapsed+=Float64(dt32);p.iterations+=1
        return Float64(dt32)
    catch
        for (s,b) in zip((c,h),backup), (a,z) in zip((s.rho,s.ux,s.uy,s.T),b);copyto!(a,z);end
        c.last_global_dt,h.last_global_dt=previous_dt
        rethrow()
    end
end
end
