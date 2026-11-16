# The pseudo-transient Newton-Krylov airfoil march (cmesh_steady_newton_ka.jl) on a C+H mesh, the topology of a blunt
# trailing edge. EXPERIMENTAL AND OPT-IN: no default solver path calls it. Included into module CMeshSolverKA.
# CPU, Float64, plain SST, the same requirements as the single-block march.
#
# Two standard blocks:
#   C  the C block (NIW wake stations, NIF surface cells). Its two wake rows j = 1 are the legs: their j = 1 faces are
#      interfaces with the strip instead of the wake cut of a sharp mesh.
#   H  the strip, stored with the BASE AS ITS WALL: i runs across the strip from the upper leg to the lower leg
#      (NS cells), j runs downstream from the base (NW cells). In this layout it is an ordinary block with NIW = 0 and
#      NIF = NS, so the wall flux, the wall loads, the SST wall rows, the wall distance and the far boundary (the
#      outlet plane) are the single-block code. Its two i ends are the interfaces. newton_pair_strip_nodes converts
#      the generators' strip (row 1 on the lower leg, column 1 on the base) to this layout.
# Coupling: ghost cells of each block hold the other block's cells; the reconstruction tables of the interface cells
# carry the true neighbour geometry; the interface flux is computed once (from the C side) and written to both blocks;
# the linear systems carry the cross-block terms explicitly. Preconditioners: the line relaxation and two-level cycle
# of the single-block march carried over to two blocks (wake lines run through the strip; the coarse level holds both
# blocks). Entry points: newton_pair_build, newton_pair_init, newton_pair_step!, newton_pair_march!,
# newton_pair_cold_run. Development results only: not validated against reference data.

struct NewtonPairSSTGeo{A};xxi::A;yxi::A;xet::A;yet::A;dE::A;dW::A;dN::A;dS::A;end

mutable struct NewtonPair
    c::KSol;h::KSol;NW::Int;NS::Int
    sign::NTuple{2,Float64}                  # H i-face flux = sign x C j-face flux, (lower leg, upper leg)
    dist::Matrix{Float64}                    # centre distance across the interface, [station, side]
    gc::NewtonPairSSTGeo;gh::NewtonPairSSTGeo
end

"C cell (i, 1) and strip cell (i, k) on the two sides of the interface at wake station k; side 1 = lower leg, 2 = upper leg."
@inline _newton_pair_cells(p::NewtonPair,k,side)=side==1 ? (p.NW+1-k,p.NS) : (p.NW+p.c.NIF+k,1)

"Nodes of the strip in the base-as-wall layout from the mesher's strip (row 1 on the lower leg, column 1 on the base)."
newton_pair_strip_nodes(strip)=(X=permutedims(strip.X[:,end:-1:1]),Y=permutedims(strip.Y[:,end:-1:1]),NIW=0,NIF=strip.NJ,NI=strip.NJ,NJ=strip.NI)

function _newton_pair_sst_geo(s::KSol)
    NI,NJ,NIW,NIF=s.NI,s.NJ,s.NIW,s.NIF;xc,yc,vol,wd=s.xc,s.yc,s.vol,s.wdist
    A()=fill!(similar(s.dtf),0);q=NewtonPairSSTGeo(A(),A(),A(),A(),A(),A(),A(),A())
    for j in 1:NJ,i in 1:NI
        q.xxi[i,j],q.yxi[i,j],q.xet[i,j],q.yet[i,j]=_sst_index_metrics(xc,yc,s.Sjx,s.Sjy,wd,i,j,NI,NJ,NIW,NIF,Int32(1))
        SfE=hypot(s.Six[i+1,j],s.Siy[i+1,j]);SfW=hypot(s.Six[i,j],s.Siy[i,j]);SfN=hypot(s.Sjx[i,j+1],s.Sjy[i,j+1]);SfS=hypot(s.Sjx[i,j],s.Sjy[i,j])
        q.dE[i,j]=i+1<=NI ? hypot(xc[i+1,j]-xc[i,j],yc[i+1,j]-yc[i,j])+1e-30 : vol[i,j]/SfE+1e-30
        q.dW[i,j]=i-1>=1 ? hypot(xc[i-1,j]-xc[i,j],yc[i-1,j]-yc[i,j])+1e-30 : vol[i,j]/SfW+1e-30
        q.dN[i,j]=j+1<=NJ ? hypot(xc[i,j+1]-xc[i,j],yc[i,j+1]-yc[i,j])+1e-30 : vol[i,j]/SfN+1e-30
        q.dS[i,j]=j>1 ? hypot(xc[i,j-1]-xc[i,j],yc[i,j-1]-yc[i,j])+1e-30 :
            (i<=NIW||i>NIW+NIF) ? hypot(xc[NI+1-i,1]-xc[i,1],yc[NI+1-i,1]-yc[i,1])+1e-30 : vol[i,j]/SfS+1e-30
    end
    q
end

# Point face f of cell (i, j) at the ghost cell beyond it with the true displacement (dx, dy) and redo the cell's
# least-squares weights (same formula as build_airfoil_muscl_geometry).
function _newton_pair_patch_muscl!(s::KSol,i,j,f,dx,dy)
    a=_AIRFOIL_MUSCL[s].geometry;g=s.g
    ii,jj=f==1 ? (i-1,j) : f==2 ? (i+1,j) : f==3 ? (i,j-1) : (i,j+1)
    a.neighbors[i,j,1,f]=ii+g;a.neighbors[i,j,2,f]=jj+g
    a.displacements[i,j,1,f]=dx;a.displacements[i,j,2,f]=dy
    d=[(Float64(a.displacements[i,j,1,m]),Float64(a.displacements[i,j,2,m])) for m in 1:4];w=[1/(x*x+y*y) for (x,y) in d]
    a11=sum(w[m]*d[m][1]^2 for m in 1:4);a12=sum(w[m]*d[m][1]*d[m][2] for m in 1:4);a22=sum(w[m]*d[m][2]^2 for m in 1:4)
    det=a11*a22-a12*a12;det>0||error("singular reconstruction stencil at an interface cell")
    for m in 1:4
        a.weights[i,j,1,m]=(a22*d[m][1]-a12*d[m][2])*w[m]/det;a.weights[i,j,2,m]=(a11*d[m][2]-a12*d[m][1])*w[m]/det
    end
end

"""
    newton_pair_build(c, h, cn, hn) -> NewtonPair

`c`, `h`: prepared solvers of the C block and of the strip in the base-as-wall layout (both with the airfoil MUSCL
workspace set); `cn`, `hn`: their node sets. Checks that the interfaces conform, sets the interface geometry of the
reconstruction and of the SST stencils, and fills the ghost cells.
"""
function newton_pair_build(c::KSol,h::KSol,cn,hn)
    NW=c.NIW;NS=h.NI;NIF=c.NIF;g=c.g
    h.NIW==0&&h.NIF==NS&&h.NJ==NW&&h.g==g&&NW>=g&&NS>=g||throw(ArgumentError("strip block does not match the C block"))
    eltype(c.rho)==Float64&&eltype(h.rho)==Float64||throw(ArgumentError("the pair march needs Float64 blocks"))
    for k in 1:NW+1
        (hn.X[1,k],hn.Y[1,k])==(cn.X[NW+NIF+k,1],cn.Y[NW+NIF+k,1])&&(hn.X[NS+1,k],hn.Y[NS+1,k])==(cn.X[NW+2-k,1],cn.Y[NW+2-k,1])||
            error("strip edge node $k is not the C block's leg node")
    end
    sg=zeros(2);dist=zeros(NW,2)
    p=NewtonPair(c,h,NW,NS,(0.0,0.0),dist,_newton_pair_sst_geo(c),_newton_pair_sst_geo(h))
    for side in 1:2,k in 1:NW
        ic,ih=_newton_pair_cells(p,k,side);ihf=side==1 ? NS+1 : 1
        a=(c.Sjx[ic,1],c.Sjy[ic,1]);b=(h.Six[ihf,k],h.Siy[ihf,k]);s=sign(a[1]*b[1]+a[2]*b[2])
        hypot(b[1]-s*a[1],b[2]-s*a[2])<=1e-10*hypot(a...)||error("interface face $k of side $side does not conform")
        k==1 ? (sg[side]=s) : sg[side]==s||error("interface orientation changes along the leg")
        dx=h.xc[ih,k]-c.xc[ic,1];dy=h.yc[ih,k]-c.yc[ic,1];dist[k,side]=hypot(dx,dy)
        _newton_pair_patch_muscl!(c,ic,1,3,dx,dy);_newton_pair_patch_muscl!(h,ih,k,side==1 ? 2 : 1,-dx,-dy)
        # SST stencils: the ghost state is the other block's cell, so the metric span and the face distance use its centre
        p.gc.xet[ic,1]=0.5*(c.xc[ic,2]-h.xc[ih,k]);p.gc.yet[ic,1]=0.5*(c.yc[ic,2]-h.yc[ih,k]);p.gc.dS[ic,1]=dist[k,side]+1e-30
        if side==1
            p.gh.xxi[ih,k]=0.5*(c.xc[ic,1]-h.xc[ih-1,k]);p.gh.yxi[ih,k]=0.5*(c.yc[ic,1]-h.yc[ih-1,k]);p.gh.dE[ih,k]=dist[k,side]+1e-30
        else
            p.gh.xxi[ih,k]=0.5*(h.xc[ih+1,k]-c.xc[ic,1]);p.gh.yxi[ih,k]=0.5*(h.yc[ih+1,k]-c.yc[ic,1]);p.gh.dW[ih,k]=dist[k,side]+1e-30
        end
    end
    p.sign=(sg[1],sg[2])
    fill!(h.gammat,1);h.gam=c.gam
    newton_pair_bc!(p);newton_pair_sst_bc!(p)
    p
end

# ---- ghost cells ---------------------------------------------------------------------------------------
function _newton_pair_copy_ghosts!(p::NewtonPair,fc,fh)
    g=p.c.g;NW,NS,NIF=p.NW,p.NS,p.c.NIF
    @inbounds for q in 1:g,k in 1:NW
        il=NW+1-k;iu=NW+NIF+k
        fc[il+g,g+1-q]=fh[NS+1-q+g,k+g];fc[iu+g,g+1-q]=fh[q+g,k+g]
        fh[g+1-q,k+g]=fc[iu+g,q+g];fh[NS+q+g,k+g]=fc[il+g,q+g]
    end
end
function newton_pair_bc!(p::NewtonPair)
    applyBC_ka!(p.c,Int32(2));applyBC_ka!(p.h,Int32(2))
    for n in (:rho,:ux,:uy,:T);_newton_pair_copy_ghosts!(p,getproperty(p.c,n),getproperty(p.h,n));end
end
function newton_pair_sst_bc!(p::NewtonPair)
    _newton_sst_bc!(p.c);_newton_sst_bc!(p.h)
    for n in (:tk,:tom,:nut,:tnl);_newton_pair_copy_ghosts!(p,getproperty(p.c,n),getproperty(p.h,n));end
end

# ---- residual ------------------------------------------------------------------------------------------
# One flux per interface face: reconstructed states of both blocks through the same thermal face flux as every other
# face, thin-layer viscous terms between the two cell centres (as the public code does on the wake cut of a sharp mesh).
function newton_pair_interface_flux!(p::NewtonPair)
    c,h=p.c,p.h;g=c.g;NS=p.NS;sc=_AIRFOIL_MUSCL[c].states;sh=_AIRFOIL_MUSCL[h].states
    CS2=_gas_cs2(Float64);sst=c.sst;cpg=0.5*(5+c.drot)*CS2
    Threads.@threads for k in 1:p.NW
        @inbounds for side in 1:2
            ic,ih=_newton_pair_cells(p,k,side);fh=side==1 ? 2 : 1;ihf=side==1 ? NS+1 : 1
            nxr=c.Sjx[ic,1];nyr=c.Sjy[ic,1];S=sqrt(nxr*nxr+nyr*nyr);nx=nxr/S;ny=nyr/S
            l=(sh[ih,k,1,fh],sh[ih,k,2,fh],sh[ih,k,3,fh],sh[ih,k,4,fh]);r=(sc[ic,1,1,3],sc[ic,1,2,3],sc[ic,1,3,3],sc[ic,1,4,3])
            Fr,Fmx,Fmy,Fe=_airfoil_thermal_face_flux(l,r,nx,ny,S,c.cxa,c.cya,c.cza,c.wa,c.drot)
            if c.nu_lam>0
                rL=h.rho[ih+g,k+g];uL=h.ux[ih+g,k+g];vL=h.uy[ih+g,k+g];TL=h.T[ih+g,k+g]
                rR=c.rho[ic+g,1+g];uR=c.ux[ic+g,1+g];vR=c.uy[ic+g,1+g];TR=c.T[ic+g,1+g]
                dn=p.dist[k,side]+1e-30;nut_f=sst ? 0.5*(h.nut[ih+g,k+g]+c.nut[ic+g,1+g]) : 0.0
                rf=0.5*(rL+rR);fmu=_mufac(0.5*(TL+TR),c.suth)
                mu_eff=c.air_sutherland!=Int32(0) ? c.rho_inf*c.nu_lam*fmu+rf*nut_f : rf*(c.nu_lam*fmu+nut_f)
                k_eff=c.air_sutherland!=Int32(0) ? cpg*(c.rho_inf*c.nu_lam*fmu/0.71+rf*nut_f/0.9) : rf*cpg*(c.nu_lam*fmu/0.71+nut_f/0.9)
                mu=mu_eff*S/dn;du=uR-uL;dv=vR-vL;dT=TR-TL;ufx=0.5*(uL+uR);ufy=0.5*(vL+vR)
                Fmx-=mu*du;Fmy-=mu*dv;Fe-=mu*(ufx*du+ufy*dv)+k_eff*S/dn*dT
                if c.rhok==Int32(1)&&sst
                    kf=0.5*(h.tk[ih+g,k+g]+c.tk[ic+g,1+g])
                    Fmx+=2/3*rf*kf*nxr;Fmy+=2/3*rf*kf*nyr;Fe+=2/3*rf*kf*(ufx*nxr+ufy*nyr)
                end
            end
            c.Fj_r[ic,1]=Fr;c.Fj_mx[ic,1]=Fmx;c.Fj_my[ic,1]=Fmy;c.Fj_e[ic,1]=Fe
            s=p.sign[side];h.Fi_r[ihf,k]=s*Fr;h.Fi_mx[ihf,k]=s*Fmx;h.Fi_my[ihf,k]=s*Fmy;h.Fi_e[ihf,k]=s*Fe
        end
    end
end

"Steady flow residual of both blocks (dc_* of each), boundary and ghost cells refreshed first."
function newton_pair_residual!(p::NewtonPair)
    newton_pair_bc!(p)
    for s in (p.c,p.h);_raw_residual_body_ka!(s,Int32(2),Int32(1),false,nothing;store_divergence=false);end
    newton_pair_interface_flux!(p)
    for s in (p.c,p.h)
        _k_residual!(s.backend)(s.dc_r,s.dc_mx,s.dc_my,s.dc_e,s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e,s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,
            Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    end
    nothing
end
newton_pair_loads(p::NewtonPair)=(a=loads_ka(p.c);b=loads_ka(p.h);(Float64(a[1])+Float64(b[1]),Float64(a[2])+Float64(b[2])))
function newton_pair_flow_l2(p::NewtonPair)
    c=p.c;r=max(Float64(c.rho_inf),1e-12);u=max(hypot(Float64(c.uxi),Float64(c.uyi)),1e-3);e=max(Float64(c.eint*c.T_inf),u*u)
    scales=(r,r*u,r*u,r*e);n=length(p.c.vol)+length(p.h.vol)
    ntuple(k->sqrt(sum(mapreduce((a,v)->abs2(a/v),+,_airfoil_residual(s)[k],s.vol) for s in (p.c,p.h))/n)/scales[k],4)
end

# ---- implicit SST on the pair --------------------------------------------------------------------------
# The public coefficient kernel with the stencil geometry read from arrays (NewtonPairSSTGeo) instead of being formed from
# the block's own centres and its wake-cut partner; identical arithmetic away from the interfaces.
@kernel function _k_newton_pair_sst_coeffs!(akW,akE,akS,akN,awW,awE,awS,awN,dk,sk,dw,sw,sf2,
    @Const(tk),@Const(tom),@Const(nut),@Const(rho),@Const(T),@Const(ux),@Const(uy),
    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),@Const(wdist),@Const(gammat),
    @Const(mxxi),@Const(myxi),@Const(mxet),@Const(myet),@Const(dE),@Const(dW),@Const(dN),@Const(dS),
    nu_inf,rho_inf,suth,air_sutherland::Int32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        βs=0.09f0;a1=0.31f0
        σk1=0.85f0;σw1=0.5f0;β1=0.075f0;γ1=0.5532f0
        σk2=1.0f0;σw2=0.856f0;β2=0.0828f0;γ2=0.4403f0
        ia=i+g;ja=j+g
        nu=_air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        k0=tk[ia,ja];w0=max(tom[ia,ja],1.0f-6);d=max(wdist[i,j],1.0f-8)
        xxi=mxxi[i,j];yxi=myxi[i,j];xet=mxet[i,j];yet=myet[i,j]
        J=xxi*yet-xet*yxi;J=abs(J)<1.0f-30 ? oftype(J,1.0f-30) : J
        duxi=0.5f0*(ux[ia+1,ja]-ux[ia-1,ja]);duet=0.5f0*(ux[ia,ja+1]-ux[ia,ja-1])
        dvxi=0.5f0*(uy[ia+1,ja]-uy[ia-1,ja]);dvet=0.5f0*(uy[ia,ja+1]-uy[ia,ja-1])
        dudx=(duxi*yet-duet*yxi)/J;dudy=(-duxi*xet+duet*xxi)/J
        dvdx=(dvxi*yet-dvet*yxi)/J;dvdy=(-dvxi*xet+dvet*xxi)/J
        Sxy=0.5f0*(dudy+dvdx);S2=2.0f0*(dudx*dudx+dvdy*dvdy+2.0f0*Sxy*Sxy);Ssr=sqrt(S2)
        Ω=abs(dudy-dvdx)
        sqk=sqrt(max(k0,0.0f0))
        dkxi=0.5f0*(tk[ia+1,ja]-tk[ia-1,ja]);dket=0.5f0*(tk[ia,ja+1]-tk[ia,ja-1])
        dwxi=0.5f0*(tom[ia+1,ja]-tom[ia-1,ja]);dwet=0.5f0*(tom[ia,ja+1]-tom[ia,ja-1])
        dkdx=(dkxi*yet-dket*yxi)/J;dkdy=(-dkxi*xet+dket*xxi)/J
        dwdx=(dwxi*yet-dwet*yxi)/J;dwdy=(-dwxi*xet+dwet*xxi)/J
        gkw=dkdx*dwdx+dkdy*dwdy
        CDkw=max(2.0f0*σw2/w0*gkw,1.0f-10)
        arg1=min(max(sqk/(βs*w0*d),500.0f0*nu/(d*d*w0)),4.0f0*σw2*k0/(CDkw*d*d))
        arg1=min(arg1,4.0f0);F1=tanh(arg1^4)
        arg2=min(max(2.0f0*sqk/(βs*w0*d),500.0f0*nu/(d*d*w0)),4.0f0);F2=tanh(arg2^2)
        slim=SST_VARIANT==1 ? Ω : Ssr
        nut_t=a1*k0/max(a1*w0,slim*F2)
        pkcap=(SST_VARIANT==1 ? 20.0f0 : 10.0f0)*βs*k0*w0
        Pk=min(nut_t*(SST_VARIANT==1 ? S2 : Ssr*Ω),pkcap)*gammat[i,j]
        σk=F1*σk1+(1.0f0-F1)*σk2;σw=F1*σw1+(1.0f0-F1)*σw2
        βc=F1*β1+(1.0f0-F1)*β2;γc=F1*γ1+(1.0f0-F1)*γ2
        iv=1/vol[i,j]
        SfE=hypot(Six[i+1,j],Siy[i+1,j]);nfE=0.5f0*(nut[ia,ja]+nut[ia+1,ja])
        SfW=hypot(Six[i,j],Siy[i,j]);nfW=0.5f0*(nut[ia,ja]+nut[ia-1,ja])
        SfN=hypot(Sjx[i,j+1],Sjy[i,j+1]);nfN=0.5f0*(nut[ia,ja]+nut[ia,ja+1])
        SfS=hypot(Sjx[i,j],Sjy[i,j]);nfS=0.5f0*(nut[ia,ja]+nut[ia,ja-1])
        u0=ux[ia,ja];v0=uy[ia,ja]
        Uci=(u0*0.5f0*(Six[i,j]+Six[i+1,j])+v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))*iv
        Ucj=(u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1])+v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))*iv
        cW=max(Uci,zero(Uci));cE=max(-Uci,zero(Uci));cS=max(Ucj,zero(Ucj));cN=max(-Ucj,zero(Ucj))
        akE[i,j]=(nu+σk*nfE)*SfE/dE[i,j]*iv+cE;awE[i,j]=(nu+σw*nfE)*SfE/dE[i,j]*iv+cE
        akW[i,j]=(nu+σk*nfW)*SfW/dW[i,j]*iv+cW;awW[i,j]=(nu+σw*nfW)*SfW/dW[i,j]*iv+cW
        akN[i,j]=(nu+σk*nfN)*SfN/dN[i,j]*iv+cN;awN[i,j]=(nu+σw*nfN)*SfN/dN[i,j]*iv+cN
        akS[i,j]=(nu+σk*nfS)*SfS/dS[i,j]*iv+cS;awS[i,j]=(nu+σw*nfS)*SfS/dS[i,j]*iv+cS
        dk[i,j]=βs*w0;sk[i,j]=Pk
        cross=2.0f0*(1.0f0-F1)*σw2/w0*gkw
        prodw=γc/max(nut_t,1.0f-10)*Pk
        dw[i,j]=2.0f0*βc*w0+(cross<0 ? -cross/w0 : zero(cross))
        sw[i,j]=prodw+βc*w0*w0+(cross>0 ? cross : zero(cross))
        sf2[i,j]=slim*F2
    end
end
function newton_pair_sst_coeffs!(s::KSol,c::NewtonSST,q::NewtonPairSSTGeo)
    _k_newton_pair_sst_coeffs!(s.backend)(c.akW,c.akE,c.akS,c.akN,c.awW,c.awE,c.awS,c.awN,c.dk,c.sk,c.dw,c.sw,c.sf2,
        s.tk,s.tom,s.nut,s.rho,s.T,s.ux,s.uy,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.wdist,s.gammat,
        q.xxi,q.yxi,q.xet,q.yet,q.dE,q.dW,q.dN,q.dS,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
end

mutable struct NewtonPairState
    kwc;kwh;Lc;Lh;coarse;sstc;ssth;scalarc;scalarh
    d0c;d0h;invdtc;invdth;backupc;backuph;basec;baseh
    coarseh;cF;XCc::Vector{SMatrix{4,4,Float64,16}};XCh::Matrix{SMatrix{4,4,Float64,16}}   # coarse level: strip aggregates, factor, cross blocks
    jcolor_foil::Vector{Int32};wcolor::Vector{Int32}   # colours of the surface j-lines (wake entries -1) and of the wake stations
    Xc::Vector{SMatrix{4,4,Float64,16}};Xh::Matrix{SMatrix{4,4,Float64,16}}   # cross blocks: Xc[i] of C wake cell (i,1); Xh[k,side] of the strip
    sxc::Vector{Float64};sxh::Matrix{Float64}    # the same for the scalar (SST) systems
    cfl::Float64;it::Int;rows::Vector{Any};since_build::Int;built_cfl::Float64
end
function newton_pair_init(p::NewtonPair;cfl=20.,agg=(4,4))
    c,h=p.c,p.h
    kw(s)=build_airfoil_kinetic_block_workspace(s;block_length=min(64,s.NI),block_scalar_type=Float64,fullvisc_block=true)
    kwc=kw(c);kwh=kw(h);Lc=newton_lines_init(c,kwc);Lh=newton_lines_init(h,kwh)
    four(s)=ntuple(_->similar(s.dtf),4);st(s)=map(copy,(s.rho,s.ux,s.uy,s.T));Z4=zero(SMatrix{4,4,Float64,16})
    jf=copy(Lc.jcolor);jf[c.NIF+1:end].=Int32(-1);wcol=Int32[Lc.jcolor[c.NIF+p.NW+1-k] for k in 1:p.NW]
    NewtonPairState(kwc,kwh,Lc,Lh,agg===nothing ? nothing : newton_coarse_init(c;fi=agg[1],fj=agg[2]),newton_sst_init(c),newton_sst_init(h),
        newton_scalar_init(c,Lc),newton_scalar_init(h,Lh),similar(c.dtf),similar(h.dtf),similar(c.dtf),similar(h.dtf),
        st(c),st(h),four(c),four(h),agg===nothing ? nothing : newton_coarse_init(h;fi=agg[2],fj=agg[1]),nothing,
        fill(Z4,agg===nothing ? 0 : c.NI÷agg[1]),fill(Z4,agg===nothing ? 0 : p.NW÷agg[1],2),jf,wcol,fill(Z4,c.NI),fill(Z4,p.NW,2),
        zeros(c.NI),zeros(p.NW,2),cfl,0,Any[],div(typemax(Int),2),NaN)
end

# Scalar systems of both blocks from their coefficients; the interface terms are taken out of the blocks' own stencils
# (C: the south coefficient of its leg cells; strip: the west/east coefficient the single-block assembly leaves out).
function _newton_pair_sst_systems!(p::NewtonPair,m::NewtonPairState,isk::Int,invdtc,invdth)
    c,h=p.c,p.h;NS=p.NS
    for (s,cf,q,invdt) in ((c,m.sstc,m.scalarc,invdtc),(h,m.ssth,m.scalarh,invdth))
        a=isk==1 ? (cf.akW,cf.akE,cf.akS,cf.akN,cf.dk,cf.sk,s.tk,s.k_inf) : (cf.awW,cf.awE,cf.awS,cf.awN,cf.dw,cf.sw,s.tom,s.w_inf)
        _k_newton_sst_system!(s.backend)(q.cW,q.cE,q.cS,q.cN,q.cP,q.rhs,a[1],a[2],a[3],a[4],a[5],a[6],a[7],invdt,
            s.rho,s.T,s.wdist,eltype(q.cP)(a[8]),Int32(isk),s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,
            Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g);ndrange=(s.NI,s.NJ))
    end
    aW=isk==1 ? m.ssth.akW : m.ssth.awW;aE=isk==1 ? m.ssth.akE : m.ssth.awE;qc=m.scalarc;qh=m.scalarh
    for k in 1:p.NW,side in 1:2
        ic,ih=_newton_pair_cells(p,k,side)
        m.sxc[ic]=qc.cS[ic,1];qc.cS[ic,1]=0
        if isk==0&&k<=2                      # omega is prescribed in the strip's two rows behind the base
            m.sxh[k,side]=0
        else
            a=side==1 ? aE[ih,k] : aW[ih,k];qh.cP[ih,k]+=a;m.sxh[k,side]=-a
        end
    end
end
# r = b - A x on both blocks, cross terms included
function _newton_pair_scalar_residual!(rc,rh,bc,bh,xc,xh,p::NewtonPair,m::NewtonPairState)
    for (s,q,r,b,x) in ((p.c,m.scalarc,rc,bc,xc),(p.h,m.scalarh,rh,bh,xh))
        _k_newton_scalar_residual!(s.backend)(r,b,x,q.cW,q.cE,q.cS,q.cN,q.cP,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);ndrange=(s.NI,s.NJ))
    end
    @inbounds for k in 1:p.NW,side in 1:2
        ic,ih=_newton_pair_cells(p,k,side)
        rc[ic,1]-=m.sxc[ic]*xh[ih,k];rh[ih,k]-=m.sxh[k,side]*xc[ic,1]
    end
end
# Cell m of the scalar wake line at station k: (block, i, j, coefficient to the previous cell, to the next cell).
@inline function _newton_pair_sline_cell(p::NewtonPair,m_::Int,k::Int,cS,cN,hW,hE,sxc,sxh)
    NJ=p.c.NJ;NS=p.NS;NW=p.NW;NIF=p.c.NIF
    if m_<=NJ
        i=NW+1-k;j=NJ+1-m_
        (1,i,j,j<NJ ? cN[i,j] : 0.0,j>1 ? cS[i,j] : sxc[i])
    elseif m_<=NJ+NS
        ih=NS+1-(m_-NJ)
        (2,ih,k,ih==NS ? sxh[k,1] : hE[ih,k],ih>1 ? hW[ih,k] : sxh[k,2])
    else
        i=NW+NIF+k;j=m_-NJ-NS
        (1,i,j,j==1 ? sxc[i] : cS[i,j],j<NJ ? cN[i,j] : 0.0)
    end
end
# x += T^-1 r on the scalar wake lines of colour c (Thomas; r and work are overwritten on those lines)
function _newton_pair_slines_solve!(xc,xh,rc,rh,wc,wh,p::NewtonPair,cS,cN,cPc,hW,hE,cPh,sxc,sxh,wcolor,c::Int)
    count=2p.c.NJ+p.NS
    Threads.@threads for k in 1:p.NW
        wcolor[k]==c||continue
        cp=0.0;dp=0.0
        @inbounds for mm in 1:count
            blk,i,j,lo,hi=_newton_pair_sline_cell(p,mm,k,cS,cN,hW,hE,sxc,sxh)
            if blk==1
                den=cPc[i,j]-lo*cp;cp=hi/den;dp=(rc[i,j]-lo*dp)/den;wc[i,j]=cp;rc[i,j]=dp
            else
                den=cPh[i,j]-lo*cp;cp=hi/den;dp=(rh[i,j]-lo*dp)/den;wh[i,j]=cp;rh[i,j]=dp
            end
        end
        next=0.0
        @inbounds for mm in count:-1:1
            blk,i,j,_,_=_newton_pair_sline_cell(p,mm,k,cS,cN,hW,hE,sxc,sxh)
            if blk==1
                next=rc[i,j]-wc[i,j]*next;xc[i,j]+=next
            else
                next=rh[i,j]-wh[i,j]*next;xh[i,j]+=next
            end
        end
    end
end
"Line relaxation of the pair's scalar system from x = 0: surface j-lines and wake lines by colour, then the streamwise lines."
function _newton_pair_scalar_precond!(xc,xh,bc,bh,p::NewtonPair,m::NewtonPairState;sweeps::Int=1)
    c,h=p.c,p.h;qc,qh=m.scalarc,m.scalarh
    fill!(xc,0);fill!(xh,0);first=true
    resid!()=if first
        copyto!(qc.r,bc);copyto!(qh.r,bh);first=false
    else
        _newton_pair_scalar_residual!(qc.r,qh.r,bc,bh,xc,xh,p,m)
    end
    for _ in 1:sweeps
        for col in 0:qc.njcolor-1
            resid!()
            _k_newton_scalar_jline!(c.backend)(xc,qc.r,qc.work,qc.cS,qc.cN,qc.cP,m.jcolor_foil,Int32(col),Int32(c.NI),Int32(c.NJ),Int32(c.NIW),Int32(c.NIF);ndrange=c.NIW+c.NIF)
            _newton_pair_slines_solve!(xc,xh,qc.r,qh.r,qc.work,qh.work,p,qc.cS,qc.cN,qc.cP,qh.cW,qh.cE,qh.cP,m.sxc,m.sxh,m.wcolor,col)
        end
        for col in 0:1
            resid!()
            _k_newton_scalar_iline!(c.backend)(xc,qc.r,qc.work,qc.cW,qc.cE,qc.cP,Int32(col),Int32(c.NI),Int32(c.NJ);ndrange=c.NJ)
        end
        for col in 0:qh.njcolor-1
            resid!()
            _k_newton_scalar_jline!(h.backend)(xh,qh.r,qh.work,qh.cS,qh.cN,qh.cP,qh.jcolor,Int32(col),Int32(h.NI),Int32(h.NJ),Int32(h.NIW),Int32(h.NIF);ndrange=h.NIW+h.NIF)
        end
    end
end
function _newton_pair_scalar_solve!(p::NewtonPair,m::NewtonPairState,phic,phih;tol=1e-3,maxit=60)
    c,h=p.c,p.h;nc=c.NI*c.NJ;nh=h.NI*h.NJ
    Rc(v)=reshape(view(v,1:nc),c.NI,c.NJ);Rh(v)=reshape(view(v,nc+1:nc+nh),h.NI,h.NJ)
    b0=zeros(nc+nh);_newton_pair_scalar_residual!(Rc(b0),Rh(b0),m.scalarc.rhs,m.scalarh.rhs,phic,phih,p,m)
    zc=fill!(similar(phic),0);zh=fill!(similar(phih),0)
    A!(y,x)=(_newton_pair_scalar_residual!(Rc(y),Rh(y),zc,zh,Rc(x),Rh(x),p,m);y.*=-1;y)
    P!(z,v)=(_newton_pair_scalar_precond!(Rc(z),Rh(z),Rc(v),Rh(v),p,m);z)
    e,its,hist=newton_gmres(A!,P!,b0;tol,restart=maxit,maxit)
    phic.+Rc(e),phih.+Rh(e),its,hist[end]
end

"Implicit SST step on both blocks at the local pseudo-time 1/invdt (set by the caller in m.invdtc, m.invdth)."
function newton_pair_sst_step!(p::NewtonPair,m::NewtonPairState;relax=1.0,nut_relax=1.0,tol=1e-3)
    c,h=p.c,p.h
    newton_pair_sst_bc!(p);newton_pair_sst_coeffs!(c,m.sstc,p.gc);newton_pair_sst_coeffs!(h,m.ssth,p.gh)
    int(s,a)=Array(view(a,s.g+1:s.g+s.NI,s.g+1:s.g+s.NJ))
    kc,kh,wc,wh=int(c,c.tk),int(h,h.tk),int(c,c.tom),int(h,h.tom)
    _newton_pair_sst_systems!(p,m,1,m.invdtc,m.invdth);kcn,khn,kits,klin=_newton_pair_scalar_solve!(p,m,kc,kh;tol)
    _newton_pair_sst_systems!(p,m,0,m.invdtc,m.invdth);wcn,whn,wits,wlin=_newton_pair_scalar_solve!(p,m,wc,wh;tol)
    dkmax=0.0;dwmax=0.0
    for (s,cf,ko,kn,wo,wn) in ((c,m.sstc,kc,kcn,wc,wcn),(h,m.ssth,kh,khn,wh,whn))
        kn.=clamp.(ko.+relax.*(kn.-ko),1e-12,1.0);wn.=max.(wo.+relax.*(wn.-wo),1e-6)
        dkmax=max(dkmax,maximum(abs.(kn.-ko)./max.(abs.(ko),Float64(s.k_inf),1e-12)))
        dwmax=max(dwmax,maximum(abs.(wn.-wo)./max.(abs.(wo),Float64(s.w_inf),1e-6)))
        ir=s.g+1:s.g+s.NI;jr=s.g+1:s.g+s.NJ
        view(s.tk,ir,jr).=kn;view(s.tom,ir,jr).=wn;nutv=view(s.nut,ir,jr);sf2=cf.sf2
        @inbounds for j in 1:s.NJ,i in 1:s.NI
            wall=j<=2&&s.NIW<i<=s.NIW+s.NIF
            nt=wall ? kn[i,j]/max(wn[i,j],1e-6) : 0.31*kn[i,j]/max(0.31*wn[i,j],Float64(sf2[i,j]))
            nutv[i,j]+=nut_relax*(min(nt,0.15)-nutv[i,j])
        end
    end
    newton_pair_sst_bc!(p)
    (;dkmax,dwmax,kits,wits,klin,wlin)
end
"RMS of the steady k and omega equation residuals over both blocks (no pseudo-time term; prescribed omega cells left out)."
function newton_pair_sst_equation_l2(p::NewtonPair,m::NewtonPairState)
    c,h=p.c,p.h
    newton_pair_sst_bc!(p);newton_pair_sst_coeffs!(c,m.sstc,p.gc);newton_pair_sst_coeffs!(h,m.ssth,p.gh)
    int(s,a)=Array(view(a,s.g+1:s.g+s.NI,s.g+1:s.g+s.NJ));zc=fill!(similar(c.dtf),0);zh=fill!(similar(h.dtf),0)
    out=Float64[]
    for isk in (1,0)
        _newton_pair_sst_systems!(p,m,isk,zc,zh);rc=similar(zc);rh=similar(zh)
        _newton_pair_scalar_residual!(rc,rh,m.scalarc.rhs,m.scalarh.rhs,isk==1 ? int(c,c.tk) : int(c,c.tom),isk==1 ? int(h,h.tk) : int(h,h.tom),p,m)
        n=length(rc)+length(rh)
        if isk==0
            for (s,r) in ((c,rc),(h,rh)),j in 1:min(2,s.NJ),i in s.NIW+1:s.NIW+s.NIF;r[i,j]=0;n-=1;end
        end
        push!(out,sqrt((sum(abs2,rc)+sum(abs2,rh))/n))
    end
    out
end

# ---- flow Newton step ----------------------------------------------------------------------------------
function _newton_pair_split(v,p::NewtonPair)
    c,h=p.c,p.h;nc=4*c.NI*c.NJ
    newton_fields(view(v,1:nc),c.NI,c.NJ),newton_fields(view(v,nc+1:length(v)),h.NI,h.NJ)
end
function _newton_pair_fd_jv!(y,x,p::NewtonPair,m::NewtonPairState;rel=1e-7)
    c,h=p.c,p.h;oc,oh=_newton_pair_split(y,p);vc,vh=_newton_pair_split(x,p)
    r=Float64(c.rho_inf);u=max(hypot(Float64(c.uxi),Float64(c.uyi)),1e-3);E=r*(Float64(c.eint*c.T_inf)+0.5*u*u)
    sc(v)=max(maximum(abs,v[1])/r,maximum(abs,v[2])/(r*u),maximum(abs,v[3])/(r*u),maximum(abs,v[4])/E)
    scaled=max(sc(vc),sc(vh))
    scaled==0&&(fill!(y,0);return y)
    hh=rel/scaled
    newton_trial!(c,m.backupc,vc,-hh);newton_trial!(h,m.backuph,vh,-hh);newton_pair_residual!(p)
    for (s,o,v,base,d0,bk) in ((c,oc,vc,m.basec,m.d0c,m.backupc),(h,oh,vh,m.baseh,m.d0h,m.backuph))
        cur=_airfoil_residual(s)
        for k in 1:4;@. o[k]=(cur[k]-base[k])/hh+d0*v[k];end
        for (d,b) in zip((s.rho,s.ux,s.uy,s.T),bk);copyto!(d,b);end
    end
    y
end
# ---- preconditioner of the flow step -------------------------------------------------------------------
# The public two-level cycle carried over to two blocks. Line relaxation: the surface j-lines of C, then the WAKE
# lines, each running from the outer boundary down a lower-leg column of C, across the strip and up the upper-leg
# column (2 NJ + NS cells, one block-tridiagonal solve), then the streamwise lines (the i-lines of C and the j-lines of
# the strip). Coarse level: aggregates of both blocks in one sparse system.

# Cell m of the wake line at station k: (block, i, j, coupling to the previous cell, coupling to the next cell).
@inline function _newton_pair_line_cell(p::NewtonPair,m_::Int,k::Int,lj,uj,hli,hui,Xc,Xh)
    NJ=p.c.NJ;NS=p.NS;NW=p.NW;NIF=p.c.NIF;Z=zero(SMatrix{4,4,Float64,16})
    if m_<=NJ
        i=NW+1-k;j=NJ+1-m_
        (1,i,j,j<NJ ? _airfoil_block4(uj,i,j) : Z,j>1 ? _airfoil_block4(lj,i,j) : Xc[i])
    elseif m_<=NJ+NS
        ih=NS+1-(m_-NJ)
        (2,ih,k,ih==NS ? Xh[k,1] : _airfoil_block4(hui,ih,k),ih>1 ? _airfoil_block4(hli,ih,k) : Xh[k,2])
    else
        i=NW+NIF+k;j=m_-NJ-NS
        (1,i,j,j==1 ? Xc[i] : _airfoil_block4(lj,i,j),j<NJ ? _airfoil_block4(uj,i,j) : Z)
    end
end
function _newton_pair_lines_factor!(p::NewtonPair,lj,uj,hli,hui,Dc,Dh,cd,cc,hd,hc,Xc,Xh)
    count=2p.c.NJ+p.NS;Z=zero(SMatrix{4,4,Float64,16})
    Threads.@threads for k in 1:p.NW
        prev=Z
        @inbounds for mm in 1:count
            blk,i,j,lo,hi=_newton_pair_line_cell(p,mm,k,lj,uj,hli,hui,Xc,Xh)
            di=inv((blk==1 ? _airfoil_block4(Dc,i,j) : _airfoil_block4(Dh,i,j))-lo*prev);ci=mm<count ? di*hi : Z
            if blk==1
                _airfoil_block4_store!(cd,di,i,j);_airfoil_block4_store!(cc,ci,i,j)
            else
                _airfoil_block4_store!(hd,di,i,j);_airfoil_block4_store!(hc,ci,i,j)
            end
            prev=ci
        end
    end
end
# x += L^-1 r on the wake lines of colour c (r is overwritten on those lines)
function _newton_pair_lines_solve!(xc,xh,rc,rh,p::NewtonPair,lj,uj,hli,hui,cd,cc,hd,hc,Xc,Xh,wcolor,c::Int)
    count=2p.c.NJ+p.NS;B=Float64
    Threads.@threads for k in 1:p.NW
        wcolor[k]==c||continue
        prev=zero(SVector{4,B})
        @inbounds for mm in 1:count
            blk,i,j,lo,_=_newton_pair_line_cell(p,mm,k,lj,uj,hli,hui,Xc,Xh)
            if blk==1
                y=_airfoil_block4(cd,i,j)*(_airfoil_vec4(rc,i,j,B)-lo*prev);_airfoil_vec4_store!(rc,y,i,j);prev=y
            else
                y=_airfoil_block4(hd,i,j)*(_airfoil_vec4(rh,i,j,B)-lo*prev);_airfoil_vec4_store!(rh,y,i,j);prev=y
            end
        end
        next=prev
        @inbounds for mm in count:-1:1
            blk,i,j,_,_=_newton_pair_line_cell(p,mm,k,lj,uj,hli,hui,Xc,Xh)
            if mm<count
                next=blk==1 ? _airfoil_vec4(rc,i,j,B)-_airfoil_block4(cc,i,j)*next : _airfoil_vec4(rh,i,j,B)-_airfoil_block4(hc,i,j)*next
            end
            x=blk==1 ? xc : xh
            x[1][i,j]+=next[1];x[2][i,j]+=next[2];x[3][i,j]+=next[3];x[4][i,j]+=next[4]
        end
    end
end
# cross-block part of r = b - A x (the in-block parts come from the public residual kernel)
function _newton_pair_cross_residual!(rc,rh,xc,xh,p::NewtonPair,Xc,Xh)
    B=Float64
    @inbounds for k in 1:p.NW,side in 1:2
        ic,ih=_newton_pair_cells(p,k,side)
        _airfoil_vec4_store!(rc,_airfoil_vec4(rc,ic,1,B)-Xc[ic]*_airfoil_vec4(xh,ih,k,B),ic,1)
        _airfoil_vec4_store!(rh,_airfoil_vec4(rh,ih,k,B)-Xh[k,side]*_airfoil_vec4(xc,ic,1,B),ih,k)
    end
end
# r = b - A x into Lc.r, Lh.r. mode 0: everywhere; 1: the C columns of j-line colour c; 2: the C rows of parity c
# (the strip is always complete).
function _newton_pair_block_residual!(bc,bh,xc,xh,p::NewtonPair,m::NewtonPairState;mode::Int=0,c::Int=0)
    newton_block_residual!(m.Lc,bc,xc,p.c,m.kwc;mode,c);newton_block_residual!(m.Lh,bh,xh,p.h,m.kwh)
    _newton_pair_cross_residual!(m.Lc.r,m.Lh.r,xc,xh,p,m.Xc,m.Xh)
end
"One alternating zebra line sweep of the pair, continuing from the current x."
function _newton_pair_smooth!(xc,xh,bc,bh,p::NewtonPair,m::NewtonPairState;zero_start::Bool=false)
    c,h=p.c,p.h;kc,kh,Lc,Lh=m.kwc,m.kwh,m.Lc,m.Lh;first=zero_start
    resid!(mode,col)=if first
        for k in 1:4;copyto!(Lc.r[k],bc[k]);copyto!(Lh.r[k],bh[k]);end
        first=false
    else
        _newton_pair_block_residual!(bc,bh,xc,xh,p,m;mode,c=col)
    end
    for col in 0:Lc.njcolor-1
        resid!(1,col)
        _k_newton_jline_solve!(c.backend)(xc...,Lc.r...,Lc.jdinv,Lc.jc,kc.lower_j,kc.upper_j,m.jcolor_foil,Int32(col),
            Int32(c.NI),Int32(c.NJ),Int32(c.NIW),Int32(c.NIF);ndrange=c.NIW+c.NIF)
        _newton_pair_lines_solve!(xc,xh,Lc.r,Lh.r,p,kc.lower_j,kc.upper_j,kh.lower_i,kh.upper_i,Lc.jdinv,Lc.jc,Lh.idinv,Lh.ic,m.Xc,m.Xh,m.wcolor,col)
    end
    for col in 0:1
        resid!(2,col)
        _k_newton_iline_solve!(c.backend)(xc...,Lc.r...,Lc.idinv,Lc.ic,kc.lower_i,Int32(col),Int32(c.NI),Int32(c.NJ);ndrange=c.NJ)
    end
    for col in 0:Lh.njcolor-1
        resid!(0,0)
        _k_newton_jline_solve!(h.backend)(xh...,Lh.r...,Lh.jdinv,Lh.jc,kh.lower_j,kh.upper_j,Lh.jcolor,Int32(col),
            Int32(h.NI),Int32(h.NJ),Int32(h.NIW),Int32(h.NIF);ndrange=h.NIW+h.NIF)
    end
end

# Galerkin sums of one block (the loop of newton_coarse_build!, without its factorisation)
function _newton_pair_coarse_sum!(C::NewtonCoarse,s::KSol,kw)
    fi,fj=C.fi,C.fj;NI,NJ,NIW,NIF=s.NI,s.NJ,s.NIW,s.NIF
    for a in (C.diagonal,C.lower_i,C.upper_i,C.lower_j,C.upper_j);fill!(a,0);end
    dg,li,ui,lj,uj=kw.diagonal,kw.lower_i,kw.upper_i,kw.lower_j,kw.upper_j
    Threads.@threads for J in 1:C.NJ
        for j in (J-1)*fj+1:J*fj,i in 1:NI
            I=(i-1)÷fi+1
            _newton_block_add!(C.diagonal,I,J,dg,i,j)
            if i>1;_newton_block_add!((i-2)÷fi+1==I ? C.diagonal : C.lower_i,I,J,li,i,j);end
            if i<NI;_newton_block_add!(i÷fi+1==I ? C.diagonal : C.upper_i,I,J,ui,i,j);end
            if j>1
                _newton_block_add!((j-2)÷fj+1==J ? C.diagonal : C.lower_j,I,J,lj,i,j)
            elseif i<=NIW||i>NIW+NIF
                _newton_block_add!(C.lower_j,I,J,lj,i,j)
            end
            if j<NJ;_newton_block_add!(j÷fj+1==J ? C.diagonal : C.upper_j,I,J,uj,i,j);end
        end
    end
end
function _newton_pair_coarse_build!(p::NewtonPair,m::NewtonPairState)
    Cc,Ch=m.coarse,m.coarseh;fi=Cc.fi;fih=Ch.fi;Z=zero(SMatrix{4,4,Float64,16})
    _newton_pair_coarse_sum!(Cc,p.c,m.kwc);_newton_pair_coarse_sum!(Ch,p.h,m.kwh)
    fill!(m.XCc,Z);fill!(m.XCh,Z)
    for k in 1:p.NW,side in 1:2
        ic,ih=_newton_pair_cells(p,k,side);m.XCc[(ic-1)÷fi+1]+=m.Xc[ic];m.XCh[(k-1)÷fi+1,side]+=m.Xh[k,side]
    end
    Ac=newton_sparse(Cc,Cc.NI,Cc.NJ,Cc.NIW,Cc.NIF);Ah=newton_sparse(Ch,Ch.NI,Ch.NJ,Ch.NIW,Ch.NIF)
    nc=size(Ac,1);nh=size(Ah,1);idc(i,j,q)=((j-1)*Cc.NI+(i-1))*4+q;idh(i,j,q)=((j-1)*Ch.NI+(i-1))*4+q
    I1=Int[];J1=Int[];V1=Float64[];I2=Int[];J2=Int[];V2=Float64[]
    for K in 1:Ch.NJ,side in 1:2
        # coarse station K: C aggregates (NWc+1-K, 1) and (NWc+NIFc+K, 1); strip aggregates (NSc, K) and (1, K)
        Ic=side==1 ? Cc.NIW+1-K : Cc.NIW+Cc.NIF+K;Ih=side==1 ? Ch.NI : 1
        for q2 in 1:4,q1 in 1:4
            push!(I1,idc(Ic,1,q1));push!(J1,idh(Ih,K,q2));push!(V1,m.XCc[Ic][q1,q2])
            push!(I2,idh(Ih,K,q1));push!(J2,idc(Ic,1,q2));push!(V2,m.XCh[K,side][q1,q2])
        end
    end
    m.cF=lu([Ac sparse(I1,J1,V1,nc,nh);sparse(I2,J2,V2,nh,nc) Ah])
    nothing
end
function _newton_pair_coarse_correct!(xc,xh,rc,rh,m::NewtonPairState)
    Cc,Ch=m.coarse,m.coarseh
    for (C,r) in ((Cc,rc),(Ch,rh))
        fi,fj=C.fi,C.fj;q=C.rc
        Threads.@threads for J in 1:C.NJ
            @inbounds for I in 1:C.NI,k in 1:4
                acc=0.0;rk=r[k]
                for j in (J-1)*fj+1:J*fj,i in (I-1)*fi+1:I*fi;acc+=rk[i,j];end
                q[k,I,J]=acc
            end
        end
    end
    e=m.cF\vcat(vec(Cc.rc),vec(Ch.rc));nc=length(Cc.rc)
    for (C,x,off) in ((Cc,xc,0),(Ch,xh,nc))
        fi,fj=C.fi,C.fj;ev=reshape(view(e,off+1:off+4*C.NI*C.NJ),4,C.NI,C.NJ)
        Threads.@threads for J in 1:C.NJ
            @inbounds for I in 1:C.NI,k in 1:4
                v=ev[k,I,J];xk=x[k]
                for j in (J-1)*fj+1:J*fj,i in (I-1)*fi+1:I*fi;xk[i,j]+=v;end
            end
        end
    end
end

# Assemble both blocks' first-order matrices, move the interface terms into the cross blocks and factor.
function _newton_pair_precond_build!(p::NewtonPair,m::NewtonPairState;delta=0.3)
    c,h=p.c,p.h;g=h.g;NS=p.NS;kc,kh=m.kwc,m.kwh;Z=zero(SMatrix{4,4,Float64,16})
    newton_roe_build!(c,kc,m.d0c;delta);newton_roe_build!(h,kh,m.d0h;delta)
    for k in 1:p.NW,side in 1:2
        ic,ih=_newton_pair_cells(p,k,side)
        m.Xc[ic]=_airfoil_block4(kc.lower_j,ic,1);_airfoil_block4_store!(kc.lower_j,Z,ic,1)
        nu=_air_nu_local(h.rho[ih+g,k+g],h.T[ih+g,k+g],h.nu_lam,h.rho_inf,h.suth,h.air_sutherland)+(h.sst ? h.nut[ih+g,k+g] : 0.0)
        ihf=side==1 ? NS+1 : 1;rv=2nu*(h.Six[ihf,k]^2+h.Siy[ihf,k]^2)/h.vol[ih,k]
        m.Xh[k,side]=side==1 ? _airfoil_block4(kh.face_i_right,ihf,k)-rv*_AIRFOIL_KV4 : -_airfoil_block4(kh.face_i_left,ihf,k)-rv*_AIRFOIL_KV4
    end
    newton_lines_factor!(m.Lc,c,kc);newton_lines_factor!(m.Lh,h,kh)
    _newton_pair_lines_factor!(p,kc.lower_j,kc.upper_j,kh.lower_i,kh.upper_i,kc.diagonal,kh.diagonal,m.Lc.jdinv,m.Lc.jc,m.Lh.idinv,m.Lh.ic,m.Xc,m.Xh)
    m.coarse===nothing||_newton_pair_coarse_build!(p,m)
    nothing
end
"z = P(v): pre-smoothing, joint coarse correction, post-smoothing, from z = 0."
function _newton_pair_precond!(z,v,p::NewtonPair,m::NewtonPairState;pre=1,post=1)
    zc,zh=_newton_pair_split(z,p);vc,vh=_newton_pair_split(v,p)
    fill!(z,0)
    for n in 1:pre;_newton_pair_smooth!(zc,zh,vc,vh,p,m;zero_start=n==1);end
    if m.coarse!==nothing
        if pre==0
            for k in 1:4;copyto!(m.Lc.r[k],vc[k]);copyto!(m.Lh.r[k],vh[k]);end
        else
            _newton_pair_block_residual!(vc,vh,zc,zh,p,m)
        end
        _newton_pair_coarse_correct!(zc,zh,m.Lc.r,m.Lh.r,m)
    end
    for _ in 1:post;_newton_pair_smooth!(zc,zh,vc,vh,p,m);end
    z
end

"Flow Newton step of the pair at pseudo-CFL m.cfl with frozen eddy viscosity."
function newton_pair_flow_step!(p::NewtonPair,m::NewtonPairState;lintol=0.05,maxit=40,fraction=0.2,lag=3,pre=1,post=1,delta=0.3)
    c,h=p.c,p.h;t0=time()
    newton_pair_residual!(p)
    for (s,bk,base) in ((c,m.backupc,m.basec),(h,m.backuph,m.baseh))
        for (b,d) in zip(bk,(s.rho,s.ux,s.uy,s.T));copyto!(b,d);end
        for (b,r) in zip(base,_airfoil_residual(s));copyto!(b,r);end
    end
    l2=newton_pair_flow_l2(p)
    newton_d0!(m.d0c,c,m.cfl);newton_d0!(m.d0h,h,m.cfl)
    if m.since_build>=lag||m.built_cfl!=m.cfl
        _newton_pair_precond_build!(p,m;delta);m.since_build=0;m.built_cfl=m.cfl
    end
    m.since_build+=1
    tbuild=time()-t0;t0=time()
    b=vcat(map(vec,m.basec)...,map(vec,m.baseh)...)
    A!(y,x)=_newton_pair_fd_jv!(y,x,p,m);P!(z,v)=_newton_pair_precond!(z,v,p,m;pre,post)
    x,its,hist=newton_gmres(A!,P!,b;tol=lintol,restart=maxit,maxit)
    dc,dh=_newton_pair_split(x,p);tlin=time()-t0
    alpha=Float64(min(newton_physical_alpha(c,m.backupc,dc,fraction),newton_physical_alpha(h,m.backuph,dh,fraction)));ok=false
    for _ in 1:6
        newton_trial!(c,m.backupc,dc,alpha);newton_trial!(h,m.backuph,dh,alpha);newton_pair_bc!(p)
        ok=_airfoil_physical_state(c.rho,c.ux,c.uy,c.T)&&_airfoil_physical_state(h.rho,h.ux,h.uy,h.T);ok&&break
        alpha/=2
    end
    if !ok
        for (s,bk) in ((c,m.backupc),(h,m.backuph)),(d,b0) in zip((s.rho,s.ux,s.uy,s.T),bk);copyto!(d,b0);end
        newton_pair_bc!(p)
    end
    (;l2,its,lin=hist[end],alpha,ok,tbuild,tlin)
end

"One coupled iteration of the pair: flow Newton step, then an implicit SST step at the same local pseudo-time."
function newton_pair_step!(p::NewtonPair,m::NewtonPairState;turb_ratio=1.0,sst_relax=1.0,nut_relax=1.0,sst_tol=1e-3,kwargs...)
    r=newton_pair_flow_step!(p,m;kwargs...)
    @. m.invdtc=m.d0c/p.c.vol/turb_ratio
    @. m.invdth=m.d0h/p.h.vol/turb_ratio
    t0=time();t=newton_pair_sst_step!(p,m;relax=sst_relax,nut_relax,tol=sst_tol)
    m.it+=1;cl,cd=newton_pair_loads(p)
    merge(r,(;it=m.it,cfl=m.cfl,dk=t.dkmax,dw=t.dwmax,kits=t.kits,wits=t.wits,tsst=time()-t0,cl,cd))
end

newton_pair_state_vec(p::NewtonPair)=vcat(newton_state_vec(p.c),newton_state_vec(p.h))
function newton_pair_state_set!(p::NewtonPair,x)
    nc=7*p.c.NI*p.c.NJ;old=newton_state_vec(p.c)
    newton_state_set!(p.c,view(x,1:nc))||return false
    if !newton_state_set!(p.h,view(x,nc+1:length(x)))
        newton_state_set!(p.c,old);newton_pair_bc!(p);newton_pair_sst_bc!(p);return false
    end
    newton_pair_bc!(p);newton_pair_sst_bc!(p);true
end
function newton_pair_six(p::NewtonPair,m::NewtonPairState)
    newton_pair_residual!(p);[newton_pair_flow_l2(p)...,newton_pair_sst_equation_l2(p,m)...]
end
newton_pair_mass(p::NewtonPair)=abs(sum(p.c.dc_r)+sum(p.h.dc_r))/(max(Float64(p.c.rho_inf),1e-12)*max(hypot(Float64(p.c.uxi),Float64(p.c.uyi)),1e-3))

"March of the pair: same controls as newton_rans_march! (CFL ramp, extrapolation, far-field vortex, stop on settled loads, `on_row`)."
function newton_pair_march!(p::NewtonPair,m::NewtonPairState;nsteps=100,cflmax=1e3,growth=1.5,budget=Inf,every=5,csv=nothing,label="",
        ref=nothing,window=20,load_tol=1e-4,min_steps=window,quiet=false,rre=nothing,vortex_relax=0.0,on_row=nothing,kwargs...)
    c,h=p.c,p.h;t0=time();io=csv===nothing ? nothing : open(csv,"w")
    io===nothing||println(io,"step,seconds,cfl,its,rho,mx,my,E,k,omega,dk,dw,mass,CL,CD")
    loads=Tuple{Float64,Float64}[];converged=false;nlocal=0
    xs=Vector{Float64}[];scale=rre===nothing ? nothing : vcat(newton_state_scale(c),newton_state_scale(h));since=0;nrre=0
    try
        for n in 1:nsteps
            r=newton_pair_step!(p,m;kwargs...);nlocal+=1
            if vortex_relax>0&&r.ok
                c.gam+=oftype(c.gam,vortex_relax)*(oftype(c.gam,0.5*hypot(c.uxi,c.uyi)*r.cl)-c.gam);h.gam=c.gam
            end
            if rre!==nothing&&r.cfl>=cflmax&&r.ok&&r.alpha>=0.99
                push!(xs,newton_pair_state_vec(p));length(xs)>rre[1]+2&&popfirst!(xs);since+=1
                if since>=rre[2]&&length(xs)==rre[1]+2
                    newton_pair_state_set!(p,newton_rre(xs,scale))&&(nrre+=1);empty!(xs);since=0
                    cl2,cd2=newton_pair_loads(p);r=merge(r,(;cl=cl2,cd=cd2))
                end
            elseif rre!==nothing
                empty!(xs);since=0
            end
            push!(loads,(r.cl,r.cd));length(loads)>window&&popfirst!(loads)
            if length(loads)==window&&nlocal>=min_steps
                cls=first.(loads);cds=last.(loads)
                converged=(maximum(cls)-minimum(cls))/max(abs(r.cl),0.1)<load_tol&&(maximum(cds)-minimum(cds))/max(abs(r.cd),0.001)<load_tol
            end
            if n%every==0||n==1||converged||n==nsteps
                six=newton_pair_six(p,m);mass=newton_pair_mass(p)
                row=(;step=m.it,seconds=time()-t0,cfl=r.cfl,its=r.its,six,dk=r.dk,dw=r.dw,mass,cl=r.cl,cd=r.cd);push!(m.rows,row)
                io===nothing||(println(io,join((row.step,row.seconds,row.cfl,row.its,six...,row.dk,row.dw,row.mass,row.cl,row.cd),','));flush(io))
                on_row===nothing||on_row(row)
                if !quiet
                    Printf.@printf("%s%4d t=%6.1f cfl=%6.0f its=%2d a=%.2f sst its %d/%d [build %.2f lin %.2f sst %.2f] six%s=%s mass=%.1e CL=%.6f CD=%.6f\n",
                        label,m.it,time()-t0,r.cfl,r.its,r.alpha,r.kits,r.wits,r.tbuild,r.tlin,r.tsst,ref===nothing ? "" : "/cold",
                        string(round.(ref===nothing ? six : six./ref;sigdigits=3)),mass,row.cl,row.cd)
                    flush(stdout)
                end
            end
            converged&&break
            m.cfl=r.ok&&r.alpha>=0.99 ? min(cflmax,m.cfl*growth) : max(1.,m.cfl*(r.ok ? max(0.5,r.alpha) : 0.25))
            time()-t0>budget&&break
        end
    finally
        io===nothing||close(io)
    end
    (;converged,steps=nlocal,seconds=time()-t0,extrapolations=nrre)
end

"""
    newton_pair_cold_run(nodes, make_c, make_h; levels=(4,2,1), ...)

Cold start with mesh sequencing on a C+H mesh. `nodes` = (X, Y, NIW, NIF, NI, NJ, strip) of the fine mesh (strip as the
mesher returns it); `make_c(nodes)` and `make_h(strip nodes in the base-as-wall layout)` return prepared solvers.
`on_row(k, f, row)`, if given, receives every diagnostic row of level number k (coarsening f).
"""
function newton_pair_cold_run(nodes,make_c,make_h;levels=(4,2,1),nsteps=(120,120,160),cflmax=(1e3,1e3,1e3),load_tol=(1e-3,3e-4,1e-4),
        window=(10,15,20),rre=(3,8),delta=0.3,outdir=nothing,every=10,lintol=0.05,maxit=40,quiet=false,vortex_relax=0.0,growth=1.5,
        on_row=nothing)
    prev=nothing;pf=0;t00=time();out=nothing;summary=Dict{String,Any}();lv=Any[]
    for (k,f) in enumerate(levels)
        t0=time()
        cn=newton_coarsen_nodes((X=nodes.X,Y=nodes.Y,NIW=nodes.NIW,NIF=nodes.NIF,NI=nodes.NI,NJ=nodes.NJ),f)
        st=nodes.strip;(st.NI%f==0&&st.NJ%f==0)||throw(ArgumentError("strip not divisible by $f"))
        hn=newton_pair_strip_nodes((X=st.X[1:f:end,1:f:end],Y=st.Y[1:f:end,1:f:end],NI=st.NI÷f,NJ=st.NJ÷f))
        c=make_c(cn);h=make_h(hn);p=newton_pair_build(c,h,cn,hn)
        m=newton_pair_init(p;cfl=prev===nothing ? 5. : 20.,agg=(4÷f,4÷f))
        ref=newton_pair_six(p,m)                        # the six residuals of the freshly initialised level
        if prev!==nothing
            newton_prolong!(c,prev.c,pf÷f);newton_prolong!(h,prev.h,pf÷f);c.gam=oftype(c.gam,prev.c.gam);h.gam=c.gam
            newton_pair_bc!(p);newton_pair_sst_bc!(p)
        end
        setup=time()-t0;cells=cn.NI*cn.NJ+hn.NI*hn.NJ
        quiet||Printf.@printf("LEVEL 1/%d: C %d x %d + strip %d x %d = %d cells, setup %.1fs\n",f,cn.NI,cn.NJ,hn.NJ,hn.NI,cells,setup)
        r=newton_pair_march!(p,m;nsteps=nsteps[k],cflmax=cflmax[k],growth,every,vortex_relax,label="L$f ",ref,quiet,
            load_tol=load_tol[k],window=window[k],rre,delta,lintol,maxit,on_row=on_row===nothing ? nothing : row->on_row(k,f,row),
            csv=outdir===nothing ? nothing : joinpath(outdir,"history_L$f.csv"))
        cl,cd=newton_pair_loads(p)
        quiet||Printf.@printf("LEVEL 1/%d done: loads stable=%s, %d steps, %d extrapolations, %.1fs level, %.1fs total, CL=%.6f CD=%.6f\n",
            f,r.converged,r.steps,r.extrapolations,time()-t0,time()-t00,cl,cd)
        flush(stdout)
        push!(lv,Dict("coarsening"=>f,"cells"=>cells,"steps"=>r.steps,"loads_stable"=>r.converged,"extrapolations"=>r.extrapolations,
            "seconds"=>time()-t0,"setup_seconds"=>setup,"CL"=>cl,"CD"=>cd,"cold_reference_six"=>ref))
        prev=p;pf=f;out=(;p,m,ref)
    end
    six=newton_pair_six(out.p,out.m);cl,cd=newton_pair_loads(out.p);base=loads_ka(out.p.h)
    summary["levels"]=lv;summary["solve_seconds"]=time()-t00;summary["final_six_residuals"]=six;summary["final_six_over_cold"]=six./out.ref
    summary["final_CL"]=cl;summary["final_CD"]=cd;summary["final_mass_imbalance"]=newton_pair_mass(out.p)
    summary["base_CL"]=Float64(base[1]);summary["base_CD"]=Float64(base[2])
    (;p=out.p,m=out.m,ref=out.ref,summary)
end
