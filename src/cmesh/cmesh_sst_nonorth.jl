# Opt-in, conservative nonorthogonal SST scalar diffusion. Included by
# CMeshSolverKA after KE_ALPHA is defined. This staged file is unapplied.

function _ensure_sst_nonorth_scratch!(s)
    size(s.sst_gkx)==(s.NI,s.NJ) && return nothing
    b=s.backend
    z(nx,ny)=KA.allocate(b,Float32,nx,ny)
    s.sst_gkx=z(s.NI,s.NJ);s.sst_gky=z(s.NI,s.NJ)
    s.sst_gwx=z(s.NI,s.NJ);s.sst_gwy=z(s.NI,s.NJ)
    s.sst_sigk=z(s.NI,s.NJ);s.sst_sigw=z(s.NI,s.NJ)
    s.sst_nu=z(s.NI,s.NJ)
    s.sst_Fik=z(s.NI+1,s.NJ);s.sst_Fiw=z(s.NI+1,s.NJ)
    s.sst_Fjk=z(s.NI,s.NJ+1);s.sst_Fjw=z(s.NI,s.NJ+1)
    nothing
end

@inline function _sst_nonorth_flux(pL::Float32,pR::Float32,
        gxL::Float32,gyL::Float32,gxR::Float32,gyR::Float32,
        dx::Float32,dy::Float32,sx::Float32,sy::Float32,
        gamma::Float32)
    a=(sx*dx+sy*dy)/max(dx*dx+dy*dy,1.0f-30)
    gx=0.5f0*(gxL+gxR); gy=0.5f0*(gyL+gyR)
    gamma*(a*(pR-pL)+(sx-a*dx)*gx+(sy-a*dy)*gy)
end

@kernel function _k_sst_nonorth_prep!(gkx,gky,gwx,gwy,sigk,sigw,nul,
        @Const(tk),@Const(tom),@Const(rho),@Const(T),
        @Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),@Const(wdist),
        nu_inf::Float32,rho_inf::Float32,suth::Float32,air_sutherland::Int32,
        keps::Float32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,
        corrected_wall_metric::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        xxi,yxi,xet,yet=_sst_index_metrics(xc,yc,Sjx,Sjy,wdist,
            i,j,NI,NJ,NIW,NIF,corrected_wall_metric)
        J=xxi*yet-xet*yxi;J=abs(J)<1.0f-30 ? 1.0f-30 : J
        dki=0.5f0*(tk[ia+1,ja]-tk[ia-1,ja])
        dkj=0.5f0*(tk[ia,ja+1]-tk[ia,ja-1])
        dwi=0.5f0*(tom[ia+1,ja]-tom[ia-1,ja])
        dwj=0.5f0*(tom[ia,ja+1]-tom[ia,ja-1])
        kx=(dki*yet-dkj*yxi)/J;ky=(-dki*xet+dkj*xxi)/J
        wx=(dwi*yet-dwj*yxi)/J;wy=(-dwi*xet+dwj*xxi)/J
        gkx[i,j]=kx;gky[i,j]=ky;gwx[i,j]=wx;gwy[i,j]=wy
        nu=_air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        nul[i,j]=nu
        k0=tk[ia,ja];w0=max(tom[ia,ja],1.0f-6)
        d=max(wdist[i,j],1.0f-8)
        cdkw=max(2.0f0*0.856f0/w0*(kx*wx+ky*wy),1.0f-10)
        arg1=min(max(sqrt(max(k0,0.0f0))/(0.09f0*w0*d),
                     500.0f0*nu/(d*d*w0)),
                 4.0f0*0.856f0*k0/(cdkw*d*d))
        F1=tanh(min(arg1,4.0f0)^4)
        sigk[i,j]=keps>0.0f0 ? KE_ALPHA : F1*0.85f0+(1.0f0-F1)*1.0f0
        sigw[i,j]=keps>0.0f0 ? KE_ALPHA : F1*0.5f0+(1.0f0-F1)*0.856f0
    end
end

@kernel function _k_sst_nonorth_i!(Fik,Fiw,
        @Const(tk),@Const(tom),@Const(nut),
        @Const(gkx),@Const(gky),@Const(gwx),@Const(gwy),
        @Const(sigk),@Const(sigw),@Const(nul),
        @Const(xc),@Const(yc),@Const(Six),@Const(Siy),
        NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if 2<=i<=NI && j<=NJ
        l=i-1;r=i;iaL=l+g;iaR=r+g;ja=j+g
        dx=xc[r,j]-xc[l,j];dy=yc[r,j]-yc[l,j]
        sx=Six[i,j];sy=Siy[i,j]
        nu=0.5f0*(nul[l,j]+nul[r,j])
        nutf=0.5f0*(nut[iaL,ja]+nut[iaR,ja])
        gammak=nu+0.5f0*(sigk[l,j]+sigk[r,j])*nutf
        gammaw=nu+0.5f0*(sigw[l,j]+sigw[r,j])*nutf
        Fik[i,j]=_sst_nonorth_flux(tk[iaL,ja],tk[iaR,ja],
            gkx[l,j],gky[l,j],gkx[r,j],gky[r,j],dx,dy,sx,sy,gammak)
        Fiw[i,j]=_sst_nonorth_flux(tom[iaL,ja],tom[iaR,ja],
            gwx[l,j],gwy[l,j],gwx[r,j],gwy[r,j],dx,dy,sx,sy,gammaw)
    end
end

@kernel function _k_sst_nonorth_j!(Fjk,Fjw,
        @Const(tk),@Const(tom),@Const(nut),
        @Const(gkx),@Const(gky),@Const(gwx),@Const(gwy),
        @Const(sigk),@Const(sigw),@Const(nul),
        @Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),
        NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && 2<=j<=NJ
        l=j-1;r=j;ia=i+g;jaL=l+g;jaR=r+g
        dx=xc[i,r]-xc[i,l];dy=yc[i,r]-yc[i,l]
        sx=Sjx[i,j];sy=Sjy[i,j]
        nu=0.5f0*(nul[i,l]+nul[i,r])
        nutf=0.5f0*(nut[ia,jaL]+nut[ia,jaR])
        gammak=nu+0.5f0*(sigk[i,l]+sigk[i,r])*nutf
        gammaw=nu+0.5f0*(sigw[i,l]+sigw[i,r])*nutf
        Fjk[i,j]=_sst_nonorth_flux(tk[ia,jaL],tk[ia,jaR],
            gkx[i,l],gky[i,l],gkx[i,r],gky[i,r],dx,dy,sx,sy,gammak)
        Fjw[i,j]=_sst_nonorth_flux(tom[ia,jaL],tom[ia,jaR],
            gwx[i,l],gwy[i,l],gwx[i,r],gwy[i,r],dx,dy,sx,sy,gammaw)
    elseif i<=NIW && j==1
        # A wake-cut face has two mirrored index-space copies. Compute once on
        # the lower leg and assign the exact negative to its upper partner.
        ip=NI+Int32(1)-i;ia=i+g;iaP=ip+g;ja=Int32(1)+g
        dx=xc[i,1]-xc[ip,1];dy=yc[i,1]-yc[ip,1]
        sx=Sjx[i,1];sy=Sjy[i,1]
        nu=0.5f0*(nul[i,1]+nul[ip,1])
        nutf=0.5f0*(nut[ia,ja]+nut[iaP,ja])
        gammak=nu+0.5f0*(sigk[i,1]+sigk[ip,1])*nutf
        gammaw=nu+0.5f0*(sigw[i,1]+sigw[ip,1])*nutf
        fk=_sst_nonorth_flux(tk[iaP,ja],tk[ia,ja],
            gkx[ip,1],gky[ip,1],gkx[i,1],gky[i,1],dx,dy,sx,sy,gammak)
        fw=_sst_nonorth_flux(tom[iaP,ja],tom[ia,ja],
            gwx[ip,1],gwy[ip,1],gwx[i,1],gwy[i,1],dx,dy,sx,sy,gammaw)
        Fjk[i,1]=fk;Fjk[ip,1]=-fk
        Fjw[i,1]=fw;Fjw[ip,1]=-fw
    end
end
