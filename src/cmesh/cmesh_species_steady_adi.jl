# ---- Steady mode, approximate factorisation (ADI) in both directions (W3a-steady, 30 Sep 2026) ----------------------
# The j-line relaxation (cmesh_species_steady_line.jl) treats only the wall-normal direction implicitly; its local step
# is the tangential explicit limit times a factor <= 0.9, and on a 128 x 48 CPU case the loads kept a slow +-1%
# oscillation after 20,000 iterations at the stable factor 0.36 (memory/project_docs/2026-09-30_W2_STEADY_MODE_FINDINGS.md). Here the
# tangential direction is implicit as well, by the classical approximate factorisation of each stage's increment:
#     (d0 + J_i) d0^-1 (d0 + J_j) D = rhs,   d0 = V/dtau,   dtau = scale * margin * ADI_CFL * V / Lambda_i,
# solved as one tridiagonal system per row j along i (C-mesh rows run open from outlet to outlet), a scaling by d0, and
# the j-line solve of cmesh_species_steady_line.jl (through the wake cut). J_i uses the i-face rates (the AUSM face-speed
# bound times the face length plus twice the viscous conductance over the density) exactly as J_j uses the j-face rates;
# the SST equations get the same factorisation with their own coefficients. ADI_CFL may exceed 1: both directions are
# implicit. As for the j-line variant, only the iteration operator changes; a converged state solves rhs = 0. The
# j-line variant's code is untouched; this path is selected by build_species_steady(c; adi=true).

const ADI_CFL = Ref{Float32}(2.0f0)

@kernel function _k_adi_flow_coeffs!(lamW,lamE,lamS,lamN,d0,dtau,@Const(rho),@Const(ux),@Const(uy),@Const(am),@Const(p),
                                     @Const(nut),@Const(vol),@Const(xc),@Const(yc),
                                     @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),
                                     minf2::Float32,nu::Float32,wallmode::Int32,sst_on::Int32,fullvisc::Int32,
                                     acfl::Float32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        rw=_species_face_rate(rho,ux,uy,am,p,ia-1,ja,ia,ja,r,Six[i,j],Siy[i,j],minf2)
        re=_species_face_rate(rho,ux,uy,am,p,ia,ja,ia+1,ja,r,Six[i+1,j],Siy[i+1,j],minf2)
        rs=_species_face_rate(rho,ux,uy,am,p,ia,ja-1,ia,ja,r,Sjx[i,j],Sjy[i,j],minf2)
        rn=_species_face_rate(rho,ux,uy,am,p,ia,ja,ia,ja+1,r,Sjx[i,j+1],Sjy[i,j+1],minf2)
        vw=0.0f0;ve=0.0f0;vs=0.0f0;vn=0.0f0
        if nu>0.0f0
            if i>1
                dx=xc[i,j]-xc[i-1,j];dy=yc[i,j]-yc[i-1,j];dn=sqrt(dx*dx+dy*dy)+1.0f-30
                nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia-1,ja]+nut[ia,ja]) : nu
                vw=_viscous_conductance(0.5f0*(r+rho[ia-1,ja]),nf,Six[i,j],Siy[i,j],dn)
            end
            if i<NI
                dx=xc[i+1,j]-xc[i,j];dy=yc[i+1,j]-yc[i,j];dn=sqrt(dx*dx+dy*dy)+1.0f-30
                nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia+1,ja]) : nu
                ve=_viscous_conductance(0.5f0*(r+rho[ia+1,ja]),nf,Six[i+1,j],Siy[i+1,j],dn)
            end
            if j>1
                dx=xc[i,j]-xc[i,j-1];dy=yc[i,j]-yc[i,j-1];dn=sqrt(dx*dx+dy*dy)+1.0f-30
                nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja-1]+nut[ia,ja]) : nu
                vs=_viscous_conductance(0.5f0*(r+rho[ia,ja-1]),nf,Sjx[i,j],Sjy[i,j],dn)
            elseif i<=NIW || i>NIW+NIF
                ip=NI+1-i;dx=xc[i,1]-xc[ip,1];dy=yc[i,1]-yc[ip,1];dn=sqrt(dx*dx+dy*dy)+1.0f-30
                nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja-1]+nut[ia,ja]) : nu
                vs=_viscous_conductance(0.5f0*(r+rho[ia,ja-1]),nf,Sjx[i,1],Sjy[i,1],dn)
            elseif wallmode==Int32(2)
                S=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1]);dn=0.5f0*vol[i,1]/S+1.0f-30
                vs=(4.0f0/3.0f0)*_viscous_conductance(r,nu,Sjx[i,1],Sjy[i,1],dn)
            end
            if j<NJ
                dx=xc[i,j+1]-xc[i,j];dy=yc[i,j+1]-yc[i,j];dn=sqrt(dx*dx+dy*dy)+1.0f-30
                nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia,ja+1]) : nu
                vn=_viscous_conductance(0.5f0*(r+rho[ia,ja+1]),nf,Sjx[i,j+1],Sjy[i,j+1],dn)
            end
            if fullvisc==Int32(1) && (i==1 || i==NI)
                side=i==1 ? Int32(1) : Int32(2);f=side==1 ? Int32(1) : NI+Int32(1)
                sg=side==1 ? -1f0 : 1f0;ig=side==1 ? g : NI+g+Int32(1)
                dx,dy,_,_=CMeshSolverKA.CMeshPressureOutlet._pressure_end_geometry(
                    Six[f,j],Siy[f,j],Sjx[i,j],Sjy[i,j],Sjx[i,j+1],Sjy[i,j+1],sg)
                nf=nu+(sst_on==Int32(1) ? nut[ig,ja] : 0f0)
                c=CMeshSolverKA.CMeshPressureOutlet._pressure_end_conductance(rho[ig,ja],nf,Six[f,j],Siy[f,j],dx,dy)
                side==1 ? (vw+=c) : (ve+=c)
            end
        end
        k=(8.0f0/3.0f0)/r
        lw=rw+k*vw; le=re+k*ve; ls=rs+k*vs; ln=rn+k*vn; Li=lw+le
        good=isfinite(Li) && Li>0.0f0 && isfinite(ls) && ls>=0.0f0 && isfinite(ln) && ln>=0.0f0
        dt=acfl*vol[i,j]/Li
        lamW[i,j]=good ? lw : 0.0f0; lamE[i,j]=good ? le : 0.0f0
        lamS[i,j]=good ? ls : 0.0f0; lamN[i,j]=good ? ln : 0.0f0
        dtau[i,j]=good && isfinite(dt) && dt>0.0f0 ? dt : 0.0f0
        d0[i,j]=good ? vol[i,j]/dt : Inf32
    end
end

@kernel function _k_adi_sst_coeffs!(lamWk,lamEk,lamWw,lamEw,lamSk,lamNk,lamSw,lamNw,d0k,d0w,@Const(dtau),@Const(Fmi),@Const(Fmj),
                                    @Const(rate),@Const(rho),@Const(nut),@Const(sigk),@Const(sigw),@Const(xc),@Const(yc),
                                    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),nu::Float32,wallmode::Int32,
                                    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=rho[i+g,j+g]
        wk,ww=_sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g)
        ek,ew=_sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i+1,j,NI,g)
        sk,sw=_sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf)
        nk,nw=j<NJ ? _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf) : (0.0f0,0.0f0)
        mw=abs(Fmi[i,j]);me=abs(Fmi[i+1,j]);ms=abs(Fmj[i,j]);mn=abs(Fmj[i,j+1])
        lamWk[i,j]=(mw+2.0f0*wk)/r; lamEk[i,j]=(me+2.0f0*ek)/r
        lamWw[i,j]=(mw+2.0f0*ww)/r; lamEw[i,j]=(me+2.0f0*ew)/r
        lamSk[i,j]=(ms+2.0f0*sk)/r; lamNk[i,j]=(mn+2.0f0*nk)/r
        lamSw[i,j]=(ms+2.0f0*sw)/r; lamNw[i,j]=(mn+2.0f0*nw)/r
        dt=dtau[i,j];src=vol[i,j]*rate[i,j]
        base=dt>0.0f0 ? vol[i,j]/dt : Inf32
        d0k[i,j]=base+src; d0w[i,j]=base+src
    end
end

# Thomas algorithm along each row j (i = 1..NI, open at both outlet ends), in place for ncomp components of X. One
# thread per row; cp is Float64 scratch.
@kernel function _k_iline_solve!(X,cp,@Const(lamW),@Const(lamE),@Const(d0),ncomp::Int32,NI::Int32,NJ::Int32)
    jg=@index(Global,Linear)
    j=Int32(jg)
    @inbounds if j<=NJ
        cprev=0.0
        for i in Int32(1):NI
            a=i>Int32(1) ? Float64(lamW[i,j]) : 0.0
            c=i<NI ? Float64(lamE[i,j]) : 0.0
            den=Float64(d0[i,j])+Float64(lamW[i,j])+Float64(lamE[i,j])+a*cprev
            cprev=-c/den
            cp[i,j]=cprev
            if i==Int32(1)
                for k in Int32(1):ncomp; X[i,j,k]=X[i,j,k]/den; end
            else
                for k in Int32(1):ncomp; X[i,j,k]=(X[i,j,k]+a*X[i-Int32(1),j,k])/den; end
            end
        end
        for i in NI-Int32(1):Int32(-1):Int32(1)
            for k in Int32(1):ncomp; X[i,j,k]-=cp[i,j]*X[i+Int32(1),j,k]; end
        end
    end
end

@kernel function _k_scale_d0!(X,@Const(d0),ncomp::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        for k in Int32(1):ncomp; X[i,j,k]*=Float64(d0[i,j]); end
    end
end

# The ADI SST trial: _k_sst_trial_line! with a positivity limiter on the pseudo-time update. With ADI_CFL 2-5 on the
# 128 x 48 CPU case the unlimited update drove k or omega negative at 3-4 attempts per iteration (all sst_trial at stage
# 1), so each of rho k and rho omega may at most fall to ADI_SST_FLOOR times the smaller of its stage-base and current
# values within a stage. A converged state has a zero increment and is never clipped: only the path changes.
const ADI_SST_FLOOR = Ref{Float32}(0.5f0)
@kernel function _k_sst_trial_adi!(ktrial,wtrial,status,@Const(rk),@Const(rw),@Const(k0),@Const(w0),
                                   @Const(DK),@Const(carrier_trial),@Const(rho),weight::Float32,floorf::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        rn=Float32(carrier_trial[i,j,1])
        K=k0[i,j]+weight*((rk[i,j]-k0[i,j])-Float32(DK[i,j,1]))
        W=w0[i,j]+weight*((rw[i,j]-w0[i,j])-Float32(DK[i,j,2]))
        K=max(K,floorf*min(k0[i,j],rk[i,j]))
        W=max(W,floorf*min(w0[i,j],rw[i,j]))
        k=K/rn;w=W/rn
        atfloor=rk[i,j]<=1.0f-12*rho[ia,ja]*(1.0f0+4.0f0*eps(Float32))
        good=isfinite(rn) && rn>0.0f0 && isfinite(K) && (K>=0.0f0 || atfloor) && isfinite(W) &&
            isfinite(k) && isfinite(w) && w>=1.0f-6
        ktrial[i,j]=K;wtrial[i,j]=W;status[i,j]=good ? 0.0f0 : 1.0f0
    end
end

function _adi_workspace(s)
    z32()=fill!(similar(s.dtf),0.0f0)
    (;_line_workspace(s)...,adi=true,lamW=z32(),lamE=z32(),lamWk=z32(),lamEk=z32(),lamWw=z32(),lamEw=z32())
end

# (d0 + J_i) d0^-1 (d0 + J_j) D = rhs for the flow components in L.X, and for rho k, rho omega in L.DK.
function _adi_solve_flow!(c,L)
    s=c.solver;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    _k_iline_solve!(s.backend)(L.X,L.cp,L.lamW,L.lamE,L.d0,Int32(4),ni,nj;ndrange=s.NJ)
    _k_scale_d0!(s.backend)(L.X,L.d0,Int32(4),ni,nj;ndrange=(s.NI,s.NJ))
    _k_line_solve!(s.backend)(L.X,L.cp,L.lamS,L.lamN,L.d0,Int32(4),ni,nj,niw,nif;ndrange=s.NIF+s.NIW)
end
function _adi_solve_sst!(c,L)
    s=c.solver;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    Dk=@view(L.DK[:,:,1:1]);Dw=@view(L.DK[:,:,2:2])
    _k_iline_solve!(s.backend)(Dk,L.cp,L.lamWk,L.lamEk,L.d0k,Int32(1),ni,nj;ndrange=s.NJ)
    _k_iline_solve!(s.backend)(Dw,L.cp,L.lamWw,L.lamEw,L.d0w,Int32(1),ni,nj;ndrange=s.NJ)
    _k_scale_d0!(s.backend)(Dk,L.d0k,Int32(1),ni,nj;ndrange=(s.NI,s.NJ))
    _k_scale_d0!(s.backend)(Dw,L.d0w,Int32(1),ni,nj;ndrange=(s.NI,s.NJ))
    _k_line_solve!(s.backend)(Dk,L.cp,L.lamSk,L.lamNk,L.d0k,Int32(1),ni,nj,niw,nif;ndrange=s.NIF+s.NIW)
    _k_line_solve!(s.backend)(Dw,L.cp,L.lamSw,L.lamNw,L.d0w,Int32(1),ni,nj,niw,nif;ndrange=s.NIF+s.NIW)
end

# One speculative ADI attempt: _enqueue_species_steady_line! with the ADI coefficients and solves.
function _enqueue_species_steady_adi!(c::SpeciesKA,w::SpeciesSteadyKA,cfl::Float32,scale::Float32;wallmode::Int32)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g); L=w.line
    A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    specs=_batch_specs(c,reactive,viscous)
    base=_stage_base(1)
    _residual_deferred!(c,wallmode,base)
    CMeshSolverKA.prepare_dt_ka!(s,cfl;slot=:batched)
    _mixture_cap!(c,cfl;slot=_clock_slice(c,base+_S_MIX))
    reactive && _steady_source!(c,base)
    viscous && _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC))
    s.sst && _sst_residual!(c;slot=_clock_slice(c,base+_S_SST_CAP),faces_slot=_status_slice(c,base+_S_SST_FACES))
    _slot_batch!(c,specs.residual[1])
    a2=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2,1.0f-6))
    _k_adi_flow_coeffs!(s.backend)(L.lamW,L.lamE,L.lamS,L.lamN,L.d0,s.dtf,s.rho,s.ux,s.uy,s.am,c.pressure,s.nut,s.vol,s.xc,s.yc,
        s.Six,s.Siy,s.Sjx,s.Sjy,minf2,s.nu_lam,wallmode,Int32(s.sst),s.fullvisc,scale*STEADY_MARGIN[]*ADI_CFL[],
        ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ))
    if s.sst
        a=c.turbulence
        _k_adi_sst_coeffs!(s.backend)(L.lamWk,L.lamEk,L.lamWw,L.lamEw,L.lamSk,L.lamNk,L.lamSw,L.lamNw,L.d0k,L.d0w,s.dtf,
            s.Fi_r,s.Fj_r,a.rate,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,wallmode,
            ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s);ndrange=(s.NI,s.NJ))
    end
    copyto!(c.conserved.U0,c.conserved.U)
    s.sst && _sst_snapshot!(c)
    for (stage,(wt,_)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
        base=_stage_base(stage)
        if stage>1
            _residual_deferred!(c,wallmode,base)
            _mixture_cap!(c,cfl;stage,slot=_clock_slice(c,base+_S_MIX))
        end
        face_states=c.reconstruction===nothing ? nothing : c.reconstruction.states
        _k_species_flux_i!(s.backend)(c.Fi,s.Fi_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,nothing,ni,nj,g;ndrange=(s.NI+1,s.NJ))
        _k_species_flux_j!(s.backend)(c.Fj,s.Fj_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,nothing,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
        if stage>1
            reactive && _steady_source!(c,base)
            viscous && _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC))
            s.sst && _sst_residual!(c;slot=_clock_slice(c,base+_S_SST_CAP),faces_slot=_status_slice(c,base+_S_SST_FACES))
            _slot_batch!(c,specs.residual[stage])
        end
        _k_line_rhs!(s.backend)(L.X,s.Fi_r,s.Fj_r,s.Fi_mx,s.Fj_mx,s.Fi_my,s.Fj_my,c.Fi,c.Fj,c.source,Int32(reactive),
            ni,nj;ndrange=(s.NI,s.NJ))
        _adi_solve_flow!(c,L)
        _k_species_trial_line!(s.backend)(c.conserved.Utrial,_status_slice(c,base+_S_TRIAL),c.conserved.U,c.conserved.U0,
            L.X,wt,Float64.(s.outlet_eos),Int32(c.eos_single),ni,nj;ndrange=(s.NI,s.NJ))
        _slot_max!(_slot(c,base+_S_TRIAL),_status_slice(c,base+_S_TRIAL))
        if s.sst
            a=c.turbulence
            _k_sst_line_rhs!(s.backend)(L.DK,a.dck,a.dcw,ni,nj;ndrange=(s.NI,s.NJ))
            _adi_solve_sst!(c,L)
            _k_sst_trial_adi!(s.backend)(a.ktrial,a.wtrial,_status_slice(c,base+_S_SST_TRIAL),a.rk,a.rw,a.k0,a.w0,L.DK,
                c.conserved.Utrial,s.rho,wt,ADI_SST_FLOOR[],ni,nj,g;ndrange=(s.NI,s.NJ))
            _slot_max!(_slot(c,base+_S_SST_TRIAL),_status_slice(c,base+_S_SST_TRIAL))
        end
        _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
            Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
        s.sst && _sst_commit!(c)
        _project!(c;batched=true)
        _slot_batch!(c,specs.project[stage])
    end
    s.sst && _sst_prepare!(c;slot=_status_slice(c,_SLOT_POST+3))
    _check_state_deferred!(c,_status_slice(c,_SLOT_POST+4))
    reactive && _steady_final_source!(c)
    _slot_batch!(c,specs.post)
    nothing
end
