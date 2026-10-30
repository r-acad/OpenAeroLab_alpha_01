# ---- Steady mode, wall-normal line relaxation (speed-up plan W3a-steady, 30 Sep 2026) -------------------------------
# memory/project_docs/2026-09-30_W2_STEADY_MODE_FINDINGS.md: plain local time stepping gains little on these C-meshes, because the thin
# cells at the wall and along the wake cut limit the local step in the wall-normal (j) direction exactly as they limit
# the physical one. This variant of the steady iteration keeps the SSPRK3 stages and the admission, and replaces each
# stage's explicit increment dtf/V * rhs by the solution of a scalar tridiagonal system along each j line:
#     (V/dtau + lamS + lamN) D_j - lamS D_{j-1} - lamN D_{j+1} = rhs_j        (per conserved component)
# lamS/lamN are upper bounds of the j-face flux derivatives (the AUSM face-speed bound times the face length, which the
# mixture clock already uses, plus twice the viscous conductance over the density), and V/dtau is set by the i faces
# alone: dtau = scale * margin * cfl * V / Lambda_i. With lamS = lamN = 0 this is the explicit stage; the stiffness of the
# thin j direction is treated implicitly (Swanson, Turkel and Rossow's RK/implicit construction, one direction). The
# SST equations get their own lines: j-face coefficients (|carrier mass flux| + the SST diffusion conductance) over the
# density, and the source stiffness bound `rate` (cmesh_species_sst.jl) point-implicit in the diagonal.
# The foil columns form lines j = 1..NJ (wall and outer faces in the diagonal only); each wake-column pair (i, NI+1-i)
# forms one line of 2 NJ cells through the wake cut, so the cut's thin cells are coupled across it.
# Only the iteration operator changes: a converged state solves the same discrete equations (rhs = 0). Nothing here is
# reached from the physical-time step.

# Outlet-end safeguard of the line relaxation (30 Sep 2026): the first and last LINE_END_COLUMNS columns (the C-mesh's
# outlet ends, where the wake cut's thin cells meet the fixed-pressure outlet) take LINE_END_FACTOR of the local step.
# At CFL 0.9 x margin 1.0 the 128 x 48 CPU case had Cp excursions to -5.8 (Cp_v -6.0) in exactly those cells. With the
# factor 0.4 the run survived but rejected 0.4-0.55 attempts per iteration, so the default is off (1.0): the working
# setting is CFL 0.45 x margin 0.8, which needs no safeguard.
const LINE_END_COLUMNS = Ref{Int}(3)
const LINE_END_FACTOR = Ref{Float32}(1.0f0) # default off: at CFL 0.9 with 0.4 the 128 x 48 case moved its rejections elsewhere (0.4-0.55 per iteration, 30 Sep); CFL 0.45 x 0.8 needed neither
@kernel function _k_line_flow_coeffs!(lamS,lamN,d0,dtau,@Const(rho),@Const(ux),@Const(uy),@Const(am),@Const(p),
                                      @Const(nut),@Const(vol),@Const(xc),@Const(yc),
                                      @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),
                                      minf2::Float32,nu::Float32,wallmode::Int32,sst_on::Int32,fullvisc::Int32,
                                      scfl::Float32,endc::Int32,endf::Float32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        rw=_species_face_rate(rho,ux,uy,am,p,ia-1,ja,ia,ja,r,Six[i,j],Siy[i,j],minf2)
        re=_species_face_rate(rho,ux,uy,am,p,ia,ja,ia+1,ja,r,Six[i+1,j],Siy[i+1,j],minf2)
        rs=_species_face_rate(rho,ux,uy,am,p,ia,ja-1,ia,ja,r,Sjx[i,j],Sjy[i,j],minf2)
        rn=_species_face_rate(rho,ux,uy,am,p,ia,ja,ia,ja+1,r,Sjx[i,j+1],Sjy[i,j+1],minf2)
        # viscous conductances per face, as _k_species_viscous_dt! forms them
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
        k=(8.0f0/3.0f0)/r   # twice the 4/3 normal-stress bound, per unit density
        Li=rw+re+k*(vw+ve); ls=rs+k*vs; ln=rn+k*vn
        good=isfinite(Li) && Li>0.0f0 && isfinite(ls) && ls>=0.0f0 && isfinite(ln) && ln>=0.0f0
        # the outlet-end columns (i <= endc or i > NI - endc) take endf of the step: the fixed-pressure outlet meeting the thin
        # wake-cut cells is where the tangential explicit part goes unstable first (CPU 128 x 48 at CFL 0.9, 30 Sep)
        dt=(i<=endc || i>NI-endc ? endf : 1.0f0)*scfl*vol[i,j]/Li
        lamS[i,j]=good ? ls : 0.0f0; lamN[i,j]=good ? ln : 0.0f0
        dtau[i,j]=good && isfinite(dt) && dt>0.0f0 ? dt : 0.0f0
        d0[i,j]=good ? vol[i,j]/dt : Inf32
    end
end

@kernel function _k_line_sst_coeffs!(lamSk,lamNk,lamSw,lamNw,d0k,d0w,@Const(dtau),@Const(Fmj),@Const(rate),
                                     @Const(rho),@Const(nut),@Const(sigk),@Const(sigw),@Const(xc),@Const(yc),
                                     @Const(Sjx),@Const(Sjy),@Const(vol),nu::Float32,wallmode::Int32,
                                     NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=rho[i+g,j+g]
        sk,sw=_sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf)
        nk,nw=j<NJ ? _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf) : (0.0f0,0.0f0)
        ms=abs(Fmj[i,j]);mn=abs(Fmj[i,j+1])
        lamSk[i,j]=(ms+2.0f0*sk)/r; lamNk[i,j]=(mn+2.0f0*nk)/r
        lamSw[i,j]=(ms+2.0f0*sw)/r; lamNw[i,j]=(mn+2.0f0*nw)/r
        dt=dtau[i,j];src=vol[i,j]*rate[i,j]
        base=dt>0.0f0 ? vol[i,j]/dt : Inf32
        d0k[i,j]=base+src; d0w[i,j]=base+src
    end
end

# Right-hand sides of the flow stage (the trial kernel's rhs), Float64, into X[i,j,1:4].
@kernel function _k_line_rhs!(X,@Const(Fir),@Const(Fjr),@Const(Fix),@Const(Fjx),@Const(Fiy),@Const(Fjy),
                              @Const(Fiq),@Const(Fjq),@Const(source),reactive::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        X[i,j,1]=_species_divergence64(Fir,Fjr,i,j)
        X[i,j,2]=_species_divergence64(Fix,Fjx,i,j)
        X[i,j,3]=_species_divergence64(Fiy,Fjy,i,j)
        X[i,j,4]=_species_divergence64(Fiq,Fjq,i,j)-(reactive!=Int32(0) ? Float64(source[i,j]) : 0.0)
    end
end

# The cell at position m of line l, with its couplings: returns (i, j, a, c) where a couples to position m-1 and c to
# m+1 (both as positive lambdas, entered with a minus sign). Foil line l: column NIW+l, j = m. Wake line l: pair
# (l, NI+1-l): m = 1..NJ runs down column NI+1-l from j = NJ to the cut, m = NJ+1..2NJ runs up column l from the cut.
@inline function _line_cell(l,m,lamS,lamN,NI,NJ,NIW,NIF)
    if l<=NIF
        i=NIW+l;j=m
        (i,j,j>1 ? lamS[i,j] : 0.0f0,j<NJ ? lamN[i,j] : 0.0f0)
    else
        w=l-NIF;ip=NI+Int32(1)-w
        if m<=NJ
            j=NJ+Int32(1)-m
            (ip,j,j<NJ ? lamN[ip,j] : 0.0f0,lamS[ip,j])   # previous = north neighbour, next = south (the cut at j = 1)
        else
            j=m-NJ
            (w,j,lamS[w,j],j<NJ ? lamN[w,j] : 0.0f0)       # previous = south (the cut at j = 1), next = north
        end
    end
end

# Thomas algorithm along each line, for ncomp components of X (in place: rhs in, increment out). cp is Float64 scratch
# for the modified super-diagonal. One thread per line.
@kernel function _k_line_solve!(X,cp,@Const(lamS),@Const(lamN),@Const(d0),ncomp::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    l=Int32(lg)
    @inbounds if l<=NIF+NIW
        M=l<=NIF ? NJ : Int32(2)*NJ
        cprev=0.0
        for m in Int32(1):M
            i,j,a,c=_line_cell(l,m,lamS,lamN,NI,NJ,NIW,NIF)
            b=Float64(d0[i,j])+Float64(lamS[i,j])+Float64(lamN[i,j])
            den=b+Float64(a)*cprev   # c' is stored negative: den = b - A_m c'_{m-1} with A_m = -a
            cprev=-Float64(c)/den
            cp[i,j]=cprev
            if m==Int32(1)
                for k in Int32(1):ncomp; X[i,j,k]=X[i,j,k]/den; end
            else
                ip_,jp_,_,_=_line_cell(l,m-Int32(1),lamS,lamN,NI,NJ,NIW,NIF)
                for k in Int32(1):ncomp; X[i,j,k]=(X[i,j,k]+Float64(a)*X[ip_,jp_,k])/den; end
            end
        end
        for m in M-Int32(1):Int32(-1):Int32(1)
            i,j,_,_=_line_cell(l,m,lamS,lamN,NI,NJ,NIW,NIF)
            inx,jnx,_,_=_line_cell(l,m+Int32(1),lamS,lamN,NI,NJ,NIW,NIF)
            for k in Int32(1):ncomp; X[i,j,k]-=cp[i,j]*X[inx,jnx,k]; end
        end
    end
end

# The line stage trial: _k_species_trial! with the solved increment D in place of dtf/V * rhs; same admissibility.
@kernel function _k_species_trial_line!(trial,status,@Const(U),@Const(U0),@Const(D),w::Float32,
                                        ep::NTuple{7,Float64},single::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        weight=Float64(w)
        for k in 1:4
            trial[i,j,k]=U0[i,j,k]+weight*((U[i,j,k]-U0[i,j,k])-D[i,j,k])
        end
        r=trial[i,j,1];mx=trial[i,j,2];my=trial[i,j,3];qn=trial[i,j,4]
        good=isfinite(r) && r>0 && isfinite(mx) && isfinite(my) && isfinite(qn) && 0<=qn<=r
        if good
            u=mx/r;v=my/r;pv,A,rl,rv,psm,lq,afloor=ep
            branch=single==Int32(1) ? _phase_state_single(r,Float32(_alpha(qn/r,rl,rv)),ep)[1] :
                                      _phase_state(r,_alpha(qn/r,rl,rv),pv,A,rl,rv,psm,afloor,lq)[1]
            good=branch && isfinite(u) && isfinite(v) && isfinite(Float32(u)^2+Float32(v)^2) &&
                 isfinite(Float32(r)) && Float32(r)>0f0
        end
        status[i,j]=good ? 0f0 : 1f0
    end
end

# The line SST trial: _k_sst_species_trial! with the solved increments in place of dtf/V * dc; same admissibility.
@kernel function _k_sst_trial_line!(ktrial,wtrial,status,@Const(rk),@Const(rw),@Const(k0),@Const(w0),
                                    @Const(DK),@Const(carrier_trial),@Const(rho),weight::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        rn=Float32(carrier_trial[i,j,1])
        K=k0[i,j]+weight*((rk[i,j]-k0[i,j])-Float32(DK[i,j,1]))
        W=w0[i,j]+weight*((rw[i,j]-w0[i,j])-Float32(DK[i,j,2]))
        k=K/rn;w=W/rn
        atfloor=rk[i,j]<=1.0f-12*rho[ia,ja]*(1.0f0+4.0f0*eps(Float32))
        good=isfinite(rn) && rn>0.0f0 && isfinite(K) && (K>=0.0f0 || atfloor) && isfinite(W) &&
            isfinite(k) && isfinite(w) && w>=1.0f-6
        ktrial[i,j]=K;wtrial[i,j]=W;status[i,j]=good ? 0.0f0 : 1.0f0
    end
end
@kernel function _k_sst_line_rhs!(DK,@Const(dck),@Const(dcw),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        DK[i,j,1]=Float64(dck[i,j]);DK[i,j,2]=Float64(dcw[i,j])
    end
end
# k and omega have different coefficients: solve each on its own (one component per launch, with its own view).
function _line_solve_sst!(c,w)
    s=c.solver;L=w.line;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    # arguments written out (no splat) so the static kernel-launch audit (preflight gate 2) can check them
    _k_line_solve!(s.backend)(@view(L.DK[:,:,1:1]),L.cp,L.lamSk,L.lamNk,L.d0k,Int32(1),ni,nj,niw,nif;ndrange=s.NIF+s.NIW)
    _k_line_solve!(s.backend)(@view(L.DK[:,:,2:2]),L.cp,L.lamSw,L.lamNw,L.d0w,Int32(1),ni,nj,niw,nif;ndrange=s.NIF+s.NIW)
end

function _line_workspace(s)
    z32()=fill!(similar(s.dtf),0.0f0)
    z64(dims...)=fill!(similar(s.dtf,Float64,dims...),0.0)
    (;lamS=z32(),lamN=z32(),d0=z32(),lamSk=z32(),lamNk=z32(),lamSw=z32(),lamNw=z32(),d0k=z32(),d0w=z32(),
      X=z64(s.NI,s.NJ,4),DK=z64(s.NI,s.NJ,2),cp=z64(s.NI,s.NJ))
end

# One speculative line-relaxed steady attempt (see the header). The coefficients and local steps are formed once, on
# the stage-1 state, and frozen for the three stages.
function _enqueue_species_steady_line!(c::SpeciesKA,w::SpeciesSteadyKA,cfl::Float32,scale::Float32;wallmode::Int32)
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
    _k_line_flow_coeffs!(s.backend)(L.lamS,L.lamN,L.d0,s.dtf,s.rho,s.ux,s.uy,s.am,c.pressure,s.nut,s.vol,s.xc,s.yc,
        s.Six,s.Siy,s.Sjx,s.Sjy,minf2,s.nu_lam,wallmode,Int32(s.sst),s.fullvisc,scale*STEADY_MARGIN[]*cfl,
        Int32(LINE_END_COLUMNS[]),LINE_END_FACTOR[],
        ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ))
    # the block variant (cmesh_species_block.jl): the flow lines take the first-order face Jacobians of this state, the
    # SST lines their upwind advection/diffusion derivatives
    block=hasproperty(L,:JL)
    block && _blk_jacobians!(c,L,wallmode)
    if s.sst
        a=c.turbulence
        if block
            _blk_sst_coeffs!(c,L,wallmode)
        else
        _k_line_sst_coeffs!(s.backend)(L.lamSk,L.lamNk,L.lamSw,L.lamNw,L.d0k,L.d0w,s.dtf,s.Fj_r,a.rate,s.rho,s.nut,
            a.sigk,a.sigw,s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.nu_lam,wallmode,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s);
            ndrange=(s.NI,s.NJ))
        end
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
        if block
            _blk_line_solve!(c,L;factor=stage==1)   # frozen coefficients: factorise once per iteration
        else
            _k_line_solve!(s.backend)(L.X,L.cp,L.lamS,L.lamN,L.d0,Int32(4),ni,nj,Int32(s.NIW),Int32(s.NIF);ndrange=s.NIF+s.NIW)
        end
        _k_species_trial_line!(s.backend)(c.conserved.Utrial,_status_slice(c,base+_S_TRIAL),c.conserved.U,c.conserved.U0,
            L.X,wt,Float64.(s.outlet_eos),Int32(c.eos_single),ni,nj;ndrange=(s.NI,s.NJ))
        # The trial flags are reduced now, before any launch consumes the trial: on the CPU backend a launch that
        # evaluates an inadmissible state can throw (the GPU propagates NaN), and step_species_steady_ka! classifies
        # such an exception by these flags. The batch below reduces the same slices again (max: unchanged).
        _slot_max!(_slot(c,base+_S_TRIAL),_status_slice(c,base+_S_TRIAL))
        if s.sst
            a=c.turbulence
            _k_sst_line_rhs!(s.backend)(L.DK,a.dck,a.dcw,ni,nj;ndrange=(s.NI,s.NJ))
            _line_solve_sst!(c,w)
            _k_sst_trial_line!(s.backend)(a.ktrial,a.wtrial,_status_slice(c,base+_S_SST_TRIAL),a.rk,a.rw,a.k0,a.w0,L.DK,
                c.conserved.Utrial,s.rho,wt,ni,nj,g;ndrange=(s.NI,s.NJ))
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
