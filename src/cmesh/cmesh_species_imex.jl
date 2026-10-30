# ---- Time-accurate wall-normal line-implicit step (speed-up plan W3a, 30 Sep 2026) ---------------------------------
# docs/2026-09-30_W3A_BLOCK_LINE_OPERATOR.md. An opt-in PHYSICAL-TIME step, not the default: step_species_ka! is
# unchanged and never reaches this file; the driver selects it with TOPO_SPECIES_IMEX=1.
#
# The explicit step's global dt is set by the wall-normal acoustic limit of the thin first-layer cells, at the wall
# and along the wake cut. Here each SSPRK3 stage is linearly implicit in the j direction:
#     (V/dt I + J_j) D = rhs,   U_stage = U0 + w ((U - U0) - D)
# with rhs the stage's full flux divergence minus source (the explicit trial's) and J_j the block operator of the
# first-order j faces (cmesh_species_block.jl), frozen on the step's stage-1 state. Every stage is thus an IMEX Euler
# step (i faces explicit, j faces linearly implicit), combined by the SSPRK3 weights: third order in the explicit part,
# first order in the implicit part, and stiff j modes decay by at least 1/3 per step. The global dt comes from the
# i-direction clocks only: the mixture face bound of the two i faces and their outward carrier flux, the viscous i
# conductances, the SST's i advection/diffusion. By default the phase-change source stays explicit with its own clock
# (a cavitating run is then still bound by it). With source=:implicit (W3a second stage) its Jacobian V dgamma/d(r,q),
# including the EOS pressure feedback on its own driving force, enters the q row of every diagonal block, coupled to the
# neighbours' pressure through the face Jacobians' q columns (lesson 2 of the plan: never a segregated source); its
# clock then leaves the global dt, and the trial admissibility (0 <= q <= r, an admissible EOS state) is the safeguard
# the drain limits were. The SST source stiffness is point-implicit (the bound `rate` in the diagonal of the SST lines). Stages 2 and 3 re-check the same i clocks, and every stage's trial carries the explicit
# path's admissibility; a failure restores the base state and retries at half the step, as step_species_ka! does.
# Refused: the conservation ledger (its accounting assumes the explicit stage fluxes), kinetic faces, and a solver
# GLOBAL_DT other than 1. Accuracy against the explicit path is a registered bridge, not a property of this code.
# Strip topology (stage 4, 7 Oct 2026; memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 5): a strip pair's attempt is
# _enqueue_pair_imex! (cmesh_species_strip.jl), whose wake lines run through the strip; without a strip nothing here changes.

mutable struct SpeciesImexKA{L,A,X}
    const line::L       # the block workspace (_block_workspace)
    const scratch::A    # the explicit SST cap, computed by _sst_residual! and not used
    const source_scratch::A # the explicit source cap when the source is implicit (only its sign is used)
    const implicit_source::Bool # W3a second stage: the phase source linearised into the line systems
    const onset::A          # per-cell onset-crossing flags of the current stage (reactive runs)
    attempts::Int
    rejections::Int
    last_rejection::Union{Nothing,SpeciesNumericalRejection}
    dt_hint::Float32        # the last admitted dt: the next first attempt is capped at DT_GROWTH x it
    # strip topology (stage 4, 7 Oct 2026; memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 5 item 6): nothing on a cache without
    # a strip (its type parameter Nothing), or the strip block's share (line, scratch, source_scratch, onset), sized by the
    # strip's carrier (_imex_strip_workspace, cmesh_species_strip.jl)
    const strip::X
end

function _admit_imex(c::SpeciesKA)
    # strip topology: refused by name until stage 4 (5 Oct 2026); since stage 4 (7 Oct 2026, plan section 5) the wake line runs
    # through the strip (_enqueue_pair_imex!, cmesh_species_strip.jl), and a strip pair is refused here only for what every
    # cache is (plan 5 item 6: nothing new; the ledger, kinetic faces, refused at the pair's build, and the margin, below)
    LEDGER_ENABLED[] && throw(ArgumentError("the line-implicit step refuses the conservation ledger (its accounting assumes explicit stage fluxes); set LEDGER_ENABLED[] = false"))
    c.kinetic===nothing || throw(ArgumentError("the line-implicit step supports the AUSM+-up faces only"))
    # The stage-1 clock margin (3 Oct 2026) is defined for the explicit step's proposal. This step forms its proposal from
    # its i-direction clocks and its own growth cap (DT_GROWTH), and its W3a admission was measured without a margin, so a
    # cache built with one is refused here, by name, at build and at every step (memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/FIX.md).
    c.stage_margin>0.0f0 && throw(ArgumentError("the line-implicit step refuses the stage clock margin (stage_clock_margin = $(c.stage_margin), driver TOPO_SPECIES_STAGE_CLOCK_MARGIN): it is defined for the explicit step only; build the cache without it"))
    CMeshSolverKA.GLOBAL_DT[]==1 || throw(ArgumentError("the line-implicit step needs the solver's GLOBAL_DT = 1"))
    # The Schnerr-Sauer source (10 Oct 2026) is refused by name in version 1: the implicit source Jacobian (_k_blk_source_jac!) is
    # a finite difference through the Merkle _phase_source, which it does not extend, and the W3a admission was measured on Merkle.
    c.source_model==Int32(1) && throw(ArgumentError("the line-implicit step does not support the Schnerr-Sauer source (source_model=:schnerr_sauer, driver TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer) in version 1: its implicit source Jacobian is a finite difference through the Merkle source; use the explicit step"))
    nothing
end

"""Workspace of the time-accurate wall-normal line-implicit step (cmesh_species_imex.jl) for one species cache.
`source=:implicit` (W3a second stage) linearises the phase-change source into the line systems (its Jacobian in the q
row of each diagonal block, coupled to the pressure through the face Jacobians' q columns) and removes the source clock
from the global dt; `:explicit` (default) keeps the source explicit with its own clock."""
function build_species_imex(c::SpeciesKA;source::Symbol=:explicit)
    _admit_imex(c)
    s=c.solver
    s.NI==2*s.NIW+s.NIF || throw(ArgumentError("the line-implicit step needs the C-mesh layout NI = 2 NIW + NIF"))
    # strip topology (stage 4, plan section 5 item 6): the layout check per block, the strip block's against the C block's
    # wake stations (_imex_strip_layout, cmesh_species_strip.jl); without a strip resolved at compile time
    c.strip===nothing || _imex_strip_layout(c)
    source in (:explicit,:implicit) || throw(ArgumentError("source must be :explicit or :implicit"))
    SpeciesImexKA(_block_workspace(s),fill!(similar(s.dtf),0.0f0),fill!(similar(s.dtf),0.0f0),source===:implicit,
                  fill!(similar(s.dtf,s.NI,s.NJ),0.0f0),0,0,nothing,Inf32,_imex_strip_workspace(c))
end

# The mixture face clock of the two i faces and their outward carrier flux (reconstructed face states when present).
@kernel function _k_imex_face_dt!(cap,states,@Const(rho),@Const(ux),@Const(uy),@Const(am),@Const(p),@Const(Fi),@Const(vol),
                                  @Const(Six),@Const(Siy),cfl::Float32,minf2::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        west=states!==nothing && i>1 ? _rate(_state(states,i-1,j,2),_state(states,i,j,1),r,Six[i,j],Siy[i,j],minf2) :
            _species_face_rate(rho,ux,uy,am,p,ia-1,ja,ia,ja,r,Six[i,j],Siy[i,j],minf2)
        east=states!==nothing && i<NI ? _rate(_state(states,i,j,2),_state(states,i+1,j,1),r,Six[i+1,j],Siy[i+1,j],minf2) :
            _species_face_rate(rho,ux,uy,am,p,ia,ja,ia+1,ja,r,Six[i+1,j],Siy[i+1,j],minf2)
        rate=west+east+(max(Fi[i+1,j],0.0f0)+max(-Fi[i,j],0.0f0))/r
        dt=cfl*vol[i,j]/rate
        cap[i,j]=isfinite(rate) && rate>0.0f0 && isfinite(dt) && dt>0.0f0 ? dt : 0.0f0
    end
end

# The viscous clock of the i faces (and of the exact outlet end faces), as _k_species_viscous_dt! forms them.
# Strip topology (stage 4, 7 Oct 2026; plan section 5 item 5): with the strip block's context (ctx.iwall == 1) its i = 1 face
# is the base, a no-slip wall on an i face: the 4/3 Stokes conductance of _k_species_viscous_dt!'s base-wall term (d1 = 0.5
# vol/|S_i1|) instead of an outlet end face, and the end-face term on the outlet side of the context's `sides` mask only, as
# the explicit clock; ctx === nothing (every existing launch) compiles to today's kernel.
@kernel function _k_imex_viscous_dt!(cap,@Const(rho),@Const(nut),@Const(vol),@Const(xc),@Const(yc),
                                     @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),nu::Float32,sst_on::Int32,fullvisc::Int32,
                                     NI::Int32,NJ::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g; ja=j+g; r=rho[ia,ja]; conductance=0.0f0
        if i>1
            dx=xc[i,j]-xc[i-1,j]; dy=yc[i,j]-yc[i-1,j]; dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia-1,ja]+nut[ia,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia-1,ja]),nf,Six[i,j],Siy[i,j],dn)
        elseif ctx!==nothing && ctx.iwall==Int32(1) && ctx.wallmode==Int32(2)
            S=sqrt(Six[1,j]*Six[1,j]+Siy[1,j]*Siy[1,j])
            dn=0.5f0*vol[1,j]/S+1.0f-30
            conductance+=(4.0f0/3.0f0)*_viscous_conductance(r,nu,Six[1,j],Siy[1,j],dn)
        end
        if i<NI
            dx=xc[i+1,j]-xc[i,j]; dy=yc[i+1,j]-yc[i,j]; dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia+1,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia+1,ja]),nf,Six[i+1,j],Siy[i+1,j],dn)
        end
        if fullvisc==Int32(1) && nu>0.0f0 && (i==1 || i==NI) &&
           (ctx===nothing || ((ctx.sides>>(i==1 ? Int32(0) : Int32(1)))&Int32(1))==Int32(1))
            side=i==1 ? Int32(1) : Int32(2);f=side==1 ? Int32(1) : NI+Int32(1)
            sg=side==1 ? -1f0 : 1f0;ig=side==1 ? g : NI+g+Int32(1)
            dx,dy,_,_=CMeshSolverKA.CMeshPressureOutlet._pressure_end_geometry(
                Six[f,j],Siy[f,j],Sjx[i,j],Sjy[i,j],Sjx[i,j+1],Sjy[i,j+1],sg)
            nf=nu+(sst_on==Int32(1) ? nut[ig,ja] : 0f0)
            conductance+=CMeshSolverKA.CMeshPressureOutlet._pressure_end_conductance(rho[ig,ja],nf,Six[f,j],Siy[f,j],dx,dy)
        end
        mass=r*vol[i,j]; Lnu=conductance/mass
        good=isfinite(mass) && mass>0.0f0 && isfinite(Lnu) && Lnu>=0.0f0
        cap[i,j]=good ? (Lnu>0.0f0 ? 0.25f0/Lnu : Inf32) : 0.0f0
    end
end

# The SST clock of the i faces: _k_sst_species_residual!'s cap without the j faces and without the source stiffness,
# which the SST lines treat implicitly.
# Strip topology (stage 4, 7 Oct 2026; plan section 5 item 5): with the strip block's context its i = 1 face is the base wall,
# whose two-point conductance (the wall branch of _sst_i_diff, _sst_i_diff_strip) the flux kernel stores in `terms` and which,
# without a reconstruction, is formed here as the flux kernel forms it; under the limited correction the base wall takes no
# allowance (it is uncorrected, _sst_corrected_i, as the explicit V2 clock counts it). ctx === nothing: today's kernel.
@kernel function _k_imex_sst_cap!(cap,@Const(Fmi),@Const(rk),@Const(rw),faces,terms,@Const(rho),@Const(nut),
                                  @Const(sigk),@Const(sigw),@Const(xc),@Const(yc),@Const(Six),@Const(Siy),@Const(vol),
                                  nu::Float32,NI::Int32,NJ::Int32,g::Int32,climit::Float32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        if terms===nothing
            ek,ew=ctx===nothing ? _sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i+1,j,NI,g) :
                                  _sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i+1,j,NI,g,ctx)
            wk,ww=ctx===nothing ? _sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g) :
                                  _sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g,ctx)
        else
            ek=terms[i+1,j,1,1];ew=terms[i+1,j,2,1];wk=terms[i,j,1,1];ww=terms[i,j,2,1]
        end
        mass=rho[i+g,j+g]*vol[i,j]
        fe=max(Fmi[i+1,j],0.0f0);fw=max(-Fmi[i,j],0.0f0)
        if faces===nothing
            R=(fe+fw+max(ek+wk,ew+ww))/mass
            good=isfinite(mass) && mass>0.0f0 && isfinite(R) && R>=0.0f0
        else
            ok=fe*_sst_own_face_value(rk,rho,faces,i,j,2,1,NI,NJ,g)+fw*_sst_own_face_value(rk,rho,faces,i,j,1,1,NI,NJ,g)
            ow=fe*_sst_own_face_value(rw,rho,faces,i,j,2,2,NI,NJ,g)+fw*_sst_own_face_value(rw,rho,faces,i,j,1,2,NI,NJ,g)
            K=rk[i,j];W=rw[i,j];V=vol[i,j]
            atfloor=K<=1.0f-12*rho[i+g,j+g]*(1.0f0+4.0f0*eps(Float32))
            Rk=(atfloor ? 0.0f0 : (ok/K)/V)+(ek+wk)/mass
            Rw=(ow/W)/V+(ew+ww)/mass
            if terms!==nothing && climit>=0f0
                # limited correction (opt-in, 2 Oct 2026): as _k_sst_species_residual!, (1+climit) x the two-point term.
                # Its face set is already the corrected one (V2, 3 Oct 2026): this clock counts only the i faces, which
                # are corrected (_sst_corrected_i) except at the open ends i = 1 and i = NI+1, whose two-point coefficient
                # is exactly zero; the uncorrected j = 1 faces never enter it. So V2 leaves it bitwise unchanged.
                if ctx!==nothing && ctx.iwall==Int32(1) && i==1
                    # the strip's base wall (stage 4): uncorrected, with a nonzero two-point coefficient: no allowance on it
                    Rk+=atfloor ? 0.0f0 : climit*ek/mass;Rw+=climit*ew/mass
                else
                    Rk+=atfloor ? 0.0f0 : climit*(ek+wk)/mass;Rw+=climit*(ew+ww)/mass
                end
            elseif terms!==nothing
                dk=max(terms[i+1,j,3,1],0.0f0)+max(-terms[i,j,3,1],0.0f0)
                dw=max(terms[i+1,j,4,1],0.0f0)+max(-terms[i,j,4,1],0.0f0)
                Rk+=(atfloor ? 0.0f0 : (dk/K)/V);Rw+=(dw/W)/V
            end
            R=max(Rk,Rw)
            good=isfinite(mass) && mass>0.0f0 && isfinite(K) && K>0.0f0 && isfinite(W) && W>0.0f0 &&
                isfinite(V) && V>0.0f0 && isfinite(ok) && ok>=0.0f0 && isfinite(ow) && ow>=0.0f0 && isfinite(R) && R>=0.0f0
        end
        cap[i,j]=good ? (R>0.0f0 ? 0.25f0/R : Inf32) : 0.0f0
    end
end

# Onset crossing (30 Sep 2026; memory/project_docs/2026-09-30_W3A_BLOCK_LINE_OPERATOR.md, section 7): a vapour-free cell whose stage-1
# state is at or above p_v may not reach a trial pressure below ONSET_FRACTION x p_v within one step. The source is
# identically zero above p_v, so no linearisation at the base state knows that a large step will cross the kink; the
# explicit step crosses it gradually. A crossing is rejected and retried at half the step, like a clock; once a cell is
# below p_v its source is active (and, with source=:implicit, treated implicitly). Flag slot: offset 3 of each stage
# block, the retired residual-time slot the steady mode also uses (the two modes are exclusive).
const ONSET_FRACTION = Ref{Float32}(0.5f0)
const _S_IMEX_ONSET = 3
@kernel function _k_imex_onset!(flags,@Const(Utrial),@Const(U0),ep::NTuple{7,Float64},single::Int32,frac::Float32,
                                NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        flag=0.0f0
        if U0[i,j,4]==0.0
            pv,A,rl,rv,psm,lq,afloor=ep
            r0=U0[i,j,1]
            g0,p0,_=single==Int32(1) ? _phase_state_single(r0,0.0f0,ep) : _phase_state(r0,0.0,pv,A,rl,rv,psm,afloor,lq)
            if g0 && Float64(p0)>=pv
                rt=Utrial[i,j,1];qt=Utrial[i,j,4]
                ok=isfinite(rt) && rt>0.0 && isfinite(qt) && 0.0<=qt<=rt
                if ok
                    gt,pt,_=single==Int32(1) ? _phase_state_single(rt,Float32(_alpha(qt/rt,rl,rv)),ep) :
                                               _phase_state(rt,_alpha(qt/rt,rl,rv),pv,A,rl,rv,psm,afloor,lq)
                    ok=gt && Float64(pt)>=Float64(frac)*pv
                end
                flag=ok ? 0.0f0 : 1.0f0
            end
        end
        flags[i,j]=flag
    end
end
# The deferred decision, then the onset flags of the three stages (only where the attempt would otherwise be admitted).
function _imex_decision(h,max_dt,recon,reactive,viscous,sst,inlet)
    d=_deferred_decision(h,max_dt,recon,reactive,viscous,sst,inlet)
    (d isa Float32 && reactive) || return d
    for stage in 1:3
        h[_stage_base(stage)+_S_IMEX_ONSET]==0x00000000 || return SpeciesNumericalRejection(stage,:onset_crossing,d)
    end
    d
end

# The line-implicit stage trial: _k_species_trial_line!'s update and admissibility, with a vapour flush (30 Sep 2026,
# found on the U25-condition CPU runs). The implicit upwind solve spreads vapour downstream as an exponentially small,
# correctly positive tail, which reaches values like 1e-49 far from any cavity. The reconstructed face mass fractions are
# Float32, whose subnormal floor is 1.4e-45: a cell holding 1e-49 can be charged an outflow computed from a face value of
# 1e-45, which exceeds its content at practically any dt, so the trial refused it all the way down the retries. A trial
# partial density within Q_FLUSH x rho of zero, on either side, is set to exactly zero. The vapour inventory then changes
# by at most Q_FLUSH relative per cell and step, twenty orders below the ledger's tolerance; the explicit step never
# creates such tails and does not flush.
const Q_FLUSH = Ref{Float64}(1.0e-30)
# Step growth (30 Sep 2026): a step's first attempt is capped at DT_GROWTH x the last admitted dt, so that a run whose
# step is limited by a rejection (the onset crossing in a cavitating transient) does not retry the full i-clock step at
# every step. With a steady admitted dt the cap does not bind.
const DT_GROWTH = Ref{Float32}(2.0f0)
@kernel function _k_imex_trial!(trial,status,@Const(U),@Const(U0),@Const(D),w::Float32,
                                ep::NTuple{7,Float64},single::Int32,qflush::Float64,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        weight=Float64(w)
        for k in 1:4
            trial[i,j,k]=U0[i,j,k]+weight*((U[i,j,k]-U0[i,j,k])-D[i,j,k])
        end
        r=trial[i,j,1];qn=trial[i,j,4]
        if isfinite(r) && r>0 && isfinite(qn) && abs(qn)<=qflush*r
            qn=0.0; trial[i,j,4]=0.0
        end
        mx=trial[i,j,2];my=trial[i,j,3]
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

# The implicit source's clock slot: floatmax where the explicit cap is positive, 0 where the source state is unresolved.
@kernel function _k_imex_source_gate!(slot,@Const(cap),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        slot[i,j]=cap[i,j]>0.0f0 ? floatmax(Float32) : 0.0f0
    end
end

# The stage's clocks into their slots (i faces only; the source clock unchanged unless the source is implicit), and the
# stage's source term and SST residual, which the line right-hand sides consume.
function _imex_caps!(c,w,cfl,base,reactive,viscous)
    s=c.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    a2=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2,1.0f-6))
    r=c.reconstruction
    states=r===nothing ? nothing : r.states
    _k_imex_face_dt!(s.backend)(_clock_slice(c,base+_S_MIX),states,s.rho,s.ux,s.uy,s.am,c.pressure,s.Fi_r,s.vol,
        s.Six,s.Siy,cfl,minf2,ni,nj,g;ndrange=(s.NI,s.NJ))
    if reactive
        if w.implicit_source
            # the source term as before; its clock leaves the global dt: the slot takes floatmax where the explicit cap is
            # positive (a resolved source state) and 0 where it is not (the unresolved-state refusal stays)
            _source_terms!(c;slot=w.source_scratch)
            _k_imex_source_gate!(s.backend)(_clock_slice(c,base+_S_SOURCE),w.source_scratch,ni,nj;ndrange=(s.NI,s.NJ))
        else
            _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE))
        end
    end
    viscous && _k_imex_viscous_dt!(s.backend)(_clock_slice(c,base+_S_VISC),s.rho,s.nut,s.vol,s.xc,s.yc,
        s.Six,s.Siy,s.Sjx,s.Sjy,s.nu_lam,Int32(s.sst),s.fullvisc,ni,nj,g,nothing;ndrange=(s.NI,s.NJ))
    if s.sst
        a=c.turbulence
        _sst_residual!(c;slot=w.scratch,faces_slot=_status_slice(c,base+_S_SST_FACES))
        _k_imex_sst_cap!(s.backend)(_clock_slice(c,base+_S_SST_CAP),s.Fi_r,a.rk,a.rw,
            r===nothing ? nothing : r.sst_states,r===nothing ? nothing : r.sst_diffusion_terms,
            s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.vol,s.nu_lam,ni,nj,g,c.sst_climit,nothing;ndrange=(s.NI,s.NJ))
    end
    nothing
end

# One speculative line-implicit attempt: _enqueue_species_deferred!'s structure with the i clocks, the frozen block
# operator and the line solves in place of the explicit increments.
function _enqueue_species_imex!(c::SpeciesKA,w::SpeciesImexKA,cfl::Float32;wallmode::Int32)
    # a strip pair's attempt (stage 4, plan section 5: the wake line through the strip, cmesh_species_strip.jl); without a strip
    # this test is resolved at compile time
    c.strip===nothing || return _enqueue_pair_imex!(c,w,cfl;wallmode)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g); L=w.line
    A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    recon=c.reconstruction!==nothing
    specs=_batch_specs(c,reactive,viscous)
    base=_stage_base(1)
    _residual_deferred!(c,wallmode,base)
    fill!(s.dtf,floatmax(Float32))           # the base solver's dt field (all faces) does not bind this step
    _imex_caps!(c,w,cfl,base,reactive,viscous)
    _slot_batch!(c,specs.residual[1])
    # -1: no stage-1 clock margin on this integrator (refused by _admit_imex); _imex_decision applies none either
    _k_species_fill_dt!(s.backend)(s.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,-1.0f0,Int32(base),
        Int32(reactive),Int32(viscous),Int32(s.sst),ni,nj;ndrange=(s.NI,s.NJ))
    _blk_jacobians!(c,L,wallmode)
    w.implicit_source && _blk_source_jac!(c,L)
    s.sst && _blk_sst_coeffs!(c,L,wallmode)
    copyto!(c.conserved.U0,c.conserved.U)
    s.sst && _sst_snapshot!(c)
    for (stage,(wt,_)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
        base=_stage_base(stage)
        stage>1 && _residual_deferred!(c,wallmode,base)
        face_states=c.reconstruction===nothing ? nothing : c.reconstruction.states
        _k_species_flux_i!(s.backend)(c.Fi,s.Fi_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,nothing,ni,nj,g;ndrange=(s.NI+1,s.NJ))
        _k_species_flux_j!(s.backend)(c.Fj,s.Fj_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,nothing,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
        if stage>1
            _imex_caps!(c,w,cfl,base,reactive,viscous)
            _slot_batch!(c,specs.residual[stage])
        end
        _k_line_rhs!(s.backend)(L.X,s.Fi_r,s.Fj_r,s.Fi_mx,s.Fj_mx,s.Fi_my,s.Fj_my,c.Fi,c.Fj,c.source,Int32(reactive),
            ni,nj;ndrange=(s.NI,s.NJ))
        # the increment as a flux difference (conservation of the explicit stage), with the implicit source if selected
        _blk_line_solve_conservative!(c,L;GS=w.implicit_source ? L.GS : nothing,factor=stage==1)
        _k_imex_trial!(s.backend)(c.conserved.Utrial,_status_slice(c,base+_S_TRIAL),c.conserved.U,c.conserved.U0,
            L.X,wt,Float64.(s.outlet_eos),Int32(c.eos_single),Q_FLUSH[],ni,nj;ndrange=(s.NI,s.NJ))
        # reduced before any launch consumes the trial (the CPU backend throws where the GPU propagates NaN)
        _slot_max!(_slot(c,base+_S_TRIAL),_status_slice(c,base+_S_TRIAL))
        if reactive   # the onset crossing of vapour-free cells (see _k_imex_onset!)
            _k_imex_onset!(s.backend)(w.onset,c.conserved.Utrial,c.conserved.U0,Float64.(s.outlet_eos),Int32(c.eos_single),
                ONSET_FRACTION[],ni,nj;ndrange=(s.NI,s.NJ))
            _slot_max!(_slot(c,base+_S_IMEX_ONSET),w.onset)
        end
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
    _slot_batch!(c,specs.post)
    nothing
end

# Diagnostic on retry exhaustion (stderr only; the state is already restored): the cells whose phase trial the LAST
# attempt refused, per stage, with their restored base state. The trial values themselves are not kept.
function _imex_exhaustion_report(c)
    s=c.solver
    try
        U=Array(c.conserved.U);P=Array(c.pressure);g=s.g
        for stage in 1:3
            st=Array(_status_slice(c,_stage_base(stage)+_S_TRIAL))
            bad=findall(>(0),@view st[1:s.NI,1:s.NJ])
            isempty(bad) && continue
            println(stderr,"LINE-IMPLICIT EXHAUSTION: stage $stage phase trial refused in $(length(bad)) cells; first cells (base state):")
            for I in bad[1:min(end,8)]
                i,j=Tuple(I)
                println(stderr,"   (",i,",",j,") r=",U[i,j,1]," q=",U[i,j,4]," u=",U[i,j,2]/U[i,j,1]," v=",U[i,j,3]/U[i,j,1],
                        " p=",P[i+g,j+g])
            end
        end
    catch err
        println(stderr,"LINE-IMPLICIT EXHAUSTION: report failed: ",sprint(showerror,err))
    end
    nothing
end

"""One accepted physical-time step of the wall-normal line-implicit integrator (see the header of cmesh_species_imex.jl).

The global dt is the minimum of the i-direction clocks at `cfl` (and `max_dt`); the phase-change source keeps its
explicit clock. A stage clock or trial rejection restores the exact base state and retries at half the step, at most
eight times (SpeciesRetryExhausted afterwards, base state restored). An unexpected flag or a backend error restores the
base state and throws. Advances c.steps, c.elapsed and last_global_dt like step_species_ka!; sets c.last_retries and
c.last_rejection. No ledger values are published (NaN)."""
function step_species_imex_ka!(c::SpeciesKA,w::SpeciesImexKA,cfl::Float32;max_dt::Float32=Inf32,wallmode::Int32=c.wallmode)
    wallmode==c.wallmode || throw(ArgumentError("species wallmode is fixed by its cache; rebuild or restore explicitly"))
    isfinite(cfl) && 0<cfl<=CFL_MAX[] && max_dt>0 || throw(ArgumentError("invalid species CFL or time cap"))
    size(w.scratch)==size(c.solver.dtf) && typeof(w.scratch)==typeof(c.solver.dtf) ||
        throw(ArgumentError("line-implicit workspace does not belong to this cache"))
    # strip topology (stage 4, plan section 5 item 6): a strip pair's workspace carries the strip block's share, sized by its
    # carrier, and only a pair's does (_imex_strip_owner, cmesh_species_strip.jl)
    _imex_strip_owner(c,w) || throw(ArgumentError("line-implicit workspace does not belong to this cache (its strip block's share)"))
    _admit_imex(c)
    _check_owner_host(c)
    isfinite(c.elapsed) && c.elapsed>=0 && 0<=c.steps<typemax(Int) || throw(ArgumentError("invalid species physical clock"))
    metadata=_step_metadata(c)
    try
        _copy_step_state!(c.backup,_rollback_arrays(c),c.solver.backend;sync=false)
    catch
        c.valid=false
        rethrow()
    end
    s,t=c.solver,c.tem
    reactive=_reactive(c); viscous=s.nu_lam>0.0f0; recon=c.reconstruction!==nothing
    restore()=try
        _restore_step!(c,metadata)
    catch
        c.valid=false
        rethrow()
    end
    cap=min(max_dt,DT_GROWTH[]*w.dt_hint); last=nothing
    for halvings in 0:8
        w.attempts+=1
        fill!(c.step_limit,reinterpret(UInt32,cap))
        outcome=try
            _enqueue_species_imex!(c,w,cfl;wallmode)
            _imex_decision(_read_attempt!(c),cap,recon,reactive,viscous,s.sst,c.inlet_present)
        catch err
            err isa InterruptException && rethrow()
            # as in the steady line variant: an exception after a flagged trial is that trial's rejection
            h=_read_attempt!(c)
            dt_try=only(Array(@view s.dtf[1:1,1:1]))
            restore()
            flagged=any(h[_stage_base(k)+o]!=0x00000000 for k in 1:3 for o in (_S_TRIAL,_S_SST_TRIAL))
            flagged && isfinite(dt_try) && dt_try>0.0f0 || rethrow()
            SpeciesNumericalRejection(0,:trial_exception,dt_try)
        end
        if outcome isa Float32
            next_time=c.elapsed+Float64(outcome)
            if !(isfinite(next_time) && next_time>c.elapsed)
                restore()
                throw(ErrorException("line-implicit step: the admitted dt does not advance the physical clock"))
            end
            s.last_global_dt=outcome
            c.strip===nothing || (c.strip.solver.last_global_dt=outcome)   # a strip pair: the strip carrier holds the pair's step
            _publish_no_ledger!(c)
            c.steps+=1; c.elapsed=next_time
            c.strip===nothing || (c.strip.steps=c.steps)
            c.last_retries=halvings; c.last_rejection=last; w.last_rejection=last
            outcome<max_dt && (w.dt_hint=outcome)   # a step shortened to a frame target does not throttle the next ones
            return nothing
        end
        restore()
        if outcome isa SpeciesNumericalRejection
            w.rejections+=1; last=outcome
            next_dt=0.5f0*outcome.dt
            if halvings==8 || !(isfinite(next_dt) && 0.0f0<next_dt<outcome.dt) || metadata.elapsed+Float64(next_dt)==metadata.elapsed
                c.last_retries=halvings; c.last_rejection=outcome
                _imex_exhaustion_report(c)
                c.strip===nothing || _imex_exhaustion_report_strip(c)   # a strip pair: the strip block's refused cells too
                throw(SpeciesRetryExhausted(halvings,outcome))
            end
            cap=min(max_dt,next_dt)
        else
            throw(ErrorException("line-implicit step flagged a failure the explicit path would replay; base state restored (flag slots " *
                                 string(findall(!=(0x00000000),@view c.readback.admission_only[1:ADMISSION_SLOTS])) * ")"))
        end
    end
    error("unreachable line-implicit retry state")
end
