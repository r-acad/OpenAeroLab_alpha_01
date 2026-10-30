# ---- Steady pseudo-time mode (speed-up plan W2, 30 Sep 2026; memory/project_docs/2026-09-30_SPEEDUP_PLAN.md) -----------------
# An opt-in iteration for STEADY WETTED targets only: every cell advances on its own clock (local time stepping),
# which is a relaxation to the steady state and NOT a time-accurate march. Nothing here is called by
# step_species_ka!, so the physical-time path is unchanged; the driver reaches it only with TOPO_SPECIES_STEADY=1.
#
# What is reused, unchanged: the deferred attempt's residuals, faces, reconstruction, SST, boundary projections,
# trial/commit kernels, status flags and batched slot reductions, the rollback backup and the retry idea. The SSPRK
# stage kernels already read a per-cell dtf, and every clock producer already writes per-cell caps into its bank
# slice before any minimum is formed. What differs:
#   * the fill: dtf[i,j] = scale * STEADY_MARGIN * min(prepare_dt field, margin * mixture cap, viscous, SST caps)
#     at (i,j), from the stage-1 bank slices, instead of one global minimum;
#   * stages 2 and 3: a per-cell check dtf <= cap (1 + 16 eps) against that stage's caps, reduced into the flag slot
#     _stage_base(stage)+3; a violation, like a failed trial, rejects the attempt and the retry halves every local
#     step (scale), at most eight times;
#   * the phase-change source must be INACTIVE: its term is evaluated at every stage as in the physical-time step,
#     and any nonzero value (a cell holding vapour, or below the saturation pressure) is refused, from the flag slot
#     _stage_base(stage)+4 (lesson 1 of the plan: pseudo-time only for truly steady targets). Its clock is then not
#     a stability limit and does not enter the local steps: a vapour-free cell above saturation keeps a finite
#     source clock 0.25/D from the destruction rate although its source term is exactly zero (found on the first
#     CPU run, 30 Sep; the first version required that clock to be +Inf and refused a pure-liquid state);
#   * no physical clock: c.steps, c.elapsed and last_global_dt are never advanced (lesson 6: a pseudo-time mode must
#     never print physical time). The iteration count lives in SpeciesSteadyKA;
#   * refusals: the conservation ledger (its accounting weights one dt) and kinetic faces (untested here).
#     Unexpected flags (never a replay here) restore the base state and throw.
# Slots 3 and 4 of each stage block are the residual-time trio's, retired on 22 Sep and unused since.
# The converged state must still be bridged against the time-accurate reference before any arm relies on it.

# Safety factor on every local cap (pseudo-time only; not used by the physical-time path).
const STEADY_MARGIN = Ref{Float32}(0.8f0)
const _S_LOCAL_CLOCK = 3   # per-stage flag: a local step exceeds this stage's caps (stages 2 and 3)
const _S_SOURCE_ACTIVE = 4 # per-stage flag: a nonzero phase-change source term
const _SLOT_SOURCE_FINAL = _SLOT_POST+5 # flag: a nonzero source term on the attempt's final state
# Where a nonzero source appears decides what it means (30 Sep 2026, found by the line relaxation's first CPU run): on
# the accepted base state (stage 1) the target is not wetted and the mode refuses; on a stage-2/3 or final state it is
# an overshoot of the relaxation, which is rejected and retried with halved local steps like any failed trial.
_steady_final_source!(c)=(_source_terms!(c;slot=c.source_dt); _slot_max!(_slot(c,_SLOT_SOURCE_FINAL),c.source))

mutable struct SpeciesSteadyKA{A,L}
    const flags::A   # per-cell local-clock violations (interior), scratch
    const line::L    # nothing, or the wall-normal line relaxation's workspace (cmesh_species_steady_line.jl)
    iterations::Int  # accepted pseudo-time iterations -- never physical steps
    attempts::Int
    rejections::Int
    last_scale::Float32
    last_rejection::Union{Nothing,SpeciesNumericalRejection}
end

function _admit_steady(c::SpeciesKA)
    # strip topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 3.3): refused in every variant
    c.strip===nothing || throw(ArgumentError("steady pseudo-time mode does not support the strip topology (CMeshSpeciesStripTopologyV1); use the physical-time step"))
    LEDGER_ENABLED[] && throw(ArgumentError("steady pseudo-time mode refuses the conservation ledger (its accounting weights one physical dt); set LEDGER_ENABLED[] = false"))
    c.kinetic===nothing || throw(ArgumentError("steady pseudo-time mode supports the AUSM+-up faces only"))
    CMeshSolverKA.GLOBAL_DT[]==1 || throw(ArgumentError("steady pseudo-time mode forms its local steps itself; the solver's GLOBAL_DT must stay 1"))
    # the stage clock margin is defined for the explicit physical-time step only (as the line-implicit step refuses it)
    c.stage_margin>0.0f0 && throw(ArgumentError("steady pseudo-time mode refuses the stage clock margin (stage_clock_margin = $(c.stage_margin)): its local steps carry STEADY_MARGIN on every cap; build the cache without it"))
    nothing
end

"""Workspace of the steady pseudo-time mode for one species cache (see the header of cmesh_species_steady.jl).
`line=true` selects the wall-normal line relaxation (cmesh_species_steady_line.jl): the stages solve one tridiagonal
system per j line and the local step is set by the i faces alone. `block=true` selects the same lines with the block
operator of cmesh_species_block.jl (the first-order j-face Jacobians, 4x4 per face). `adi=true` selects the approximate
factorisation in both directions (cmesh_species_steady_adi.jl): i lines, then j lines, with the local step ADI_CFL x the
i-face limit."""
function build_species_steady(c::SpeciesKA;line::Bool=false,adi::Bool=false,block::Bool=false)
    _admit_steady(c)
    s=c.solver
    count((line,adi,block))<=1 || throw(ArgumentError("choose at most one of line, block and adi"))
    (line || adi || block) && !(s.NI==2*s.NIW+s.NIF) && throw(ArgumentError("the line relaxation needs the C-mesh layout NI = 2 NIW + NIF"))
    ws=adi ? _adi_workspace(s) : block ? _block_workspace(s) : line ? _line_workspace(s) : nothing
    SpeciesSteadyKA(fill!(similar(s.dtf),0.0f0),ws,0,0,0,1.0f0,nothing)
end

@kernel function _k_species_fill_local_dt!(dtf,@Const(mix),@Const(visc),@Const(sstc),factor::Float32,scale::Float32,
                                           viscous::Int32,sst::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        dt=min(dtf[i,j],factor*mix[i,j])
        viscous==Int32(1) && (dt=min(dt,visc[i,j]))
        sst==Int32(1) && (dt=min(dt,sstc[i,j]))
        dtf[i,j]=scale*dt
    end
end

@kernel function _k_species_local_clock!(flags,@Const(dtf),@Const(mix),@Const(visc),@Const(sstc),
                                         viscous::Int32,sst::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        tol=1.0f0+16.0f0*eps(Float32); dt=dtf[i,j]
        good=dt<=mix[i,j]*tol
        viscous==Int32(1) && (good=good && dt<=visc[i,j]*tol)
        sst==Int32(1) && (good=good && dt<=sstc[i,j]*tol)
        flags[i,j]=good ? 0.0f0 : 1.0f0
    end
end

# The stage's per-cell caps (a disabled clock passes the mixture slice, which its switch never reads).
function _steady_caps(c,base,viscous)
    mix=_clock_slice(c,base+_S_MIX)
    (mix,viscous ? _clock_slice(c,base+_S_VISC) : mix,c.solver.sst ? _clock_slice(c,base+_S_SST_CAP) : mix)
end

# The source term of this stage and whether any cell's is nonzero (c.source is the interior gamma array the trial
# consumes; _slot_max! flags every nonzero element).
function _steady_source!(c,base)
    _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE))
    _slot_max!(_slot(c,base+_S_SOURCE_ACTIVE),c.source)
end

# One speculative steady attempt: _enqueue_species_deferred! with the local fill, the per-cell stage checks and the
# source-activity flags. The ledger and kinetic faces are refused before this is reached.
function _enqueue_species_steady!(c::SpeciesKA,w::SpeciesSteadyKA,cfl::Float32,scale::Float32;wallmode::Int32)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    recon=c.reconstruction!==nothing
    specs=_batch_specs(c,reactive,viscous)
    base=_stage_base(1)
    _residual_deferred!(c,wallmode,base)
    CMeshSolverKA.prepare_dt_ka!(s,cfl;slot=:batched)      # per-cell field in s.dtf; its minimum only goes to a slot
    _mixture_cap!(c,cfl;slot=_clock_slice(c,base+_S_MIX))
    reactive && _steady_source!(c,base)
    viscous && _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC))
    s.sst && _sst_residual!(c;slot=_clock_slice(c,base+_S_SST_CAP),faces_slot=_status_slice(c,base+_S_SST_FACES))
    _slot_batch!(c,specs.residual[1])
    mix,visc,sstc=_steady_caps(c,base,viscous)
    _k_species_fill_local_dt!(s.backend)(s.dtf,mix,visc,sstc,recon ? CLOCK_MARGIN[] : 1.0f0,scale*STEADY_MARGIN[],
        Int32(viscous),Int32(s.sst),ni,nj;ndrange=(s.NI,s.NJ))
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
            mix,visc,sstc=_steady_caps(c,base,viscous)
            _k_species_local_clock!(s.backend)(w.flags,s.dtf,mix,visc,sstc,Int32(viscous),Int32(s.sst),
                ni,nj;ndrange=(s.NI,s.NJ))
            _slot_max!(_slot(c,base+_S_LOCAL_CLOCK),w.flags)
        end
        _check_stage!(c,wt;retryable=true,stage,slot=_status_slice(c,base+_S_TRIAL))
        s.sst && _sst_trial!(c,wt;retryable=true,stage,slot=_status_slice(c,base+_S_SST_TRIAL))
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

# :admitted, a SpeciesNumericalRejection (local clock or trial: retry with halved local steps), :phase_change
# (a source term became nonzero: refused) or :unexpected (a flag the physical-time path would replay: refused here).
function _steady_decision(h,reactive::Bool,viscous::Bool,sst::Bool,inlet::Bool)
    flag(k)=h[k]==0x00000000
    cap(k)=reinterpret(Float32,h[ADMISSION_SLOTS+k])
    positive(k)=cap(k)>0.0f0
    clock(k)=isfinite(cap(k)) && cap(k)>0.0f0
    pressure(k)=!inlet || positive(k)
    residual(b)=flag(b+_S_SST_PREPARE) && flag(b+_S_RECON) && pressure(b+_S_FLUX_PRESSURE) && flag(b+_S_STATE)
    clock(_SLOT_DTFIELD) || return :unexpected
    for stage in 1:3
        b=_stage_base(stage)
        residual(b) && clock(b+_S_MIX) || return :unexpected
        if reactive
            positive(b+_S_SOURCE) || return :unexpected      # an unresolved source state, as in the physical-time step
            flag(b+_S_SOURCE_ACTIVE) || return stage==1 ? :phase_change : SpeciesNumericalRejection(stage,:source_activated,0.0f0)
        end
        viscous && !positive(b+_S_VISC) && return :unexpected
        sst && !(flag(b+_S_SST_FACES) && positive(b+_S_SST_CAP)) && return :unexpected
        stage>1 && !flag(b+_S_LOCAL_CLOCK) && return SpeciesNumericalRejection(stage,:local_clock,0.0f0)
        flag(b+_S_TRIAL) || return SpeciesNumericalRejection(stage,:phase_trial,0.0f0)
        sst && !flag(b+_S_SST_TRIAL) && return SpeciesNumericalRejection(stage,:sst_trial,0.0f0)
        pressure(b+_S_PROJECT_PRESSURE) && flag(b+_S_PROJECT_OUTLET) || return :unexpected
    end
    flag(_SLOT_POST+3) && flag(_SLOT_POST+4) || return :unexpected
    reactive && !flag(_SLOT_SOURCE_FINAL) && return SpeciesNumericalRejection(4,:source_activated,0.0f0)
    :admitted
end
_steady_flags(c)=@view c.readback.admission_only[1:ADMISSION_SLOTS]

"""One steady pseudo-time iteration (local time stepping; see the header of cmesh_species_steady.jl).

Advances the conserved state towards the steady solution of the same discrete equations the physical-time step
integrates, with each cell at `scale * STEADY_MARGIN * min(its own caps)` for the given `cfl`. The physical clock
(c.steps, c.elapsed, last_global_dt) is never advanced; `w.iterations` counts accepted iterations. A local-clock or
trial rejection restores the base state and retries with every local step halved, up to eight halvings
(SpeciesRetryExhausted afterwards, base state restored). A nonzero phase-change source, an unexpected flag or a
backend error restores the base state and throws.
"""
function step_species_steady_ka!(c::SpeciesKA,w::SpeciesSteadyKA,cfl::Float32;wallmode::Int32=c.wallmode)
    wallmode==c.wallmode || throw(ArgumentError("species wallmode is fixed by its cache; rebuild or restore explicitly"))
    isfinite(cfl) && 0<cfl<=CFL_MAX[] || throw(ArgumentError("invalid species CFL"))
    size(w.flags)==size(c.solver.dtf) && typeof(w.flags)==typeof(c.solver.dtf) ||
        throw(ArgumentError("steady workspace does not belong to this cache"))
    _admit_steady(c)
    _check_owner_host(c)
    metadata=_step_metadata(c)
    try
        _copy_step_state!(c.backup,_rollback_arrays(c),c.solver.backend;sync=false)
    catch
        c.valid=false
        rethrow()
    end
    s,t=c.solver,c.tem
    reactive=_reactive(c); viscous=s.nu_lam>0.0f0
    scale=1.0f0; last=nothing
    restore()=try
        _restore_step!(c,metadata)
    catch
        c.valid=false
        rethrow()
    end
    for _ in 0:8
        w.attempts+=1
        outcome=try
            if w.line===nothing
                _enqueue_species_steady!(c,w,cfl,scale;wallmode)
            else
                hasproperty(w.line,:adi) ? _enqueue_species_steady_adi!(c,w,cfl,scale;wallmode) :
                                           _enqueue_species_steady_line!(c,w,cfl,scale;wallmode)
            end
            _steady_decision(_read_attempt!(c),reactive,viscous,s.sst,c.inlet_present)
        catch err
            err isa InterruptException && rethrow()
            # The line variant reduces each stage's trial flags before any launch consumes the trial; an exception
            # after a flagged trial (the CPU backend throws where the GPU propagates NaN) is that trial's rejection.
            # Any other exception is not classified here: the base state is restored and it propagates.
            h=w.line===nothing ? nothing : _read_attempt!(c)
            restore()
            flagged=h!==nothing && any(h[_stage_base(k)+o]!=0x00000000 for k in 1:3 for o in (_S_TRIAL,_S_SST_TRIAL))
            flagged || rethrow()
            SpeciesNumericalRejection(0,:trial_exception,0.0f0)
        end
        if outcome===:admitted
            _publish_no_ledger!(c)
            w.iterations+=1; w.last_scale=scale; w.last_rejection=last
            return nothing
        end
        restore()
        if outcome isa SpeciesNumericalRejection
            w.rejections+=1; last=outcome; scale*=0.5f0
        elseif outcome===:phase_change
            throw(ArgumentError("steady pseudo-time mode refused: a phase-change source term is nonzero (vapour present or a cell below the saturation pressure); the mode is for wetted targets only"))
        else
            throw(ErrorException("steady pseudo-time iteration flagged a failure the physical-time path would replay; base state restored (flag slots " *
                                 string(findall(!=(0x00000000),_steady_flags(c))) * ")"))
        end
    end
    throw(SpeciesRetryExhausted(8,last))
end

"""Pseudo-time residuals of the last accepted steady iteration, read on the host: the RMS over interior cells of
(U - U0)/dtf for rho, rho u, rho v and q, and of (rho k, rho omega) when SST is on; with the local-step range.
U0 and the SST snapshots hold the iteration's base state and dtf its local steps until the next attempt starts."""
function species_steady_residuals(c::SpeciesKA)
    s=c.solver; ni,nj=s.NI,s.NJ
    U=Array(c.conserved.U); U0=Array(c.conserved.U0); dt=Array(s.dtf); n=ni*nj
    rms(f)=sqrt(sum(f(i,j)^2 for i in 1:ni, j in 1:nj)/n)
    r=ntuple(k->rms((i,j)->(U[i,j,k]-U0[i,j,k])/Float64(dt[i,j])),4)
    rk=rw=NaN
    if c.turbulence!==nothing
        a=c.turbulence; K=Array(a.rk); K0=Array(a.k0); W=Array(a.rw); W0=Array(a.w0)
        rk=rms((i,j)->(Float64(K[i,j])-Float64(K0[i,j]))/Float64(dt[i,j]))
        rw=rms((i,j)->(Float64(W[i,j])-Float64(W0[i,j]))/Float64(dt[i,j]))
    end
    (rho=r[1],mx=r[2],my=r[3],q=r[4],k=rk,w=rw,dt_min=Float64(minimum(dt)),dt_max=Float64(maximum(dt)))
end

# The wall-normal line relaxation variant (W3a-steady, 30 Sep 2026).
include(joinpath(@__DIR__,"cmesh_species_steady_line.jl"))
# The approximate factorisation (ADI) variant (W3a-steady, 30 Sep 2026).
include(joinpath(@__DIR__,"cmesh_species_steady_adi.jl"))
# The block operator of the line relaxation (W3a, 30 Sep 2026).
include(joinpath(@__DIR__,"cmesh_species_block.jl"))
# The time-accurate line-implicit step built on the block operator (W3a, 30 Sep 2026).
include(joinpath(@__DIR__,"cmesh_species_imex.jl"))
