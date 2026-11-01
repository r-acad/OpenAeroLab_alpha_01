# Isolated steady relaxation, never a physical-time entry or production default.
module GasStripSSTSteady
using KernelAbstractions
import ..GasStripSSTPrototype
const S=GasStripSSTPrototype
const G=S.G
const P=S.P
export SteadyPair,build_steady,step!

mutable struct SteadyPair{T}
    coupled::T
    iterations::Int
    last_rejections::Int
end

function admit(p::SteadyPair)
    S.admit(p.coupled)
    G.GLOBAL_DT[]==0 || throw(ArgumentError("Steady local pseudo-time requires GLOBAL_DT=0"))
    p.coupled.elapsed==0 && p.coupled.accepted_steps==0 && p.coupled.flow.elapsed==0 && p.coupled.flow.iterations==0 ||
        throw(ArgumentError("Steady relaxation cannot mix physical and pseudo-time clocks"))
end

function build_steady(p::S.CoupledPair)
    result=SteadyPair(p,0,0);admit(result);result
end

@kernel function cap_clock!(dt,@Const(source_dt),cap::Float32,factor::Float32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ;dt[i,j]=factor*min(dt[i,j],source_dt[i,j],cap);end
end

@kernel function clock_status!(status,@Const(used),@Const(flow),@Const(source_dt),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        u=used[i,j];cap=min(flow[i,j],source_dt[i,j])
        status[i,j]=isfinite(u) && u>0 && u<=cap*(1f0+8eps(Float32)) ? 0f0 : 1f0
    end
end

"""j-line implicit residual smoothing of one block's four gas residuals (opt-in IRS-J, 21 Oct 2026): the native
`_k_irsj!` exactly as the single-block `step_ka!` applies it, with the block's frozen local clock. Each grid line is
smoothed within its own block with zero-residual ghosts at both ends (the wall, the C/H interfaces, the outer
boundary), so the steady fixed point is unchanged; `prepare_dt_ka!` sizes the clock on the streamwise restriction
(s.irsj == 1) and the SST positivity clock still caps it. The SST residuals are not smoothed."""
function _irsj_smooth!(s)
    sston=s.sst ? Int32(1) : Int32(0)
    for a in P.residuals(s)
        G._k_irsj!(s.backend)(a,s.irs_c,s.irs_d,s.ux,s.uy,s.T,s.nut,s.Sjx,s.Sjy,s.vol,s.dtf,s.nu_lam,sston,
            s.am,Int32(0),s.am_ref,0.0f0,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,))
    end
    nothing
end

"""Local-clock SSPRK relaxation of the admitted shared six-equation residual.

The cell clocks vary, so this update has no single elapsed physical time and no
global conservative-time inventory identity. The *residual* retains exact common
face fluxes and its independent finite-volume closure. Every trial is atomic for
both blocks. These are steady iterations; they cannot establish unsteady stall.
"""
function step!(p::SteadyPair,cfl::Real;max_pseudo_dt::Real=Inf,clock_margin::Real=.9,max_retries::Int=6,stage_observer=nothing)
    admit(p)
    isfinite(cfl) && 0<cfl<=.5 && max_pseudo_dt>0 && isfinite(clock_margin) && 0<clock_margin<=1 && max_retries>=0 || throw(ArgumentError("Invalid local relaxation setting"))
    f=p.coupled;blocks=(f.c,f.h)
    irsj=f.c.solver.irsj==1
    f.c.solver.irsj==f.h.solver.irsj && f.c.solver.irsj in (0,1) || throw(ArgumentError("IRS-J must be set on both blocks or neither"))
    saved=S.backup(f)
    previous_dt=(f.c.solver.last_global_dt,f.h.solver.last_global_dt)
    for retry in 0:max_retries
        S.restore!(f,saved,previous_dt)
        try
            S.residual!(f)
            for q in blocks
                s=q.solver;G.prepare_dt_ka!(s,Float32(cfl))
                cap=Float32(max_pseudo_dt);Float64(cap)>max_pseudo_dt && (cap=prevfloat(cap))
                cap_clock!(s.backend)(s.dtf,q.source_dt,cap,Float32(clock_margin*2.0^-retry),Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
            end
            clocks=map(q->copy(q.solver.dtf),blocks)
            for q in blocks;P.snapshot!(q.solver);S._sst_snapshot!(q);end
            for (stage,(weight,other)) in enumerate(((1f0,0f0),(.25f0,.75f0),(2f0/3f0,1f0-2f0/3f0)))
                S.residual!(f)
                for (q,d) in zip(blocks,clocks)
                    s=q.solver;G.prepare_dt_ka!(s,Float32(cfl))
                    clock_status!(s.backend)(q.status,d,s.dtf,q.source_dt,Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
                    maximum(q.status)==0 || throw(S.StageReject("Local gas/SST stage clock decreased"))
                    copyto!(s.dtf,d)
                end
                stage_observer===nothing || stage_observer(f,stage,:residual)
                if irsj
                    for q in blocks;_irsj_smooth!(q.solver);end
                end
                for q in blocks;P.update!(q.solver,weight,other);end
                for q in blocks
                    s=q.solver
                    all(isfinite,s.rho) && all(isfinite,s.ux) && all(isfinite,s.uy) && all(isfinite,s.T) && minimum(s.rho)>1f-4 && minimum(s.T)>1f-3 && maximum(hypot.(s.ux,s.uy))<3.49f0 ||
                        throw(S.StageReject("Native gas safeguard activated in local relaxation"))
                    S.pack_gas!(s.backend)(q.conserved.Utrial,s.rho,s.ux,s.uy,s.T,s.eint,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
                    S._sst_trial!(q,weight;retryable=true,stage=stage)
                end
                for q in blocks;S._sst_commit!(q);end
                stage_observer===nothing || stage_observer(f,stage,:committed)
            end
            S.residual!(f)
            for q in blocks;q.solver.last_global_dt=0f0;end
            p.iterations+=1;p.last_rejections=retry
            return (iterations=p.iterations,rejections=retry,physical_elapsed=0.,clock_margin=Float64(clock_margin),
                local_dt_min=minimum(Float64(minimum(d)) for d in clocks),local_dt_max=maximum(Float64(maximum(d)) for d in clocks))
        catch err
            S.restore!(f,saved,previous_dt)
            err isa S.StageReject && retry<max_retries && continue
            rethrow()
        end
    end
    error("Unreachable local trial exhaustion")
end
end
