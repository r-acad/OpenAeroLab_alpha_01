module NativeBarotropicCompactCoupled

using SHA, Serialization
import ..NativeBarotropicEOS as EOS
import ..NativeBarotropicSource as Source
import ..NativeBarotropicPeriodic as Flow
import ..NativeBarotropicCoupled as Audit

export CompactCoupledSolver, CompactCheckpoint, CompactAdmissionError,
       step!, checkpoint, restore!, state_arrays, clock_state, audit_counts, receipt_snapshot

const F64=Flow.Solver{Float64}
const Digest=NTuple{32,UInt8}
const ZERO_DIGEST=ntuple(_->UInt8(0),32)
const TAIL_SIZE=8
const SCHEMA="native-compact-source-flow-Float64-tail8-v1"
const CHECKPOINT_SCHEMA="native-compact-source-flow-checkpoint-v1"
const EXPECTED_AUDIT="1e5ed5aeeea6942a2bb6295685b0a55f4543d388416cc503d0d468a1c2423a79"
const MODULE_PATH=abspath(@__FILE__)
const MODULE_SHA=bytes2hex(sha256(read(MODULE_PATH)))

struct CompactAdmissionError <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO,e::CompactAdmissionError)=print(io,e.status,": ",e.message)
require(ok,status,message)=ok || throw(CompactAdmissionError(status,message))
digest_data(x)=begin io=IOBuffer();serialize(io,x);convert(Digest,Tuple(sha256(take!(io)))) end
array_digest(a)=digest_data((;shape=size(a),scalar=:Float64,values=a))

struct ClockRecord
    time::Float64
    compensation::Float64
    steps::Int
    last_dt::Float64
    last_path::Symbol
end
clock_record(s)=ClockRecord(s.time,s.time_compensation,s.steps,s.last_dt,s.last_path)
clock_tuple(c::ClockRecord)=(;time=c.time,compensation=c.compensation,steps=c.steps,last_dt=c.last_dt,last_path=c.last_path)
const EMPTY_CLOCK=ClockRecord(0.0,0.0,0,0.0,:none)

struct CellSummary
    index::Int
    valid::Bool
    reason::Symbol
    q::Float64
    iterations::Int
    bracket_width::Float64
    predictor_residual::Float64
    target::Float64
    rate::Float64
    hk::Float64
end
empty_cell(index=0,q=NaN)=CellSummary(index,false,:not_returned,q,0,NaN,NaN,NaN,NaN,NaN)
cell_summary(i,r)=CellSummary(i,r.valid,r.reason,r.q,r.iterations,r.bracket_width,r.predictor_residual,r.target,r.rate,r.hk)

struct HalfReceipt
    h::Float64
    visited::Int
    accepted::Int
    reason_counts::NTuple{3,Int}
    iteration_sum::Int
    max_iterations::Int
    sum_changes::Float64
    volume_sum::Float64
    max_abs_change::Float64
    min_q::Float64
    max_q::Float64
    before_digest::Digest
    after_digest::Digest
    changes_digest::Digest
    complete::Bool
    last_cell::CellSummary
end
empty_half()=HalfReceipt(0.0,0,0,(0,0,0),0,0,0.0,0.0,0.0,Inf,-Inf,
    ZERO_DIGEST,ZERO_DIGEST,ZERO_DIGEST,false,empty_cell())

"""Fixed-size scalar evidence only. No array, String, Vector or mutable field."""
struct CompactReceipt
    kind::Symbol
    sequence::Int
    coupled_step::Int
    dt::Float64
    from_clock::ClockRecord
    to_clock::ClockRecord
    phase::Symbol
    stage::Int
    cell::Int
    status::Symbol
    flow_status::Symbol
    first_source::HalfReceipt
    second_source::HalfReceipt
    initial_cell_dt::Float64
    post_source_cell_dt::Float64
    source_delta::Int
    flow_delta::Int
    prior_root::Digest
    checkpoint_root::Digest
    prior_context_digest::Digest
    error_digest::Digest
    quarantined_prior::Bool
end
const EMPTY_RECEIPT=CompactReceipt(:empty,0,0,0.0,EMPTY_CLOCK,EMPTY_CLOCK,:entry,0,0,:none,:none,
    empty_half(),empty_half(),NaN,NaN,0,0,ZERO_DIGEST,ZERO_DIGEST,ZERO_DIGEST,ZERO_DIGEST,false)
const EMPTY_TAIL=ntuple(_->EMPTY_RECEIPT,TAIL_SIZE)
event_root(r::CompactReceipt)=digest_data((;schema=SCHEMA,prior=r.prior_root,event=r))

mutable struct PendingAttempt
    h::Float64
    from_clock::ClockRecord
    phase::Symbol
    stage::Int
    cell::Int
    source_start::Int
    flow_start::Int
    first_source::HalfReceipt
    second_source::HalfReceipt
    initial_cell_dt::Float64
    post_source_cell_dt::Float64
    flow_status::Symbol
end
PendingAttempt()=PendingAttempt(NaN,EMPTY_CLOCK,:entry,0,0,0,0,empty_half(),empty_half(),NaN,NaN,:none)

mutable struct CompactCoupledSolver{F<:F64}
    const flow::F
    const source::Source.SourceModel
    const trial::F
    const validation::Flow.Derived{Float64}
    const changes::Matrix{Float64}
    const owned_arrays::Tuple
    const trial_arrays::Tuple
    const auxiliary_arrays::Tuple
    const buffer_dataids::Tuple
    const binding::NamedTuple
    const lock::ReentrantLock
    const pending::PendingAttempt
    tail::NTuple{8,CompactReceipt}
    tail_count::Int
    receipt_count::Int
    tail_anchor::Digest
    history_root::Digest
    last_success::CompactReceipt
    last_attempt::CompactReceipt
    coupled_steps::Int
    origin_steps::Int
    failed::Bool
    committed_hash::Digest
    source_calls::Int
    flow_calls::Int
end
state_arrays(c::CompactCoupledSolver)=Flow.state_arrays(c.flow)
clock_state(c::CompactCoupledSolver)=Flow.clock_state(c.flow)
audit_counts(c::CompactCoupledSolver)=(;source=c.source_calls,flow=c.flow_calls)
receipt_snapshot(c::CompactCoupledSolver)=(;tail=c.tail,count=c.tail_count,total=c.receipt_count,
    anchor=c.tail_anchor,root=c.history_root,last_success=c.last_success,last_attempt=c.last_attempt,
    retention=:bounded_tail8_no_full_cell_history,counter_scope=:instance_lifetime_not_restored)

function verify_sources!()
    require(Audit.WRAPPER_SHA==EXPECTED_AUDIT,:source_binding,"full-audit helper identity mismatch")
    Audit.verify_sources!()
    require(bytes2hex(sha256(read(MODULE_PATH)))==MODULE_SHA,:source_binding,"compact module bytes changed")
end
compact_binding()=(;schema=SCHEMA,checkpoint_schema=CHECKPOINT_SCHEMA,compact_sha256=MODULE_SHA,
    audit_sha256=EXPECTED_AUDIT,tail_size=TAIL_SIZE,inner_history=:empty,
    counter_scope=:instance_lifetime_not_restored,retention=:bounded_scalar_commitment_only)
binding(s,source)=(;full=Audit.make_binding(s,source),compact=compact_binding())

function assert_owned!(c)
    require(isequal(binding(c.flow,c.source),c.binding),:owner_configuration,"bound flow/source configuration changed")
    shape=(c.flow.grid.nx,c.flow.grid.ny)
    current=(Audit.all_arrays(c.flow)...,Audit.all_arrays(c.trial)...,Flow.derived_arrays(c.validation)...,c.changes)
    original=(c.owned_arrays...,c.trial_arrays...,c.auxiliary_arrays...)
    require(length(current)==length(original) && all(a===b && a isa Matrix{Float64} && size(a)==shape for (a,b) in zip(current,original)),
        :owner_arrays,"original distinct Float64 buffers and exact shapes are required")
    require(isequal(map(Base.dataids,current),c.buffer_dataids),:owner_arrays,"original buffer storage identities are required")
    require(isequal(Audit.values_of(c.trial.grid),Audit.values_of(c.flow.grid)) &&
        isequal(Audit.values_of(c.trial.material),Audit.values_of(c.flow.material)) &&
        isequal(Audit.values_of(c.trial.config),Audit.values_of(c.flow.config)) &&
        isequal(Flow.source_binding(c.trial.digest),Flow.source_binding(c.flow.digest)),
        :owner_configuration,"private trial binding mismatch")
end
function validate_flow!(s,work)
    Flow.validate_clock(Flow.clock_state(s),Float64,:clock)
    require(!s.failed,:inner_failed,"inner flow must not have an outstanding failure")
    require(isempty(s.history),:inner_history,"compact mode admits only empty inner flow history")
    Flow.derive!(work,s.U,s.material)
    require(all(isequal(a,b) for (a,b) in zip(Flow.derived_arrays(work),Flow.derived_arrays(s.D))),
        :derived_state,"committed derived fields are stale")
end
function fingerprint(c;U=c.flow.U,D=c.flow.D,clock=clock_record(c.flow),inner_failed=c.flow.failed,
        failed=c.failed,steps=c.coupled_steps,origin=c.origin_steps,tail=c.tail,tail_count=c.tail_count,
        count=c.receipt_count,anchor=c.tail_anchor,root=c.history_root,last_success=c.last_success,last_attempt=c.last_attempt)
    digest_data((;schema=SCHEMA,arrays=Flow.state_arrays(U),derived=Flow.derived_arrays(D),clock,inner_failed,
        failed,steps,origin,tail,tail_count,count,anchor,root,last_success,last_attempt))
end

function CompactCoupledSolver(s::F,source::Source.SourceModel) where {F<:F64}
    verify_sources!()
    require(s.config.reconstruction in (:pc,:muscl_rho_q),:flow_mode,"unified flow requires :pc or :muscl_rho_q")
    require(isequal(Audit.values_of(s.material),Audit.values_of(source.material)),:source_material,"source and flow material must match exactly")
    require(isempty(s.history),:inner_history,"compact construction does not discard prior inner history")
    work=Flow.Derived(Float64,s.grid.nx,s.grid.ny);validate_flow!(s,work)
    trial=deepcopy(s);changes=zeros(Float64,s.grid.nx,s.grid.ny)
    owned=Audit.all_arrays(s);private=Audit.all_arrays(trial);aux=(Flow.derived_arrays(work)...,changes)
    allbuffers=(owned...,private...,aux...);shape=(s.grid.nx,s.grid.ny)
    require(all(a isa Matrix{Float64} && size(a)==shape for a in allbuffers) &&
        length(unique(objectid.(allbuffers)))==length(allbuffers) &&
        all(!Base.mightalias(allbuffers[i],allbuffers[j]) for i in eachindex(allbuffers) for j in 1:i-1),
        :owner_arrays,"all committed/private/validation buffers must have disjoint storage")
    c=CompactCoupledSolver(s,source,trial,work,changes,owned,private,aux,map(Base.dataids,allbuffers),binding(s,source),ReentrantLock(),PendingAttempt(),
        EMPTY_TAIL,0,0,ZERO_DIGEST,ZERO_DIGEST,EMPTY_RECEIPT,EMPTY_RECEIPT,0,s.steps,false,ZERO_DIGEST,0,0)
    Audit.invalidate_reconstruction!(s);c.committed_hash=fingerprint(c);c
end
CompactCoupledSolver(s,source)=throw(CompactAdmissionError(:type,"only native Float64 Flow and SourceModel are supported"))

function refresh_trial!(c)
    for (dest,src) in zip(c.trial_arrays,c.owned_arrays);copyto!(dest,src);end
    if c.flow.reconstruction_scratch!==nothing
        a=c.trial.reconstruction_scratch.diagnostics;b=c.flow.reconstruction_scratch.diagnostics
        for k in 1:fieldcount(typeof(a));setfield!(a,k,getfield(b,k));end
    end
    Audit.copy_clock!(c.trial,clock_state(c));c.trial.failed=c.flow.failed;empty!(c.trial.history)
end
function begin_attempt!(c,h)
    p=c.pending;p.h=h;p.from_clock=clock_record(c.flow);p.phase=:entry;p.stage=0;p.cell=0
    p.source_start=c.source_calls;p.flow_start=c.flow_calls;p.first_source=empty_half();p.second_source=empty_half()
    p.initial_cell_dt=NaN;p.post_source_cell_dt=NaN;p.flow_status=:none
end

function source_half!(c,half,phase)
    p=c.pending;p.phase=phase;before=array_digest(c.trial.U.q);fill!(c.changes,0.0)
    visited=0;accepted=0;reasons=(0,0,0);iterations=0;max_iterations=0
    max_change=0.0;min_q=Inf;max_q=-Inf;last=empty_cell();complete=false
    try
        for i in eachindex(c.trial.U.q)
            p.cell=i;q0=c.trial.U.q[i];last=empty_cell(i,q0)
            require(c.source_calls<typemax(Int),:counter,"source attempt counter overflow")
            c.source_calls+=1;visited+=1
            result=Source.source_step(c.source,c.trial.U.rho[i],q0,half)
            last=cell_summary(i,result)
            require(result.valid,:source_rejection,"$(result.reason) at cell $i")
            c.changes[i]=result.q-q0;c.trial.U.q[i]=result.q;accepted+=1
            k=result.reason==:accepted ? 1 : result.reason==:identity ? 2 : 3
            reasons=ntuple(j->reasons[j]+(j==k ? 1 : 0),3)
            iterations=Base.Checked.checked_add(iterations,result.iterations);max_iterations=max(max_iterations,result.iterations)
            max_change=max(max_change,abs(c.changes[i]));min_q=min(min_q,result.q);max_q=max(max_q,result.q)
        end
        Flow.derive!(c.trial.D,c.trial.U,c.trial.material);complete=true
    finally
        total=sum(c.changes)
        receipt=HalfReceipt(half,visited,accepted,reasons,iterations,max_iterations,total,total*c.trial.grid.volume,
            max_change,min_q,max_q,before,array_digest(c.trial.U.q),array_digest(c.changes),complete,last)
        if phase==:first_source;p.first_source=receipt;else;p.second_source=receipt;end
    end
end

function valid_clock(c::ClockRecord)
    Audit.valid_clock(clock_tuple(c))
end
function follows_clock(old::ClockRecord,new::ClockRecord,h)
    # Receipt validation only; actual clock advancement is exclusively Flow.step!.
    valid_clock(old) && valid_clock(new) && old.steps<typemax(Int) || return false
    y=h-old.compensation;t=old.time+y;comp=(t-old.time)-y
    new.steps==old.steps+1 && new.last_dt===h && new.last_path==:explicit_controlled &&
        isequal(new.time,t) && isequal(new.compensation,comp)
end
function valid_cell(c::CellSummary,N)
    c.index==0 && return isequal(c,empty_cell())
    1<=c.index<=N && 0<=c.iterations<=100 || return false
    !c.valid && return true
    c.reason in (:accepted,:identity,:representable_equilibrium) &&
        all(isfinite,(c.q,c.bracket_width,c.predictor_residual,c.target,c.rate,c.hk)) &&
        min(c.q,c.bracket_width,c.target,c.rate,c.hk)>=0
end
function valid_half(h::HalfReceipt,N,volume)
    isfinite(h.h) && h.h>=0 && 0<=h.accepted<=h.visited<=N || return false
    all(x->0<=x<=N,h.reason_counts) && sum(h.reason_counts)==h.accepted || return false
    0<=h.max_iterations<=100 && 0<=h.iteration_sum<=big(100)*h.accepted || return false
    all(isfinite,(h.sum_changes,h.volume_sum,h.max_abs_change)) && h.max_abs_change>=0 || return false
    isequal(h.volume_sum,h.sum_changes*volume) || return false
    if h.accepted==0
        isequal(h.min_q,Inf) && isequal(h.max_q,-Inf) || return false
    else
        isfinite(h.min_q) && isfinite(h.max_q) && 0<=h.min_q<=h.max_q || return false
    end
    valid_cell(h.last_cell,N) || return false
    !h.complete || (h.h>0 && h.accepted==h.visited==N && h.last_cell.index==N && h.last_cell.valid &&
        h.min_q<=h.last_cell.q<=h.max_q)
end
function valid_receipt(r::CompactReceipt,N,volume)
    r.kind in (:step,:failure,:restore) && r.sequence>0 && 0<=r.stage<=3 && 0<=r.cell<=N || return false
    0<=r.source_delta<=2N && 0<=r.flow_delta<=1 || return false
    valid_half(r.first_source,N,volume) && valid_half(r.second_source,N,volume) || return false
    if r.kind==:step
        return r.status==:ok && r.phase==:complete && r.stage==3 && r.cell==0 && r.flow_status==:ok &&
            r.coupled_step>0 && isfinite(r.dt) && r.dt>0 && follows_clock(r.from_clock,r.to_clock,r.dt) &&
            r.first_source.complete && r.second_source.complete && r.first_source.h===r.second_source.h &&
            r.first_source.h+r.second_source.h==r.dt && r.source_delta==2N && r.flow_delta==1 &&
            isfinite(r.initial_cell_dt) && r.initial_cell_dt>0 && isfinite(r.post_source_cell_dt) && r.post_source_cell_dt>=r.dt
    elseif r.kind==:restore
        return r.status==:restored && r.phase==:restore && r.stage==r.cell==0 && r.coupled_step>=0 &&
            valid_clock(r.to_clock) && r.source_delta==r.flow_delta==0
    end
    # Failure from-clock/counter observations may be invalid; committed state is checked separately.
    r.phase in (:entry,:first_source,:post_source_cfl,:flow,:second_source,:commit,:restore)
end
function valid_tail(tail,count,total,anchor,root,N,volume)
    0<=count<=TAIL_SIZE && total>=count && count==min(total,TAIL_SIZE) || return false
    running=anchor
    for i in 1:TAIL_SIZE
        r=tail[i]
        if i<=count
            valid_receipt(r,N,volume) && r.sequence==total-count+i && r.prior_root==running || return false
            running=event_root(r)
        else
            isequal(r,EMPTY_RECEIPT) || return false
        end
    end
    running==root
end
function tail_base(c)
    N=length(c.changes)
    good=valid_tail(c.tail,c.tail_count,c.receipt_count,c.tail_anchor,c.history_root,N,c.flow.grid.volume)
    prior=digest_data((;tail=c.tail,count=c.tail_count,total=c.receipt_count,anchor=c.tail_anchor,root=c.history_root,
        last_attempt=c.last_attempt,attempted_inner_history=c.flow.history))
    good ? (;tail=c.tail,count=c.tail_count,total=c.receipt_count,anchor=c.tail_anchor,root=c.history_root,quarantined=false,prior) :
        (;tail=EMPTY_TAIL,count=0,total=0,anchor=prior,root=prior,quarantined=true,prior)
end
function prepare_tail(base,r)
    require(base.total<typemax(Int),:counter,"receipt counter overflow")
    require(r.sequence==base.total+1 && r.prior_root==base.root,:receipt,"prospective receipt chain mismatch")
    if base.count<TAIL_SIZE
        tail=ntuple(i->i==base.count+1 ? r : base.tail[i],TAIL_SIZE);anchor=base.anchor;count=base.count+1
    else
        tail=ntuple(i->i<TAIL_SIZE ? base.tail[i+1] : r,TAIL_SIZE);anchor=event_root(base.tail[1]);count=TAIL_SIZE
    end
    (;tail,count,total=r.sequence,anchor,root=event_root(r))
end
function make_receipt(c,base,kind,status;phase=c.pending.phase,stage=c.pending.stage,cell=c.pending.cell,
        step=c.coupled_steps,to=clock_record(c.flow),checkpoint_root=ZERO_DIGEST,error_digest=ZERO_DIGEST)
    p=c.pending
    CompactReceipt(kind,Base.Checked.checked_add(base.total,1),step,p.h,p.from_clock,to,phase,stage,cell,status,p.flow_status,
        p.first_source,p.second_source,p.initial_cell_dt,p.post_source_cell_dt,
        c.source_calls-p.source_start,c.flow_calls-p.flow_start,base.root,checkpoint_root,base.prior,error_digest,base.quarantined)
end
function apply_tail!(c,next)
    c.tail=next.tail;c.tail_count=next.count;c.receipt_count=next.total;c.tail_anchor=next.anchor;c.history_root=next.root
end
function failure!(c,status,message)
    base=tail_base(c);r=make_receipt(c,base,:failure,status;error_digest=digest_data(message));next=prepare_tail(base,r)
    seal=fingerprint(c;failed=true,tail=next.tail,tail_count=next.count,count=next.total,anchor=next.anchor,root=next.root,last_attempt=r)
    c.failed=true;apply_tail!(c,next);c.last_attempt=r;c.committed_hash=seal
    (;success=false,phase=r.phase,stage=r.stage,cell=r.cell,status,message,dt=0.0,time=c.flow.time,step=c.coupled_steps)
end

"""Opt-in compact S(h/2)-actual F(h)-S(h/2). Expected refusals leave committed arrays/clock intact.
Unexpected exceptions are recorded and rethrown; OOM/asynchronous failure atomicity is not promised.
"""
function step!(c::CompactCoupledSolver,h::Float64)
    lock(c.lock) do
        c.failed && return (;success=false,phase=:entry,stage=0,cell=0,status=:failed_run,
            message="restore an admitted coupled checkpoint explicitly",dt=0.0,time=c.flow.time,step=c.coupled_steps)
        begin_attempt!(c,h);p=c.pending
        try
            verify_sources!();assert_owned!(c)
            require(isempty(c.flow.history),:owner_state,"committed inner history changed outside wrapper")
            require(fingerprint(c)==c.committed_hash,:owner_state,"committed state, clock or receipt changed outside wrapper")
            validate_flow!(c.flow,c.validation)
            require(c.source_calls>=0 && c.flow_calls>=0,:counter,"invalid instance-lifetime attempt counters")
            require(isfinite(h) && h>0,:dt,"explicit finite positive Float64 h required")
            half=h/2;require(half>0 && half+half==h,:half_time,"half interval must be positive and exactly compose h")
            require(c.coupled_steps<typemax(Int),:clock,"coupled step counter overflow")
            refresh_trial!(c);p.initial_cell_dt=Audit.cell_dt(c.trial)
            source_half!(c,half,:first_source)
            p.phase=:post_source_cfl;p.cell=0;p.post_source_cell_dt=Audit.cell_dt(c.trial)
            require(h<=p.post_source_cell_dt,:post_source_cfl,"explicit h exceeds post-first-source cell spectral estimate")
            p.phase=:flow;require(c.flow_calls<typemax(Int),:counter,"flow attempt counter overflow")
            c.flow_calls+=1;result=Flow.step!(c.trial,h);p.stage=result.stage;p.flow_status=result.status
            require(result.success,:flow_rejection,"inner flow $(result.status): $(result.message)")
            require(c.trial.steps==p.from_clock.steps+1 && c.trial.last_dt===h && c.trial.last_path==:explicit_controlled,
                :clock,"inner flow did not apply exactly one explicit h")
            p.stage=0;source_half!(c,half,:second_source)
            p.phase=:commit;p.cell=0;validate_flow!(c.trial,c.validation);verify_sources!();assert_owned!(c)
            require(isempty(c.flow.history),:owner_state,"committed inner history changed during private trial")
            require(fingerprint(c)==c.committed_hash,:owner_state,"committed state changed during private trial")
            newclock=clock_record(c.trial);base=tail_base(c)
            r=make_receipt(c,base,:step,:ok;phase=:complete,stage=3,cell=0,step=c.coupled_steps+1,to=newclock)
            next=prepare_tail(base,r)
            seal=fingerprint(c;U=c.trial.U,D=c.trial.D,clock=newclock,inner_failed=false,failed=false,
                steps=c.coupled_steps+1,tail=next.tail,tail_count=next.count,count=next.total,anchor=next.anchor,root=next.root,last_success=r,last_attempt=r)
            # All prospective receipt/digest allocation occurs before committed copies.
            Flow.copy_state!(c.flow.U,c.trial.U);Flow.copy_derived!(c.flow.D,c.trial.D)
            Audit.copy_clock!(c.flow,clock_tuple(newclock));c.flow.failed=false;Audit.invalidate_reconstruction!(c.flow)
            c.coupled_steps+=1;c.last_success=r;c.last_attempt=r;apply_tail!(c,next);c.committed_hash=seal
            (;success=true,phase=:complete,stage=3,cell=0,status=:ok,message="",dt=h,time=c.flow.time,step=c.coupled_steps)
        catch e
            if e isa CompactAdmissionError || e isa Audit.CoupledAdmissionError || e isa Flow.AdmissionError
                return failure!(c,e.status,e.message)
            end
            failure!(c,:unexpected_exception,sprint(showerror,e));rethrow()
        end
    end
end
function step!(c::CompactCoupledSolver,h)
    lock(c.lock) do
        begin_attempt!(c,NaN);failure!(c,:dt_type,"explicit timestep must be Float64; no conversion is applied")
    end
end

struct CompactCheckpoint
    data::NamedTuple
    checksum::Digest
end
function checkpoint(c::CompactCoupledSolver)
    lock(c.lock) do
        verify_sources!();assert_owned!(c)
        require(isempty(c.flow.history),:owner_state,"committed inner history changed outside wrapper")
        require(fingerprint(c)==c.committed_hash,:owner_state,"out-of-band state change")
        validate_flow!(c.flow,c.validation);Audit.assert_intervals!(c.flow,c.source)
        require(valid_tail(c.tail,c.tail_count,c.receipt_count,c.tail_anchor,c.history_root,length(c.changes),c.flow.grid.volume),
            :checkpoint_history,"invalid compact receipt tail")
        require(c.origin_steps>=0 && c.coupled_steps>=0 && c.coupled_steps==c.flow.steps-c.origin_steps,
            :checkpoint_count,"coupled/inner step counts disagree")
        require(c.coupled_steps==0 ? isequal(c.last_success,EMPTY_RECEIPT) :
            valid_receipt(c.last_success,length(c.changes),c.flow.grid.volume) && c.last_success.kind==:step &&
            c.last_success.coupled_step==c.coupled_steps && isequal(c.last_success.to_clock,clock_record(c.flow)),
            :checkpoint_last_success,"last success does not match committed clock/count")
        data=(;schema=CHECKPOINT_SCHEMA,binding=deepcopy(c.binding),arrays=map(copy,state_arrays(c)),clock=clock_record(c.flow),
            inner_failed=c.flow.failed,inner_history=:empty,coupled_steps=c.coupled_steps,origin_steps=c.origin_steps,failed=c.failed,
            tail=c.tail,tail_count=c.tail_count,receipt_count=c.receipt_count,tail_anchor=c.tail_anchor,history_root=c.history_root,
            last_success=c.last_success,counter_scope=:instance_lifetime_not_restored)
        CompactCheckpoint(data,digest_data(data))
    end
end
function restore!(c::CompactCoupledSolver,cp::CompactCheckpoint)
    lock(c.lock) do
        begin_attempt!(c,NaN);c.pending.phase=:restore
        try
            verify_sources!();assert_owned!(c);d=cp.data
            require(digest_data(d)==cp.checksum,:checkpoint_checksum,"checkpoint integrity mismatch")
            require(keys(d)==(:schema,:binding,:arrays,:clock,:inner_failed,:inner_history,:coupled_steps,:origin_steps,:failed,
                :tail,:tail_count,:receipt_count,:tail_anchor,:history_root,:last_success,:counter_scope),:checkpoint_schema,"malformed compact checkpoint fields")
            require(d.schema==CHECKPOINT_SCHEMA && isequal(d.binding,c.binding) && d.counter_scope==:instance_lifetime_not_restored,
                :checkpoint_binding,"compact schema/operator/mode/material/rates/configuration mismatch")
            shape=size(c.changes);N=length(c.changes)
            require(d.arrays isa Tuple && length(d.arrays)==4 && all(a isa Matrix{Float64} && size(a)==shape for a in d.arrays),
                :checkpoint_arrays,"exact Float64 checkpoint array shapes required")
            require(d.clock isa ClockRecord && valid_clock(d.clock),:checkpoint_clock,"invalid compact checkpoint clock")
            require(d.coupled_steps isa Int && d.origin_steps isa Int && 0<=d.origin_steps<=d.clock.steps &&
                d.coupled_steps==d.clock.steps-d.origin_steps,:checkpoint_count,"coupled/inner checkpoint counts disagree")
            require(d.inner_failed===false && d.inner_history===:empty && d.failed isa Bool &&
                d.tail isa NTuple{8,CompactReceipt} && d.tail_count isa Int && d.receipt_count isa Int &&
                d.tail_anchor isa Digest && d.history_root isa Digest &&
                valid_tail(d.tail,d.tail_count,d.receipt_count,d.tail_anchor,d.history_root,N,c.flow.grid.volume),
                :checkpoint_history,"malformed compact history")
            require(d.last_success isa CompactReceipt && (d.coupled_steps==0 ? isequal(d.last_success,EMPTY_RECEIPT) :
                valid_receipt(d.last_success,N,c.flow.grid.volume) && d.last_success.kind==:step &&
                d.last_success.coupled_step==d.coupled_steps && isequal(d.last_success.to_clock,d.clock)),
                :checkpoint_last_success,"last success does not match checkpoint clock/count")
            refresh_trial!(c)
            for (a,b) in zip(Flow.state_arrays(c.trial),d.arrays);copyto!(a,b);end
            Flow.derive!(c.trial.D,c.trial.U,c.trial.material);Audit.copy_clock!(c.trial,clock_tuple(d.clock))
            c.trial.failed=false;empty!(c.trial.history);validate_flow!(c.trial,c.validation);Audit.assert_intervals!(c.trial,c.source)
            verify_sources!();assert_owned!(c)
            base=tail_base(c);r=make_receipt(c,base,:restore,:restored;phase=:restore,stage=0,cell=0,
                step=d.coupled_steps,to=d.clock,checkpoint_root=d.history_root)
            next=prepare_tail(base,r)
            seal=fingerprint(c;U=c.trial.U,D=c.trial.D,clock=d.clock,inner_failed=false,failed=d.failed,
                steps=d.coupled_steps,origin=d.origin_steps,tail=next.tail,tail_count=next.count,count=next.total,
                anchor=next.anchor,root=next.root,last_success=d.last_success,last_attempt=r)
            Flow.copy_state!(c.flow.U,c.trial.U);Flow.copy_derived!(c.flow.D,c.trial.D);Audit.copy_clock!(c.flow,clock_tuple(d.clock))
            c.flow.failed=false;empty!(c.flow.history);Audit.invalidate_reconstruction!(c.flow)
            c.coupled_steps=d.coupled_steps;c.origin_steps=d.origin_steps;c.failed=d.failed;c.last_success=d.last_success
            c.last_attempt=r;apply_tail!(c,next);c.committed_hash=seal
            (;success=true,status=:restored,time=c.flow.time,step=c.coupled_steps)
        catch e
            if e isa CompactAdmissionError || e isa Audit.CoupledAdmissionError || e isa Flow.AdmissionError
                return failure!(c,e.status,e.message)
            end
            failure!(c,:unexpected_exception,sprint(showerror,e));rethrow()
        end
    end
end
function restore!(c::CompactCoupledSolver,cp)
    lock(c.lock) do
        begin_attempt!(c,NaN);c.pending.phase=:restore
        failure!(c,:checkpoint_type,"a complete CompactCheckpoint is required; no implicit legacy import")
    end
end

end # module
