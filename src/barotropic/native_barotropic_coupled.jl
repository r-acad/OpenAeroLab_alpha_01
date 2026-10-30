module NativeBarotropicCoupled

using SHA, Serialization
import ..NativeBarotropicEOS as EOS
import ..NativeBarotropicSource as Source
import ..NativeBarotropicPeriodic as Flow

export CoupledSolver, CoupledCheckpoint, step!, checkpoint, restore!,
       state_arrays, clock_state, source_receipt, audit_counts, CoupledAdmissionError

const Flow64 = Flow.Solver{Float64}
const EXPECTED_FLOW = "39257ce9210af2175e4f62cd5037044e29717fcfab9fdd088be48a3afb1bc0a9"
const EXPECTED_SOURCE = "808297a4eb92a130871462cfcc439cf58bf932fdb361927ffa600b825bb6dcc2"
const EXPECTED_EOS = "dcef358579539a8c29ddffbd5fde090fd69add170421b0ec52c020526f387400"
const WRAPPER_PATH = abspath(@__FILE__)
const WRAPPER_SHA = bytes2hex(sha256(read(WRAPPER_PATH)))
const SOURCE_PATH = abspath(String(first(methods(Source.source_step)).file))
const SOURCE_SHA = bytes2hex(sha256(read(SOURCE_PATH)))
const SCHEMA = "native-source-flow-coupled-Float64-unified-flow-v2"

struct CoupledAdmissionError <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO,e::CoupledAdmissionError)=print(io,e.status,": ",e.message)
require(ok,status,message)=ok || throw(CoupledAdmissionError(status,message))
flow_mode(s::Flow.Solver)=s.config.reconstruction
function invalidate_reconstruction!(s::Flow64)
    s.reconstruction_scratch===nothing || Flow.reset!(s.reconstruction_scratch.diagnostics)
    nothing
end
values_of(x)=ntuple(i->getfield(x,i),fieldcount(typeof(x)))
hash_data(x)=begin io=IOBuffer();serialize(io,x);bytes2hex(sha256(take!(io))) end
serialized_bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io) end

function verify_sources!()
    for (path,loaded,expected) in ((Flow.OPERATOR_PATH,Flow.OPERATOR_SHA256,EXPECTED_FLOW),
            (Flow.EOS_PATH,Flow.EOS_SHA256,EXPECTED_EOS),
            (SOURCE_PATH,SOURCE_SHA,EXPECTED_SOURCE),(WRAPPER_PATH,WRAPPER_SHA,WRAPPER_SHA))
        require(loaded==expected && bytes2hex(sha256(read(path)))==expected,
                :source_binding,"loaded or current file identity mismatch: $path")
    end
    Source.ieee_environment() || throw(CoupledAdmissionError(:ieee_environment,
        "source requires nearest IEEE arithmetic and preserved subnormals"))
    nothing
end
source_receipt()=(;schema=SCHEMA,julia_version=string(VERSION),scalar="Float64",
    wrapper_sha256=WRAPPER_SHA,source_sha256=SOURCE_SHA,eos_sha256=EXPECTED_EOS,
    flow_sha256=EXPECTED_FLOW,flow_checkpoint_schema=Flow.CHECKPOINT_SCHEMA,
    arithmetic="source-half_actual-flow-SSPRK3_source-half_explicit-dt_private-trial")

function all_arrays(s)
    arrays=Any[]
    for name in (:U,:D,:base,:A,:B,:scratchD,:Fx,:Fy,:rhs)
        append!(arrays,values_of(getproperty(s,name)))
    end
    if s.reconstruction_scratch!==nothing
        append!(arrays,values_of(s.reconstruction_scratch.x))
        append!(arrays,values_of(s.reconstruction_scratch.y))
    end
    Tuple(arrays)
end
function make_binding(s,source)
    (;receipt=source_receipt(),mode=flow_mode(s),grid=values_of(s.grid),
      material=values_of(s.material),config=values_of(s.config),
      source_model=deepcopy(values_of(source)),flow_receipt=Flow.source_binding(s.digest))
end

"""Exclusive CPU Float64 ownership of an existing native flow/source pair.
No source step or flow step is taken by construction. Authoritative arrays,
derived fields, clock and audit history may subsequently change only through
this wrapper or its admitted coupled restore operation. Scratch is not a
committed-state interface. This first wrapper allocates private trials.
Successful construction, commit and restore invalidate reconstruction diagnostics.
The source-free inner solver remains available independently; this wrapper is
explicitly CPU Float64 and has no default movie/foil integration.
"""
mutable struct CoupledSolver{F<:Flow64}
    const flow::F
    const source::Source.SourceModel
    const binding::NamedTuple
    const owned_arrays::Tuple
    const lock::ReentrantLock
    const history::Vector{NamedTuple}
    const last_attempt::Dict{String,Any}
    coupled_steps::Int
    origin_steps::Int
    failed::Bool
    last_success::NamedTuple
    committed_hash::String
    source_calls::Int
    flow_calls::Int
end
state_arrays(c::CoupledSolver)=Flow.state_arrays(c.flow)
clock_state(c::CoupledSolver)=Flow.clock_state(c.flow)
audit_counts(c::CoupledSolver)=(;source=c.source_calls,flow=c.flow_calls)

function committed_data(c)
    s=c.flow;mod=Flow
    (;arrays=mod.state_arrays(s),derived=mod.derived_arrays(s.D),clock=mod.clock_state(s),
      inner_failed=s.failed,inner_history=s.history,coupled_steps=c.coupled_steps,
      origin_steps=c.origin_steps,failed=c.failed,history=c.history,last_success=c.last_success)
end
fingerprint(c)=hash_data(committed_data(c))
function assert_owned!(c)
    require(isequal(make_binding(c.flow,c.source),c.binding),:owner_configuration,
            "bound flow/source configuration changed")
    arrays=all_arrays(c.flow);shape=(c.flow.grid.nx,c.flow.grid.ny)
    require(length(arrays)==length(c.owned_arrays) &&
        all(a===b && a isa Matrix{Float64} && size(a)==shape for (a,b) in zip(arrays,c.owned_arrays)),
        :owner_arrays,"original CPU Float64 arrays and shapes are required")
    nothing
end
function validate_flow!(s;coherent=true)
    mod=Flow
    mod.validate_clock(mod.clock_state(s),Float64,:clock)
    require(!s.failed,:inner_failed,"inner flow must not have an outstanding failure")
    require(s.history isa Vector{NamedTuple} && all(mod.valid_history,s.history),
            :inner_history,"malformed inner flow audit history")
    fresh=mod.Derived(Float64,s.grid.nx,s.grid.ny)
    mod.derive!(fresh,s.U,s.material)
    coherent && require(all(isequal(a,b) for (a,b) in zip(mod.derived_arrays(fresh),mod.derived_arrays(s.D))),
                        :derived_state,"committed derived fields are stale")
    nothing
end
function CoupledSolver(s::F,source::Source.SourceModel) where {F<:Flow64}
    verify_sources!()
    require(s.config.reconstruction in (:pc,:muscl_rho_q),
            :flow_mode,"unified flow requires :pc or :muscl_rho_q")
    require(isequal(values_of(s.material),values_of(source.material)),:source_material,
            "source and flow material must match exactly")
    validate_flow!(s)
    arrays=all_arrays(s);shape=(s.grid.nx,s.grid.ny)
    require(all(a isa Matrix{Float64} && size(a)==shape for a in arrays) &&
            length(unique(objectid.(arrays)))==length(arrays),:owner_arrays,
            "distinct native CPU Float64 buffers required")
    c=CoupledSolver(s,source,make_binding(s,source),arrays,ReentrantLock(),NamedTuple[],
        Dict{String,Any}(),0,s.steps,false,(;),"",0,0)
    # A previously used flow may carry completed residual diagnostics from before ownership.
    invalidate_reconstruction!(s)
    c.committed_hash=fingerprint(c);c
end
CoupledSolver(s,source)=throw(CoupledAdmissionError(:type,"only native CPU Float64 flow and SourceModel are supported"))

function copy_clock!(dest,clock)
    dest.time=clock.time;dest.time_compensation=clock.compensation;dest.steps=clock.steps
    dest.last_dt=clock.last_dt;dest.last_path=clock.last_path
end
function cell_dt(s)
    mod=Flow;mod.derive!(s.scratchD,s.U,s.material)
    mod.applied_dt(s,nothing)
end
function assert_intervals!(s,source)
    for i in eachindex(s.U.q)
        iv=Source.admit_interval(source,s.U.rho[i],s.U.q[i])
        require(iv.valid,:checkpoint_source_domain,"source interval $(iv.reason) at cell $i")
    end
end
function failure!(c,phase,stage,cell,status,message)
    c.failed=true
    event=(;kind=:failure,phase,stage,cell,status,message,at_step=c.flow.steps,
            at_time=c.flow.time,coupled_step=c.coupled_steps)
    push!(c.history,event);c.committed_hash=fingerprint(c)
    (;success=false,phase,stage,cell,status,message,dt=0.0,time=c.flow.time,step=c.coupled_steps)
end

function source_half!(c,trial,half,phase)
    before=copy(trial.U.q);changes=zeros(Float64,size(before));cells=NamedTuple[]
    report=Dict{String,Any}("phase"=>phase,"h"=>half,"q_before"=>before,
        "changes"=>changes,"cells"=>cells,"complete"=>false)
    c.last_attempt[string(phase)]=report
    for index in eachindex(before)
        c.last_attempt["cell"]=index;c.source_calls+=1
        result=Source.source_step(c.source,trial.U.rho[index],before[index],half)
        push!(cells,merge((;index),result))
        require(result.valid,:source_rejection,"$(result.reason) at cell $index")
        changes[index]=result.q-before[index];trial.U.q[index]=result.q
    end
    Flow.derive!(trial.D,trial.U,trial.material)
    report["complete"]=true
    (;h=half,q_before=before,changes,cells,q_after=copy(trial.U.q),
      volume_weighted_sum_Float64=sum(changes)*trial.grid.volume)
end

"""One fully transactional S(h/2)-actual F(h)-S(h/2) composition.
Only expected admissions return a failure record; unexpected coding/runtime
exceptions are recorded and rethrown. Source/flow calls occur only on a private
trial, and there is no public fault-injection or retry switch.
"""
function step!(c::CoupledSolver,h::Float64)
    lock(c.lock) do
        c.failed && return (;success=false,phase=:entry,stage=0,cell=0,status=:failed_run,
            message="restore an admitted coupled checkpoint explicitly",dt=0.0,time=c.flow.time,step=c.coupled_steps)
        phase=:entry;stage=0
        empty!(c.last_attempt);merge!(c.last_attempt,Dict("h"=>h,"cell"=>0,"committed"=>false))
        try
            verify_sources!();assert_owned!(c)
            require(fingerprint(c)==c.committed_hash,:owner_state,"committed state, clock or history changed outside wrapper")
            validate_flow!(c.flow)
            require(isfinite(h) && h>0,:dt,"explicit finite positive Float64 h required")
            half=h/2
            require(half>0 && half+half==h,:half_time,"half interval must be positive and exactly compose h")
            require(c.coupled_steps<typemax(Int),:clock,"coupled step counter overflow")
            trial=deepcopy(c.flow);mod=Flow;oldclock=clock_state(c)
            c.last_attempt["initial_cell_dt"]=cell_dt(trial)
            phase=:first_source
            first=source_half!(c,trial,half,phase)
            phase=:post_source_cfl;c.last_attempt["cell"]=0
            after_dt=cell_dt(trial);c.last_attempt["post_source_cell_dt"]=after_dt
            require(h<=after_dt,:post_source_cfl,"explicit h exceeds post-first-source cell spectral estimate")
            phase=:flow;c.flow_calls+=1
            result=mod.step!(trial,h);stage=result.stage
            c.last_attempt["flow_result"]=result;c.last_attempt["inner_attempt_history"]=deepcopy(trial.history)
            require(result.success,:flow_rejection,"inner flow $(result.status): $(result.message)")
            require(trial.steps==oldclock.steps+1 && trial.last_dt===h && trial.last_path==:explicit_controlled,
                    :clock,"inner flow did not apply exactly one explicit h")
            phase=:second_source;stage=0
            second=source_half!(c,trial,half,phase)
            phase=:commit;c.last_attempt["cell"]=0
            validate_flow!(trial);verify_sources!();assert_owned!(c)
            require(fingerprint(c)==c.committed_hash,:owner_state,"committed state changed during private trial")
            newclock=mod.clock_state(trial)
            success=(;kind=:step,index=c.coupled_steps+1,dt=h,from_clock=oldclock,to_clock=newclock,
                first_source=first,second_source=second,initial_cell_dt=c.last_attempt["initial_cell_dt"],
                post_source_cell_dt=after_dt)
            inner_history=deepcopy(trial.history)
            # All expected admissions have passed. No inner restore! is invoked.
            mod.copy_state!(c.flow.U,trial.U);mod.copy_derived!(c.flow.D,trial.D)
            copy_clock!(c.flow,newclock);c.flow.history=inner_history;c.flow.failed=false
            # Private-trial residual diagnostics do not describe this committed coupled state.
            invalidate_reconstruction!(c.flow)
            c.coupled_steps+=1;c.last_success=success;push!(c.history,success)
            c.last_attempt["committed"]=true;c.committed_hash=fingerprint(c)
            (;success=true,phase=:complete,stage=3,cell=0,status=:ok,message="",dt=h,time=c.flow.time,step=c.coupled_steps)
        catch e
            if e isa CoupledAdmissionError || e isa Flow.AdmissionError
                return failure!(c,phase,stage,get(c.last_attempt,"cell",0),e.status,e.message)
            end
            failure!(c,phase,stage,get(c.last_attempt,"cell",0),:unexpected_exception,sprint(showerror,e))
            rethrow()
        end
    end
end
function step!(c::CoupledSolver,h)
    lock(c.lock) do
        failure!(c,:entry,0,0,:dt_type,"explicit timestep must be Float64; no conversion is applied")
    end
end

struct CoupledCheckpoint
    data::NamedTuple
    checksum::String
end
function valid_clock(clock)
    try
        Flow.validate_clock(clock,Float64,:history_clock);true
    catch e
        e isa Flow.AdmissionError || rethrow()
        false
    end
end
function valid_half(half,dt,shape,volume)
    half isa NamedTuple && keys(half)==(:h,:q_before,:changes,:cells,:q_after,:volume_weighted_sum_Float64) || return false
    half.h isa Float64 && isfinite(half.h) && half.h>0 && half.h+half.h==dt || return false
    all(a isa Matrix{Float64} && size(a)==shape && all(isfinite,a) for a in (half.q_before,half.changes,half.q_after)) || return false
    half.cells isa Vector{NamedTuple} && length(half.cells)==prod(shape) || return false
    half.volume_weighted_sum_Float64 isa Float64 && isfinite(half.volume_weighted_sum_Float64) || return false
    isequal(half.volume_weighted_sum_Float64,sum(half.changes)*volume) || return false
    for (i,cell) in enumerate(half.cells)
        keys(cell)==(:index,:valid,:reason,:q,:iterations,:bracket_width,:predictor_residual,:target,:rate,:hk) || return false
        cell.index===i && cell.valid===true && cell.reason in (:accepted,:identity,:representable_equilibrium) || return false
        cell.iterations isa Int && 0<=cell.iterations<=100 || return false
        all(x isa Float64 && isfinite(x) for x in (cell.q,cell.bracket_width,cell.predictor_residual,cell.target,cell.rate,cell.hk)) || return false
        cell.q>=0 && cell.bracket_width>=0 && cell.target>=0 && cell.rate>=0 && cell.hk>=0 || return false
        isequal(cell.q,half.q_after[i]) && isequal(cell.q-half.q_before[i],half.changes[i]) || return false
        min(half.q_before[i],cell.target)<=cell.q<=max(half.q_before[i],cell.target) || return false
    end
    true
end
function valid_coupled_history(event,shape,volume)
    event isa NamedTuple && hasproperty(event,:kind) || return false
    if event.kind==:failure
        return keys(event)==(:kind,:phase,:stage,:cell,:status,:message,:at_step,:at_time,:coupled_step) &&
            event.phase in (:entry,:first_source,:post_source_cfl,:flow,:second_source,:commit,:restore) &&
            event.stage isa Int && 0<=event.stage<=3 && event.cell isa Int && 0<=event.cell<=prod(shape) &&
            event.status isa Symbol && event.message isa String && event.at_step isa Int && event.at_step>=0 &&
            event.at_time isa Float64 && isfinite(event.at_time) && event.at_time>=0 &&
            event.coupled_step isa Int && event.coupled_step>=0
    elseif event.kind==:restore
        return keys(event)==(:kind,:coupled_step,:time,:prior_failed,:checkpoint_checksum) &&
            event.coupled_step isa Int && event.coupled_step>=0 && event.time isa Float64 &&
            isfinite(event.time) && event.time>=0 && event.prior_failed isa Bool &&
            event.checkpoint_checksum isa String && occursin(r"^[0-9a-f]{64}$",event.checkpoint_checksum)
    elseif event.kind==:quarantined_audit
        # Retain malformed attempted history as opaque bytes, never as an
        # admitted source/flow event. Recovery does not deserialize this payload.
        return keys(event)==(:kind,:payload,:payload_sha256,:checkpoint_checksum) &&
            event.payload isa Vector{UInt8} && event.payload_sha256 isa String &&
            bytes2hex(sha256(event.payload))==event.payload_sha256 &&
            event.checkpoint_checksum isa String && occursin(r"^[0-9a-f]{64}$",event.checkpoint_checksum)
    elseif event.kind==:step
        good=keys(event)==(:kind,:index,:dt,:from_clock,:to_clock,:first_source,:second_source,:initial_cell_dt,:post_source_cell_dt) &&
            event.index isa Int && event.index>0 && event.dt isa Float64 && isfinite(event.dt) && event.dt>0 &&
            valid_clock(event.from_clock) && valid_clock(event.to_clock) &&
            valid_half(event.first_source,event.dt,shape,volume) && valid_half(event.second_source,event.dt,shape,volume) &&
            event.initial_cell_dt isa Float64 && isfinite(event.initial_cell_dt) && event.initial_cell_dt>0 &&
            event.post_source_cell_dt isa Float64 && isfinite(event.post_source_cell_dt) && event.post_source_cell_dt>=event.dt
        good || return false
        old,new=event.from_clock,event.to_clock
        y=event.dt-old.compensation;t=old.time+y;comp=(t-old.time)-y
        return old.steps<typemax(Int) && new.steps==old.steps+1 && new.last_dt===event.dt &&
            new.last_path==:explicit_controlled && isequal(new.time,t) && isequal(new.compensation,comp)
    end
    false
end
function checkpoint(c::CoupledSolver)
    lock(c.lock) do
        verify_sources!();assert_owned!(c)
        require(fingerprint(c)==c.committed_hash,:owner_state,"out-of-band state change")
        validate_flow!(c.flow);assert_intervals!(c.flow,c.source)
        shape=(c.flow.grid.nx,c.flow.grid.ny)
        require(all(x->valid_coupled_history(x,shape,c.flow.grid.volume),c.history),:checkpoint_history,"malformed coupled history")
        data=(;schema=SCHEMA,binding=deepcopy(c.binding),arrays=map(copy,state_arrays(c)),clock=clock_state(c),
            inner_failed=c.flow.failed,inner_history=deepcopy(c.flow.history),coupled_steps=c.coupled_steps,
            origin_steps=c.origin_steps,failed=c.failed,history=deepcopy(c.history),last_success=deepcopy(c.last_success))
        CoupledCheckpoint(data,hash_data(data))
    end
end
"""Restore a complete trusted coupled checkpoint, retaining its failed flag.
A failed checkpoint is an audit snapshot: restoration succeeds, but further
steps still refuse. Continuation requires a checkpoint with failed=false.
Invalid attempted audit records are retained as opaque quarantined bytes.
"""
function restore!(c::CoupledSolver,cp::CoupledCheckpoint)
    lock(c.lock) do
        try
            verify_sources!();assert_owned!(c);d=cp.data
            require(hash_data(d)==cp.checksum,:checkpoint_checksum,"checkpoint integrity mismatch")
            require(keys(d)==(:schema,:binding,:arrays,:clock,:inner_failed,:inner_history,:coupled_steps,:origin_steps,:failed,:history,:last_success),
                    :checkpoint_schema,"malformed checkpoint fields")
            require(d.schema==SCHEMA && isequal(d.binding,c.binding),:checkpoint_binding,
                    "schema/Julia/precision/operator/wrapper/mode/material/rates/configuration mismatch")
            shape=(c.flow.grid.nx,c.flow.grid.ny)
            require(d.arrays isa Tuple && length(d.arrays)==4 && all(a isa Matrix{Float64} && size(a)==shape for a in d.arrays),
                    :checkpoint_arrays,"checkpoint arrays require exact Float64 matrix shape")
            mod=Flow;mod.validate_clock(d.clock,Float64,:checkpoint_clock)
            require(d.coupled_steps isa Int && d.origin_steps isa Int && d.coupled_steps>=0 && d.origin_steps>=0 &&
                    d.origin_steps<=d.clock.steps && d.coupled_steps==d.clock.steps-d.origin_steps,
                    :checkpoint_count,"coupled/inner step counts disagree")
            require(d.inner_failed===false && d.inner_history isa Vector{NamedTuple} && all(mod.valid_history,d.inner_history) &&
                    d.failed isa Bool && d.history isa Vector{NamedTuple} && all(x->valid_coupled_history(x,shape,c.flow.grid.volume),d.history),
                    :checkpoint_history,"malformed inner/coupled failure or audit history")
            require(d.last_success isa NamedTuple && (d.coupled_steps==0 ? isempty(d.last_success) :
                valid_coupled_history(d.last_success,shape,c.flow.grid.volume) && d.last_success.kind==:step &&
                d.last_success.index==d.coupled_steps && isequal(d.last_success.to_clock,d.clock)),
                :checkpoint_last_success,"last success does not match checkpoint clock/count")
            require(isempty(d.last_success) || any(x->isequal(x,d.last_success),d.history),
                    :checkpoint_last_success,"last success is absent from coupled history")
            trial=deepcopy(c.flow)
            for (dest,src) in zip(mod.state_arrays(trial),d.arrays);copyto!(dest,src);end
            mod.derive!(trial.D,trial.U,trial.material);copy_clock!(trial,d.clock)
            trial.failed=false;trial.history=deepcopy(d.inner_history)
            validate_flow!(trial);assert_intervals!(trial,c.source)
            merged=deepcopy(d.history)
            for event in c.history
                retained=if valid_coupled_history(event,shape,c.flow.grid.volume)
                    deepcopy(event)
                else
                    payload=serialized_bytes(event)
                    (;kind=:quarantined_audit,payload,payload_sha256=bytes2hex(sha256(payload)),
                      checkpoint_checksum=cp.checksum)
                end
                any(x->isequal(x,retained),merged) || push!(merged,retained)
            end
            push!(merged,(;kind=:restore,coupled_step=d.coupled_steps,time=d.clock.time,
                prior_failed=c.failed,checkpoint_checksum=cp.checksum))
            last=deepcopy(d.last_success);inner=deepcopy(trial.history)
            mod.copy_state!(c.flow.U,trial.U);mod.copy_derived!(c.flow.D,trial.D);copy_clock!(c.flow,d.clock)
            c.flow.failed=false;c.flow.history=inner;c.coupled_steps=d.coupled_steps;c.origin_steps=d.origin_steps
            c.failed=d.failed;c.last_success=last;empty!(c.history);append!(c.history,merged)
            # Restored authoritative state has no associated completed residual.
            invalidate_reconstruction!(c.flow)
            c.committed_hash=fingerprint(c)
            (;success=true,status=:restored,time=c.flow.time,step=c.coupled_steps)
        catch e
            if e isa CoupledAdmissionError || e isa Flow.AdmissionError
                return failure!(c,:restore,0,0,e.status,e.message)
            end
            failure!(c,:restore,0,0,:unexpected_exception,sprint(showerror,e));rethrow()
        end
    end
end
function restore!(c::CoupledSolver,cp)
    lock(c.lock) do
        failure!(c,:restore,0,0,:checkpoint_type,"an admitted complete CoupledCheckpoint is required")
    end
end

end # module
