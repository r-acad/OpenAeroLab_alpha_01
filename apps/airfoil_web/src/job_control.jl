module AirfoilJobControl

export CancellationToken, JobCancelled, with_token, requested, check_cancelled,
       cancel!, process_command, track!, finish!

"A cancellation request belongs to one job, including every point in its sweep."
mutable struct CancellationToken
    lock::ReentrantLock
    cancelled::Bool
    children::IdDict{Any,Nothing}
    marker::String
    grace_seconds::Float64
end
CancellationToken(;marker="",grace_seconds=30.) = CancellationToken(ReentrantLock(), false,
    IdDict{Any,Nothing}(),String(marker),Float64(grace_seconds))
struct JobCancelled <: Exception end
Base.showerror(io::IO, ::JobCancelled) = print(io, "Analysis cancelled by the user")

const CONTEXT_KEY = :airfoil_lab_cancellation_token
current_token() = get(task_local_storage(), CONTEXT_KEY, nothing)
requested(token::Union{Nothing,CancellationToken}=current_token()) =
    token === nothing ? false : lock(() -> token.cancelled, token.lock)
check_cancelled(token::Union{Nothing,CancellationToken}=current_token()) =
    requested(token) ? throw(JobCancelled()) : nothing

function with_token(action::Function, token::CancellationToken)
    storage=task_local_storage(); had=haskey(storage,CONTEXT_KEY); previous=get(storage,CONTEXT_KEY,nothing)
    storage[CONTEXT_KEY]=token
    try
        check_cancelled(token)
        action()
    finally
        had ? (storage[CONTEXT_KEY]=previous) : delete!(storage,CONTEXT_KEY)
    end
end

# A dedicated process group on Unix, or an owned Windows Job Object, includes
# solver grandchildren. Never find processes by executable name or kill a PID
# obtained from a file. Only Process objects spawned by this job are registered.
function process_command(cmd::Cmd)
    prepared=Cmd(cmd;detach=true,windows_hide=true)
    token=current_token()
    token===nothing || isempty(token.marker) ? prepared : addenv(prepared,"AIRFOIL_CANCEL_FILE"=>token.marker)
end
mutable struct OwnedProcess
    process::Base.Process
    token::Union{Nothing,CancellationToken}
    windows_job::Ptr{Cvoid}
    finished::Bool
end

function _windows_job(process::Base.Process)
    Sys.iswindows() || return C_NULL
    job=ccall((:CreateJobObjectW,"kernel32"),Ptr{Cvoid},(Ptr{Cvoid},Ptr{UInt16}),C_NULL,C_NULL)
    job == C_NULL && error("Could not create the worker's process group")
    try
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, 64-bit Windows. LimitFlags is
        # at byte 16; KILL_ON_JOB_CLOSE also cleans up children after an error.
        limits=zeros(UInt8,144)
        GC.@preserve limits begin
            unsafe_store!(Ptr{UInt32}(pointer(limits)+16),UInt32(0x2000))
            ccall((:SetInformationJobObject,"kernel32"),Int32,
                (Ptr{Cvoid},Int32,Ptr{UInt8},UInt32),job,9,pointer(limits),length(limits)) != 0 ||
                error("Could not configure the worker's process group")
        end
        # libuv retains a process handle, preventing this PID from being reused
        # while we obtain a Windows handle to the exact child we just spawned.
        child=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x0101,0,getpid(process))
        child == C_NULL && error("Could not retain the worker process handle")
        try
            ccall((:AssignProcessToJobObject,"kernel32"),Int32,(Ptr{Cvoid},Ptr{Cvoid}),job,child) != 0 ||
                error("Could not attach the worker to its process group")
        finally
            ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),child)
        end
        job
    catch
        ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),job)
        rethrow()
    end
end

function _stop!(child::OwnedProcess)
    child.finished && return
    if Sys.iswindows() && child.windows_job != C_NULL
        ccall((:TerminateJobObject,"kernel32"),Int32,(Ptr{Cvoid},UInt32),child.windows_job,1)
    elseif !Sys.iswindows()
        # process_command creates this child's own group, so -pid cannot target
        # the web server or another job's solver.
        ccall(:kill,Int32,(Int32,Int32),-getpid(child.process),Base.SIGKILL)
    end
end

"Register a freshly spawned worker before it can emit progress or start a sweep."
function track!(process::Base.Process, token::Union{Nothing,CancellationToken}=current_token())
    child=OwnedProcess(process,token,C_NULL,false)
    try
        child.windows_job=_windows_job(process)
    catch
        process_running(process) && kill(process)
        wait(process)
        rethrow()
    end
    if token !== nothing
        lock(token.lock) do
            token.children[child]=nothing
            token.cancelled && _stop!(child)
        end
    end
    child
end

"Close the process group and release its registration after logs have drained."
function finish!(child::OwnedProcess)
    token=child.token
    token === nothing ? _finish!(child) : lock(() -> _finish!(child),token.lock)
end
function _finish!(child::OwnedProcess)
    child.finished && return
    token=child.token
    token === nothing || delete!(token.children,child)
    _stop!(child)
    child.finished=true
    if child.windows_job != C_NULL
        ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),child.windows_job)
        child.windows_job=C_NULL
    end
    nothing
end

"Request cancellation and stop only this token's currently owned process groups."
function cancel!(token::CancellationToken)
    lock(token.lock) do
        token.cancelled && return
        token.cancelled=true
        cooperative=!isempty(token.marker)
        if cooperative
            try
                mkpath(dirname(token.marker))
                write(token.marker,"Cancellation requested by Airfoil Lab\n")
            catch
                cooperative=false
            end
        end
        if cooperative && token.grace_seconds>0
            @async begin
                sleep(token.grace_seconds)
                lock(token.lock) do
                    foreach(_stop!,keys(token.children))
                end
            end
        else
            foreach(_stop!,keys(token.children))
        end
    end
    nothing
end

end
