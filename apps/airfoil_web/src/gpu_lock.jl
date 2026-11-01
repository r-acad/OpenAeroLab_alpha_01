# =====================================================================================================================
# gpu_lock.jl -- the shared GPU lock of the version-2 work for the app's low-speed workers (24 Oct 2026). Pure Julia
# (Base + Dates). The protocol uses a directory lock under the private development root:
#   * take the GPU by creating TEMP_DELETEABLE/GPU_LOCK.d with mkdir; it fails if the folder exists, and
#     then the worker waits and retries every 60 s;
#   * write owner.txt inside it (who, what, start time UTC);
#   * run, then remove the folder.
# A folder this process did not create is never removed: a lock is never broken. While waiting, the worker prints
# "GPU_WAIT,<the holder's owner line>" once per retry and, on taking the lock, "GPU_ACQUIRED,<its own owner line>"; the app
# turns these into the job status ("waiting for the GPU (held by ...)").
# =====================================================================================================================
module AirfoilGPULock
using Dates

const LOCK_DIR = get(ENV, "AIRFOIL_GPU_LOCK_DIR",
    joinpath(normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
        joinpath(@__DIR__, "..", "..", "..", "..", "02_PRIVATE_PROJECT_DEVELOPMENT"))),
        "TEMP_DELETEABLE", "GPU_LOCK.d"))
const OWNER_FILE = "owner.txt"

mutable struct GPULock
    dir::String
    owner::String
    held::Bool
end

utc_now() = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")

"The first line of the holder's owner.txt (or a placeholder while the holder has created the folder but not yet written it)."
function holder(dir::AbstractString=LOCK_DIR)
    f = joinpath(dir, OWNER_FILE)
    text = isfile(f) ? (try read(f, String) catch; "" end) : ""
    line = strip(first(split(text * "\n", '\n')))
    isempty(line) ? "an owner that has not written owner.txt yet" : String(line)
end

"""The retry interval: 60 s by the protocol. AIRFOIL_GPU_LOCK_POLL (seconds, 1-600) overrides it for a supervised test
only (24 Oct 2026: the one functional test taken between two A2 arms, whose chain re-takes the lock about 3 s after
releasing it); run.cmd never sets it."""
default_poll() = clamp(something(tryparse(Float64, get(ENV, "AIRFOIL_GPU_LOCK_POLL", "60")), 60.0), 1.0, 600.0)

"""    acquire!(what; dir, poll=default_poll(), report=60, io=stdout, max_wait=Inf) -> GPULock

Take the GPU lock, waiting while another owner holds it. A GPU_WAIT line is printed on the first refusal, whenever the
holder changes, and at least `report` s after the previous one. Registers an atexit hook that releases the lock if the
process exits without calling `release!`."""
function acquire!(what::AbstractString; dir::AbstractString=LOCK_DIR, poll::Real=default_poll(), report::Real=60,
                  io::IO=stdout, max_wait::Real=Inf)
    owner = "Airfoil Lab low-speed CFD worker (app server job): $(what); pid $(getpid()); started $(utc_now())"
    t0 = time(); last_holder = ""; last_print = -Inf
    while true
        created = try
            mkdir(dir); true          # atomic: fails when the folder exists
        catch err
            isdir(dir) || rethrow()   # any failure other than "the folder exists"
            false
        end
        if created
            write(joinpath(dir, OWNER_FILE), owner * "\n")
            lock = GPULock(String(dir), owner, true)
            atexit(() -> release!(lock))
            println(io, "GPU_ACQUIRED,", replace(owner, ',' => ';')); flush(io)
            return lock
        end
        h = holder(dir)
        if h != last_holder || time() - last_print >= report
            println(io, "GPU_WAIT,", replace(h, ',' => ';')); flush(io)
            last_holder = h; last_print = time()
        end
        time() - t0 > max_wait && error("waited $(round(Int, time() - t0)) s for the GPU lock (held by $(holder(dir)))")
        sleep(poll)
    end
end

"""    gate_wait(gate; tries=10, poll=60, io=stdout)

With the lock held, run the solver's own GPU gate (PBCav.gpu_gate refuses while a julia process is on the GPU, e.g. the
previous owner's process still exiting). While it refuses, print a GPU_WAIT line and retry every `poll` s; after `tries`
refusals rethrow (the worker then exits and its atexit hook releases the lock)."""
function gate_wait(gate; tries::Integer=10, poll::Real=60, io::IO=stdout)
    for attempt in 1:tries
        try
            return gate()
        catch err
            attempt == tries && rethrow()
            msg = first(split(sprint(showerror, err), '\n'))
            println(io, "GPU_WAIT,the lock is ours but the solver's GPU gate refuses: ", replace(msg, ',' => ';')); flush(io)
            sleep(poll)
        end
    end
end

"Remove the lock folder if this process still owns it (its owner.txt is this lock's line); idempotent."
function release!(lock::GPULock)
    lock.held || return false
    lock.held = false
    f = joinpath(lock.dir, OWNER_FILE)
    if isfile(f) && strip(read(f, String)) == lock.owner
        rm(lock.dir; recursive=true, force=true)
        return true
    end
    false
end

"""The app server's safety net after a worker exits (it may have been killed before its atexit hook ran): remove the
lock folder only when its owner.txt names this worker's pid. Returns true when it removed one."""
function release_if_pid!(pid::Integer; dir::AbstractString=LOCK_DIR)
    f = joinpath(dir, OWNER_FILE)
    isfile(f) || return false
    text = try read(f, String) catch; return false end
    occursin("Airfoil Lab low-speed CFD worker", text) && occursin("; pid $(pid);", text) || return false
    rm(dir; recursive=true, force=true)
    true
end

end # module AirfoilGPULock
