module FieldSnapshots

using UUIDs

export Schedule, save_if_due!, atomic_write, scratch_root, interval_seconds

const DEFAULT_SECONDS = 600.0
const MIN_FREE_BYTES = 1024^3
monotonic_seconds() = time_ns() / 1.0e9

function interval_seconds(value=DEFAULT_SECONDS)
    value isa Real && !(value isa Bool) && isfinite(value) && 0 <= value <= 86400 ||
        throw(ArgumentError("Snapshot interval must be between 0 and 86400 seconds (0 disables periodic snapshots)"))
    Float64(value)
end

mutable struct Schedule{F}
    interval::Float64
    clock::F
    next_due::Float64
    saved::Int
    failed::Int
end
function Schedule(seconds=DEFAULT_SECONDS; clock=monotonic_seconds)
    interval=interval_seconds(seconds)
    Schedule(interval,clock,interval==0 ? Inf : clock()+interval,0,0)
end

"""Save at the first committed step after a wall-clock deadline, never catch up in a burst.
The timer starts after solver initialization. A failed write is logged and retried
only after another interval; it must not be announced as a published artifact.
"""
function save_if_due!(write_snapshot::Function, schedule::Schedule; log=println)
    schedule.clock() >= schedule.next_due || return false
    started=schedule.clock()
    ok=false
    try
        write_snapshot()
        schedule.saved+=1
        ok=true
        log("Snapshot saved; write took $(round(schedule.clock()-started;digits=3)) s; saved snapshots: $(schedule.saved).")
    catch err
        schedule.failed+=1
        log("SNAPSHOT_WARNING: snapshot not saved: $(sprint(showerror,err)). Next attempt in $(schedule.interval/60) minutes.")
    finally
        schedule.next_due=schedule.clock()+schedule.interval
    end
    ok
end

function scratch_root()
    public=dirname(dirname(@__DIR__))
    private=abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(public),"02_PRIVATE_PROJECT_DEVELOPMENT")))
    root=joinpath(private,"TEMP_DELETEABLE","snapshot_staging")
    mkpath(root)
    root
end

"""Stage and verify a complete artifact before publishing it. Never accept a short write.
Staging files are disposable and live exclusively in the private scratch tree.
"""
function atomic_write(path::AbstractString, bytes::AbstractVector{UInt8}; reserve_bytes::Integer=MIN_FREE_BYTES)
    mkpath(dirname(path))
    free=Base.Filesystem.diskstat(dirname(path)).available
    free >= length(bytes)+reserve_bytes || throw(ArgumentError(
        "Insufficient disk space for snapshot: $(round(free/1024^3;digits=2)) GiB free; leave at least $(round(reserve_bytes/1024^3;digits=2)) GiB free"))
    staging=joinpath(scratch_root(),string(uuid4())*".part")
    try
        open(staging,"w") do io
            write(io,bytes)==length(bytes) || error("Short snapshot write")
            flush(io)
        end
        filesize(staging)==length(bytes) || error("Incomplete snapshot write")
        # Artifacts are published only after all bytes have been closed successfully.
        Base.Filesystem.rename(staging,path)
        filesize(path)==length(bytes) || error("Incomplete snapshot publication")
    finally
        isfile(staging) && rm(staging;force=true)
    end
    length(bytes)
end

end
