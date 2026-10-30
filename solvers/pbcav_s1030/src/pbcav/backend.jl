# =====================================================================================================================
# backend.jl -- the solver's data on a KernelAbstractions backend (stage P3; work item W3 of plan Amendment 12, section 32;
# 21 Oct 2026).
#
# ONE CODE BASE (plan 5.5). Every loop over cells or faces was already a KernelAbstractions kernel launched on the backend
# its arrays live on; this file moves the data there. A device run (PB_BACKEND=cuda: CUDA.jl's CUDABackend, which the
# DRIVER or a test loads -- this module never loads CUDA, so a CPU run is unchanged and never touches a GPU) builds its
# mesh on the CPU (build_faces with backend CPU()) and moves it with `to_backend`, so the metrics are the CPU's bit for bit
# and only the solver's arithmetic runs on the device; the condition set and the state likewise. The work space (pbwork:
# the momentum and pressure systems, the least-squares geometry, the line families, the Krylov work, the coupled step's
# CoupledWork, the multigrid hierarchy and its numeric plan) is allocated on cfg.backend by its own constructors.
# WHAT CROSSES TO THE HOST DURING A STEP (counted by the W3 launch census, records air/w3_gpu/): scalars only -- the FCG and
# FGMRES dots, the momentum residual and |Theta| maxima, the multigrid trigger's maximum and the refresh's three pivot
# minima, the reference cell's diagonal, the non-positive-D counts -- and the coarsest multigrid level (its few hundred
# cells' operator, read back and inverted on the host AT EVERY REFRESH, and the dense inverse UPLOADED back -- 232 x 232 =
# 53,824 values on N60R L0, about 2 refreshes per step, about 0.86 MB host-to-device per step and a synchronisation point in
# each refresh: review of W3, F9; W3's launch census counted only `_dev` uploads and missed this `copyto!` --
# PBConfig.mg_coarse = :host, plan 24.6). The
# hierarchy's host BUILD (the symbolic phase, review F5) runs once per march and once per projection start, reading the
# pressure operator back once; it is setup, not a step.
# KernelAbstractions issue 652: no kernel here (there are none).
# =====================================================================================================================

"""
    to_backend(backend, x)

`x` -- a PBMesh, BCSet or PBState -- with every array on `backend` (CPU(), or CUDA.jl's CUDABackend once the caller has
loaded CUDA, or any KernelAbstractions backend whose array type Adapt knows). Values are copied bit for bit; scalars and
metadata are kept. Arrays already on `backend` are kept, not copied (Adapt's rule: on CPU() a host object comes back with
the same arrays); `to_host` always copies a state.
"""
to_backend(backend, m::PBMesh) = Adapt.adapt(backend, m)
# BCSet and PBState are rebuilt FIELD BY FIELD from fieldnames (review of W3, F1, the merge with the W1+W2 integration, 22 Oct
# 2026): W1 added PBState's `mut`, and a positional constructor of the nine pre-W1 arrays threw MethodError on every device
# path after the merge; built from fieldnames, every array field moves (any later field included) and every other field
# (NK, t, dt, dt_prev, nstep; npdir) is kept. PBMesh's Adapt rule (mesh_faces.jl) is built the same way.
_fieldwise(T, x, f) = T((let v = getfield(x, n); v isa AbstractArray ? f(v) : v end for n in fieldnames(T))...)
to_backend(backend, bc::BCSet) = _fieldwise(BCSet, bc, A -> Adapt.adapt(backend, A))
to_backend(backend, st::PBState) = _fieldwise(PBState, st, A -> Adapt.adapt(backend, A))

"A host copy of a state (always a copy, from any backend: every array field, `mut` included)."
to_host(st::PBState) = _fieldwise(PBState, st, A -> Array(A))
"The mesh with its arrays on the host (the same arrays when they already are)."
to_host(m::PBMesh) = Adapt.adapt(Array, m)

"""
    copy_state!(dst, src) -> dst

Every field of `src` at every time level, and its scalars (t, dt, dt_prev, nstep), into `dst`, across backends, bit for bit.
"""
function copy_state!(dst::PBState, src::PBState)
    for f in STATE_FIELDS; copyto!(getfield(dst, f), getfield(src, f)); end
    dst.t = src.t; dst.dt = src.dt; dst.dt_prev = src.dt_prev; dst.nstep = src.nstep
    dst
end

"""
    state_sha(st) -> String

The SHA-256 of a state: every field at every time level in the order of STATE_FIELDS (raw Float64 bytes, column-major),
then t, dt, dt_prev and nstep. Equal hashes mean bit-for-bit equal states on any backend.
"""
function state_sha(st::PBState)
    io = IOBuffer()
    for f in STATE_FIELDS; write(io, reinterpret(UInt8, vec(Array(getfield(st, f))))); end
    write(io, reinterpret(UInt8, [st.t, st.dt, st.dt_prev])); write(io, Int64(st.nstep))
    bytes2hex(sha256(take!(io)))
end

"The KernelAbstractions backend an array lives on (its parent's, for a view)."
array_backend(A::AbstractArray) = KernelAbstractions.get_backend(A)

"""
    check_on_backend(backend, objs...; what="")

Refuse (ArgumentError) when an array field of a PBMesh, BCSet or PBState in `objs` is not on `backend` (compared by
type: a CPU mesh handed to a device configuration fails here, at setup, rather than deep inside a kernel launch).
"""
function check_on_backend(backend, objs...; what::AbstractString="")
    tb = typeof(backend)
    for o in objs
        o === nothing && continue
        for f in fieldnames(typeof(o))
            v = getfield(o, f)
            v isa AbstractArray || continue
            typeof(array_backend(v)) == tb ||
                throw(ArgumentError("$(what)$(nameof(typeof(o))).$f is on $(typeof(array_backend(v))), the configuration's " *
                                    "backend is $tb: move the mesh, condition set and state with to_backend(backend, x) first"))
        end
    end
    nothing
end

# ---- the GPU's owners (host; before any device library is loaded) ----------------------------------------------------
"""
    gpu_owners(text; resolve=pid_image) -> (julia, unknown, other)

The rows of `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` (`text`) classified for the one-GPU-owner
rule (review of W3, F11, 22 Oct 2026). A row whose process name contains "julia" is a julia owner. A row WITHOUT a usable
name (`[Insufficient Permissions]`, `[N/A]`, empty: this machine lists dwm.exe that way) is resolved through the operating
system, `resolve(pid)` (the image name, or `nothing`): julia when the image is julia, other when the system names another
image, UNKNOWN when it cannot be resolved -- an unidentified owner is never taken as free. Other named rows are other. Each
list holds (pid, name) pairs; a resolved row's name is "<image> (resolved)".
"""
function gpu_owners(text::AbstractString; resolve=pid_image)
    jl = Tuple{Int,String}[]; unk = Tuple{Int,String}[]; oth = Tuple{Int,String}[]
    for line in split(text, '\n')
        s = strip(line); isempty(s) && continue
        i = findfirst(',', s)
        pid = i === nothing ? nothing : tryparse(Int, strip(s[1:prevind(s, i)]))
        name = i === nothing ? "" : String(strip(s[nextind(s, i):end]))
        if pid === nothing
            push!(unk, (-1, String(s)))                 # an unreadable row: its owner is unknown
        elseif occursin(r"(?i)julia", name)
            push!(jl, (pid, name))
        elseif isempty(name) || startswith(name, "[")
            img = try resolve(pid) catch; nothing end
            if img === nothing || isempty(strip(img))
                push!(unk, (pid, name))
            elseif occursin(r"(?i)julia", img)
                push!(jl, (pid, string(img, " (resolved)")))
            else
                push!(oth, (pid, string(img, " (resolved)")))
            end
        else
            push!(oth, (pid, name))
        end
    end
    (julia=jl, unknown=unk, other=oth)
end

"""
    pid_image(pid) -> String or nothing

The image name of process `pid` from the operating system (Windows: `tasklist /FI "PID eq <pid>" /FO CSV /NH`; elsewhere
`ps -p <pid> -o comm=`), or `nothing` when it is not found or the query fails.
"""
function pid_image(pid::Integer)
    if Sys.iswindows()
        out = try read(`tasklist /FI "PID eq $pid" /FO CSV /NH`, String) catch; return nothing end
        s = strip(out)
        (isempty(s) || startswith(s, "INFO")) && return nothing
        m = match(r"^\"([^\"]*)\",\"(\d+)\"", s)
        m === nothing && return nothing
        parse(Int, m.captures[2]) == pid || return nothing
        String(m.captures[1])
    else
        out = try read(`ps -p $pid -o comm=`, String) catch; return nothing end
        s = strip(out); isempty(s) ? nothing : String(s)
    end
end

"""
    gpu_gate(what; smi=nothing, resolve=pid_image) -> classified rows

The one-GPU-owner gate of every CUDA entry point (the driver's PB_BACKEND=cuda, the suite's gpu and parity modes, the records'
CUDA scripts), run BEFORE CUDA.jl is loaded: refuses (ErrorException naming `what`) when nvidia-smi cannot be run, when a
julia process holds the GPU, or when a GPU process cannot be identified (`gpu_owners`). `smi` replaces nvidia-smi's output
(the suite's negative examples).
"""
function gpu_gate(what::AbstractString; smi=nothing, resolve=pid_image)
    text = smi !== nothing ? smi : try read(`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`, String) catch
        error("$what: nvidia-smi could not be run, so the GPU's owners are unknown (refused)")
    end
    o = gpu_owners(text; resolve)
    isempty(o.julia) || error("$what: the GPU is held by julia process(es) " *
                              join(["$p $n" for (p, n) in o.julia], "; ") * " (one GPU owner at a time; refused)")
    isempty(o.unknown) || error("$what: GPU process(es) " * join(["$p $n" for (p, n) in o.unknown], "; ") *
                                " could not be identified (an unknown owner is not taken as free; refused)")
    o
end
