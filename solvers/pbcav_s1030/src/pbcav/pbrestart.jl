# =====================================================================================================================
# pbrestart.jl -- the checkpoint schema PBRestart/1 (plan 6.1, EC-F9). Stage P0b, 14 Oct 2026.
#
# STANDALONE ON PURPOSE. This file defines module PBRestartIO and uses only the standard library (TOML, SHA), so a reader
# includes it in a bare Julia session and never loads PBCav's environment, and no file is tied to a module's types:
#     include(<path of this file>); using .PBRestartIO
#     hdr, arrays = read_pbrestart(path)
# PBCav includes the same file (PBCav.PBRestartIO).
#
# THE FILE.
#   line 1      "PBRestart/1 header_bytes=NNNNNNNNNN\n" (ten digits): the length of the header that follows
#   header      TOML, UTF-8, exactly header_bytes bytes:
#                 schema = "PBRestart/1", endianness = "little", writer, written_unix, data_bytes
#                 [meta]      the run's metadata; must hold sigma, p_ref, q, p_v, knobs (table), mesh_hashes (array)
#                 [[arrays]]  one per array: name, eltype (Float64 Float32 Int64 Int32 UInt8), dims, offset (bytes from
#                             the start of the data section, a multiple of 8), nbytes, sha256 (of the stored bytes)
#   data        the arrays' raw little-endian bytes in column-major order, each starting at a multiple of 8, zero-padded
# The reader refuses: another schema or endianness, an unknown eltype, a size or offset that does not fit, a digest that
# does not match, missing metadata, and bytes beyond data_bytes. The writer refuses an existing path (exclusive create:
# it writes a private temporary file and renames it, which fails if the path appeared meanwhile).
# =====================================================================================================================
module PBRestartIO

using TOML, SHA

export PBRESTART_SCHEMA, write_pbrestart, read_pbrestart, read_pbrestart_header

const PBRESTART_SCHEMA = "PBRestart/1"
const REQUIRED_META = ("sigma", "p_ref", "q", "p_v", "knobs", "mesh_hashes")
const ELTYPES = Dict("Float64" => Float64, "Float32" => Float32, "Int64" => Int64, "Int32" => Int32, "UInt8" => UInt8)
const ELNAMES = Dict(v => k for (k, v) in ELTYPES)
const LINE1_PREFIX = "PBRestart/1 header_bytes="

_pad8(n) = (8 - n % 8) % 8

"""
    write_pbrestart(path, meta, arrays; writer="PBCav") -> path

Write `arrays` (a vector of `name => Array` pairs) and the metadata table `meta` as a PBRestart/1 file. Refuses an
existing `path`, a duplicate or empty name, an unsupported element type and metadata without $(join(REQUIRED_META, ", ")).
"""
function write_pbrestart(path::AbstractString, meta::AbstractDict, arrays::AbstractVector; writer::AbstractString="PBCav")
    ispath(path) && throw(ArgumentError("PBRestart: refusing to overwrite $path (exclusive create)"))
    for k in REQUIRED_META
        haskey(meta, k) || throw(ArgumentError("PBRestart: metadata must hold $(join(REQUIRED_META, ", ")) (missing $k)"))
    end
    meta["knobs"] isa AbstractDict || throw(ArgumentError("PBRestart: metadata knobs must be a table"))
    meta["mesh_hashes"] isa AbstractVector || throw(ArgumentError("PBRestart: metadata mesh_hashes must be an array"))
    names = String[]; entries = Dict{String,Any}[]; blobs = Vector{UInt8}[]; off = 0
    for pr in arrays
        name = String(first(pr)); a = last(pr)
        isempty(name) && throw(ArgumentError("PBRestart: empty array name"))
        name in names && throw(ArgumentError("PBRestart: duplicate array name $name"))
        a isa Array || throw(ArgumentError("PBRestart: $name is a $(typeof(a)); host Arrays only (copy device arrays first)"))
        haskey(ELNAMES, eltype(a)) || throw(ArgumentError("PBRestart: $name has element type $(eltype(a)); allowed: " *
                                                         join(sort(collect(keys(ELTYPES))), ", ")))
        bytes = Vector{UInt8}(reinterpret(UInt8, vec(htol.(a))))
        push!(names, name); push!(blobs, bytes)
        push!(entries, Dict{String,Any}("name" => name, "eltype" => ELNAMES[eltype(a)], "dims" => collect(Int, size(a)),
                                       "offset" => off, "nbytes" => length(bytes), "sha256" => bytes2hex(sha256(bytes))))
        off += length(bytes) + _pad8(length(bytes))
    end
    hdr = Dict{String,Any}("schema" => PBRESTART_SCHEMA, "endianness" => "little", "writer" => String(writer),
                           "written_unix" => time(), "data_bytes" => off, "meta" => meta, "arrays" => entries)
    htxt = sprint(io -> TOML.print(io, hdr; sorted=true))
    hb = codeunits(htxt)
    length(hb) < 10^10 || throw(ArgumentError("PBRestart: header too large"))
    line1 = LINE1_PREFIX * lpad(string(length(hb)), 10, '0') * "\n"
    dir = dirname(abspath(path)); isdir(dir) || throw(ArgumentError("PBRestart: directory $dir does not exist"))
    tmp = string(abspath(path), ".partial.", getpid(), ".", time_ns())
    open(tmp, "w") do io
        write(io, line1); write(io, hb)
        for b in blobs; write(io, b); write(io, zeros(UInt8, _pad8(length(b)))); end
    end
    try
        mv(tmp, path)                       # refuses if `path` appeared meanwhile
    catch err
        rm(tmp; force=true); rethrow(err)
    end
    String(path)
end

"Read and validate the header of a PBRestart/1 file: (header::Dict, data_start::Int)."
function read_pbrestart_header(path::AbstractString)
    open(path, "r") do io
        l1 = readline(io; keep=true)
        startswith(l1, LINE1_PREFIX) && length(l1) == length(LINE1_PREFIX) + 11 && endswith(l1, "\n") ||
            throw(ArgumentError("PBRestart: $path does not start with \"$(LINE1_PREFIX)\" and ten digits (not a PBRestart/1 file)"))
        nh = parse(Int, l1[length(LINE1_PREFIX)+1:end-1])
        hb = read(io, nh); length(hb) == nh || throw(ArgumentError("PBRestart: header truncated"))
        hdr = TOML.parse(String(hb))
        get(hdr, "schema", "") == PBRESTART_SCHEMA || throw(ArgumentError("PBRestart: schema $(get(hdr, "schema", "")) is not $PBRESTART_SCHEMA"))
        get(hdr, "endianness", "") == "little" || throw(ArgumentError("PBRestart: endianness must be little"))
        meta = get(hdr, "meta", nothing)
        meta isa AbstractDict && all(k -> haskey(meta, k), REQUIRED_META) ||
            throw(ArgumentError("PBRestart: metadata must hold $(join(REQUIRED_META, ", "))"))
        hdr, position(io)
    end
end

"""
    read_pbrestart(path) -> (header::Dict, arrays::Dict{String,Array})

Read a PBRestart/1 file, verifying every array's size, offset and digest, and that no byte follows the data section.
"""
function read_pbrestart(path::AbstractString)
    hdr, d0 = read_pbrestart_header(path)
    nd = hdr["data_bytes"]
    filesize(path) == d0 + nd || throw(ArgumentError("PBRestart: file is $(filesize(path)) bytes, header says $(d0 + nd)"))
    out = Dict{String,Array}()
    open(path, "r") do io
        for e in get(hdr, "arrays", Any[])
            name = e["name"]; T = get(ELTYPES, e["eltype"], nothing)
            T === nothing && throw(ArgumentError("PBRestart: $name has unknown eltype $(e["eltype"])"))
            dims = Tuple(Int.(e["dims"])); nb = e["nbytes"]; off = e["offset"]
            all(>=(0), dims) && nb == prod(dims; init=1)*sizeof(T) && off % 8 == 0 && 0 <= off && off + nb <= nd ||
                throw(ArgumentError("PBRestart: $name's dims, nbytes or offset do not fit"))
            haskey(out, name) && throw(ArgumentError("PBRestart: duplicate array $name"))
            seek(io, d0 + off); b = read(io, nb)
            length(b) == nb && bytes2hex(sha256(b)) == e["sha256"] ||
                throw(ArgumentError("PBRestart: $name's bytes do not match their sha256"))
            v = ltoh.(collect(reinterpret(T, b)))
            out[name] = reshape(v, dims)
        end
    end
    hdr, out
end

end # module PBRestartIO
