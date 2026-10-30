# ==============================================================================
# lbm_slice_format.jl — the numeric slice file (".lbmslice", format version 1)
# ==============================================================================
# lbm-io-opts (27 Oct 2026). Self-contained: Base + the TOML standard library
# only, so a reader (the web app, a test) can `include` this file on its own.
#
# File layout (all integers and floats little-endian):
#   bytes  0..7   magic  "LBMSLICE"
#   bytes  8..15  UInt64 H = length in bytes of the TOML header text
#   bytes 16..23  UInt64 D = byte offset of the data section from the file start
#                 (D is a multiple of 64 and D >= 24 + H; the gap is zero bytes)
#   bytes 24..    the TOML header (UTF-8, H bytes)
#   bytes D..     the arrays, in the order of the header's [[arrays]] tables; each
#                 array starts at D + arrays[i].offset, holds n1*n2 elements of
#                 arrays[i].dtype ("float32" or "uint8") in column-major order:
#                 element (i, k) is at linear index i + (k - 1) * n1, with i along
#                 axis1 and k along axis2.
# The header documents everything else (plane, origin, spacing, units, the level
# combination rule); see docs/_records/2026-10-27_lbm_io_opts/README.md.
# ==============================================================================

module LBMSliceFormat

import TOML

export LBMSLICE_MAGIC, LBMSLICE_FORMAT_VERSION, write_lbmslice, read_lbmslice

const LBMSLICE_MAGIC = "LBMSLICE"
const LBMSLICE_FORMAT_VERSION = 1
const _ALIGN = 64

_dtype_name(::Type{Float32}) = "float32"
_dtype_name(::Type{UInt8}) = "uint8"
_dtype_type(name::AbstractString) = name == "float32" ? Float32 :
                                    name == "uint8" ? UInt8 :
                                    error("lbmslice: unknown dtype $(repr(name))")

"""
    write_lbmslice(path, header, arrays)

Write one slice file. `header` is a Dict of TOML-representable values (it must not
contain the key `arrays`); `arrays` is a vector of `(name, matrix, units, description)`
with matrices of identical size (n1, n2) and element type Float32 or UInt8. The
`arrays` table, `n1`, `n2`, `format` and `format_version` are added here. The file
is written to `path * ".partial"` and renamed into place, so a reader never sees a
half-written file.
"""
function write_lbmslice(path::AbstractString, header::AbstractDict, arrays)
    ENDIAN_BOM == 0x04030201 || error("lbmslice: writer assumes a little-endian host")
    haskey(header, "arrays") && error("lbmslice: header must not define `arrays`")
    isempty(arrays) && error("lbmslice: no arrays")
    n1, n2 = size(arrays[1][2])
    tables = Dict{String,Any}[]
    offset = 0
    for (name, a, units, description) in arrays
        size(a) == (n1, n2) || error("lbmslice: array $name has size $(size(a)), expected ($n1, $n2)")
        T = eltype(a)
        (T === Float32 || T === UInt8) || error("lbmslice: array $name has element type $T")
        nbytes = n1 * n2 * sizeof(T)
        push!(tables, Dict{String,Any}("name" => String(name), "dtype" => _dtype_name(T),
                                       "offset" => offset, "nbytes" => nbytes,
                                       "units" => String(units), "description" => String(description)))
        offset += nbytes
        offset = cld(offset, _ALIGN) * _ALIGN
    end
    h = Dict{String,Any}(String(k) => v for (k, v) in header)
    h["format"] = "LUDWIG_LBM_SLICE"
    h["format_version"] = LBMSLICE_FORMAT_VERSION
    h["n1"] = n1
    h["n2"] = n2
    h["arrays"] = tables
    text = sprint(io -> TOML.print(io, h; sorted=true))
    hbytes = codeunits(text)
    data_offset = cld(24 + length(hbytes), _ALIGN) * _ALIGN
    tmp = string(path, ".partial")
    try
        open(tmp, "w") do io
            write(io, codeunits(LBMSLICE_MAGIC))
            write(io, htol(UInt64(length(hbytes))))
            write(io, htol(UInt64(data_offset)))
            write(io, hbytes)
            write(io, zeros(UInt8, data_offset - 24 - length(hbytes)))
            pos = 0
            for (t, (_, a, _, _)) in zip(tables, arrays)
                pad = t["offset"] - pos
                pad > 0 && write(io, zeros(UInt8, pad))
                write(io, a)
                pos = t["offset"] + t["nbytes"]
            end
        end
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)   # never leave a half-written file behind
        rethrow()
    end
    return path
end

"""
    read_lbmslice(path) -> (header::Dict{String,Any}, arrays::Dict{String,Matrix})

Read a slice file written by `write_lbmslice`, checking the magic, the format
version and that every array lies inside the file.
"""
function read_lbmslice(path::AbstractString)
    ENDIAN_BOM == 0x04030201 || error("lbmslice: reader assumes a little-endian host")
    open(path, "r") do io
        magic = String(read(io, 8))
        magic == LBMSLICE_MAGIC || error("lbmslice: bad magic $(repr(magic)) in $path")
        hlen = Int(ltoh(read(io, UInt64)))
        data_offset = Int(ltoh(read(io, UInt64)))
        (data_offset >= 24 + hlen && data_offset % 64 == 0) || error("lbmslice: bad data offset in $path")
        header = TOML.parse(String(read(io, hlen)))
        header["format_version"] == LBMSLICE_FORMAT_VERSION ||
            error("lbmslice: unsupported format_version $(header["format_version"])")
        n1 = header["n1"]; n2 = header["n2"]
        fsize = filesize(path)
        arrays = Dict{String,Matrix}()
        for t in header["arrays"]
            T = _dtype_type(t["dtype"])
            t["nbytes"] == n1 * n2 * sizeof(T) || error("lbmslice: array $(t["name"]) has a bad size")
            data_offset + t["offset"] + t["nbytes"] <= fsize ||
                error("lbmslice: array $(t["name"]) extends past the end of $path")
            seek(io, data_offset + t["offset"])
            a = Matrix{T}(undef, n1, n2)
            read!(io, a)
            arrays[t["name"]] = a
        end
        return header, arrays
    end
end

end # module LBMSliceFormat
