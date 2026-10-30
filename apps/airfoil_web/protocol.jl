module AirfoilWire

"""Small, dependency-free MessagePack and BSON codecs for the app's JSON-like payloads.

Supported values are dictionaries with string keys, arrays, strings, byte arrays,
finite numbers, booleans, and `nothing`. BSON is the binary JSON format used here.
"""

export encode, decode, media_type, wire_format

const MAX_BYTES = 64 * 1024 * 1024
const MAX_DEPTH = 64

media_type(format::Symbol) = format === :msgpack ? "application/msgpack" :
    format === :bson ? "application/bson" : throw(ArgumentError("unknown wire format"))

function wire_format(s::AbstractString)
    t = lowercase(split(String(s), ';'; limit=2)[1]) |> strip
    t in ("application/msgpack", "application/x-msgpack") && return :msgpack
    t in ("application/bson", "application/x-bson") && return :bson
    throw(ArgumentError("expected application/msgpack or application/bson, got '$t'"))
end

function _u16be(io, x::Integer)
    0 <= x <= typemax(UInt16) || throw(ArgumentError("integer out of range"))
    write(io, UInt8((x >> 8) & 0xff), UInt8(x & 0xff))
end
function _u32be(io, x::Integer)
    0 <= x <= typemax(UInt32) || throw(ArgumentError("integer out of range"))
    for k in 3:-1:0; write(io, UInt8((x >> (8k)) & 0xff)); end
end
function _u64be(io, x::Integer)
    for k in 7:-1:0; write(io, UInt8((UInt64(x) >> (8k)) & 0xff)); end
end
function _u32le(io, x::Integer)
    for k in 0:3; write(io, UInt8((UInt32(x) >> (8k)) & 0xff)); end
end
function _u64le(io, x::Integer)
    for k in 0:7; write(io, UInt8((UInt64(x) >> (8k)) & 0xff)); end
end
function _readn(io, n::Integer)
    0 <= n <= MAX_BYTES || throw(ArgumentError("wire length out of range: $n"))
    bytes = read(io, n)
    length(bytes) == n || throw(EOFError())
    bytes
end
function _be(io, n::Int)
    x = UInt64(0)
    for _ in 1:n; x = (x << 8) | read(io, UInt8); end
    x
end
function _le(io, n::Int)
    x = UInt64(0)
    for k in 0:n-1; x |= UInt64(read(io, UInt8)) << (8k); end
    x
end
_i32bits(x) = reinterpret(Int32, UInt32(x))
_i64bits(x) = reinterpret(Int64, UInt64(x))

function _pack(io, x, depth::Int)
    depth <= MAX_DEPTH || throw(ArgumentError("MessagePack nesting too deep"))
    if x === nothing
        write(io, 0xc0)
    elseif x isa Bool
        write(io, x ? 0xc3 : 0xc2)
    elseif x isa Integer
        if 0 <= x <= 127
            write(io, UInt8(x))
        elseif -32 <= x < 0
            write(io, reinterpret(UInt8, Int8(x)))
        elseif 0 <= x <= 255
            write(io, 0xcc, UInt8(x))
        elseif 0 <= x <= 65535
            write(io, 0xcd); _u16be(io, x)
        elseif typemin(Int32) <= x <= typemax(Int32)
            write(io, 0xd2); _u32be(io, reinterpret(UInt32, Int32(x)))
        else
            write(io, 0xd3); _u64be(io, reinterpret(UInt64, Int64(x)))
        end
    elseif x isa AbstractFloat
        isfinite(x) || throw(ArgumentError("non-finite numbers are not wire values"))
        write(io, 0xcb); _u64be(io, reinterpret(UInt64, Float64(x)))
    elseif x isa AbstractString
        b = codeunits(String(x)); n = length(b)
        n <= MAX_BYTES || throw(ArgumentError("string too large"))
        if n <= 31
            write(io, UInt8(0xa0 | n))
        elseif n <= 255
            write(io, 0xd9, UInt8(n))
        elseif n <= 65535
            write(io, 0xda); _u16be(io, n)
        else
            write(io, 0xdb); _u32be(io, n)
        end
        write(io, b)
    elseif x isa AbstractVector{UInt8}
        n = length(x); n <= MAX_BYTES || throw(ArgumentError("binary value too large"))
        if n <= 255
            write(io, 0xc4, UInt8(n))
        elseif n <= 65535
            write(io, 0xc5); _u16be(io, n)
        else
            write(io, 0xc6); _u32be(io, n)
        end
        write(io, x)
    elseif x isa AbstractVector || x isa Tuple
        n = length(x); n <= MAX_BYTES || throw(ArgumentError("array too large"))
        if n <= 15
            write(io, UInt8(0x90 | n))
        elseif n <= 65535
            write(io, 0xdc); _u16be(io, n)
        else
            write(io, 0xdd); _u32be(io, n)
        end
        for v in x; _pack(io, v, depth+1); end
    elseif x isa AbstractDict || x isa NamedTuple
        pairs_x = collect(pairs(x)); n = length(pairs_x)
        n <= MAX_BYTES || throw(ArgumentError("map too large"))
        if n <= 15
            write(io, UInt8(0x80 | n))
        elseif n <= 65535
            write(io, 0xde); _u16be(io, n)
        else
            write(io, 0xdf); _u32be(io, n)
        end
        for (k,v) in pairs_x
            _pack(io, string(k), depth+1); _pack(io, v, depth+1)
        end
    else
        throw(ArgumentError("unsupported MessagePack value $(typeof(x))"))
    end
end

function _unpack(io, depth::Int)
    depth <= MAX_DEPTH || throw(ArgumentError("MessagePack nesting too deep"))
    b = read(io, UInt8)
    b <= 0x7f && return Int(b)
    b >= 0xe0 && return Int(reinterpret(Int8, b))
    if 0xa0 <= b <= 0xbf; return String(_readn(io, Int(b & 0x1f))); end
    if 0x90 <= b <= 0x9f; return [_unpack(io, depth+1) for _ in 1:Int(b & 0x0f)]; end
    if 0x80 <= b <= 0x8f; return _unpack_map(io, Int(b & 0x0f), depth); end
    b == 0xc0 && return nothing
    b == 0xc2 && return false
    b == 0xc3 && return true
    b == 0xcc && return Int(_be(io, 1))
    b == 0xcd && return Int(_be(io, 2))
    b == 0xce && return Int(_be(io, 4))
    b == 0xcf && return _be(io, 8)
    b == 0xd0 && return Int(reinterpret(Int8, UInt8(_be(io, 1))))
    b == 0xd1 && return Int(reinterpret(Int16, UInt16(_be(io, 2))))
    b == 0xd2 && return Int(_i32bits(_be(io, 4)))
    b == 0xd3 && return _i64bits(_be(io, 8))
    b == 0xca && return Float64(reinterpret(Float32, UInt32(_be(io, 4))))
    b == 0xcb && return reinterpret(Float64, UInt64(_be(io, 8)))
    b == 0xd9 && return String(_readn(io, Int(_be(io, 1))))
    b == 0xda && return String(_readn(io, Int(_be(io, 2))))
    b == 0xdb && return String(_readn(io, Int(_be(io, 4))))
    b == 0xc4 && return _readn(io, Int(_be(io, 1)))
    b == 0xc5 && return _readn(io, Int(_be(io, 2)))
    b == 0xc6 && return _readn(io, Int(_be(io, 4)))
    b == 0xdc && return [_unpack(io, depth+1) for _ in 1:Int(_be(io, 2))]
    b == 0xdd && return [_unpack(io, depth+1) for _ in 1:Int(_be(io, 4))]
    b == 0xde && return _unpack_map(io, Int(_be(io, 2)), depth)
    b == 0xdf && return _unpack_map(io, Int(_be(io, 4)), depth)
    throw(ArgumentError("unsupported MessagePack tag 0x$(string(b, base=16))"))
end
function _unpack_map(io, n::Int, depth::Int)
    n <= MAX_BYTES || throw(ArgumentError("map too large"))
    d = Dict{String,Any}()
    for _ in 1:n
        k = _unpack(io, depth+1)
        k isa String || throw(ArgumentError("map keys must be strings"))
        haskey(d, k) && throw(ArgumentError("duplicate map key"))
        d[k] = _unpack(io, depth+1)
    end
    d
end

function _cstring(io, s::AbstractString)
    occursin('\0', s) && throw(ArgumentError("BSON key contains NUL"))
    write(io, codeunits(String(s))); write(io, 0x00)
end
function _bson_doc(x, depth::Int)
    depth <= MAX_DEPTH || throw(ArgumentError("BSON nesting too deep"))
    body = IOBuffer()
    pairs_x = x isa AbstractDict || x isa NamedTuple ? pairs(x) : (string(i-1)=>v for (i,v) in enumerate(x))
    for (k,v) in pairs_x
        key = string(k)
        if v === nothing
            write(body, 0x0a); _cstring(body, key)
        elseif v isa Bool
            write(body, 0x08); _cstring(body, key); write(body, v ? 0x01 : 0x00)
        elseif v isa Integer
            if typemin(Int32) <= v <= typemax(Int32)
                write(body, 0x10); _cstring(body, key); _u32le(body, reinterpret(UInt32, Int32(v)))
            else
                write(body, 0x12); _cstring(body, key); _u64le(body, reinterpret(UInt64, Int64(v)))
            end
        elseif v isa Real
            isfinite(v) || throw(ArgumentError("non-finite numbers are not wire values"))
            write(body, 0x01); _cstring(body, key); _u64le(body, reinterpret(UInt64, Float64(v)))
        elseif v isa AbstractString
            b = codeunits(String(v)); length(b)+1 <= MAX_BYTES || throw(ArgumentError("BSON string too large"))
            write(body, 0x02); _cstring(body, key); _u32le(body, length(b)+1); write(body, b); write(body, 0x00)
        elseif v isa AbstractVector{UInt8}
            write(body, 0x05); _cstring(body, key); _u32le(body, length(v)); write(body, 0x00); write(body, v)
        elseif v isa AbstractVector || v isa Tuple
            write(body, 0x04); _cstring(body, key); write(body, _bson_doc(v, depth+1))
        elseif v isa AbstractDict || v isa NamedTuple
            write(body, 0x03); _cstring(body, key); write(body, _bson_doc(v, depth+1))
        else
            throw(ArgumentError("unsupported BSON value $(typeof(v))"))
        end
    end
    bytes = take!(body); n = length(bytes)+5
    n <= MAX_BYTES || throw(ArgumentError("BSON document too large"))
    io = IOBuffer(); _u32le(io, n); write(io, bytes); write(io, 0x00); take!(io)
end
function _readcstring(io)
    bytes = UInt8[]
    while true
        length(bytes) <= MAX_BYTES || throw(ArgumentError("BSON key too long"))
        b = read(io, UInt8)
        b == 0x00 && return String(bytes)
        push!(bytes, b)
    end
end
function _decode_bson_doc(io, depth::Int; asarray::Bool=false)
    depth <= MAX_DEPTH || throw(ArgumentError("BSON nesting too deep"))
    start = position(io); n = Int(_i32bits(_le(io, 4)))
    5 <= n <= MAX_BYTES || throw(ArgumentError("invalid BSON document length"))
    stop = start+n
    d = Dict{String,Any}()
    while position(io) < stop-1
        tag = read(io, UInt8); key = _readcstring(io)
        haskey(d, key) && throw(ArgumentError("duplicate BSON key"))
        value = if tag == 0x01
            reinterpret(Float64, UInt64(_le(io, 8)))
        elseif tag == 0x02
            slen = Int(_i32bits(_le(io, 4)))
            slen >= 1 || throw(ArgumentError("invalid BSON string length"))
            b = _readn(io, slen); b[end] == 0x00 || throw(ArgumentError("unterminated BSON string"))
            String(b[1:end-1])
        elseif tag == 0x03
            _decode_bson_doc(io, depth+1)
        elseif tag == 0x04
            _decode_bson_doc(io, depth+1; asarray=true)
        elseif tag == 0x05
            len = Int(_i32bits(_le(io, 4))); subtype = read(io, UInt8)
            subtype == 0x00 || throw(ArgumentError("unsupported BSON binary subtype"))
            _readn(io, len)
        elseif tag == 0x08
            b = read(io, UInt8); b in (0x00, 0x01) || throw(ArgumentError("invalid BSON bool")); b == 0x01
        elseif tag == 0x0a
            nothing
        elseif tag == 0x10
            Int(_i32bits(_le(io, 4)))
        elseif tag == 0x12
            _i64bits(_le(io, 8))
        else
            throw(ArgumentError("unsupported BSON tag 0x$(string(tag, base=16))"))
        end
        d[key] = value
    end
    read(io, UInt8) == 0x00 && position(io) == stop || throw(ArgumentError("invalid BSON terminator or length"))
    if asarray
        length(d) <= MAX_BYTES || throw(ArgumentError("BSON array too large"))
        keys_expected = Set(string(i) for i in 0:length(d)-1)
        Set(keys(d)) == keys_expected || throw(ArgumentError("BSON array indices are not consecutive"))
        return [d[string(i)] for i in 0:length(d)-1]
    end
    d
end

function encode(x, format::Symbol)::Vector{UInt8}
    if format === :msgpack
        io = IOBuffer(); _pack(io, x, 0); return take!(io)
    elseif format === :bson
        x isa AbstractDict || x isa NamedTuple || throw(ArgumentError("BSON root must be a document"))
        return _bson_doc(x, 0)
    end
    throw(ArgumentError("unknown wire format"))
end
function decode(bytes::AbstractVector{UInt8}, format::Symbol)
    length(bytes) <= MAX_BYTES || throw(ArgumentError("payload too large"))
    io = IOBuffer(bytes)
    x = format === :msgpack ? _unpack(io, 0) : format === :bson ? _decode_bson_doc(io, 0) :
        throw(ArgumentError("unknown wire format"))
    eof(io) || throw(ArgumentError("trailing wire bytes"))
    x
end

end
