# The octree mesh: levels of complete 2x2x2 octs in Morton order, slot kinds, oct neighbour tables, cut links,
# ghost and restriction lists, and the `mesh.oct` file. DESIGN.md section 3. Nothing here touches populations:
# the mesh is immutable once built, and the step kernels read it.

# slot kinds
const K_NONE = 0x00       # slot not in use (padding of a boundary oct, or a level-1 slot outside the domain)
const K_FLUID = 0x01      # stepped leaf
const K_SOLID = 0x02      # inside a body; never stepped
const K_COVERED = 0x03    # parent of a finer oct; carries a restricted state when listed in `restrict_slot`
const K_GHOST = 0x04      # fine slot filled by interpolation from the coarser level; never stepped
const K_BOUNDARY = 0x05   # domain-boundary slot of the level-1 padding layer; its face tag says which condition

# boundary face tags; a boundary slot's `face` is the bitmask of the faces it lies beyond (bit tag - 1)
const FACE_XMIN = 0x01; const FACE_XMAX = 0x02; const FACE_YMIN = 0x03; const FACE_YMAX = 0x04; const FACE_ZMIN = 0x05; const FACE_ZMAX = 0x06
@inline face_bit(tag::UInt8) = UInt8(1) << (tag - 0x01)

"Index into a 27-entry neighbour table of the oct offset (dx, dy, dz), each in -1:1."
@inline nbr_index(dx::Integer, dy::Integer, dz::Integer) = Int(1 + (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1))
const NBR_SELF = nbr_index(0, 0, 0)
"Local offsets of slot k (1..8) inside its oct."
@inline slot_offset(k::Integer) = (Int((k - 1) & 1), Int(((k - 1) >> 1) & 1), Int(((k - 1) >> 2) & 1))
@inline slot_index(ox::Integer, oy::Integer, oz::Integer) = Int(1 + ox + 2oy + 4oz)
"Linear slot id of slot k of oct o."
@inline slot_id(k::Integer, o::Integer) = Int(k) + 8 * (Int(o) - 1)
@inline slot_oct(s::Integer) = (Int(s) - 1) ÷ 8 + 1
@inline slot_k(s::Integer) = (Int(s) - 1) % 8 + 1

"Morton key of an oct origin given in oct units (non-negative, < 2^21 per axis)."
function morton(ix::Integer, iy::Integer, iz::Integer)
    function spread(v::UInt64)
        v &= 0x1fffff
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v << 8)) & 0x100f00f00f00f00f
        v = (v | (v << 4)) & 0x10c30c30c30c30c3
        v = (v | (v << 2)) & 0x1249249249249249
        v
    end
    spread(UInt64(ix)) | (spread(UInt64(iy)) << 1) | (spread(UInt64(iz)) << 2)
end

"""
One refinement level. Octs are in Morton order of their origin (in level cells, even numbers; the padding layer of
level 1 may have negative origins). Per-slot arrays are (8, nocts).
"""
mutable struct OctLevel
    level::Int
    dx::Float64
    n::NTuple{3,Int}                 # cells per axis of the full level grid (domain / dx)
    nocts::Int
    origin::Matrix{Int32}            # (3, nocts), 0-based level cells
    real::BitVector                  # (nocts): a real oct (its parent cell is refined); false for ghost and boundary octs
    kind::Matrix{UInt8}              # (8, nocts)
    face::Matrix{UInt8}              # (8, nocts): bitmask of the domain faces a K_BOUNDARY slot lies beyond (bit tag - 1)
    onbr::Matrix{Int32}              # (27, nocts): neighbour oct, 0 = none
    cutidx::Matrix{Int32}            # (8, nocts): 0 or the cut-cell index c of a K_FLUID slot
    special::Matrix{UInt32}          # (8, nocts): bit (j - 1) set when direction j of the pull needs care: its source slot is a boundary slot, or the link opposite to j is cut
    cut_slot::Vector{Int32}          # (ncutcells): slot id of cut cell c
    cut_off::Vector{Int32}           # (ncutcells + 1): CSR offsets into the link arrays
    cut_dir::Vector{UInt8}           # direction i (2..19) of the cut link
    cut_q::Vector{Float32}           # fraction to the surface along the link
    cut_tri::Vector{Int32}           # triangle id of the crossing
    cut_partner::Vector{Bool}        # the Bouzidi partner x - c_i is a plain fluid slot
    ghost_slot::Vector{Int32}        # (nghost): slot ids of K_GHOST slots
    ghost_donor::Matrix{Int32}       # (64, nghost): donor slot ids at level - 1 on the 4x4x4 stencil (0 = unused); a trilinear ghost uses the inner 8
    ghost_w::Matrix{Float32}         # (64, nghost): donor weights (cubic Lagrange, or trilinear renormalised over the admissible donors)
    restrict_slot::Vector{Int32}     # (nrestrict): K_COVERED slot ids that carry a restricted fine state
    restrict_src::Matrix{Int32}      # (64, nrestrict): source slot ids at level + 1 on the 4x4x4 stencil around the centre (0 = unused)
    restrict_cubic::BitVector        # (nrestrict): the full stencil is admissible (cubic weights); else the inner 8 are averaged
    octmap::Dict{UInt64,Int32}       # Morton key of (origin / 2) -> oct index
end

function OctLevel(level::Int, dx::Float64, n::NTuple{3,Int})
    OctLevel(level, dx, n, 0, zeros(Int32, 3, 0), falses(0), zeros(UInt8, 8, 0), zeros(UInt8, 8, 0), zeros(Int32, 27, 0), zeros(Int32, 8, 0), zeros(UInt32, 8, 0),
             Int32[], Int32[1], UInt8[], Float32[], Int32[], Bool[], Int32[], zeros(Int32, 64, 0), zeros(Float32, 64, 0),
             Int32[], zeros(Int32, 64, 0), falses(0), Dict{UInt64,Int32}())
end

# The 4x4x4 transfer stencil (DESIGN.md A4): node n = 1 + di + 4 dj + 16 dk with di, dj, dk in 0:3 over the cells
# at -1, 0, 1, 2 from the base cell of each axis. The inner 2x2x2 (di, dj, dk in 1:2) are the children of a
# restricted slot and the trilinear donors of a ghost.
const STENCIL_INNER = ntuple(m -> (di = 1 + (m - 1) & 1; dj = 1 + ((m - 1) >> 1) & 1; dk = 1 + ((m - 1) >> 2) & 1; 1 + di + 4dj + 16dk), 8)
"Lagrange weights of the nodes -1, 0, 1, 2 for a point at t in [0, 1] from node 0."
@inline cubic_weights(t) = (-t * (1 - t) * (2 - t) / 6, (1 + t) * (1 - t) * (2 - t) / 2, (1 + t) * t * (2 - t) / 2, -(1 + t) * t * (1 - t) / 6)
"Tensor weights of the cubic restriction: the fine cells at ±0.5 and ±1.5 fine cells from the coarse centre per axis."
const RESTRICT_W = ntuple(n -> (w = cubic_weights(0.5); di = (n - 1) & 3; dj = ((n - 1) >> 2) & 3; dk = (n - 1) >> 4; w[di + 1] * w[dj + 1] * w[dk + 1]), 64)

"""
Oct key of an oct origin (level cells, even, possibly -2 for the padding layer): shifted by one oct so it is
non-negative. Positions beyond the padding layer get a sentinel key that no oct has.
"""
@inline function oct_key(ox::Integer, oy::Integer, oz::Integer)
    sx = (ox >> 1) + 1; sy = (oy >> 1) + 1; sz = (oz >> 1) + 1
    (sx < 0 || sy < 0 || sz < 0) && return typemax(UInt64)
    morton(sx, sy, sz)
end

"Oct index containing level cell (i, j, k) (0-based), or 0."
@inline function find_oct(L::OctLevel, i::Integer, j::Integer, k::Integer)
    get(L.octmap, oct_key(i & ~1, j & ~1, k & ~1), Int32(0))
end
"Slot id of level cell (i, j, k), or 0."
@inline function find_slot(L::OctLevel, i::Integer, j::Integer, k::Integer)
    o = find_oct(L, i, j, k)
    o == 0 ? 0 : slot_id(slot_index(Int(i & 1), Int(j & 1), Int(k & 1)), Int(o))
end
"Level cell coordinates (0-based) of slot s."
@inline function slot_cell(L::OctLevel, s::Integer)
    o = slot_oct(s); k = slot_k(s); off = slot_offset(k)
    (Int(L.origin[1, o]) + off[1], Int(L.origin[2, o]) + off[2], Int(L.origin[3, o]) + off[3])
end
"Centre of slot s in metres."
@inline function slot_centre(L::OctLevel, s::Integer)
    c = slot_cell(L, s)
    ((c[1] + 0.5) * L.dx, (c[2] + 0.5) * L.dx, (c[3] + 0.5) * L.dx)
end

"The whole mesh."
struct OctMesh
    domain::V3                       # size in metres; the origin is (0, 0, 0)
    offset::V3                       # translation applied to the STL
    periodic::NTuple{3,Bool}
    levels::Vector{OctLevel}
    triangles::Vector{Tri}           # placed triangles (for the surface fields and the renderer)
    config_sha::String
    stl_sha::String
    counts::Dict{String,Any}         # the generator's summary (cells per kind and level, timings, reports)
end

nlevels(m::OctMesh) = length(m.levels)
finest(m::OctMesh) = m.levels[end]

# ---------------------------------------------------------------- mesh.oct file
const MESH_MAGIC = "OCTQ19M\0"
const MESH_FORMAT = 5                 # 5: the inner ghosts only are listed, depth-2 restricted slots average their children (A8)

# pull-source tables: slot k pulls direction i from the oct at offset SRC_OFF[k, i] (an index into the 27-entry
# neighbour table) and the slot SRC_CHILD[k, i] of that oct
function _source_tables()
    off = Matrix{Int8}(undef, 8, Q); child = Matrix{Int8}(undef, 8, Q)
    for k in 1:8, i in 1:Q
        x, y, z = slot_offset(k)
        p = (x - CX[i], y - CY[i], z - CZ[i])
        o = map(v -> v < 0 ? -1 : v > 1 ? 1 : 0, p)
        off[k, i] = nbr_index(o[1], o[2], o[3])
        child[k, i] = slot_index(p[1] - 2o[1], p[2] - 2o[2], p[3] - 2o[3])
    end
    off, child
end
const SRC_OFF, SRC_CHILD = _source_tables()
"The (oct offset index, slot) of the same-level neighbour of slot k along (dx, dy, dz) in -1:1 per axis."
@inline function neighbour_slot(k::Integer, dx::Integer, dy::Integer, dz::Integer)
    x, y, z = slot_offset(k)
    nx = x + dx; ny = y + dy; nz = z + dz
    ox = nx < 0 ? -1 : nx > 1 ? 1 : 0; oy = ny < 0 ? -1 : ny > 1 ? 1 : 0; oz = nz < 0 ? -1 : nz > 1 ? 1 : 0
    nbr_index(ox, oy, oz), slot_index(nx - 2ox, ny - 2oy, nz - 2oz)
end

function _write_array(io, a::AbstractArray)
    write(io, Int64(ndims(a)))
    for s in size(a); write(io, Int64(s)); end
    write(io, a)
end
function _read_array(io, ::Type{T}) where T
    nd = Int(read(io, Int64)); dims = ntuple(_ -> Int(read(io, Int64)), nd)
    a = Array{T}(undef, dims...); read!(io, a); a
end
function _write_string(io, s::String); b = codeunits(s); write(io, Int64(length(b))); write(io, b); end
function _read_string(io); n = Int(read(io, Int64)); String(read(io, n)); end

"Write the mesh; the file ends with the SHA-256 of everything before it."
function write_mesh(path::AbstractString, m::OctMesh)
    buf = IOBuffer()
    write(buf, codeunits(MESH_MAGIC)); write(buf, Int64(MESH_FORMAT))
    for v in m.domain; write(buf, Float64(v)); end
    for v in m.offset; write(buf, Float64(v)); end
    for p in m.periodic; write(buf, UInt8(p)); end
    _write_string(buf, m.config_sha); _write_string(buf, m.stl_sha)
    _write_string(buf, sprint(io -> TOML.print(io, m.counts)))
    write(buf, Int64(length(m.triangles)))
    tri = Array{Float64}(undef, 3, 3, length(m.triangles))
    for (t, T) in enumerate(m.triangles), v in 1:3, a in 1:3; tri[a, v, t] = T[v][a]; end
    _write_array(buf, tri)
    write(buf, Int64(length(m.levels)))
    for L in m.levels
        write(buf, Int64(L.level)); write(buf, Float64(L.dx)); for v in L.n; write(buf, Int64(v)); end
        write(buf, Int64(L.nocts))
        _write_array(buf, L.origin); _write_array(buf, UInt8.(L.real)); _write_array(buf, L.kind); _write_array(buf, L.face); _write_array(buf, L.onbr); _write_array(buf, L.cutidx); _write_array(buf, L.special)
        _write_array(buf, L.cut_slot); _write_array(buf, L.cut_off); _write_array(buf, L.cut_dir); _write_array(buf, L.cut_q); _write_array(buf, L.cut_tri)
        _write_array(buf, UInt8.(L.cut_partner))
        _write_array(buf, L.ghost_slot); _write_array(buf, L.ghost_donor); _write_array(buf, L.ghost_w)
        _write_array(buf, L.restrict_slot); _write_array(buf, L.restrict_src); _write_array(buf, UInt8.(L.restrict_cubic))
    end
    body = take!(buf)
    digest = sha256(body)
    tmp = path * ".tmp"
    open(tmp, "w") do io; write(io, body); write(io, digest); end
    mv(tmp, path; force=true)
    bytes2hex(digest)
end

"Read a mesh written by `write_mesh`; verifies the trailer hash and the format."
function read_mesh(path::AbstractString)
    bytes = read(path)
    length(bytes) > 32 || error("$path is not a mesh file")
    # Hash the payload as a view and parse from the original byte vector. A large
    # mesh can be several GB; slicing `bytes[1:end-32]` made a full second copy
    # before the state was even allocated.
    digest = @view bytes[end-31:end]
    sha256(@view bytes[1:end-32]) == digest || error("$path: hash mismatch (the file is damaged or truncated)")
    io = IOBuffer(bytes)
    String(read(io, length(MESH_MAGIC))) == MESH_MAGIC || error("$path: not an OctreeQ19 mesh")
    fmt = Int(read(io, Int64)); fmt == MESH_FORMAT || error("$path: mesh format $fmt, expected $MESH_FORMAT")
    domain = ntuple(_ -> read(io, Float64), 3); offset = ntuple(_ -> read(io, Float64), 3)
    periodic = ntuple(_ -> read(io, UInt8) != 0, 3)
    config_sha = _read_string(io); stl_sha = _read_string(io); counts = TOML.parse(_read_string(io))
    nt = Int(read(io, Int64)); tri = _read_array(io, Float64)
    triangles = [((tri[1, 1, t], tri[2, 1, t], tri[3, 1, t]), (tri[1, 2, t], tri[2, 2, t], tri[3, 2, t]), (tri[1, 3, t], tri[2, 3, t], tri[3, 3, t])) for t in 1:nt]
    nl = Int(read(io, Int64)); levels = OctLevel[]
    for _ in 1:nl
        level = Int(read(io, Int64)); dx = read(io, Float64); n = ntuple(_ -> Int(read(io, Int64)), 3); nocts = Int(read(io, Int64))
        L = OctLevel(level, dx, n)
        L.nocts = nocts
        L.origin = _read_array(io, Int32); L.real = BitVector(vec(_read_array(io, UInt8)) .!= 0); L.kind = _read_array(io, UInt8); L.face = _read_array(io, UInt8); L.onbr = _read_array(io, Int32); L.cutidx = _read_array(io, Int32); L.special = _read_array(io, UInt32)
        L.cut_slot = vec(_read_array(io, Int32)); L.cut_off = vec(_read_array(io, Int32)); L.cut_dir = vec(_read_array(io, UInt8)); L.cut_q = vec(_read_array(io, Float32)); L.cut_tri = vec(_read_array(io, Int32))
        L.cut_partner = vec(_read_array(io, UInt8)) .!= 0
        L.ghost_slot = vec(_read_array(io, Int32)); L.ghost_donor = _read_array(io, Int32); L.ghost_w = _read_array(io, Float32)
        L.restrict_slot = vec(_read_array(io, Int32)); L.restrict_src = _read_array(io, Int32); L.restrict_cubic = BitVector(vec(_read_array(io, UInt8)) .!= 0)
        rebuild_octmap!(L)
        push!(levels, L)
    end
    OctMesh(domain, offset, periodic, levels, triangles, config_sha, stl_sha, counts)
end

function rebuild_octmap!(L::OctLevel)
    empty!(L.octmap); sizehint!(L.octmap, L.nocts)
    for o in 1:L.nocts
        L.octmap[oct_key(L.origin[1, o], L.origin[2, o], L.origin[3, o])] = Int32(o)
    end
    L
end

"""
The leaf containing the point p (metres): (level, slot id), or (0, 0) outside the domain. Searches from the finest
level down: the first real oct slot that is a leaf (fluid or solid) wins; covered slots and ghost octs defer to
their own level's children or parent.
"""
function locate_leaf(m::OctMesh, p::V3)
    for l in nlevels(m):-1:1
        L = m.levels[l]
        i = floor(Int, p[1] / L.dx); j = floor(Int, p[2] / L.dx); k = floor(Int, p[3] / L.dx)
        (i < 0 || j < 0 || k < 0 || i >= L.n[1] || j >= L.n[2] || k >= L.n[3]) && return (0, 0)
        o = find_oct(L, i, j, k)
        o == 0 && continue
        L.real[o] || continue
        s = slot_id(slot_index(i & 1, j & 1, k & 1), Int(o))
        kd = L.kind[slot_k(s), o]
        (kd == K_FLUID || kd == K_SOLID) && return (l, s)
    end
    (0, 0)
end

"""
Counts per kind of a level. Fluid, solid and covered are counted over real octs only (leaves and parents that
partition the volume); ghost and boundary over the other octs, and `solid_ghost` is the number of ghost-oct slots
that lie inside a body.
"""
function kind_counts(L::OctLevel)
    c = zeros(Int, 6); sg = 0
    for o in 1:L.nocts, k in 1:8
        kd = L.kind[k, o]
        if L.real[o]
            c[Int(kd) + 1] += 1
        elseif kd == K_SOLID
            sg += 1
        else
            c[Int(kd) + 1] += 1
        end
    end
    Dict("none" => c[1], "fluid" => c[2], "solid" => c[3], "covered" => c[4], "ghost" => c[5], "boundary" => c[6], "solid_ghost" => sg)
end
