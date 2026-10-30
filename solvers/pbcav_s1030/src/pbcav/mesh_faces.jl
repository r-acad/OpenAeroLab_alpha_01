# =====================================================================================================================
# mesh_faces.jl -- version 2's mesh: Float64 metrics, true area centroids, owner-neighbour face lists, the special-face
# lists, line maps, wall distance, box and periodic meshes, and bilinear subdivision for nested levels (plan 4.1, 6.1,
# 10.1). Stage P0b, 14 Oct 2026.
#
# THE MESHES ARE VERSION 1'S. The C-mesh (and the strip block behind a blunt base) comes from version 1's generator,
# src/cmesh/cmesh_gen.jl, included read-only, through the same generate_cmesh call the version-1 driver makes
# (tools/cmesh_cav_movie.jl:714-720): `generate_v1(call)`. Its node arrays hash to version 1's MESH COORDINATE RECEIPT
# bit for bit (N60R: x 3af02802, y a01a759c, strip x aff44ff4, strip y c12b469a). Nothing here writes to the generator's
# state or to its arrays.
#
# WHAT IS NEW AGAINST VERSION 1 (plan 4.1):
#   * Float64 throughout (v1 stores its metrics in Float32, which caps continuity near 1e-7);
#   * TRUE AREA CENTROIDS (the polygon centroid), not v1's vertex average; the vertex average is kept beside them (xv, yv)
#     for the cross-checks against v1 and the P0a operators;
#   * one owner and one neighbour per internal face, each face evaluated once: the wake cut (cell i against NI+1-i) and the
#     strip interfaces (strip column st against C columns NIW+1-st and NIW+NIF+st) are ordinary internal faces, so v1's
#     ghost copies across them are not needed;
#   * a fixed four-slot cell->face table (cface) so every per-cell sum is a deterministic gather (no atomics, plan 5.5).
#
# CONVENTIONS.
#   Cells (the P0a harness's numbering, so its operators and line lists compare index for index):
#     C block cell (i, j) -> (i-1)*NJ + j, i = 1..NI (lower wake leg, foil, upper wake leg), j = 1..NJ (wall outwards);
#     strip cell (st, js) -> NI*NJ + (st-1)*NS + js, st = 1..NIs (base -> outlet), js = 1..NS (lower leg -> upper leg);
#     box cell (i, j)     -> (i-1)*ny + j.
#   Nodes: C block (i, j) -> (i-1)*(NJ+1) + j; strip (st, js) -> (NI+1)*(NJ+1) + (st-1)*(NS+1) + js; box (i-1)*(ny+1)+j.
#   A face is the segment from node a to node b; its area vector is the right-hand normal S = (yb - ya, -(xb - xa)),
#   |S| = the face length. Internal faces point from the owner to the neighbour; boundary faces point out of the cell.
#   With these node orders S is, bit for bit, version 1's face vector (cmesh_metrics.jl: S_i = (Y[i,j+1]-Y[i,j],
#   -(X[i,j+1]-X[i,j])), S_j = (-(Y[i+1,j]-Y[i,j]), X[i+1,j]-X[i,j])), so Float32(S) equals v1's stored metric exactly.
#   Internal face ORDER is the P0a harness's (docs/_records/2026-10-13_pressure_based/p0a/p0a_common_20261014.jl,
#   facemesh): C i faces (j outer, i inner), C j faces, then the cut (sharp) or the interfaces (station order, lower then
#   upper), then strip i faces and strip j faces; periodic faces of a box come last.
#   Boundary face order: foil wall, base wall, outer j faces (i order; inlet / lower slip / upper slip / far field by the
#   generator's kind), the C outlets (for each j the i = 1 end then the i = NI+1 end), the strip outlet; box sides W, E, S, N.
#   The fixed-pressure list `dirichlet` is in the P0a order: far field, then the C outlets, then the strip outlet.
#   cface[k, c], k = 1..4 the cell's i-, i+, j-, j+ side in its own block: +f (owner of internal face f), -f (neighbour of
#   internal face f), or nf + b (boundary face b).
#   A periodic internal face carries the translation t that takes the neighbour's cell to the owner's side:
#   d = x_N + t - x_P. Every other face has t = 0.
#
# FLOATING POINT RUNS IN KERNELS. The face topology (who neighbours whom) is integer index arithmetic done once per mesh on
# the host. Every floating-point quantity (areas, centroids, face vectors, weights, angles, wall distance) is computed by a
# KernelAbstractions kernel on the backend the caller names; the arrays live on that backend.
# =====================================================================================================================

include(joinpath(@__DIR__, "..", "cmesh", "cmesh_gen.jl"))
using .CMeshGen

"The repository root (two directories above this file)."
const PB_REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ---- face and boundary tags ------------------------------------------------------------------------------------------
# Internal-face tags 1-6 are the P0a harness's (FT_CI ... FT_SJ), so the census compares tag for tag.
const FT_CI = 0x01    # C block i face
const FT_CJ = 0x02    # C block j face
const FT_CUT = 0x03   # the wake cut of a sharp C-mesh: upper-leg cell (NI+1-i, 1) against lower-leg cell (i, 1)
const FT_IF = 0x04    # strip interface: lower (C (NIW+1-st, 1) -> strip (st, 1)) or upper (strip (st, NS) -> C (NIW+NIF+st, 1))
const FT_SI = 0x05    # strip i face
const FT_SJ = 0x06    # strip j face
const FT_BI = 0x07    # box i face
const FT_BJ = 0x08    # box j face
const FT_PERI = 0x09  # periodic face across the box's i ends (owner (nx, j), neighbour (1, j), t = (Lx, 0))
const FT_PERJ = 0x0a  # periodic face across the box's j ends (owner (i, ny), neighbour (i, 1), t = (0, Ly))

const BT_FOIL = 0x01     # foil wall (C block j = 1, foil columns)
const BT_BASE = 0x02     # blunt base wall (the strip's i = 1 faces)
const BT_INLET = 0x03    # channel inlet (C block j = NJ+1, OUTER_INLET)
const BT_LOWER = 0x04    # lower tunnel wall (OUTER_LOWER_WALL)
const BT_UPPER = 0x05    # upper tunnel wall (OUTER_UPPER_WALL)
const BT_FAR = 0x06      # far field (OUTER_FARFIELD: a C-mesh without a channel)
const BT_OUTLET = 0x07   # C block outlet (its i = 1 and i = NI+1 faces)
const BT_SOUTLET = 0x08  # strip outlet (its i = NIs+1 faces)
const BT_W = 0x09; const BT_E = 0x0a; const BT_S = 0x0b; const BT_N = 0x0c    # box sides
# INTERIOR WALLS ("BAFFLES", 24 Oct 2026; opt-in, `build_faces(...; baffle)`): an internal face cut into two boundary faces, one
# per side, both no-slip walls (a zero-thickness plate on a grid line, such as a Gurney flap; baffle.jl). FRONT is the side the
# cut names front (a Gurney plate: the side facing upstream, its foil column), BACK the other. Without a baffle no face has them.
const BT_BAFFLE_FRONT = 0x0d
const BT_BAFFLE_BACK = 0x0e

"Threshold of the high-angle face list (plan 4.1: the limiter choice is between a global limiter and one active above 70 degrees)."
const HIGH_ANGLE_DEG = 70.0

h64(A) = bytes2hex(sha256(reinterpret(UInt8, vec(A))))

# ---- the version-1 generator call ------------------------------------------------------------------------------------
"""
    CMeshCall(; foil, n_foil, n_normal, dn1, growth, smooth, n_wake, le_ds, te, H, xin, xout, alpha)

One call of version 1's generator, field for field the version-1 driver's (TOPO_MESH = n_foil,n_normal,dn1,growth,smooth,
n_wake; TOPO_MESH_LE_DS; the trailing-edge keywords; TOPO_CHANNEL_H/XMIN/XMAX; TOPO_ALPHA). `te` is `nothing` or the
NamedTuple of generator keywords (n_base, te_ds, wake_near_ds, wake_near_len, wake_near_growth, te_topology). `foil` is a
path, absolute or relative to the repository root.
"""
Base.@kwdef struct CMeshCall
    foil::String
    n_foil::Int; n_normal::Int; dn1::Float64; growth::Float64 = 1.20; smooth::Int = 80; n_wake::Int
    le_ds::Float64 = 0.0
    te::Union{Nothing,NamedTuple} = nothing
    H::Float64; xin::Float64; xout::Float64; alpha::Float64
end

"The foil path of a call, resolved against the repository root."
foil_path(c::CMeshCall) = isabspath(c.foil) ? c.foil : joinpath(PB_REPO, c.foil)

"N60R's mesh (docs/2026-10-09_N60R_REGISTRATION.md; docs/_records/2026-10-09_n60r/launch_n60r.sh): the strip mesh."
const N60R_CALL = CMeshCall(; foil="tools/naca66mod_t09_f020_a08_blunt.dat", n_foil=256, n_normal=108, dn1=5e-5,
    growth=1.20, smooth=80, n_wake=136, le_ds=3e-4,
    te=(n_base=8, te_ds=5.62e-4, wake_near_ds=7.4925e-4, wake_near_len=0.05994, wake_near_growth=1.05, te_topology=:strip),
    H=5.0, xin=-5.0, xout=6.83, alpha=4.0)
"LE256 (the committed multigrid study's sharp mesh; the P0a harness's :LE256)."
const LE256_CALL = CMeshCall(; foil="tools/naca66mod_t09_f020_a08_sharp.dat", n_foil=256, n_normal=100, dn1=1e-4,
    growth=1.20, smooth=80, n_wake=64, le_ds=3e-4, te=nothing, H=5.0, xin=-5.0, xout=6.83, alpha=3.0)
"N60R's registered coordinate hashes (x, y, strip x, strip y), from its MESH COORDINATE RECEIPT."
const N60R_RECEIPT = ("3af028027f166a2ede9121720872008455791f3792aa8557a16a8989b1c88a91",
                      "a01a759cdc528d8644d50dd3ceea742e25d5e8898ebc87e30ad6288b2f93bfa5",
                      "aff44ff45cdac43892a67267e1665e223e228cf919b7e24f1745e2a63113af1f",
                      "c12b469a355613c12da7f5a9a9b4dfa652eee5b7202f1be653d3b1ddf0b8a720")

"""
    generate_v1(call::CMeshCall) -> CMeshGen.ChannelMesh

Version 1's mesh: `generate_cmesh` with the version-1 driver's fixed arguments (R_ff 25, L_wake 25, bl_freeze 6, the
channel's xi_ref 0.5 and ref (0.5, 0)). Version 2 builds only channel meshes (plan section 9).
"""
function generate_v1(c::CMeshCall)
    c.H > 0 || throw(ArgumentError("version 2 builds channel meshes only (plan section 9): H must be > 0 (got $(c.H))"))
    af = read_airfoil_dat(foil_path(c))
    ch = ChannelSpec(; height=c.H, xi_in=c.xin, xi_out=c.xout, xi_ref=0.5, ref=(0.5, 0.0), angle_deg=c.alpha)
    kw = c.te === nothing ? (;) : c.te
    generate_cmesh(af; n_foil=c.n_foil, n_wake=c.n_wake, n_normal=c.n_normal, dn1=c.dn1, growth_max=c.growth,
                   R_ff=25.0, L_wake=25.0, smooth_iters=c.smooth, bl_freeze=6, le_ds=c.le_ds, channel=ch, kw...)
end

# ---- block descriptions (what subdivision acts on) -------------------------------------------------------------------
"""
    CBlocks

A C-mesh as node arrays, with the strip block when there is one: what `subdivide` acts on and `build_faces` reads.
`hashes` are the GENERATED mesh's coordinate hashes (x, y, strip x, strip y; "" without a strip) and `level` the number of
bilinear subdivisions applied since.
"""
struct CBlocks
    label::String
    NI::Int; NJ::Int; NIW::Int; NIF::Int
    X::Matrix{Float64}; Y::Matrix{Float64}
    kind::Vector{UInt8}
    NIs::Int; NS::Int
    Xs::Matrix{Float64}; Ys::Matrix{Float64}
    level::Int
    hashes::NTuple{4,String}
end

"""
    cblocks(m::ChannelMesh; label="") -> CBlocks

The node arrays of a generated mesh (copied: the generator's arrays are never aliased) and its coordinate hashes.
"""
function cblocks(m::ChannelMesh; label::AbstractString="")
    s = m.strip
    Xs = s === nothing ? zeros(0, 0) : copy(s.X); Ys = s === nothing ? zeros(0, 0) : copy(s.Y)
    hs = (h64(m.X), h64(m.Y), s === nothing ? "" : h64(s.X), s === nothing ? "" : h64(s.Y))
    CBlocks(String(label), m.NI, m.NJ, m.NIW, m.NIF, copy(m.X), copy(m.Y), copy(m.outer_face_kind),
            s === nothing ? 0 : s.NI, s === nothing ? 0 : s.NJ, Xs, Ys, 0, hs)
end

"""
    subdivide_nodes(X) -> F

Bilinear subdivision of a node array: each quad into four at its edge midpoints and its vertex average (the P0a harness's
construction, so N60Rx2 is the same mesh). The coarse nodes are kept, the children of a quad tile it exactly, and a
conforming interface (identical nodes on both sides) stays conforming.
"""
function subdivide_nodes(X::AbstractMatrix{Float64})
    n1, n2 = size(X) .- 1; F = zeros(2n1 + 1, 2n2 + 1)
    for i in 1:n1+1, j in 1:n2+1; F[2i-1, 2j-1] = X[i, j]; end
    for i in 1:n1, j in 1:n2+1; F[2i, 2j-1] = 0.5*(X[i, j] + X[i+1, j]); end
    for i in 1:n1+1, j in 1:n2; F[2i-1, 2j] = 0.5*(X[i, j] + X[i, j+1]); end
    for i in 1:n1, j in 1:n2; F[2i, 2j] = 0.25*(X[i, j] + X[i+1, j] + X[i, j+1] + X[i+1, j+1]); end
    F
end

"""
    subdivide(b::CBlocks, k=1) -> CBlocks

`k` bilinear subdivisions of both blocks (the plan's nested MMS levels, 10.1): NI, NJ, NIW, NIF, NIs and NS double each
time, each outer kind is repeated twice, the strip interface stays conforming.
"""
function subdivide(b::CBlocks, k::Integer=1)
    k >= 0 || throw(ArgumentError("subdivision count must be >= 0 (got $k)"))
    k == 0 && return b
    s = b.NS > 0
    c = CBlocks(b.label, 2b.NI, 2b.NJ, 2b.NIW, 2b.NIF, subdivide_nodes(b.X), subdivide_nodes(b.Y), repeat(b.kind, inner=2),
                2b.NIs, 2b.NS, s ? subdivide_nodes(b.Xs) : zeros(0, 0), s ? subdivide_nodes(b.Ys) : zeros(0, 0),
                b.level + 1, b.hashes)
    subdivide(c, k - 1)
end

"The current node arrays' hashes (x, y, strip x, strip y; equal to `hashes` at level 0)."
node_hashes(b::CBlocks) = (h64(b.X), h64(b.Y), b.NS > 0 ? h64(b.Xs) : "", b.NS > 0 ? h64(b.Ys) : "")

"""
    QuadBox

A box mesh as one node array: uniform, or smoothly distorted, and periodic in either direction. When periodic in i its
last node column is the first translated by exactly (Lx, 0) (and likewise in j), so the periodic faces match bit for bit.
"""
struct QuadBox
    label::String
    X::Matrix{Float64}; Y::Matrix{Float64}
    Lx::Float64; Ly::Float64
    periodic::NTuple{2,Bool}
    walls::NTuple{4,Bool}        # (W, E, S, N) sides counted as walls for the wall distance
    level::Int
end

"""
    box_blocks(nx, ny; Lx=1, Ly=1, x0=0, y0=0, periodic=(false, false), distort=0.0, walls=nothing, label="box")

A box of nx x ny cells. `distort` a (|a| < 1/(2 pi) keeps every cell convex) moves the nodes by
x = xi + a Lx sin(2 pi xi/Lx) sin(2 pi eta/Ly), y = eta + a Ly sin(2 pi xi/Lx) sin(2 pi eta/Ly) (relative to x0, y0),
which vanishes on the four sides, so the sides stay straight and a periodic box stays periodic. `walls` defaults to every
non-periodic side.
"""
function box_blocks(nx::Integer, ny::Integer; Lx::Real=1.0, Ly::Real=1.0, x0::Real=0.0, y0::Real=0.0,
                    periodic::NTuple{2,Bool}=(false, false), distort::Real=0.0, walls=nothing, label::AbstractString="box")
    nx >= 1 && ny >= 1 || throw(ArgumentError("a box needs nx, ny >= 1 (got $nx, $ny)"))
    Lx > 0 && Ly > 0 || throw(ArgumentError("a box needs Lx, Ly > 0"))
    periodic[1] && nx < 2 && throw(ArgumentError("a box periodic in i needs nx >= 2"))
    periodic[2] && ny < 2 && throw(ArgumentError("a box periodic in j needs ny >= 2"))
    abs(distort) < 1/(2pi) || throw(ArgumentError("|distort| must be < 1/(2 pi) to keep the cells convex (got $distort)"))
    X = zeros(nx + 1, ny + 1); Y = zeros(nx + 1, ny + 1)
    for i in 1:nx+1, j in 1:ny+1
        xi = Lx*(i - 1)/nx; eta = Ly*(j - 1)/ny
        g = distort*sin(2pi*xi/Lx)*sin(2pi*eta/Ly)
        X[i, j] = x0 + xi + Lx*g; Y[i, j] = y0 + eta + Ly*g
    end
    if periodic[1]; for j in 1:ny+1; X[nx+1, j] = X[1, j] + Lx; Y[nx+1, j] = Y[1, j]; end; end
    if periodic[2]; for i in 1:nx+1; X[i, ny+1] = X[i, 1]; Y[i, ny+1] = Y[i, 1] + Ly; end; end
    w = walls === nothing ? (!periodic[1], !periodic[1], !periodic[2], !periodic[2]) : walls
    QuadBox(String(label), X, Y, Float64(Lx), Float64(Ly), periodic, w, 0)
end

"""
    quad_blocks(X, Y; label="quad", walls=(false, false, false, false)) -> QuadBox

A box-topology mesh from given node arrays (both (nx+1) x (ny+1), i along the first index): a stretched structured grid read
from a file, such as NASA TMR's flat-plate grids (W1, 21 Oct 2026; `read_plot3d_2d`). Not periodic; its sides are W (i = 1),
E, S (j = 1), N as for `box_blocks`. `walls` are the sides counted as walls for mesh.wdist (default none: an SST case takes
its wall distance from its condition set's no-slip faces, assemble_komega.jl). build_faces refuses a non-convex cell.
"""
function quad_blocks(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; label::AbstractString="quad",
                     walls::NTuple{4,Bool}=(false, false, false, false))
    size(X) == size(Y) && all(size(X) .>= 2) || throw(ArgumentError("quad_blocks: X and Y must be equal-sized node arrays, at least 2 x 2"))
    Xf = Float64.(X); Yf = Float64.(Y)
    all(isfinite, Xf) && all(isfinite, Yf) || throw(ArgumentError("quad_blocks: non-finite node coordinates"))
    QuadBox(String(label), Xf, Yf, maximum(Xf) - minimum(Xf), maximum(Yf) - minimum(Yf), (false, false), walls, 0)
end

"""
    read_plot3d_2d(path) -> (X, Y)

A formatted 2-D PLOT3D grid file with one block (NASA TMR's flat-plate grids, flatplate_clust2*.p2dfmt): `nbl` (1), then
`idim jdim`, then x(i, j) with i fastest, then y(i, j); read in Float64 (as TMR asks). Host setup, once per mesh.
"""
function read_plot3d_2d(path::AbstractString)
    tok = split(read(path, String))
    nbl = parse(Int, tok[1]); nbl == 1 || throw(ArgumentError("read_plot3d_2d: one block expected (got $nbl) in $path"))
    ni = parse(Int, tok[2]); nj = parse(Int, tok[3]); n = ni*nj
    length(tok) == 3 + 2n || throw(ArgumentError("read_plot3d_2d: $path holds $(length(tok) - 3) values, expected $(2n)"))
    v = parse.(Float64, @view tok[4:end])
    (reshape(v[1:n], ni, nj), reshape(v[n+1:2n], ni, nj))
end

"`k` bilinear subdivisions of a box (nested levels; a periodic box stays periodic, its translations exact)."
function subdivide(b::QuadBox, k::Integer=1)
    k >= 0 || throw(ArgumentError("subdivision count must be >= 0 (got $k)"))
    k == 0 && return b
    X = subdivide_nodes(b.X); Y = subdivide_nodes(b.Y); nx, ny = size(X) .- 1
    if b.periodic[1]; for j in 1:ny+1; X[nx+1, j] = X[1, j] + b.Lx; Y[nx+1, j] = Y[1, j]; end; end
    if b.periodic[2]; for i in 1:nx+1; X[i, ny+1] = X[i, 1]; Y[i, ny+1] = Y[i, 1] + b.Ly; end; end
    subdivide(QuadBox(b.label, X, Y, b.Lx, b.Ly, b.periodic, b.walls, b.level + 1), k - 1)
end

# ---- the mesh -----------------------------------------------------------------------------------------------------------
"""
    PBMesh

Version 2's mesh on one backend. Counts: `nc` cells, `nf` internal faces, `nb` boundary faces. Array fields (AF Float64,
AI Int32, AU UInt8, MI Int32 matrices) live on the backend; the rest is metadata.

Cells: `vol` area, `xc, yc` true area centroid, `xv, yv` vertex average (v1's centre), `wdist` distance to the nearest
wall face, `cnode` (4 x nc) the nodes counter-clockwise, `cface` (4 x nc) the i-, i+, j-, j+ faces (codes in the header),
`cblk, ci, cj` the block (0 C block or box, 1 strip) and the indices in it.

Internal faces (owner `own` -> neighbour `nei`): `fna, fnb` nodes, `ftag`, `fsta` (station: cut i, interface +st lower /
-st upper, else 0), `ftx, fty` periodic translation, `Sx, Sy` area vector, `Smag` = |S|, `xf, yf` face centroid,
`dx, dy` = x_N + t - x_P, `fw` the owner's linear-interpolation weight (S.(x_N + t - x_f))/(S.d), `fa` = |S|^2/(S.d)
(the two-point coefficient), `kx, ky` = S - fa d (the non-orthogonal remainder, over-relaxed split), `theta` the angle
between S and d in degrees, `skx, sky` = x_f - x_f' the skewness vector from x_f', the point where the line from x_P to
x_N + t crosses the face (x_f' = x_P + (1 - fw) d, where the weight fw interpolates exactly), to the face centroid x_f.

Boundary faces (owner `bown`, outward): `bna, bnb`, `btag`, `bsta` (foil station / base row / outer column / outlet row),
`bSx, bSy, bSmag`, `bxf, byf`, `bdx, bdy` = x_f - x_P, `ba` = |S|^2/(S.d_b), `bkx, bky`.

Lines (CSR): `jl_ptr, jl_cell` (the P0a j-line order: the wake lines through the cut or the strip, then the foil columns;
box columns), `jl_fptr, jl_face` the internal faces between consecutive cells (and the closing face of a cyclic line),
`jl_cyclic`; the same for i-lines (`il_*`: C rows, then strip rows; box rows).

Special lists: `cut`, `if_lower[st]`, `if_upper[st]` (internal faces by station), `outlet_rows` and `high_angle` (internal
faces), `foil`, `base`, `inlet`, `slip_lower`, `slip_upper`, `farfield`, `outlets`, `dirichlet` (boundary faces),
`corner_cells` (cells touching a trailing-edge corner node, or an end node of a baffle), `baffle` (the boundary faces of the
baffles, BT_BAFFLE_FRONT and BT_BAFFLE_BACK, in the order the cuts made them; empty without a baffle; `bsta` of a baffle face
is its row along the plate, 1 at the root, negative on plate 2).
"""
struct PBMesh{AF<:AbstractVector{Float64},AI<:AbstractVector{Int32},AU<:AbstractVector{UInt8},MI<:AbstractMatrix{Int32}}
    label::String
    family::Symbol
    level::Int
    NI::Int; NJ::Int; NIW::Int; NIF::Int; NIs::Int; NS::Int
    nblk::Int
    blk_off::NTuple{2,Int}; blk_ni::NTuple{2,Int}; blk_nj::NTuple{2,Int}
    periodic::NTuple{2,Bool}
    Lper::NTuple{2,Float64}
    hashes::NTuple{4,String}
    n_outlet_rows::Int
    nc::Int; nf::Int; nb::Int; nn::Int
    xn::AF; yn::AF
    cnode::MI; cface::MI; cblk::AU; ci::AI; cj::AI
    vol::AF; xc::AF; yc::AF; xv::AF; yv::AF; wdist::AF
    own::AI; nei::AI; fna::AI; fnb::AI; ftag::AU; fsta::AI; ftx::AF; fty::AF
    Sx::AF; Sy::AF; Smag::AF; xf::AF; yf::AF; dx::AF; dy::AF; fw::AF; fa::AF; kx::AF; ky::AF; theta::AF; skx::AF; sky::AF
    bown::AI; bna::AI; bnb::AI; btag::AU; bsta::AI
    bSx::AF; bSy::AF; bSmag::AF; bxf::AF; byf::AF; bdx::AF; bdy::AF; ba::AF; bkx::AF; bky::AF
    jl_ptr::AI; jl_cell::AI; jl_fptr::AI; jl_face::AI; jl_cyclic::AU
    il_ptr::AI; il_cell::AI; il_fptr::AI; il_face::AI; il_cyclic::AU
    cut::AI; if_lower::AI; if_upper::AI; outlet_rows::AI; high_angle::AI
    foil::AI; base::AI; inlet::AI; slip_lower::AI; slip_upper::AI; farfield::AI; outlets::AI; dirichlet::AI
    corner_cells::AI
    baffle::AI
end

function Adapt.adapt_structure(to, m::PBMesh)
    PBMesh(map(f -> (v = getfield(m, f); v isa AbstractArray ? adapt(to, v) : v), fieldnames(PBMesh))...)
end

"Cell index of (block, i, j) on a mesh (block 0 the C block or box, 1 the strip)."
cell_index(m::PBMesh, blk::Integer, i::Integer, j::Integer) = m.blk_off[blk+1] + (i - 1)*m.blk_nj[blk+1] + j

# ---- the host-side topology (integers only) -------------------------------------------------------------------------
mutable struct _Topo
    xn::Vector{Float64}; yn::Vector{Float64}
    cnode::Matrix{Int32}; cface::Matrix{Int32}; cblk::Vector{UInt8}; ci::Vector{Int32}; cj::Vector{Int32}
    own::Vector{Int32}; nei::Vector{Int32}; fna::Vector{Int32}; fnb::Vector{Int32}; ftag::Vector{UInt8}
    fsta::Vector{Int32}; ftx::Vector{Float64}; fty::Vector{Float64}
    fslot::Vector{NTuple{2,Int8}}                     # (owner slot, neighbour slot) in cface
    bown::Vector{Int32}; bna::Vector{Int32}; bnb::Vector{Int32}; btag::Vector{UInt8}; bsta::Vector{Int32}
    bslot::Vector{Int8}
    jlines::Vector{Vector{Int32}}; jcyc::Vector{Bool}
    ilines::Vector{Vector{Int32}}; icyc::Vector{Bool}
    wall_tags::Vector{UInt8}
    corner_nodes::Vector{Int32}
end
_Topo(nn, nc) = _Topo(zeros(nn), zeros(nn), zeros(Int32, 4, nc), zeros(Int32, 4, nc), zeros(UInt8, nc), zeros(Int32, nc),
                      zeros(Int32, nc), Int32[], Int32[], Int32[], Int32[], UInt8[], Int32[], Float64[], Float64[],
                      NTuple{2,Int8}[], Int32[], Int32[], Int32[], UInt8[], Int32[], Int8[],
                      Vector{Int32}[], Bool[], Vector{Int32}[], Bool[], UInt8[], Int32[])

function _iface!(t::_Topo, P, N, a, b, tag, slotP, slotN; st=0, tx=0.0, ty=0.0)
    push!(t.own, P); push!(t.nei, N); push!(t.fna, a); push!(t.fnb, b); push!(t.ftag, tag); push!(t.fsta, st)
    push!(t.ftx, tx); push!(t.fty, ty); push!(t.fslot, (Int8(slotP), Int8(slotN)))
    nothing
end
function _bface!(t::_Topo, P, a, b, tag, slot; st=0)
    push!(t.bown, P); push!(t.bna, a); push!(t.bnb, b); push!(t.btag, tag); push!(t.bsta, st); push!(t.bslot, Int8(slot))
    nothing
end

"The C-mesh topology in the P0a face order (see the header)."
function _topo_cmesh(b::CBlocks)
    NI, NJ, NIW, NIF, NIs, NS = b.NI, b.NJ, b.NIW, b.NIF, b.NIs, b.NS
    size(b.X) == (NI + 1, NJ + 1) && size(b.Y) == size(b.X) && length(b.kind) == NI ||
        throw(ArgumentError("C block arrays do not match NI = $NI, NJ = $NJ"))
    NI == 2NIW + NIF || throw(ArgumentError("C block needs NI = 2 NIW + NIF (got $NI, $NIW, $NIF)"))
    strip = NS > 0
    strip && (NIs == NIW && size(b.Xs) == (NIs + 1, NS + 1) && size(b.Ys) == size(b.Xs) ||
              throw(ArgumentError("strip block needs NIs = NIW = $NIW and (NIs+1) x (NS+1) nodes (got $NIs, $(size(b.Xs)))")))
    nnC = (NI + 1)*(NJ + 1); nn = nnC + (strip ? (NIs + 1)*(NS + 1) : 0); nc = NI*NJ + NIs*NS
    t = _Topo(nn, nc)
    c(i, j) = Int32((i - 1)*NJ + j); s(st, js) = Int32(NI*NJ + (st - 1)*NS + js)
    n(i, j) = Int32((i - 1)*(NJ + 1) + j); ns(st, js) = Int32(nnC + (st - 1)*(NS + 1) + js)
    for i in 1:NI+1, j in 1:NJ+1; t.xn[n(i, j)] = b.X[i, j]; t.yn[n(i, j)] = b.Y[i, j]; end
    for i in 1:NI, j in 1:NJ
        k = c(i, j); t.cnode[:, k] .= (n(i, j), n(i+1, j), n(i+1, j+1), n(i, j+1)); t.cblk[k] = 0x00
        t.ci[k] = i; t.cj[k] = j
    end
    if strip
        for st in 1:NIs+1, js in 1:NS+1; t.xn[ns(st, js)] = b.Xs[st, js]; t.yn[ns(st, js)] = b.Ys[st, js]; end
        for st in 1:NIs, js in 1:NS
            k = s(st, js); t.cnode[:, k] .= (ns(st, js), ns(st+1, js), ns(st+1, js+1), ns(st, js+1)); t.cblk[k] = 0x01
            t.ci[k] = st; t.cj[k] = js
        end
    end
    # internal faces, P0a order
    for j in 1:NJ, i in 2:NI; _iface!(t, c(i-1, j), c(i, j), n(i, j), n(i, j+1), FT_CI, 2, 1); end
    for j in 2:NJ, i in 1:NI; _iface!(t, c(i, j-1), c(i, j), n(i+1, j), n(i, j), FT_CJ, 4, 3); end
    if !strip
        for i in 1:NIW; _iface!(t, c(NI+1-i, 1), c(i, 1), n(i+1, 1), n(i, 1), FT_CUT, 3, 3; st=i); end
    else
        for st in 1:NIW
            _iface!(t, c(NIW+1-st, 1), s(st, 1), ns(st+1, 1), ns(st, 1), FT_IF, 3, 3; st=st)
            _iface!(t, s(st, NS), c(NIW+NIF+st, 1), ns(st+1, NS+1), ns(st, NS+1), FT_IF, 4, 3; st=-st)
        end
        for js in 1:NS, st in 2:NIs; _iface!(t, s(st-1, js), s(st, js), ns(st, js), ns(st, js+1), FT_SI, 2, 1); end
        for js in 2:NS, st in 1:NIs; _iface!(t, s(st, js-1), s(st, js), ns(st+1, js), ns(st, js), FT_SJ, 4, 3); end
    end
    # boundary faces
    for i in NIW+1:NIW+NIF; _bface!(t, c(i, 1), n(i, 1), n(i+1, 1), BT_FOIL, 3; st=i-NIW); end
    for js in 1:NS; _bface!(t, s(1, js), ns(1, js+1), ns(1, js), BT_BASE, 1; st=js); end
    for i in 1:NI
        k = b.kind[i]
        tag = k == OUTER_INLET ? BT_INLET : k == OUTER_LOWER_WALL ? BT_LOWER : k == OUTER_UPPER_WALL ? BT_UPPER :
              k == OUTER_FARFIELD ? BT_FAR : throw(ArgumentError("outer face $i has kind $k, not a C-block outer kind"))
        _bface!(t, c(i, NJ), n(i+1, NJ+1), n(i, NJ+1), tag, 4; st=i)
    end
    for j in 1:NJ
        _bface!(t, c(1, j), n(1, j+1), n(1, j), BT_OUTLET, 1; st=j)
        _bface!(t, c(NI, j), n(NI+1, j), n(NI+1, j+1), BT_OUTLET, 2; st=j)
    end
    for js in 1:NS; _bface!(t, s(NIs, js), ns(NIs+1, js), ns(NIs+1, js+1), BT_SOUTLET, 2; st=js); end
    # lines (the P0a / W3a mesh_lines order)
    for i in 1:NIW
        ip = NI + 1 - i; st = NIW + 1 - i
        L = Int32[c(ip, j) for j in NJ:-1:1]
        strip && append!(L, Int32[s(st, js) for js in NS:-1:1])
        append!(L, Int32[c(i, j) for j in 1:NJ]); push!(t.jlines, L); push!(t.jcyc, false)
    end
    for i in NIW+1:NIW+NIF; push!(t.jlines, Int32[c(i, j) for j in 1:NJ]); push!(t.jcyc, false); end
    for j in 1:NJ; push!(t.ilines, Int32[c(i, j) for i in 1:NI]); push!(t.icyc, false); end
    for js in 1:NS; push!(t.ilines, Int32[s(st, js) for st in 1:NIs]); push!(t.icyc, false); end
    t.wall_tags = [BT_FOIL, BT_BASE]
    t.corner_nodes = Int32[n(NIW+1, 1), n(NIW+NIF+1, 1)]
    t
end

"The box topology (periodic faces last among the internal faces)."
function _topo_box(b::QuadBox)
    nx, ny = size(b.X) .- 1
    nn = (nx + 1)*(ny + 1); nc = nx*ny; t = _Topo(nn, nc)
    c(i, j) = Int32((i - 1)*ny + j); n(i, j) = Int32((i - 1)*(ny + 1) + j)
    for i in 1:nx+1, j in 1:ny+1; t.xn[n(i, j)] = b.X[i, j]; t.yn[n(i, j)] = b.Y[i, j]; end
    for i in 1:nx, j in 1:ny
        k = c(i, j); t.cnode[:, k] .= (n(i, j), n(i+1, j), n(i+1, j+1), n(i, j+1)); t.ci[k] = i; t.cj[k] = j
    end
    for j in 1:ny, i in 2:nx; _iface!(t, c(i-1, j), c(i, j), n(i, j), n(i, j+1), FT_BI, 2, 1); end
    for j in 2:ny, i in 1:nx; _iface!(t, c(i, j-1), c(i, j), n(i+1, j), n(i, j), FT_BJ, 4, 3); end
    if b.periodic[1]
        for j in 1:ny; _iface!(t, c(nx, j), c(1, j), n(nx+1, j), n(nx+1, j+1), FT_PERI, 2, 1; st=j, tx=b.Lx); end
    end
    if b.periodic[2]
        for i in 1:nx; _iface!(t, c(i, ny), c(i, 1), n(i+1, ny+1), n(i, ny+1), FT_PERJ, 4, 3; st=i, ty=b.Ly); end
    end
    if !b.periodic[1]
        for j in 1:ny; _bface!(t, c(1, j), n(1, j+1), n(1, j), BT_W, 1; st=j); end
        for j in 1:ny; _bface!(t, c(nx, j), n(nx+1, j), n(nx+1, j+1), BT_E, 2; st=j); end
    end
    if !b.periodic[2]
        for i in 1:nx; _bface!(t, c(i, 1), n(i, 1), n(i+1, 1), BT_S, 3; st=i); end
        for i in 1:nx; _bface!(t, c(i, ny), n(i+1, ny+1), n(i, ny+1), BT_N, 4; st=i); end
    end
    for i in 1:nx; push!(t.jlines, Int32[c(i, j) for j in 1:ny]); push!(t.jcyc, b.periodic[2]); end
    for j in 1:ny; push!(t.ilines, Int32[c(i, j) for i in 1:nx]); push!(t.icyc, b.periodic[1]); end
    t.wall_tags = UInt8[tag for (tag, w) in zip((BT_W, BT_E, BT_S, BT_N), b.walls) if w]
    t
end

"Fill cface from the face lists; refuse a cell side claimed twice or never."
function _fill_cface!(t::_Topo)
    nf = length(t.own); nc = size(t.cface, 2)
    for f in 1:nf
        sP, sN = t.fslot[f]; P = t.own[f]; N = t.nei[f]
        t.cface[sP, P] == 0 || throw(ArgumentError("cell $P side $sP claimed twice"))
        t.cface[sN, N] == 0 || throw(ArgumentError("cell $N side $sN claimed twice"))
        t.cface[sP, P] = Int32(f); t.cface[sN, N] = Int32(-f)
    end
    for q in eachindex(t.bown)
        P = t.bown[q]; sP = t.bslot[q]
        t.cface[sP, P] == 0 || throw(ArgumentError("cell $P side $sP claimed twice (boundary face $q)"))
        t.cface[sP, P] = Int32(nf + q)
    end
    nmiss = count(==(0), t.cface)
    nmiss == 0 || throw(ArgumentError("$nmiss cell sides have no face"))
    nothing
end

"""The internal face joining cells u and v (searched on u's four sides), periodic or not as asked (in a periodic box of
two cells a pair shares both an ordinary and a periodic face), or 0."""
function _face_between_any(t::_Topo, u::Integer, v::Integer; periodic::Bool=false)
    nf = length(t.own)
    for k in 1:4
        code = t.cface[k, u]
        f = 0 < code <= nf ? Int(code) : code < 0 ? Int(-code) : 0
        f == 0 && continue
        (t.ftag[f] in (FT_PERI, FT_PERJ)) == periodic || continue
        other = t.own[f] == u ? t.nei[f] : t.own[f]
        other == v && return Int32(f)
    end
    Int32(0)
end

"CSR of a line family and the internal face between consecutive cells (and the closing face of a cyclic line)."
function _lines_csr(t::_Topo, lines, cyc)
    ptr = Int32[1]; cells = Int32[]; fptr = Int32[1]; faces = Int32[]
    for (L, cy) in zip(lines, cyc)
        append!(cells, L); push!(ptr, Int32(length(cells) + 1))
        for k in 1:length(L)-1
            f = _face_between_any(t, L[k], L[k+1])
            # a cyclic line opened by a baffle (`_cut_faces!`) runs through its periodic face (only reached where the ordinary
            # search finds none, which before baffles was an error: every existing mesh's lines are unchanged)
            f > 0 || (f = _face_between_any(t, L[k], L[k+1]; periodic=true))
            f > 0 || throw(ArgumentError("line cells $(L[k]) and $(L[k+1]) share no internal face"))
            push!(faces, f)
        end
        if cy
            f = _face_between_any(t, L[end], L[1]; periodic=true)
            f > 0 || throw(ArgumentError("cyclic line cells $(L[end]) and $(L[1]) share no internal face"))
            push!(faces, f)
        end
        push!(fptr, Int32(length(faces) + 1))
    end
    ptr, cells, fptr, faces, UInt8.(cyc)
end

# ---- interior walls: baffles (24 Oct 2026; opt-in) --------------------------------------------------------------------
"""
    BaffleCut(; dir, line, lo, hi, plate=1, front=:minus)

One straight run of internal faces of one grid line of block 0 (the C block or a box), cut into walls (`build_faces(...;
baffle)`). `dir = :i`: the i faces on node column `line` (between cells (line-1, j) and (line, j)) for rows j = lo..hi;
`dir = :j`: the j faces on node row `line` (between cells (i, line-1) and (i, line)) for columns i = lo..hi (boxes only: a
C-mesh's j-lines are its fold lines, which a j cut would break). `plate` (1 or 2) names the plate the faces belong to (the
sign of their `bsta`); `front` is the side whose boundary faces are tagged BT_BAFFLE_FRONT: `:minus` the cell on the lower
index side, `:plus` the other.
"""
Base.@kwdef struct BaffleCut
    dir::Symbol; line::Int; lo::Int; hi::Int; plate::Int = 1; front::Symbol = :minus
end

"""THE CUT (`build_faces(...; baffle)`): each face of `cuts` (on a topology `t` of block-0 row length `nj`, node row length
`nj + 1`) leaves the internal list (the others keep their order) and becomes two boundary faces appended after every other
boundary face, cut by cut: the owner's (the face's own nodes, so S points out of it; the owner's cface slot) for each row, then
the neighbour's (the nodes reversed; its slot), tagged front or back by the cut's `front`, `bsta` = the row along the cut from
`lo` (1), negative on plate 2. The lines (j and i) are split where consecutive cells were joined by a cut face; a cyclic line
with a cut is opened there (rotated to start after its first cut, then split at the others). Refuses a face that is not a
block-0 i or j face (cut, interface, strip and periodic faces cannot be cut), a face cut twice, and a C-mesh j cut. Host
integer work, once per mesh. Returns the cut node ends (for `corner_cells`)."""
function _cut_faces!(t::_Topo, cuts, ni::Integer, nj::Integer; cmesh::Bool)
    nf = length(t.own)
    key = Dict{Tuple{Int32,Int32},Int}()
    for f in 1:nf; key[(t.own[f], t.nei[f])] = f; end
    cid(i, j) = Int32((i - 1)*nj + j); nid(i, j) = Int32((i - 1)*(nj + 1) + j)
    iface, jface = cmesh ? (FT_CI, FT_CJ) : (FT_BI, FT_BJ)
    cutset = Set{Int}(); ends = Int32[]; recs = Tuple{Int,Int,Int,Symbol}[]    # (face, plate, row, front)
    for c in cuts
        c.dir in (:i, :j) || throw(ArgumentError("BaffleCut: dir must be :i or :j (got :$(c.dir))"))
        c.front in (:minus, :plus) || throw(ArgumentError("BaffleCut: front must be :minus or :plus (got :$(c.front))"))
        c.plate in (1, 2) || throw(ArgumentError("BaffleCut: plate must be 1 or 2 (got $(c.plate))"))
        cmesh && c.dir === :j && throw(ArgumentError("BaffleCut: a C-mesh baffle lies on a C column (dir = :i); a j cut would " *
                                                     "break the j-lines' fold through the wake (the multigrid's fold lines)"))
        nline, nalong = c.dir === :i ? (ni, nj) : (nj, ni)
        (2 <= c.line <= nline && 1 <= c.lo <= c.hi <= nalong) ||
            throw(ArgumentError("BaffleCut: needs 2 <= line <= $nline and 1 <= lo <= hi <= $nalong for a $(c.dir) cut of this block " *
                                "(got line $(c.line), lo $(c.lo), hi $(c.hi))"))
        for r in c.lo:c.hi
            P, N = c.dir === :i ? (cid(c.line - 1, r), cid(c.line, r)) : (cid(r, c.line - 1), cid(r, c.line))
            f = (c.line >= 2 && r >= 1) ? get(key, (P, N), 0) : 0
            (f > 0 && t.ftag[f] == (c.dir === :i ? iface : jface)) ||
                throw(ArgumentError("BaffleCut: no block-0 $(c.dir) face between cells $P and $N (line $(c.line), position $r)"))
            f in cutset && throw(ArgumentError("BaffleCut: face $f is cut twice"))
            push!(cutset, f); push!(recs, (f, c.plate, r - c.lo + 1, c.front))
        end
        push!(ends, c.dir === :i ? nid(c.line, c.lo) : nid(c.lo, c.line), c.dir === :i ? nid(c.line, c.hi + 1) : nid(c.hi + 1, c.line))
    end
    pairs = Set{Tuple{Int32,Int32}}((min(t.own[f], t.nei[f]), max(t.own[f], t.nei[f])) for f in cutset)
    # the boundary faces: per cut, the owner side's rows, then the neighbour side's
    q0 = 1
    for c in cuts
        n = c.hi - c.lo + 1; rs = recs[q0:q0+n-1]; q0 += n
        for side in (:minus, :plus), (f, plate, row, front) in rs
            tag = side === front ? BT_BAFFLE_FRONT : BT_BAFFLE_BACK
            st = plate == 2 ? -row : row
            if side === :minus      # the owner (the lower-index cell): the face's own orientation points out of it
                _bface!(t, t.own[f], t.fna[f], t.fnb[f], tag, t.fslot[f][1]; st)
            else
                _bface!(t, t.nei[f], t.fnb[f], t.fna[f], tag, t.fslot[f][2]; st)
            end
        end
    end
    # the internal faces kept, in their order
    keep = [f for f in 1:nf if !(f in cutset)]
    for fld in (:own, :nei, :fna, :fnb, :ftag, :fsta, :ftx, :fty, :fslot)
        setfield!(t, fld, getfield(t, fld)[keep])
    end
    # the lines split at the cuts
    function split_lines(lines, cyc)
        out = Vector{Vector{Int32}}(); oc = Bool[]
        iscut(a, b) = (min(a, b), max(a, b)) in pairs
        for (L, cy) in zip(lines, cyc)
            n = length(L)
            brk = [k for k in 1:n-1 if iscut(L[k], L[k+1])]
            closing = cy && n >= 2 && iscut(L[n], L[1])
            if isempty(brk) && !closing
                push!(out, L); push!(oc, cy); continue
            end
            if cy && !closing                    # open the cyclic line at its first cut: start after it
                k1 = brk[1]; L = vcat(L[k1+1:n], L[1:k1]); brk = [k for k in 1:n-1 if iscut(L[k], L[k+1])]
            end
            a = 1
            for k in vcat(brk, n)
                push!(out, L[a:k]); push!(oc, false); a = k + 1
            end
        end
        out, oc
    end
    t.jlines, t.jcyc = split_lines(t.jlines, t.jcyc)
    t.ilines, t.icyc = split_lines(t.ilines, t.icyc)
    append!(t.wall_tags, (BT_BAFFLE_FRONT, BT_BAFFLE_BACK))
    ends
end

# ---- the geometry kernels (Float64, on the backend) -------------------------------------------------------------------
@kernel function _k_cell_geom!(vol, xc, yc, xv, yv, @Const(cnode), @Const(xn), @Const(yn))
    c = @index(Global)
    @inbounds begin
        a1 = cnode[1, c]; a2 = cnode[2, c]; a3 = cnode[3, c]; a4 = cnode[4, c]
        x1 = xn[a1]; x2 = xn[a2]; x3 = xn[a3]; x4 = xn[a4]
        y1 = yn[a1]; y2 = yn[a2]; y3 = yn[a3]; y4 = yn[a4]
        # the vertex average in version 1's summation order, (i,j) + (i+1,j) + (i,j+1) + (i+1,j+1) (cmesh_metrics.jl), so
        # Float32(xv) is v1's stored centre bit for bit
        mx = 0.25*(x1 + x2 + x4 + x3); my = 0.25*(y1 + y2 + y4 + y3)
        # polygon (shoelace) area and centroid, relative to the vertex average to keep the round-off small
        u1 = x1 - mx; u2 = x2 - mx; u3 = x3 - mx; u4 = x4 - mx
        w1 = y1 - my; w2 = y2 - my; w3 = y3 - my; w4 = y4 - my
        k1 = u1*w2 - u2*w1; k2 = u2*w3 - u3*w2; k3 = u3*w4 - u4*w3; k4 = u4*w1 - u1*w4
        A2 = k1 + k2 + k3 + k4
        cx = (u1 + u2)*k1 + (u2 + u3)*k2 + (u3 + u4)*k3 + (u4 + u1)*k4
        cy = (w1 + w2)*k1 + (w2 + w3)*k2 + (w3 + w4)*k3 + (w4 + w1)*k4
        vol[c] = 0.5*A2
        xc[c] = mx + cx/(3*A2); yc[c] = my + cy/(3*A2)
        xv[c] = mx; yv[c] = my
    end
end

@kernel function _k_face_geom!(Sx, Sy, Smag, xf, yf, dx, dy, fw, fa, kx, ky, theta, skx, sky,
                               @Const(own), @Const(nei), @Const(fna), @Const(fnb), @Const(ftx), @Const(fty),
                               @Const(xn), @Const(yn), @Const(xc), @Const(yc))
    f = @index(Global)
    @inbounds begin
        a = fna[f]; b = fnb[f]; P = own[f]; N = nei[f]
        sx = yn[b] - yn[a]; sy = -(xn[b] - xn[a])
        mx = 0.5*(xn[a] + xn[b]); my = 0.5*(yn[a] + yn[b])
        xN = xc[N] + ftx[f]; yN = yc[N] + fty[f]
        ddx = xN - xc[P]; ddy = yN - yc[P]
        sd = sx*ddx + sy*ddy; s2 = sx*sx + sy*sy; sm = sqrt(s2)
        Sx[f] = sx; Sy[f] = sy; Smag[f] = sm; xf[f] = mx; yf[f] = my; dx[f] = ddx; dy[f] = ddy
        w = (sx*(xN - mx) + sy*(yN - my))/sd; fw[f] = w
        q = s2/sd; fa[f] = q; kx[f] = sx - q*ddx; ky[f] = sy - q*ddy
        theta[f] = acosd(clamp(sd/(sm*sqrt(ddx*ddx + ddy*ddy)), -1.0, 1.0))
        # skewness (review F1, 14 Oct): from the crossing point x_f' = x_P + (1 - w) d to the face centroid
        skx[f] = mx - (xc[P] + (1 - w)*ddx); sky[f] = my - (yc[P] + (1 - w)*ddy)
    end
end

@kernel function _k_bface_geom!(Sx, Sy, Smag, xf, yf, dx, dy, ba, kx, ky,
                                @Const(own), @Const(fna), @Const(fnb), @Const(xn), @Const(yn), @Const(xc), @Const(yc))
    f = @index(Global)
    @inbounds begin
        a = fna[f]; b = fnb[f]; P = own[f]
        sx = yn[b] - yn[a]; sy = -(xn[b] - xn[a])
        mx = 0.5*(xn[a] + xn[b]); my = 0.5*(yn[a] + yn[b])
        ddx = mx - xc[P]; ddy = my - yc[P]
        sd = sx*ddx + sy*ddy; s2 = sx*sx + sy*sy
        Sx[f] = sx; Sy[f] = sy; Smag[f] = sqrt(s2); xf[f] = mx; yf[f] = my; dx[f] = ddx; dy[f] = ddy
        q = s2/sd; ba[f] = q; kx[f] = sx - q*ddx; ky[f] = sy - q*ddy
    end
end

"The smallest corner cross product of each quad (positive at every corner = strictly convex and counter-clockwise)."
@kernel function _k_convexity!(cvx, @Const(cnode), @Const(xn), @Const(yn))
    c = @index(Global)
    @inbounds begin
        m = Inf
        for k in 1:4
            a = cnode[k, c]; b = cnode[mod1(k + 1, 4), c]; e = cnode[mod1(k + 2, 4), c]
            cr = (xn[b] - xn[a])*(yn[e] - yn[b]) - (yn[b] - yn[a])*(xn[e] - xn[b])
            m = min(m, cr)
        end
        cvx[c] = m
    end
end

"Distance from each cell's true centroid to the nearest wall segment (point-segment, brute force over the wall faces)."
@kernel function _k_wdist!(wdist, @Const(xc), @Const(yc), @Const(wax), @Const(way), @Const(wbx), @Const(wby), nw::Int32)
    c = @index(Global)
    @inbounds begin
        px = xc[c]; py = yc[c]; dmin = Inf
        for q in 1:nw
            ax = wax[q]; ay = way[q]; ex = wbx[q] - ax; ey = wby[q] - ay
            tt = clamp(((px - ax)*ex + (py - ay)*ey)/max(ex*ex + ey*ey, 1e-300), 0.0, 1.0)
            d = sqrt((px - (ax + tt*ex))^2 + (py - (ay + tt*ey))^2)
            dmin = min(dmin, d)
        end
        wdist[c] = dmin
    end
end

_dev(backend, v::AbstractArray) = (a = KernelAbstractions.allocate(backend, eltype(v), size(v)); copyto!(a, v); a)
_zeros(backend, T, n) = KernelAbstractions.zeros(backend, T, n)
_host(v::AbstractArray) = Array(v)

"""THE CPU LAUNCH SHAPE (work item W7, plan Amendment 13; 22 Oct 2026). KernelAbstractions' CPU backend gives each Julia thread
whole workgroups, so a 256-item workgroup puts a kernel over at most 256 items (most line-solve colours, the reductions' chunk
passes, the coarse multigrid levels) on ONE thread. `:fixed` (THE DEFAULT, the path before W7): workgroups of 256 on every
backend. `:balanced`: on CPU() a kernel of n < 256 x nthreads items gets workgroups of max(1, cld(n, nthreads)) items, so
every thread gets work; every other launch is as before. No kernel of the solver uses @localmem or @synchronize (each work
item is independent), so the workgroup size changes no result, bit for bit (W7's records check it). Set by
`pb_cpu_launch!` (the driver's PB_CPU_LAUNCH)."""
const PB_CPU_LAUNCH = Ref(:fixed)
function pb_cpu_launch!(mode::Symbol)
    mode in (:fixed, :balanced) || throw(ArgumentError("pb_cpu_launch!: :fixed or :balanced (got :$mode)"))
    PB_CPU_LAUNCH[] = mode
end
@inline function _wgsize(backend, n::Integer)
    (backend isa CPU && PB_CPU_LAUNCH[] === :balanced) || return 256
    nt = Threads.nthreads()
    n >= 256*nt ? 256 : max(1, cld(Int(n), nt))
end
"The kernel launch itself (n > 0): the one place a kernel is instantiated (W7's census wraps `_launch!` around it)."
_launch_kernel!(kern, backend, n::Integer, args...) = kern(backend, _wgsize(backend, n))(args...; ndrange=n)
"Launch a kernel over n items (no launch when n = 0)."
function _launch!(kern, backend, n::Integer, args...)
    n > 0 && _launch_kernel!(kern, backend, n, args...)
    nothing
end
"Launch a kernel over n items in workgroups of `wg` items (W7: the kernels that use @localmem; n a multiple of wg)."
function _launch_wg!(kern, backend, n::Integer, wg::Integer, args...)
    n > 0 && kern(backend, Int(wg))(args...; ndrange=n)
    nothing
end

"""THE SYNCHRONISATIONS OF A STEP (W7, plan Amendment 13; 22 Oct 2026). Every assembler, sweep, cycle and reduction of the
solver ends with a device synchronisation (`pb_sync`). On the CPU a KernelAbstractions launch returns when its kernel has
finished, so a synchronisation is a no-op there whatever the setting. On a GPU every launch, copy and host read of the solver
is ordered on one stream, and every host read is a synchronising copy, so the synchronisations change no result: they only
stop the host from queueing launches ahead of the device. `:all` (THE DEFAULT, the path before W7): every synchronisation
runs; `:reads`: on a non-CPU backend `pb_sync` returns at once and the stream is drained only by the host reads themselves
(and the explicit synchronisations of the diagnostics). Set from PBConfig.sync by each step (`pb_sync_mode!`)."""
const PB_SYNC = Ref(:all)
function pb_sync_mode!(mode::Symbol)
    mode in (:all, :reads) || throw(ArgumentError("PBConfig.sync must be :all or :reads (got :$mode)"))
    PB_SYNC[] = mode
end
@inline pb_sync(backend) = (PB_SYNC[] === :all || backend isa CPU) ? KernelAbstractions.synchronize(backend) : nothing

"""THE GRAPH BACKEND (W7, PBConfig.graphs): this module never loads CUDA, so the caller that has (the driver, a test, a W7
script) installs the three functions a captured graph needs: `capture(f)` (CUDA.capture), `instantiate(g)`
(CUDA.instantiate) and `launch(exec)` (CUDA.launch). `nothing` (the default): graphs = :on is refused on a GPU."""
const PB_GRAPH = Ref{Any}(nothing)
pb_graph_backend!(capture, instantiate, launch) = (PB_GRAPH[] = (capture=capture, instantiate=instantiate, launch=launch); nothing)

# ---- build ------------------------------------------------------------------------------------------------------------
"""
    build_faces(b; backend=CPU(), n_outlet_rows=6*2^level) -> PBMesh

The faces, metrics, lines and special lists of a C-mesh (`b::CBlocks`, or a generated `ChannelMesh`) or a box
(`b::QuadBox`), built on the CPU and moved to `backend` (review of W3, F6). Refuses an internal or boundary face whose area vector does not point along the
centroid-to-centroid direction (S.d <= 0), and a cell that is not strictly convex. `n_outlet_rows` is the number of cell
columns beside each outlet whose faces form the outlet-row list (C-meshes; plan 4.1, 10.1). The default 6 is measured: on
N60R every face above 70 degrees
(13, all C-block j faces of rows 13-14, max 82.28 degrees, as audit NP measured) lies in C columns 1-6 or 523-528, and
tools/test_pbcav.jl checks that the outlet-row list holds every high-angle face on N60R and LE256. On a nested level the
default doubles with each subdivision, so the list covers the same region (N60Rx2: 12 columns).
"""
build_faces(m::ChannelMesh; label::AbstractString="", kw...) = build_faces(cblocks(m; label); kw...)

function build_faces(b::Union{CBlocks,QuadBox}; backend=CPU(), n_outlet_rows::Union{Nothing,Integer}=nothing, baffle=nothing)
    # EVERY MESH IS BUILT ON THE CPU AND MOVED (review of W3, F6, 22 Oct 2026): on a device the geometry kernels would let FMA
    # contraction move the metrics by ulps and change the receipts; so a device `backend` gets the CPU build moved by
    # to_backend (backend.jl), bit for bit, on every path (the driver, freeair_mesh, box_mesh, any caller). CPU() unchanged.
    backend isa CPU || return to_backend(backend, build_faces(b; backend=CPU(), n_outlet_rows, baffle))
    cm = b isa CBlocks
    n_outlet_rows = n_outlet_rows === nothing ? 6*2^b.level : Int(n_outlet_rows)
    t = cm ? _topo_cmesh(b) : _topo_box(b)
    # BAFFLES (24 Oct 2026; opt-in): `baffle` is nothing (the default: nothing below runs, the mesh is bit for bit the one
    # before), a vector of BaffleCut, or a BafflePlan (baffle.jl: its `cuts`, made on the plan's own blocks, so pass
    # `plan.blocks` as `b`; refused otherwise)
    if baffle !== nothing && !(baffle isa AbstractVector)
        baffle.blocks === b || throw(ArgumentError("build_faces: a BafflePlan is built on its own blocks (pass plan.blocks)"))
    end
    cuts = baffle === nothing ? BaffleCut[] : baffle isa AbstractVector ? collect(BaffleCut, baffle) : collect(BaffleCut, baffle.cuts)
    bends = isempty(cuts) ? Int32[] : _cut_faces!(t, cuts, cm ? b.NI : size(b.X, 1) - 1, cm ? b.NJ : size(b.X, 2) - 1; cmesh=cm)
    cm && append!(t.corner_nodes, bends)
    _fill_cface!(t)
    nc = size(t.cnode, 2); nf = length(t.own); nb = length(t.bown); nn = length(t.xn)
    # geometry on the backend
    xn = _dev(backend, t.xn); yn = _dev(backend, t.yn); cnode = _dev(backend, t.cnode)
    vol, xc, yc, xv, yv, wdist = (_zeros(backend, Float64, nc) for _ in 1:6)
    _launch!(_k_cell_geom!, backend, nc, vol, xc, yc, xv, yv, cnode, xn, yn)
    own = _dev(backend, t.own); nei = _dev(backend, t.nei); fna = _dev(backend, t.fna); fnb = _dev(backend, t.fnb)
    ftx = _dev(backend, t.ftx); fty = _dev(backend, t.fty)
    Sx, Sy, Smag, xf, yf, dx, dy, fw, fa, kx, ky, theta, skx, sky = (_zeros(backend, Float64, nf) for _ in 1:14)
    _launch!(_k_face_geom!, backend, nf, Sx, Sy, Smag, xf, yf, dx, dy, fw, fa, kx, ky, theta, skx, sky,
             own, nei, fna, fnb, ftx, fty, xn, yn, xc, yc)
    bown = _dev(backend, t.bown); bna = _dev(backend, t.bna); bnb = _dev(backend, t.bnb)
    bSx, bSy, bSmag, bxf, byf, bdx, bdy, ba, bkx, bky = (_zeros(backend, Float64, nb) for _ in 1:10)
    _launch!(_k_bface_geom!, backend, nb, bSx, bSy, bSmag, bxf, byf, bdx, bdy, ba, bkx, bky,
             bown, bna, bnb, xn, yn, xc, yc)
    KernelAbstractions.synchronize(backend)
    # wall distance to the wall faces (C-mesh: foil and base; box: the sides named walls)
    wq = findall(in(t.wall_tags), t.btag)
    if isempty(wq)
        fill!(wdist, Inf)
    else
        wax = _dev(backend, t.xn[t.bna[wq]]); way = _dev(backend, t.yn[t.bna[wq]])
        wbx = _dev(backend, t.xn[t.bnb[wq]]); wby = _dev(backend, t.yn[t.bnb[wq]])
        _launch!(_k_wdist!, backend, nc, wdist, xc, yc, wax, way, wbx, wby, Int32(length(wq)))
        KernelAbstractions.synchronize(backend)
    end
    # refusals (setup: read back the sign tests)
    cvx = _zeros(backend, Float64, nc)
    _launch!(_k_convexity!, backend, nc, cvx, cnode, xn, yn); KernelAbstractions.synchronize(backend)
    vh = _host(vol); fah = _host(fa); bah = _host(ba)
    nneg = count(v -> !(v > 0), vh)
    nneg == 0 || throw(ArgumentError("build_faces: $nneg cells with non-positive or non-finite area"))
    ncon = count(v -> !(v > 0), _host(cvx))
    ncon == 0 || throw(ArgumentError("build_faces: $ncon cells are not strictly convex (a corner cross product <= 0); " *
                                     "version 1's generator smooths until every cell passes this test"))
    nbad = count(v -> !(v > 0) || !isfinite(v), fah); nbadb = count(v -> !(v > 0) || !isfinite(v), bah)
    nbad == 0 && nbadb == 0 ||
        throw(ArgumentError("build_faces: $nbad internal and $nbadb boundary faces with S.d <= 0 (true centroids)"))
    # lines
    jl = _lines_csr(t, t.jlines, t.jcyc); il = _lines_csr(t, t.ilines, t.icyc)
    # special lists
    thh = _host(theta)
    high = Int32.(findall(>(HIGH_ANGLE_DEG), thh))
    if cm
        NI, NJ, NIW, NIF, NIs, NS = b.NI, b.NJ, b.NIW, b.NIF, b.NIs, b.NS
        cut = Int32.(findall(==(FT_CUT), t.ftag)); cut = cut[sortperm(t.fsta[cut])]
        ifl = Int32.(findall(q -> t.ftag[q] == FT_IF && t.fsta[q] > 0, eachindex(t.ftag))); ifl = ifl[sortperm(t.fsta[ifl])]
        ifu = Int32.(findall(q -> t.ftag[q] == FT_IF && t.fsta[q] < 0, eachindex(t.ftag))); ifu = ifu[sortperm(-t.fsta[ifu])]
        rowcell = falses(nc)
        for k in 1:nc
            i = t.ci[k]
            rowcell[k] = t.cblk[k] == 0x00 ? (i <= n_outlet_rows || i > NI - n_outlet_rows) : (i > NIs - n_outlet_rows)
        end
        orows = Int32.(findall(f -> rowcell[t.own[f]] || rowcell[t.nei[f]], 1:nf))
        corner = Int32[]
        for k in 1:nc
            any(a -> any(cn -> t.xn[a] == t.xn[cn] && t.yn[a] == t.yn[cn], t.corner_nodes), view(t.cnode, :, k)) &&
                push!(corner, Int32(k))
        end
        dims = (NI, NJ, NIW, NIF, NIs, NS); nblk = NS > 0 ? 2 : 1
        boff = (0, NI*NJ); bni = (NI, NIs); bnj = (NJ, NS); per = (false, false); Lper = (0.0, 0.0)
        hashes = b.hashes; fam = :cmesh; lab = b.label; lev = b.level
    else
        nx, ny = size(b.X) .- 1
        cut = Int32[]; ifl = Int32[]; ifu = Int32[]; orows = Int32[]
        # a box has no corner list, except the cells touching a baffle's end nodes (empty without a baffle)
        corner = Int32[k for k in 1:nc if any(a -> a in bends, view(t.cnode, :, k))]
        dims = (nx, ny, 0, 0, 0, 0); nblk = 1; boff = (0, nx*ny); bni = (nx, 0); bnj = (ny, 0)
        per = b.periodic; Lper = (b.Lx, b.Ly); hashes = (h64(b.X), h64(b.Y), "", ""); fam = :box; lab = b.label
        lev = b.level
    end
    bl(tag) = Int32.(findall(==(tag), t.btag))
    outl = Int32.(findall(in((BT_OUTLET, BT_SOUTLET)), t.btag))
    dirl = vcat(bl(BT_FAR), outl)
    d(v) = _dev(backend, v)
    PBMesh(lab, fam, lev, dims..., nblk, boff, bni, bnj, per, Lper, hashes, Int(cm ? n_outlet_rows : 0),
           nc, nf, nb, nn, xn, yn,
           cnode, d(t.cface), d(t.cblk), d(t.ci), d(t.cj), vol, xc, yc, xv, yv, wdist,
           own, nei, fna, fnb, d(t.ftag), d(t.fsta), ftx, fty,
           Sx, Sy, Smag, xf, yf, dx, dy, fw, fa, kx, ky, theta, skx, sky,
           bown, bna, bnb, d(t.btag), d(t.bsta), bSx, bSy, bSmag, bxf, byf, bdx, bdy, ba, bkx, bky,
           d(jl[1]), d(jl[2]), d(jl[3]), d(jl[4]), d(jl[5]), d(il[1]), d(il[2]), d(il[3]), d(il[4]), d(il[5]),
           d(cut), d(ifl), d(ifu), d(orows), d(high),
           d(bl(BT_FOIL)), d(bl(BT_BASE)), d(bl(BT_INLET)), d(bl(BT_LOWER)), d(bl(BT_UPPER)), d(bl(BT_FAR)),
           d(outl), d(dirl), d(corner), d(Int32.(findall(in((BT_BAFFLE_FRONT, BT_BAFFLE_BACK)), t.btag))))
end

"""`box_mesh(nx, ny; kw...)` = `build_faces(box_blocks(nx, ny; kw...))` (keywords `backend`, `level` and `baffle`, a vector
of BaffleCut on the subdivided box, taken out)."""
function box_mesh(nx::Integer, ny::Integer; backend=CPU(), level::Integer=0, baffle=nothing, kw...)
    build_faces(subdivide(box_blocks(nx, ny; kw...), level); backend, baffle)
end

# ---- queries ----------------------------------------------------------------------------------------------------------
"""
    face_census(m::PBMesh) -> NamedTuple

Counts by kind: cells, internal faces by tag (C i, C j, cut, interface, strip i, strip j, box i, box j, periodic i/j),
boundary faces by tag, the special lists, the lines.
"""
function face_census(m::PBMesh)
    ft = _host(m.ftag); bt = _host(m.btag)
    cf(x) = count(==(x), ft); cb(x) = count(==(x), bt)
    (cells=m.nc, internal=m.nf, boundary=m.nb, ci=cf(FT_CI), cj=cf(FT_CJ), cut=cf(FT_CUT), interface=cf(FT_IF),
     si=cf(FT_SI), sj=cf(FT_SJ), bi=cf(FT_BI), bj=cf(FT_BJ), peri=cf(FT_PERI), perj=cf(FT_PERJ),
     foil=cb(BT_FOIL), base=cb(BT_BASE), inlet=cb(BT_INLET), lower=cb(BT_LOWER), upper=cb(BT_UPPER), far=cb(BT_FAR),
     outlet=cb(BT_OUTLET), soutlet=cb(BT_SOUTLET), W=cb(BT_W), E=cb(BT_E), S=cb(BT_S), N=cb(BT_N),
     outlet_rows=length(m.outlet_rows), high_angle=length(m.high_angle), corner_cells=length(m.corner_cells),
     jlines=length(m.jl_ptr) - 1, ilines=length(m.il_ptr) - 1)
end

"""
    parent_cells(fine::PBMesh, coarse::PBMesh) -> Vector{Int32}

For nested levels (`fine` one bilinear subdivision of `coarse`): the coarse cell holding each fine cell.
"""
function parent_cells(fine::PBMesh, coarse::PBMesh)
    fine.family == coarse.family && fine.level == coarse.level + 1 && fine.nblk == coarse.nblk &&
        all(fine.blk_ni[k] == 2coarse.blk_ni[k] && fine.blk_nj[k] == 2coarse.blk_nj[k] for k in 1:fine.nblk) ||
        throw(ArgumentError("parent_cells needs a fine mesh one subdivision of the coarse mesh"))
    blk = _host(fine.cblk); ci = _host(fine.ci); cj = _host(fine.cj)
    Int32[cell_index(coarse, blk[k], cld(ci[k], 2), cld(cj[k], 2)) for k in 1:fine.nc]
end

"""
    special_cells(m::PBMesh) -> BitVector

The cells the MMS norms report with and without (plan 10.1): those touching a cut, interface, outlet-row or high-angle
face, a foil, base or outlet face, and the trailing-edge corner cells.
"""
function special_cells(m::PBMesh)
    s = falses(m.nc); own = _host(m.own); nei = _host(m.nei); bown = _host(m.bown)
    for f in vcat(_host(m.cut), _host(m.if_lower), _host(m.if_upper), _host(m.outlet_rows), _host(m.high_angle))
        s[own[f]] = true; s[nei[f]] = true
    end
    for q in vcat(_host(m.foil), _host(m.base), _host(m.outlets)); s[bown[q]] = true; end
    s[_host(m.corner_cells)] .= true
    s
end
