# ============================================================================================
# CMESH2D metrics (A0, 2026-07-14 — memory/project_docs/2026-07-14_CMESH2D_solver_plan.md §1)
# FP64 host precompute from the mesh nodes, FP32 storage by default; optional scalar_type=Float64 retains original precision. SHARED face-normal storage: each face
# normal computed ONCE and read by both adjacent cells — the discrete GCL (Σ S_outward = 0 per
# cell) then holds to FP32-storage roundoff by construction (per-cell recomputation breaks
# freestream preservation via FP32 rounding — plan §1 "non-negotiable").
#
# Conventions (interior-sized; ghosts carry STATE only):
#   i-faces: between cells (i-1,j) and (i,j), array (NI+1)×NJ, normal points +i:
#            S_i = ( Y[i,j+1]-Y[i,j],  -(X[i,j+1]-X[i,j]) )
#   j-faces: between cells (i,j-1) and (i,j), array NI×(NJ+1), normal points +j:
#            S_j = ( -(Y[i+1,j]-Y[i,j]),  X[i+1,j]-X[i,j] )
#   vol: quad area 0.5·|d1×d2| (cross-diagonals); xc,yc cell centers (BC/vortex physical coords);
#   wdist: true min distance to the airfoil surface polyline (SST F1/F2 + trip);
#   gammat: trip intermittency (laminar ahead of the trip station, per surface side).
# ============================================================================================

module CMeshMetrics

export CMetrics, build_metrics, build_strip_metrics, gcl_check

import ..CMeshGen: CMesh, ChannelMesh, StripBlock, OUTER_FARFIELD, OUTER_INLET,
                   OUTER_LOWER_WALL, OUTER_UPPER_WALL, OUTER_STRIP_INTERFACE

struct CMetrics{T<:AbstractFloat}
    NI::Int; NJ::Int; NIW::Int; NIF::Int
    Six::Matrix{T}; Siy::Matrix{T}   # (NI+1) x NJ
    Sjx::Matrix{T}; Sjy::Matrix{T}   # NI x (NJ+1)
    vol::Matrix{T}                          # NI x NJ
    xc::Matrix{T};  yc::Matrix{T}     # NI x NJ
    wdist::Matrix{T}                        # NI x NJ
    gammat::Matrix{T}                       # NI x NJ trip intermittency
    outer_face_kind::Vector{UInt8}                # NI outer j-face tags (0 = legacy far field)
    channel_provenance::Bool                      # true only for validated ChannelMesh geometry
end

# `wall_segments` replaces the foil-range wall polyline (the strip block: its wall is the base segment); `extra_segments`
# are appended to it (the C block of a strip mesh: the base segment CL-CU its open foil range no longer closes). Each
# segment is a tuple (ax, ay, bx, by). Both default to nothing: the foil-range loop of every other mesh, statement for
# statement (strip topology, 5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 4.5).
function _build_metrics(X::Matrix{Float64}, Y::Matrix{Float64}, NIW::Int, NIF::Int,
                        kinds::Vector{UInt8}; channel_provenance::Bool=false,
                        trip_upper::Float64=0.03,
                        trip_lower::Float64=0.05,
                        trip_side::Symbol=:y_sign,
                        wall_segments::Union{Nothing,Vector{NTuple{4,Float64}}}=nothing,
                        extra_segments::Union{Nothing,Vector{NTuple{4,Float64}}}=nothing,
                        scalar_type::Type{T}=Float32) where {T<:AbstractFloat}
    T in (Float32,Float64) || throw(ArgumentError("metric scalar_type must be Float32 or Float64"))
    NI = size(X,1)-1; NJ = size(X,2)-1
    length(kinds) == NI || throw(ArgumentError(
        "outer_face_kind has length $(length(kinds)); expected one tag for each of $NI outer faces"))
    Six = zeros(T, NI+1, NJ); Siy = zeros(T, NI+1, NJ)
    Sjx = zeros(T, NI, NJ+1); Sjy = zeros(T, NI, NJ+1)
    vol = zeros(T, NI, NJ);   xc  = zeros(T, NI, NJ); yc = zeros(T, NI, NJ)
    for j in 1:NJ, i in 1:NI+1                                   # i-faces (FP64 arithmetic, FP32 store)
        Six[i,j] =  T(Y[i,j+1] - Y[i,j])
        Siy[i,j] = -T(X[i,j+1] - X[i,j])
    end
    for j in 1:NJ+1, i in 1:NI                                   # j-faces
        Sjx[i,j] = -T(Y[i+1,j] - Y[i,j])
        Sjy[i,j] =  T(X[i+1,j] - X[i,j])
    end
    # ⛔⛔ NaN IS NOT A NEGATIVE AREA, AND THIS GATE COULD NOT SEE IT (fixed 2026-08-05).
    # The gate was `a <= 0 && (nneg += 1)` followed by `@assert nneg == 0`. In IEEE arithmetic
    # `NaN <= 0` is FALSE -- MEASURED, not assumed -- so a cell whose area came out NaN did not
    # increment the counter and the assertion PASSED. A mesh containing thousands of NaN coordinates
    # therefore cleared the fold gate, and the solver then marched on it.
    #
    # WHY THAT MATTERED MORE THAN IT LOOKS. The same `x > tol && error(...)` shape guards the
    # GLOBAL-CLOCK RECEIPT in three dynamics tools, and `NaN > tol` is false too -- so a corrupt
    # state printed "GLOBAL CLOCK ENGAGED: dtf spread hi/lo = NaN" and the tool went on to emit a
    # dynamics verdict. That receipt exists BECAUSE a movie once marched on local time stepping while
    # printing "this IS physical time", voiding two dynamics verdicts. A guard that passes on NaN does
    # not close that hole. All four sites are fixed together; this is the mesh-side one.
    #
    # ⚠ ADD-ONLY BY CONSTRUCTION: this counts, it does not alter `vol`, `xc`, `yc` or any returned
    # array. Verified rather than asserted -- tools/_attic/retired_verifiers/verify_metrics_inert.jl hashes all ten returned
    # arrays for the production mesh and reports them bitwise unchanged.
    nneg = 0; nnan = 0
    for j in 1:NJ, i in 1:NI
        d1x = X[i+1,j+1]-X[i,j]; d1y = Y[i+1,j+1]-Y[i,j]
        d2x = X[i,j+1]-X[i+1,j]; d2y = Y[i,j+1]-Y[i+1,j]
        a = 0.5*(d1x*d2y - d1y*d2x)
        isnan(a) ? (nnan += 1) : (a <= 0 && (nneg += 1))
        vol[i,j] = T(a)
        xc[i,j] = T(0.25*(X[i,j]+X[i+1,j]+X[i,j+1]+X[i+1,j+1]))
        yc[i,j] = T(0.25*(Y[i,j]+Y[i+1,j]+Y[i,j+1]+Y[i+1,j+1]))
    end
    @assert nnan == 0 "build_metrics: $nnan cell volumes are NaN -- the mesh contains non-finite " *
                      "coordinates. This is NOT a fold; it is a corrupt generator output, and the " *
                      "old `a <= 0` test could not see it (NaN <= 0 is false)."
    @assert nneg == 0 "build_metrics: $nneg non-positive cell volumes (mesh gate should have caught this)"

    # wall distance: brute-force min over the NIF+1 wall SEGMENTS (point-segment distance), once. The segment list is the
    # foil range's NIF segments in index order (or `wall_segments` in their place) plus `extra_segments`; on the default
    # path it holds exactly the NIF (ax, ay, bx, by) values the loop used to read from wx/wy, in the same order, so dmin is
    # the same Float64.
    wdist = fill(T(Inf), NI, NJ)
    wx = [X[NIW+k, 1] for k in 1:NIF+1]; wy = [Y[NIW+k, 1] for k in 1:NIF+1]
    segs = wall_segments === nothing ? [(wx[k], wy[k], wx[k+1], wy[k+1]) for k in 1:NIF] : copy(wall_segments)
    extra_segments === nothing || append!(segs, extra_segments)
    for j in 1:NJ, i in 1:NI
        px = Float64(xc[i,j]); py = Float64(yc[i,j]); dmin = Inf
        for (ax, ay, bx, by) in segs
            ex = bx-ax; ey = by-ay
            t = clamp(((px-ax)*ex + (py-ay)*ey) / max(ex*ex+ey*ey, 1e-30), 0.0, 1.0)
            d = hypot(px-(ax+t*ex), py-(ay+t*ey))
            d < dmin && (dmin = d)
        end
        wdist[i,j] = T(dmin)
    end

    # Trip intermittency: per-column chord station, laminar (0) ahead of the trip,
    # then a 2%c ramp. The historical y-sign rule remains the default for
    # existing solver cases. The optional wall-order rule follows the foil
    # from lower TE to LE to upper TE, so cambered/translated/rotated sections
    # cannot accidentally exchange the upper and lower trip locations.
    trip_side in (:y_sign, :wall_order) || throw(ArgumentError(
        "trip_side must be :y_sign or :wall_order (got $trip_side)"))
    le_node = 0
    lex = ley = chordx = chordy = chord2 = 0.0
    if trip_side === :wall_order
        first_wall = NIW+1; last_wall = NIW+NIF+1
        tex = 0.5*(X[first_wall,1]+X[last_wall,1])
        tey = 0.5*(Y[first_wall,1]+Y[last_wall,1])
        le_node = argmax(i -> (X[i,1]-tex)^2 + (Y[i,1]-tey)^2,
                         first_wall:last_wall)
        first_wall < le_node < last_wall || throw(ArgumentError(
            "wall-order trip assignment requires a leading-edge node inside the foil range"))
        lex = X[le_node,1]; ley = Y[le_node,1]
        chordx = tex-lex; chordy = tey-ley
        chord2 = chordx^2+chordy^2
        isfinite(chord2) && chord2 > 0 || throw(ArgumentError(
            "wall-order trip assignment requires a finite, nonzero chord"))
    end
    gammat = ones(T, NI, NJ)
    for i in NIW+1:NIW+NIF
        xw = 0.5*(X[i,1] + X[i+1,1]); yw = 0.5*(Y[i,1] + Y[i+1,1])
        xtrip = trip_side === :wall_order ?
            (i < le_node ? trip_lower : trip_upper) :
            (yw >= 0 ? trip_upper : trip_lower)
        # Preserve the legacy raw-x station by default. Wall-order mode uses
        # the LE-to-TE projection, which is invariant to rigid-body rotation.
        xstation = trip_side === :wall_order ?
            ((xw-lex)*chordx+(yw-ley)*chordy)/chord2 : xw
        gam = T(clamp((xstation - xtrip)/0.02, 0.0, 1.0))
        for j in 1:NJ; gammat[i,j] = gam; end
    end

    CMetrics(NI, NJ, NIW, NIF, Six, Siy, Sjx, Sjy, vol, xc, yc, wdist, gammat, kinds,
             channel_provenance)
end

"""
build_metrics(X, Y, NIW, NIF; trip_upper, trip_lower) -> CMetrics
X,Y: node coords (NI+1)x(NJ+1) (Float64 from the generator).

The raw-matrix overload is deliberately legacy-only and always produces an all-far-field mask. A
channel mask is accepted only through `build_metrics(::ChannelMesh)`, where its geometry and tag
contract can be validated. This prevents a mask from being attached to an unrelated legacy mesh.
"""
function build_metrics(X::Matrix{Float64}, Y::Matrix{Float64}, NIW::Int, NIF::Int;
                       trip_upper::Float64=0.03, trip_lower::Float64=0.05,
                       trip_side::Symbol=:y_sign, scalar_type::Type{T}=Float32) where {T<:AbstractFloat}
    NI = size(X,1)-1
    _build_metrics(X, Y, NIW, NIF, fill(OUTER_FARFIELD, NI);
                   trip_upper=trip_upper, trip_lower=trip_lower, trip_side=trip_side, scalar_type=scalar_type)
end

"""Build legacy metrics from a mesh object; all outer j-faces retain far-field semantics."""
function build_metrics(mesh::CMesh; trip_upper::Float64=0.03, trip_lower::Float64=0.05,
                       trip_side::Symbol=:y_sign, scalar_type::Type{T}=Float32) where {T<:AbstractFloat}
    build_metrics(mesh.X, mesh.Y, mesh.NIW, mesh.NIF;
                  trip_upper=trip_upper, trip_lower=trip_lower, trip_side=trip_side, scalar_type=scalar_type)
end

function _validate_channel_mask(mesh::ChannelMesh)
    NI, NJ = mesh.NI, mesh.NJ
    kinds = mesh.outer_face_kind
    length(kinds) == NI || throw(ArgumentError(
        "ChannelMesh outer_face_kind has length $(length(kinds)); expected $NI"))
    all(k -> k == OUTER_INLET || k == OUTER_LOWER_WALL || k == OUTER_UPPER_WALL, kinds) ||
        throw(ArgumentError("ChannelMesh mask must contain only inlet/lower-wall/upper-wall tags"))
    first(kinds) == OUTER_LOWER_WALL || throw(ArgumentError(
        "ChannelMesh outer mask must start on the lower wall"))
    last(kinds) == OUTER_UPPER_WALL || throw(ArgumentError(
        "ChannelMesh outer mask must end on the upper wall"))
    count(==(OUTER_INLET), kinds) > 0 || throw(ArgumentError(
        "ChannelMesh outer mask has no inlet faces"))
    rank(k) = k == OUTER_LOWER_WALL ? 1 : k == OUTER_INLET ? 2 : 3
    all(rank(kinds[i]) <= rank(kinds[i+1]) for i in 1:NI-1) || throw(ArgumentError(
        "ChannelMesh outer mask must be one contiguous lower -> inlet -> upper sequence"))

    ch = mesh.channel
    esx, esy = ch.streamwise; enx, eny = -esy, esx
    px, py = ch.ref; h = 0.5*ch.height
    coord(x, y) = (ch.xi_ref + (x-px)*esx + (y-py)*esy,
                   (x-px)*enx + (y-py)*eny)
    scale = max(1.0, ch.height, abs(ch.xi_in), abs(ch.xi_out))
    tol = 4096eps(Float64)*scale
    for i in 1:NI
        xi1, eta1 = coord(mesh.X[i,NJ+1], mesh.Y[i,NJ+1])
        xi2, eta2 = coord(mesh.X[i+1,NJ+1], mesh.Y[i+1,NJ+1])
        k = kinds[i]
        if k == OUTER_LOWER_WALL
            max(abs(eta1+h), abs(eta2+h)) <= tol || throw(ArgumentError(
                "ChannelMesh face $i is tagged lower-wall but is not on eta=-H/2"))
        elseif k == OUTER_INLET
            max(abs(xi1-ch.xi_in), abs(xi2-ch.xi_in)) <= tol || throw(ArgumentError(
                "ChannelMesh face $i is tagged inlet but is not on xi=xi_in"))
        else
            max(abs(eta1-h), abs(eta2-h)) <= tol || throw(ArgumentError(
                "ChannelMesh face $i is tagged upper-wall but is not on eta=+H/2"))
        end
    end
    nothing
end

"The base segment CL-CU of a strip mesh, from the C block's corner nodes (the strip's column 1 holds the same bits, P1')."
_base_segment(mesh::ChannelMesh) = (mesh.X[mesh.NIW+1, 1], mesh.Y[mesh.NIW+1, 1],
                                    mesh.X[mesh.NIW+mesh.NIF+1, 1], mesh.Y[mesh.NIW+mesh.NIF+1, 1])

"""Build metrics from a channel mesh after validating its geometry/tag contract. With the strip topology the C block's
wall polyline (its open surface) gains the base segment CL-CU, the strip's wall, so the leg cells beside the base see it
(plan section 4.5); without a strip the call is today's, statement for statement."""
function build_metrics(mesh::ChannelMesh; trip_upper::Float64=0.03, trip_lower::Float64=0.05,
                       trip_side::Symbol=:y_sign, scalar_type::Type{T}=Float32) where {T<:AbstractFloat}
    _validate_channel_mask(mesh)
    _build_metrics(mesh.X, mesh.Y, mesh.NIW, mesh.NIF, copy(mesh.outer_face_kind);
                   channel_provenance=true,
                   trip_upper=trip_upper, trip_lower=trip_lower, trip_side=trip_side, scalar_type=scalar_type,
                   extra_segments=mesh.strip === nothing ? nothing : [_base_segment(mesh)])
end

"""
    build_strip_metrics(mesh::ChannelMesh) -> CMetrics

The strip block's metrics (strip topology, plan section 6.2 item 7): the same face vectors, volumes and centroids as the
C block's, on the strip's (NI+1) x (NJ+1) nodes, with `NIW = NI` (every column is a wake column: the solver's index tests
then call every j = 1 face an interface and no face a foil wall), `NIF = 0`, every outer tag `OUTER_STRIP_INTERFACE` (the
j = NJ+1 faces are the upper interface, never a channel face, so the channel mask contract is not applied), channel
provenance, the wall distance to the base segment (the strip's column 1, the nearest wall of every strip cell) and
`gammat` all ones (no foil column). Refused on a mesh without a strip block.
"""
function build_strip_metrics(mesh::ChannelMesh; trip_upper::Float64=0.03, trip_lower::Float64=0.05, scalar_type::Type{T}=Float32) where {T<:AbstractFloat}
    s = mesh.strip
    s === nothing && throw(ArgumentError("build_strip_metrics needs a strip mesh (te_topology=:strip); this mesh has no strip block"))
    _build_metrics(s.X, s.Y, s.NI, 0, fill(OUTER_STRIP_INTERFACE, s.NI);
                   channel_provenance=true, trip_upper=trip_upper, trip_lower=trip_lower, scalar_type=scalar_type,
                   wall_segments=[_base_segment(mesh)])
end

"""
gcl_check(m) -> (max_closure, mean_closure)
Discrete GCL: for every cell, Σ outward face normals = 0 exactly (up to FP32 storage roundoff),
normalized by the cell's face-length scale. THE metric-correctness gate precursor: a uniform flow
through closed faces produces zero net flux only if this holds.
"""
function gcl_check(m::CMetrics)
    worst = 0.0; s = 0.0
    for j in 1:m.NJ, i in 1:m.NI
        sx = Float64(m.Six[i+1,j]) - Float64(m.Six[i,j]) + Float64(m.Sjx[i,j+1]) - Float64(m.Sjx[i,j])
        sy = Float64(m.Siy[i+1,j]) - Float64(m.Siy[i,j]) + Float64(m.Sjy[i,j+1]) - Float64(m.Sjy[i,j])
        ref = abs(m.Six[i,j]) + abs(m.Siy[i,j]) + abs(m.Sjx[i,j]) + abs(m.Sjy[i,j]) + 1e-30
        c = hypot(sx, sy)/ref
        worst = max(worst, c); s += c
    end
    (worst, s/(m.NI*m.NJ))
end

end # module
