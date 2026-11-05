# The mesh generator: DESIGN.md section 4. From placed triangles and a MeshSpec to an OctMesh.
#   1. refinement marks per level (dense bit masks): surface bands, local surface levels, wake, regions
#   2. balance and buffers (fine to coarse dilation by 2), domain-face margin, sibling closure
#   3. octs per level, ghost octs around refined regions, boundary octs around the root grid, neighbour tables
#   4. classification by a flood fill over uncut links from the inlet face; ghost slots by ray parity
#   5. cut links (Bouzidi q and triangle id), ghost donors and weights, restriction lists
#   6. counts, reports and checks
using Base.Threads

struct Region
    kind::Symbol                 # :box, :sphere, :cylinder
    level::Int
    lo::V3; hi::V3               # box
    centre::V3; radius::Float64  # sphere; cylinder: axis from lo to hi, radius
end
Region(kind::Symbol, level::Int; lo=(0.0, 0.0, 0.0), hi=(0.0, 0.0, 0.0), centre=(0.0, 0.0, 0.0), radius=0.0) = Region(kind, level, lo, hi, centre, radius)

struct SurfaceLevel
    level::Int
    lo::V3; hi::V3
end

Base.@kwdef struct MeshSpec
    domain::V3                        # requested size; the root grid rounds it up to even cell counts
    periodic::NTuple{3,Bool} = (false, false, false)
    levels::Int                       # L, the finest level of the mesh
    dx_finest::Float64
    surface_level::Int = levels       # level of the surface band everywhere (<= levels)
    band_fine::Int = 8                # band of the surface level, in its cells
    band::Int = 6                     # band of each coarser level, in its cells
    wake_length::Float64 = 0.0        # in units of the body's x extent; 0 disables
    wake_level_offset::Int = 1        # wake level = surface_level - offset
    wake_margin::Float64 = 0.1        # lateral and vertical margin, in units of the body extent
    regions::Vector{Region} = Region[]
    surface_levels::Vector{SurfaceLevel} = SurfaceLevel[]
    faces::NTuple{6,Symbol} = (:inlet, :outlet, :slip, :slip, :slip, :slip)   # -x +x -y +y -z +z
    seal_cavities::Bool = false       # keep the fluid cells inside the surface by parity out of the flood fill (leaky models)
end

# ---------------------------------------------------------------- helpers on dense masks
function chunk_ranges(n::Int, nt::Int)
    r = Vector{UnitRange{Int}}(undef, nt); base, rem = divrem(n, nt); start = 1
    for t in 1:nt
        len = base + (t <= rem ? 1 : 0); r[t] = start:(start + len - 1); start += len
    end
    r
end

"Separable Chebyshev dilation by r cells (periodic axes wrap)."
function dilate!(M::BitArray{3}, r::Int, periodic::NTuple{3,Bool})
    nx, ny, nz = size(M)
    for axis in 1:3
        idx = findall(M)
        n = (nx, ny, nz)[axis]
        for I in idx
            i, j, k = Tuple(I)
            for s in -r:r
                s == 0 && continue
                v = (i, j, k)[axis] + s
                if periodic[axis]
                    v = mod1(v, n)
                elseif v < 1 || v > n
                    continue
                end
                axis == 1 ? (M[v, j, k] = true) : axis == 2 ? (M[i, v, k] = true) : (M[i, j, v] = true)
            end
        end
    end
    M
end

function parent_mask(R::BitArray{3})
    P = falses(size(R, 1) ÷ 2, size(R, 2) ÷ 2, size(R, 3) ÷ 2)
    for I in findall(R)
        i, j, k = Tuple(I); P[(i + 1) ÷ 2, (j + 1) ÷ 2, (k + 1) ÷ 2] = true
    end
    P
end
"Sibling closure: every child of a refined parent."
function children_mask(P::BitArray{3})
    R = falses(2size(P, 1), 2size(P, 2), 2size(P, 3))
    for I in findall(P)
        i, j, k = Tuple(I)
        for dk in 0:1, dj in 0:1, di in 0:1
            R[2i - 1 + di, 2j - 1 + dj, 2k - 1 + dk] = true
        end
    end
    R
end

@inline in_box(p::V3, lo::V3, hi::V3) = lo[1] <= p[1] <= hi[1] && lo[2] <= p[2] <= hi[2] && lo[3] <= p[3] <= hi[3]
function in_region(p::V3, r::Region)
    r.kind == :box && return in_box(p, r.lo, r.hi)
    r.kind == :sphere && return vnorm2(vsub(p, r.centre)) <= r.radius^2
    # cylinder: axis lo -> hi
    a = vsub(r.hi, r.lo); l2 = vnorm2(a); l2 == 0 && return false
    t = vdot(vsub(p, r.lo), a) / l2; (t < 0 || t > 1) && return false
    vnorm2(vsub(vsub(p, r.lo), vmul(a, t))) <= r.radius^2
end

# ---------------------------------------------------------------- the generator
"""
Build the mesh. `tri` are the placed triangles (already translated by `offset`). Returns an OctMesh whose `counts`
holds the summary, reports and timings.
"""
function generate(tri_input::Vector{Tri}, spec::MeshSpec; offset::V3=(0.0, 0.0, 0.0), config_sha::String="", stl_sha::String="")
    t_start = time()
    L = spec.levels
    1 <= spec.surface_level <= L || error("surface_level must be in 1..levels")
    dxs = [spec.dx_finest * 2.0^(L - l) for l in 1:L]
    # a case without geometry (the periodic-box tests): one triangle far outside answers every query with "far"
    no_geometry = isempty(tri_input)
    far = 1e6 * (1 + maximum(spec.domain))
    tri_orig = no_geometry ? Tri[((far, far, far), (far + 1, far, far), (far, far + 1, far))] : tri_input
    # periodic axes: the surface near a periodic face is replicated across it for the queries, so that cells near
    # the wrapped side see it; the mesh keeps the original triangles and their order, cut links map back to them.
    # A body that spans the whole period (an extruded section) is periodic by construction: its overhang beyond
    # the domain (end caps included) is dropped, not wrapped. A body shorter than the period (a sphere on the face)
    # is wrapped whole.
    tri = copy(tri_orig); tri_src = collect(1:length(tri_orig))
    overhang_dropped = 0
    for a in 1:3
        spec.periodic[a] || continue
        blo_a = minimum(min(T[1][a], T[2][a], T[3][a]) for T in tri); bhi_a = maximum(max(T[1][a], T[2][a], T[3][a]) for T in tri)
        if bhi_a - blo_a >= spec.domain[a] - 1e-9
            keep = [max(T[1][a], T[2][a], T[3][a]) > 0 && min(T[1][a], T[2][a], T[3][a]) < spec.domain[a] for T in tri]
            overhang_dropped += count(!, keep)
            tri = tri[keep]; tri_src = tri_src[keep]
        end
        margin = spec.band_fine * dxs[L] + sum((spec.band * dxs[m] for m in 2:L-1); init=0.0) + 4dxs[1]
        extra = Tri[]; src = Int[]
        for (t, T) in enumerate(tri)
            lo = min(T[1][a], T[2][a], T[3][a]); hi = max(T[1][a], T[2][a], T[3][a])
            sh = ntuple(b -> b == a ? spec.domain[a] : 0.0, 3)
            lo < margin && (push!(extra, (vadd(T[1], sh), vadd(T[2], sh), vadd(T[3], sh))); push!(src, tri_src[t]))
            hi > spec.domain[a] - margin && (push!(extra, (vsub(T[1], sh), vsub(T[2], sh), vsub(T[3], sh))); push!(src, tri_src[t]))
        end
        append!(tri, extra); append!(tri_src, src)
    end
    bvh = build_bvh(tri)
    blo, bhi = bounds(tri_orig)
    normals = [tri_normal_area(t)[1] for t in tri]
    areas = [tri_normal_area(t)[2] for t in tri_orig]
    # surface orientation: +1 when the triangle normals point into the fluid, -1 when into the body, 0 when the
    # surface is open (a plate: both sides fluid, no inside). Both sides of sampled triangles are probed one fine
    # cell away: a closed surface has exactly one inside side, an open one has none.
    orient = let st = new_stack(), n = min(length(tri_orig), 2000), step = max(1, length(tri_orig) ÷ n), pos = 0, neg = 0, open = 0
        for t in 1:step:length(tri_orig)
            c = vmul(vadd(vadd(tri_orig[t][1], tri_orig[t][2]), tri_orig[t][3]), 1 / 3)
            a = is_inside(bvh, vadd(c, vmul(normals[t], spec.dx_finest)), st)
            b = is_inside(bvh, vsub(c, vmul(normals[t], spec.dx_finest)), st)
            if a && !b
                neg += 1          # the normal points into the body
            elseif b && !a
                pos += 1          # the normal points into the fluid
            else
                open += 1
            end
        end
        total = max(pos + neg + open, 1)
        open / total > 0.5 ? 0 : pos >= neg ? 1 : -1
    end
    inner_band = 1.0        # cells of refinement kept inside the body beyond the surface (what the gather reads; the sibling closure adds up to one more)
    # root grid, even counts; periodic axes must divide exactly
    n1 = ntuple(3) do a
        c = spec.domain[a] / dxs[1]
        if spec.periodic[a]
            round(Int, c) ≈ c && isinteger(round(Int, c) / 2) || error("periodic axis $a: the domain ($(spec.domain[a]) m) must be an even number of coarse cells ($(dxs[1]) m)")
            round(Int, c)
        else
            v = ceil(Int, c - 1e-9); v + (v % 2)
        end
    end
    domain = ntuple(a -> n1[a] * dxs[1], 3)
    sizes = [ntuple(a -> n1[a] * 2^(l - 1), 3) for l in 1:L]
    nt = nthreads()
    stacks = [new_stack() for _ in 1:nt]
    timings = Dict{String,Float64}()
    reports = Dict{String,Any}()

    # ---- 1. marks
    t0 = time()
    # band distance of level l when the local surface level is Ls: d_l = b_{Ls} dx_{Ls} + sum_{m=l}^{Ls-1} b_m dx_m
    band_distance(l, Ls) = l > Ls ? -1.0 : spec.band_fine * dxs[Ls] + sum((spec.band * dxs[m] for m in l:Ls-1); init=0.0)
    local_surface_level(p::V3) = max(spec.surface_level, maximum((s.level for s in spec.surface_levels if in_box(p, s.lo, s.hi)); init=0))
    R = Vector{BitArray{3}}(undef, L)
    R[1] = trues(sizes[1]...)
    halfdiag = [sqrt(3) * dxs[l] / 2 for l in 1:L]
    nquery = zeros(Int, nt)
    ex = vsub(bhi, blo)
    wake_level = spec.wake_length > 0 ? spec.surface_level - spec.wake_level_offset : 0
    for l in 2:L
        R[l] = falses(sizes[l]...)
        cand = findall(R[l - 1]); ranges = chunk_ranges(length(cand), nt)
        lists = [Int[] for _ in 1:nt]
        dxp = dxs[l - 1]; dxc = dxs[l]; hdc = halfdiag[l]
        dmax = band_distance(l, L)        # the largest band distance any local surface level can give
        @threads :static for t in 1:nt
            st = stacks[t]; lst = lists[t]
            for ci in ranges[t]
                i, j, k = Tuple(cand[ci])
                pc = ((i - 0.5) * dxp, (j - 0.5) * dxp, (k - 0.5) * dxp)
                dmax < 0 && continue
                dp = nearest_distance(bvh, pc, st; bound=dmax + 2hdc + dxp); nquery[t] += 1
                dp > dmax + 2hdc && continue
                for ck in 0:1, cj in 0:1, cci in 0:1
                    ii = 2i - 1 + cci; jj = 2j - 1 + cj; kk = 2k - 1 + ck
                    cc = ((ii - 0.5) * dxc, (jj - 0.5) * dxc, (kk - 0.5) * dxc)
                    dl = band_distance(l, local_surface_level(cc))
                    dl < 0 && continue
                    dc, ti, qn = nearest(bvh, cc, st; bound=dl + hdc + dxc); nquery[t] += 1
                    dc - hdc <= dl || continue
                    # one-sided band: deeper than `inner_band` cells inside the body, nothing is refined
                    if orient != 0 && ti > 0
                        side = orient * vdot(vsub(cc, qn), normals[ti])
                        side < -(inner_band * dxc + hdc) && continue
                    end
                    push!(lst, LinearIndices(sizes[l])[ii, jj, kk])
                end
            end
        end
        for lst in lists, idx in lst
            R[l][idx] = true
        end
        # wake box
        if wake_level >= l
            x0 = bhi[1]; x1 = min(domain[1], bhi[1] + spec.wake_length * ex[1])
            y0 = max(0.0, blo[2] - spec.wake_margin * ex[2]); y1 = min(domain[2], bhi[2] + spec.wake_margin * ex[2])
            z0 = max(0.0, blo[3] - spec.wake_margin * ex[3]); z1 = min(domain[3], bhi[3] + spec.wake_margin * ex[3])
            for k in max(1, floor(Int, z0 / dxc) + 1):min(sizes[l][3], ceil(Int, z1 / dxc)),
                j in max(1, floor(Int, y0 / dxc) + 1):min(sizes[l][2], ceil(Int, y1 / dxc)),
                i in max(1, floor(Int, x0 / dxc) + 1):min(sizes[l][1], ceil(Int, x1 / dxc))
                R[l][i, j, k] = true
            end
        end
        # user regions (over their bounding boxes)
        for r in spec.regions
            r.level >= l || continue
            lo = r.kind == :box ? r.lo : r.kind == :sphere ? vsub(r.centre, (r.radius, r.radius, r.radius)) : vsub(vmin(r.lo, r.hi), (r.radius, r.radius, r.radius))
            hi = r.kind == :box ? r.hi : r.kind == :sphere ? vadd(r.centre, (r.radius, r.radius, r.radius)) : vadd(vmax(r.lo, r.hi), (r.radius, r.radius, r.radius))
            for k in max(1, floor(Int, lo[3] / dxc) + 1):min(sizes[l][3], ceil(Int, hi[3] / dxc)),
                j in max(1, floor(Int, lo[2] / dxc) + 1):min(sizes[l][2], ceil(Int, hi[2] / dxc)),
                i in max(1, floor(Int, lo[1] / dxc) + 1):min(sizes[l][1], ceil(Int, hi[1] / dxc))
                in_region(((i - 0.5) * dxc, (j - 0.5) * dxc, (k - 0.5) * dxc), r) && (R[l][i, j, k] = true)
            end
        end
    end
    timings["mark"] = time() - t0

    # ---- 2. balance, buffers, domain-face margin, sibling closure
    t0 = time()
    for l in L:-1:2
        # Open/domain-wall faces retain the coarse buffer. Symmetry is a local
        # reflecting condition at every level, so surface refinement reaches it.
        R[l] .&= .!face_margin_mask(sizes[l], l, spec.periodic, spec.faces)
        P = parent_mask(R[l]); dilate!(P, 2, spec.periodic)
        P .&= .!face_margin_mask(sizes[l - 1], l - 1, spec.periodic, spec.faces)
        R[l - 1] .|= P
        R[l] = children_mask(parent_mask(R[l]))
        R[l] .&= .!face_margin_mask(sizes[l], l, spec.periodic, spec.faces)
        R[l] = children_mask(parent_mask(R[l]))
    end
    # (a second pass guarantees the buffer after the sibling closure and the face margin)
    for l in L:-1:2
        P = parent_mask(R[l]); dilate!(P, 2, spec.periodic); P .&= .!face_margin_mask(sizes[l - 1], l - 1, spec.periodic, spec.faces)
        R[l - 1] .|= P
    end
    timings["balance"] = time() - t0

    # ---- 3. octs, ghost octs, boundary octs, neighbour tables
    t0 = time()
    refined = Vector{BitArray{3}}(undef, L)       # refined[l]: cells of level l that are parents of level l+1 octs
    for l in 1:L-1
        refined[l] = parent_mask(R[l + 1])
    end
    refined[L] = falses(1, 1, 1)
    levels = OctLevel[]
    for l in 1:L
        Lv = OctLevel(l, dxs[l], sizes[l])
        origins = Tuple{Int,Int,Int}[]
        if l == 1
            for k in 0:2:sizes[1][3]-1, j in 0:2:sizes[1][2]-1, i in 0:2:sizes[1][1]-1
                push!(origins, (i, j, k))
            end
        else
            for I in findall(refined[l - 1])
                i, j, k = Tuple(I); push!(origins, (2(i - 1), 2(j - 1), 2(k - 1)))
            end
        end
        # ghost octs: neighbour positions of real octs that hold no real oct (inside the domain or wrapped)
        real = Set{Tuple{Int,Int,Int}}(origins)
        ghosts = Set{Tuple{Int,Int,Int}}()
        boundaries = Set{Tuple{Int,Int,Int}}()
        for (ox, oy, oz) in origins, dz in -1:1, dy in -1:1, dx in -1:1
            (dx == 0 && dy == 0 && dz == 0) && continue
            p = (ox + 2dx, oy + 2dy, oz + 2dz)
            pw, inside = wrap_origin(p, sizes[l], spec.periodic)
            if !inside
                # Fine octs may meet a symmetry face. Supply real boundary
                # padding at that level, never an AMR ghost or periodic wrap.
                push!(boundaries, pw)
                continue
            end
            pw in real && continue
            l == 1 && continue                     # every in-domain position of level 1 is a real oct
            push!(ghosts, pw)
        end
        allorigins = vcat(origins, collect(ghosts), collect(boundaries))
        kinds = vcat(fill(UInt8(0), length(origins)), fill(K_GHOST, length(ghosts)), fill(K_BOUNDARY, length(boundaries)))
        order = sortperm([morton((o[1] >> 1) + 1, (o[2] >> 1) + 1, (o[3] >> 1) + 1) for o in allorigins])
        allorigins = allorigins[order]; kinds = kinds[order]
        nocts = length(allorigins)
        Lv.nocts = nocts
        Lv.real = BitVector(kinds .== 0)
        Lv.origin = Matrix{Int32}(undef, 3, nocts); Lv.kind = Matrix{UInt8}(undef, 8, nocts); Lv.face = zeros(UInt8, 8, nocts)
        Lv.onbr = zeros(Int32, 27, nocts); Lv.cutidx = zeros(Int32, 8, nocts)
        for o in 1:nocts
            Lv.origin[:, o] .= allorigins[o]
            Lv.octmap[oct_key(allorigins[o]...)] = Int32(o)
            if kinds[o] == K_BOUNDARY
                Lv.kind[:, o] .= K_BOUNDARY
                for k in 1:8
                    c = (allorigins[o][1] + slot_offset(k)[1], allorigins[o][2] + slot_offset(k)[2], allorigins[o][3] + slot_offset(k)[3])
                    # a bitmask of the domain faces the padding slot lies beyond (an edge or corner slot touches
                    # two or three; the step combines their conditions, DESIGN.md A7)
                    m = UInt8(0)
                    c[1] < 0 && (m |= face_bit(FACE_XMIN)); c[1] >= sizes[l][1] && (m |= face_bit(FACE_XMAX))
                    c[2] < 0 && (m |= face_bit(FACE_YMIN)); c[2] >= sizes[l][2] && (m |= face_bit(FACE_YMAX))
                    c[3] < 0 && (m |= face_bit(FACE_ZMIN)); c[3] >= sizes[l][3] && (m |= face_bit(FACE_ZMAX))
                    Lv.face[k, o] = m
                end
            elseif kinds[o] == K_GHOST
                Lv.kind[:, o] .= K_GHOST
            else
                for k in 1:8
                    c = (allorigins[o][1] + slot_offset(k)[1], allorigins[o][2] + slot_offset(k)[2], allorigins[o][3] + slot_offset(k)[3])
                    Lv.kind[k, o] = (l < L && refined[l][c[1] + 1, c[2] + 1, c[3] + 1]) ? K_COVERED : K_FLUID   # fluid for now; the flood fill decides
                end
            end
        end
        # neighbour tables
        for o in 1:nocts, dz in -1:1, dy in -1:1, dx in -1:1
            p = (Int(Lv.origin[1, o]) + 2dx, Int(Lv.origin[2, o]) + 2dy, Int(Lv.origin[3, o]) + 2dz)
            pw, _ = wrap_origin(p, sizes[l], spec.periodic)
            Lv.onbr[nbr_index(dx, dy, dz), o] = get(Lv.octmap, oct_key(pw...), Int32(0))
        end
        push!(levels, Lv)
    end
    timings["octs"] = time() - t0

    # ---- 4. classification: flood fill over uncut face links from the inlet face
    t0 = time()
    near = [near_surface_mask(levels[l], bvh, stacks, 2.0) for l in 1:L]   # slots within 2 cell diagonals of the surface
    # ---- 4a. the crossings of every link of the near leaf slots, computed once in parallel: q and triangle per
    # direction (q = 1: no crossing). The flood fill and the cut-link pass look them up; a segment is shared by its
    # two cells, so one side's row answers for both.
    nearrow = Vector{Matrix{Int32}}(undef, L); cross_q = Vector{Matrix{Float32}}(undef, L); cross_tri = Vector{Matrix{Int32}}(undef, L)
    for l in 1:L
        Lv = levels[l]
        rows = zeros(Int32, 8, Lv.nocts); slots = Int[]
        for o in 1:Lv.nocts, k in 1:8
            if near[l][k, o] && Lv.kind[k, o] != K_BOUNDARY && Lv.kind[k, o] != K_COVERED
                push!(slots, slot_id(k, o)); rows[k, o] = Int32(length(slots))
            end
        end
        cq = ones(Float32, Q - 1, length(slots)); ct = zeros(Int32, Q - 1, length(slots))
        ranges = chunk_ranges(length(slots), nt)
        @threads :static for t in 1:nt
            st = stacks[t]
            for r in ranges[t]
                pc = slot_centre(Lv, slots[r])
                for i in 2:Q
                    target = (pc[1] + CX[i] * Lv.dx, pc[2] + CY[i] * Lv.dx, pc[3] + CZ[i] * Lv.dx)
                    q, ti = first_crossing(bvh, pc, target, st)
                    cq[i - 1, r] = Float32(q); ct[i - 1, r] = Int32(ti)
                end
            end
        end
        nearrow[l] = rows; cross_q[l] = cq; cross_tri[l] = ct
    end
    timings["crossings"] = time() - t0
    t0 = time()
    # ---- 4b. sealed cavities (spec.seal_cavities, off by default): every fluid slot whose centre lies inside the
    # body's bounding box (one coarse cell of margin) and inside the surface by parity is kept out of the fill, so the
    # interiors of a leaky model (open panels, nested shells) stay solid instead of flooding, ringing at an impulsive
    # start and cutting the skin's triangles from inside. The parity (majority of three axis rays) is fooled only by
    # open shells; the count is reported as `sealed_cells`. (The C212 run of 3 Nov 2026, c212/README.md.)
    sealed = [falses(8, levels[l].nocts) for l in 1:L]
    nsealed = 0
    if spec.seal_cavities && !no_geometry
        mlo = vsub(blo, (dxs[1], dxs[1], dxs[1])); mhi = vadd(bhi, (dxs[1], dxs[1], dxs[1]))
        for l in 1:L
            Lv = levels[l]
            cand = [(o, k) for o in 1:Lv.nocts, k in 1:8 if Lv.kind[k, o] == K_FLUID && in_box(slot_centre(Lv, slot_id(k, o)), mlo, mhi)]
            ranges = chunk_ranges(length(cand), nt); flags = falses(length(cand))
            @threads :static for t in 1:nt
                st = stacks[t]
                for ci in ranges[t]
                    o, k = cand[ci]; flags[ci] = is_inside(bvh, slot_centre(Lv, slot_id(k, o)), st)
                end
            end
            for (ci, (o, k)) in enumerate(cand)
                flags[ci] && (sealed[l][k, o] = true; nsealed += 1)
            end
        end
    end
    reports["sealed_cells"] = nsealed
    fluid_reached = [falses(8, levels[l].nocts) for l in 1:L]
    queue = Tuple{Int,Int}[]                       # (level, slot id)
    inlet_axis, inlet_side = 1, 0                  # -x face
    # the fill starts from the leaves of the x = 0 plane that lie outside the geometry (parity), at any level
    st0 = stacks[1]
    for l in 1:L
        Lv = levels[l]
        for o in 1:Lv.nocts
            Lv.real[o] || continue
            Int(Lv.origin[1, o]) == 0 || continue
            for k in 1:8
                slot_offset(k)[1] == 0 || continue
                Lv.kind[k, o] == K_FLUID || continue
                !no_geometry && is_inside(bvh, slot_centre(Lv, slot_id(k, o)), st0) && continue   # every start cell, near or not
                if !fluid_reached[l][k, o]
                    fluid_reached[l][k, o] = true; push!(queue, (l, slot_id(k, o)))
                end
            end
        end
    end
    isempty(queue) && error("no fluid cell on the inlet face: the geometry covers the inlet or the domain is solid")
    while !isempty(queue)
        (l, s) = pop!(queue)
        Lv = levels[l]; o = slot_oct(s); k = slot_k(s)
        c = slot_cell(Lv, s)
        for (i, e) in ((2, (1, 0, 0)), (3, (-1, 0, 0)), (4, (0, 1, 0)), (5, (0, -1, 0)), (6, (0, 0, 1)), (7, (0, 0, -1)))
            nc = (c[1] + e[1], c[2] + e[2], c[3] + e[3])
            # the same-level neighbour slot through the oct table
            off = slot_offset(k); nx = off[1] + e[1]; ny = off[2] + e[2]; nz = off[3] + e[3]
            dxo = nx < 0 ? -1 : nx > 1 ? 1 : 0; dyo = ny < 0 ? -1 : ny > 1 ? 1 : 0; dzo = nz < 0 ? -1 : nz > 1 ? 1 : 0
            no = Lv.onbr[nbr_index(dxo, dyo, dzo), o]
            no == 0 && continue
            nk = slot_index(nx - 2dxo, ny - 2dyo, nz - 2dzo)
            nkind = Lv.kind[nk, no]
            # a link crossing the surface is cut (precomputed for near slots; the segment is shared by both cells)
            row = nearrow[l][k, o]; rown = nearrow[l][nk, no]
            ((row != 0 && cross_q[l][i - 1, row] < 1.0f0) || (rown != 0 && cross_q[l][OPP[i] - 1, rown] < 1.0f0)) && continue
            if nkind == K_FLUID
                if !fluid_reached[l][nk, no] && !sealed[l][nk, no]
                    fluid_reached[l][nk, no] = true; push!(queue, (l, slot_id(nk, no)))
                end
            elseif nkind == K_COVERED
                # descend to the 4 children on the near face (recursively through covered children)
                push_near_children!(queue, fluid_reached, sealed, levels, l, (Int(Lv.origin[1, no]) + slot_offset(nk)[1], Int(Lv.origin[2, no]) + slot_offset(nk)[2], Int(Lv.origin[3, no]) + slot_offset(nk)[3]), e)
            elseif nkind == K_GHOST && l > 1
                # the neighbour is a coarser leaf: its parent cell at level l-1
                Lp = levels[l - 1]
                ps = find_slot(Lp, nc[1] >> 1, nc[2] >> 1, nc[3] >> 1)
                ps == 0 && continue
                po = slot_oct(ps); pk = slot_k(ps)
                if Lp.kind[pk, po] == K_FLUID && !fluid_reached[l - 1][pk, po] && !sealed[l - 1][pk, po]
                    fluid_reached[l - 1][pk, po] = true; push!(queue, (l - 1, ps))
                end
            end
        end
    end
    nsolid_leaves = 0
    for l in 1:L
        Lv = levels[l]
        for o in 1:Lv.nocts, k in 1:8
            if Lv.kind[k, o] == K_FLUID && !fluid_reached[l][k, o]
                Lv.kind[k, o] = K_SOLID; nsolid_leaves += 1
            end
        end
    end
    # ghost slots: solid when their parent leaf is solid, else by the parity of their centre (every ghost, not only
    # the near ones: a ghost deep inside a body must not keep donors)
    for l in 2:L
        Lv = levels[l]; Lp = levels[l - 1]
        gl = [(o, k) for o in 1:Lv.nocts, k in 1:8 if Lv.kind[k, o] == K_GHOST]
        ranges = chunk_ranges(length(gl), nt); flags = falses(length(gl))
        @threads :static for t in 1:nt
            st = stacks[t]
            for gi in ranges[t]
                o, k = gl[gi]
                c = slot_cell(Lv, slot_id(k, o))
                ps = find_slot(Lp, c[1] >> 1, c[2] >> 1, c[3] >> 1)
                if ps != 0 && Lp.kind[slot_k(ps), slot_oct(ps)] == K_SOLID
                    flags[gi] = true
                else
                    flags[gi] = is_inside(bvh, slot_centre(Lv, slot_id(k, o)), st)
                end
            end
        end
        for (gi, (o, k)) in enumerate(gl)
            flags[gi] && (Lv.kind[k, o] = K_SOLID)
        end
    end
    # covered slots whose children are all solid stay covered (the kinds then partition the volume by leaves), but
    # they carry no state: the link pass treats them as solid neighbours
    covered_solid = [falses(8, levels[l].nocts) for l in 1:L]
    for l in L-1:-1:1
        Lv = levels[l]; Lf = levels[l + 1]
        for o in 1:Lv.nocts, k in 1:8
            Lv.kind[k, o] == K_COVERED || continue
            c = slot_cell(Lv, slot_id(k, o))
            allsolid = true
            for dk in 0:1, dj in 0:1, di in 0:1
                cs = find_slot(Lf, 2c[1] + di, 2c[2] + dj, 2c[3] + dk)
                cs == 0 && continue
                ck = Lf.kind[slot_k(cs), slot_oct(cs)]
                (ck == K_SOLID || (ck == K_COVERED && covered_solid[l + 1][slot_k(cs), slot_oct(cs)])) || (allsolid = false)
            end
            covered_solid[l][k, o] = allsolid
        end
    end
    # leak check: fluid leaves near the surface whose centre is inside by parity
    leaks = Tuple{Int,V3}[]
    for l in 1:L
        Lv = levels[l]
        cand = [(o, k) for o in 1:Lv.nocts, k in 1:8 if Lv.kind[k, o] == K_FLUID && near[l][k, o]]
        ranges = chunk_ranges(length(cand), nt); flags = falses(length(cand))
        @threads :static for t in 1:nt
            st = stacks[t]
            for ci in ranges[t]
                o, k = cand[ci]; flags[ci] = is_inside(bvh, slot_centre(Lv, slot_id(k, o)), st)
            end
        end
        for (ci, (o, k)) in enumerate(cand)
            flags[ci] && length(leaks) < 1000 && push!(leaks, (l, slot_centre(Lv, slot_id(k, o))))
        end
    end
    reports["leak_cells"] = length(leaks)
    reports["leak_examples"] = [Dict("level" => l, "centre_m" => collect(p)) for (l, p) in leaks[1:min(10, length(leaks))]]
    timings["classify"] = time() - t0

    # ---- 5. cut links, ghost donors, restriction lists
    t0 = time()
    missing_partner = 0; cut_links = 0; wall_cells = 0; triangles_hit = falses(length(tri_orig))
    for l in 1:L
        Lv = levels[l]
        cand = [(o, k) for o in 1:Lv.nocts, k in 1:8 if Lv.kind[k, o] == K_FLUID && near[l][k, o]]
        ranges = chunk_ranges(length(cand), nt)
        lists = [Tuple{Int,UInt8,Float32,Int32,Bool}[] for _ in 1:nt]     # (slot, dir, q, tri, partner_ok)
        @threads :static for t in 1:nt
            for ci in ranges[t]
                o, k = cand[ci]; s = slot_id(k, o); off = slot_offset(k)
                row = nearrow[l][k, o]
                for i in 2:Q
                    nx = off[1] + CX[i]; ny = off[2] + CY[i]; nz = off[3] + CZ[i]
                    dxo = nx < 0 ? -1 : nx > 1 ? 1 : 0; dyo = ny < 0 ? -1 : ny > 1 ? 1 : 0; dzo = nz < 0 ? -1 : nz > 1 ? 1 : 0
                    no = Lv.onbr[nbr_index(dxo, dyo, dzo), o]
                    q = Float64(cross_q[l][i - 1, row]); ti = Int(cross_tri[l][i - 1, row])
                    cut = q < 1.0
                    if !cut && no != 0
                        nk = slot_index(nx - 2dxo, ny - 2dyo, nz - 2dzo)
                        # a solid neighbour with no detected crossing (a tangent or a tessellation gap): cut at the half link
                        (Lv.kind[nk, no] == K_SOLID || (Lv.kind[nk, no] == K_COVERED && covered_solid[l][nk, no])) && (cut = true; q = 0.5; ti = 0)
                    elseif !cut && no == 0
                        cut = true; q = 0.5; ti = 0        # no slot there at all: treat as a wall at the half link
                    end
                    cut || continue
                    # the Bouzidi partner x - c_i
                    px = off[1] - CX[i]; py = off[2] - CY[i]; pz = off[3] - CZ[i]
                    pxo = px < 0 ? -1 : px > 1 ? 1 : 0; pyo = py < 0 ? -1 : py > 1 ? 1 : 0; pzo = pz < 0 ? -1 : pz > 1 ? 1 : 0
                    po = Lv.onbr[nbr_index(pxo, pyo, pzo), o]
                    ok = false
                    if po != 0
                        pk = slot_index(px - 2pxo, py - 2pyo, pz - 2pzo)
                        pkind = Lv.kind[pk, po]
                        if pkind == K_FLUID || pkind == K_GHOST || (pkind == K_COVERED && !covered_solid[l][pk, po])
                            ok = cross_q[l][OPP[i] - 1, row] >= 1.0f0
                        end
                    end
                    push!(lists[t], (s, UInt8(i), Float32(max(q, 1e-6)), Int32(ti > 0 ? tri_src[ti] : 0), ok))
                end
            end
        end
        links = sort!(vcat(lists...); by=x -> (x[1], x[2]))
        slots = unique(first.(links))
        Lv.cut_slot = Int32.(slots); Lv.cut_off = zeros(Int32, length(slots) + 1)
        Lv.cut_dir = [x[2] for x in links]; Lv.cut_q = [x[3] for x in links]; Lv.cut_tri = [x[4] for x in links]; Lv.cut_partner = [x[5] for x in links]
        ci = 0
        for (li, x) in enumerate(links)
            if li == 1 || x[1] != links[li - 1][1]
                ci += 1; Lv.cut_off[ci] = li
                Lv.cutidx[slot_k(x[1]), slot_oct(x[1])] = Int32(ci)
            end
            x[4] > 0 && (triangles_hit[x[4]] = true)
            x[5] || (missing_partner += 1)
        end
        Lv.cut_off[end] = length(links) + 1
        cut_links += length(links); wall_cells += length(slots)
    end
    reports["periodic_overhang_triangles_dropped"] = overhang_dropped
    reports["unreached_triangles"] = count(!, triangles_hit)
    reports["unreached_triangle_area_fraction"] = sum(areas[t] for t in eachindex(tri_orig) if !triangles_hit[t]; init=0.0) / max(sum(areas), eps())
    reports["surface_orientation"] = orient
    reports["cut_links"] = cut_links; reports["wall_cells"] = wall_cells; reports["missing_partner_links"] = missing_partner
    # ---- 5a. transfers between levels (DESIGN.md A4: cubic stencils)
    # (1) a covered slot may carry a restricted state when it has a fluid descendant and lies within 2 cells of a
    #     same-level fluid leaf; the lists themselves are built in (3), after the ghost kinds are final
    restricted = [falses(8, levels[l].nocts) for l in 1:L]
    depth1 = [falses(8, levels[l].nocts) for l in 1:L]           # within one cell of a same-level fluid leaf (A8)
    for l in 1:L-1
        Lv = levels[l]
        for o in 1:Lv.nocts, k in 1:8
            (Lv.kind[k, o] == K_COVERED && !covered_solid[l][k, o]) || continue
            c = slot_cell(Lv, slot_id(k, o))
            found = false; near1 = false
            for dk in -2:2, dj in -2:2, di in -2:2
                cc = wrap_cell((c[1] + di, c[2] + dj, c[3] + dk), Lv.n, spec.periodic)
                (cc[1] < 0 || cc[2] < 0 || cc[3] < 0) && continue
                ns = find_slot(Lv, cc...)
                if ns != 0 && Lv.kind[slot_k(ns), slot_oct(ns)] == K_FLUID
                    found = true
                    max(abs(di), abs(dj), abs(dk)) <= 1 && (near1 = true)
                end
            end
            restricted[l][k, o] = found; depth1[l][k, o] = near1
        end
    end
    # the inner ghosts (A8): a ghost slot within one lattice link of a fine fluid slot of its level; these and the
    # outer ghosts that a cubic restriction stencil references are listed, filled and admitted as donors or sources
    # (the step reads nothing beyond one cell); the other slots of a ghost oct keep their kind and are never read
    inner = [falses(8, levels[l].nocts) for l in 1:L]
    for l in 2:L
        Lv = levels[l]
        for o in 1:Lv.nocts, k in 1:8
            Lv.kind[k, o] == K_GHOST || continue
            for j in 2:Q
                so = Lv.onbr[SRC_OFF[k, j], o]
                if so != 0 && Lv.kind[SRC_CHILD[k, j], so] == K_FLUID
                    inner[l][k, o] = true; break
                end
            end
        end
    end
    # (2) restriction lists from the finest level down (a level's sources must be known to be restricted before the
    #     coarser level admits them): the 4x4x4 fine cells around the centre with cubic weights when all are fluid,
    #     ghost or restricted covered slots, else the admissible children averaged; a slot without an admissible
    #     child carries no state and is dropped (its neighbours see it as solid). The ghosts a cubic stencil
    #     references are marked for the ghost lists; a stencil whose ghost loses its donors is demoted in (4a).
    referenced = [falses(8, levels[l].nocts) for l in 1:L]
    for l in L-1:-1:1
        Lv = levels[l]; Lf = levels[l + 1]
        rs = [slot_id(k, o) for o in 1:Lv.nocts, k in 1:8 if restricted[l][k, o]]
        src = zeros(Int32, 64, length(rs)); cubic = falses(length(rs))
        for (ri, s) in enumerate(rs)
            c = slot_cell(Lv, s)
            complete = true
            for dk in 0:3, dj in 0:3, di in 0:3
                cc = wrap_cell((2c[1] - 1 + di, 2c[2] - 1 + dj, 2c[3] - 1 + dk), Lf.n, spec.periodic)
                cs = (cc[1] < 0 || cc[2] < 0 || cc[3] < 0) ? 0 : find_slot(Lf, cc...)
                ok = false
                if cs != 0
                    ck = Lf.kind[slot_k(cs), slot_oct(cs)]
                    ok = ck == K_FLUID || ck == K_GHOST || (ck == K_COVERED && restricted[l + 1][slot_k(cs), slot_oct(cs)])
                end
                ok ? (src[1 + di + 4dj + 16dk, ri] = Int32(cs)) : (complete = false)
            end
            # the cubic form for the slots within one cell of the coarse fluid; a depth-2 slot feeds only the
            # -0.039 node of the ghost stencils and averages its children (A8), listing only them as sources
            cubic[ri] = complete && depth1[l][slot_k(s), slot_oct(s)]
            if !cubic[ri]
                for n in 1:64
                    (n in STENCIL_INNER) || (src[n, ri] = Int32(0))
                end
            end
        end
        keep = [any(src[n, ri] != 0 for n in STENCIL_INNER) for ri in eachindex(rs)]
        for ri in eachindex(rs)
            keep[ri] || (restricted[l][slot_k(rs[ri]), slot_oct(rs[ri])] = false)
        end
        for ri in eachindex(rs)
            (keep[ri] && cubic[ri]) || continue
            for n in 1:64
                cs = src[n, ri]; cs == 0 && continue
                Lf.kind[slot_k(Int(cs)), slot_oct(Int(cs))] == K_GHOST && (referenced[l + 1][slot_k(Int(cs)), slot_oct(Int(cs))] = true)
            end
        end
        reports["restricted_without_fluid_children_level_$l"] = count(!, keep)
        reports["restricted_cubic_level_$l"] = count(cubic[keep])
        reports["restricted_depth1_level_$l"] = count(depth1[l][slot_k(rs[ri]), slot_oct(rs[ri])] for ri in eachindex(rs) if keep[ri])
        Lv.restrict_slot = Int32.(rs[keep]); Lv.restrict_src = src[:, keep]; Lv.restrict_cubic = cubic[keep]
    end
    listed = [inner[l] .| referenced[l] for l in 1:L]
    # (3) ghost donors: the 4x4x4 coarse cells around the ghost centre with cubic Lagrange weights when every node
    #     is a fluid or restricted covered cell (and, for near ghosts, not behind the surface); else the trilinear 8
    #     over the admissible subset (coarse ghosts allowed), renormalised; no admissible donor at all: solid
    donor_slot = function (Lp, c, flags, glist, allow_ghost, isnear, pc, st)
        ci, cj, ck = wrap_cell(c, Lp.n, spec.periodic)
        (ci < 0 || cj < 0 || ck < 0) && return 0
        ds = find_slot(Lp, ci, cj, ck); ds == 0 && return 0
        kd = Lp.kind[slot_k(ds), slot_oct(ds)]
        (kd == K_FLUID || (kd == K_COVERED && flags[slot_k(ds), slot_oct(ds)]) || (allow_ghost && kd == K_GHOST && glist[slot_k(ds), slot_oct(ds)])) || return 0
        if isnear
            qd, _ = first_crossing(bvh, pc, slot_centre(Lp, ds), st)
            qd < 1.0 && return 0
        end
        ds
    end
    set_ghost_donors! = function (Lv, Lp, gi, s, flags, glist, st)
        pc = slot_centre(Lv, s); isnear = near[Lv.level][slot_k(s), slot_oct(s)]
        fx = pc[1] / Lp.dx - 0.5; fy = pc[2] / Lp.dx - 0.5; fz = pc[3] / Lp.dx - 0.5
        i0 = floor(Int, fx); j0 = floor(Int, fy); k0 = floor(Int, fz)
        tx = fx - i0; ty = fy - j0; tz = fz - k0
        wx = cubic_weights(tx); wy = cubic_weights(ty); wz = cubic_weights(tz)
        fill!(view(Lv.ghost_donor, :, gi), Int32(0)); fill!(view(Lv.ghost_w, :, gi), 0f0)
        complete = true
        for dk in 0:3, dj in 0:3, di in 0:3
            ds = donor_slot(Lp, (i0 - 1 + di, j0 - 1 + dj, k0 - 1 + dk), flags, glist, false, isnear, pc, st)
            ds == 0 && (complete = false; break)
            n = 1 + di + 4dj + 16dk
            Lv.ghost_donor[n, gi] = Int32(ds); Lv.ghost_w[n, gi] = Float32(wx[di + 1] * wy[dj + 1] * wz[dk + 1])
        end
        complete && return :cubic
        fill!(view(Lv.ghost_donor, :, gi), Int32(0)); fill!(view(Lv.ghost_w, :, gi), 0f0)
        wsum = 0.0
        for dk in 0:1, dj in 0:1, di in 0:1
            ds = donor_slot(Lp, (i0 + di, j0 + dj, k0 + dk), flags, glist, true, isnear, pc, st)
            ds == 0 && continue
            w = (di == 1 ? tx : 1 - tx) * (dj == 1 ? ty : 1 - ty) * (dk == 1 ? tz : 1 - tz)
            n = 1 + (di + 1) + 4(dj + 1) + 16(dk + 1)
            Lv.ghost_donor[n, gi] = Int32(ds); Lv.ghost_w[n, gi] = Float32(w); wsum += w
        end
        wsum > 0 || return :none
        for n in 1:64; Lv.ghost_w[n, gi] = Float32(Lv.ghost_w[n, gi] / wsum); end
        :trilinear
    end
    for l in 2:L
        Lv = levels[l]; Lp = levels[l - 1]
        gl = [slot_id(k, o) for o in 1:Lv.nocts, k in 1:8 if Lv.kind[k, o] == K_GHOST && listed[l][k, o]]
        Lv.ghost_slot = Int32.(gl); Lv.ghost_donor = zeros(Int32, 64, length(gl)); Lv.ghost_w = zeros(Float32, 64, length(gl))
        modes = fill(:none, length(gl))
        ranges = chunk_ranges(length(gl), nt)
        @threads :static for t in 1:nt
            st = stacks[t]
            for gi in ranges[t]
                modes[gi] = set_ghost_donors!(Lv, Lp, gi, gl[gi], restricted[l - 1], listed[l - 1], st)
            end
        end
        for gi in eachindex(gl)
            modes[gi] == :none && (Lv.kind[slot_k(gl[gi]), slot_oct(gl[gi])] = K_SOLID; listed[l][slot_k(gl[gi]), slot_oct(gl[gi])] = false)
        end
        keep = modes .!= :none
        Lv.ghost_slot = Lv.ghost_slot[keep]; Lv.ghost_donor = Lv.ghost_donor[:, keep]; Lv.ghost_w = Lv.ghost_w[:, keep]
        reports["ghosts_cubic_level_$l"] = count(==(:cubic), modes); reports["ghosts_trilinear_level_$l"] = count(==(:trilinear), modes)
    end
    # (4a) a cubic restriction stencil whose ghost source lost its donors in (3) (solid now, unlisted) averages its
    #      children instead
    for l in 1:L-1
        Lv = levels[l]; Lf = levels[l + 1]; demoted = 0
        for ri in 1:length(Lv.restrict_slot)
            Lv.restrict_cubic[ri] || continue
            bad = false
            for n in 1:64
                cs = Lv.restrict_src[n, ri]; cs == 0 && continue
                ck = Lf.kind[slot_k(Int(cs)), slot_oct(Int(cs))]
                (ck == K_SOLID || (ck == K_GHOST && !listed[l + 1][slot_k(Int(cs)), slot_oct(Int(cs))])) && (bad = true; break)
            end
            bad || continue
            Lv.restrict_cubic[ri] = false; demoted += 1
            for n in 1:64
                (n in STENCIL_INNER) || (Lv.restrict_src[n, ri] = Int32(0))
            end
        end
        reports["restricted_demoted_level_$l"] = demoted
        reports["restricted_cubic_level_$l"] -= demoted
    end
    # (4) a covered ghost donor must be restricted; the drop of (3) touches only slots away from the interfaces (an
    #     inner ghost's stencil ends two coarse cells inside the fine region, which the balance keeps fluid), so
    #     this rebuilds nothing in practice and fails loudly if the invariant ever breaks
    for l in 2:L
        Lv = levels[l]; Lp = levels[l - 1]; rebuilt = 0
        for gi in 1:length(Lv.ghost_slot)
            for n in 1:64
                ds = Lv.ghost_donor[n, gi]; ds == 0 && continue
                dk_ = slot_k(Int(ds)); do_ = slot_oct(Int(ds))
                (Lp.kind[dk_, do_] == K_COVERED && !restricted[l - 1][dk_, do_]) || continue
                rebuilt += 1
                set_ghost_donors!(Lv, Lp, gi, Int(Lv.ghost_slot[gi]), restricted[l - 1], listed[l - 1], stacks[1]) == :none && error("generate: a ghost of level $l lost its donors to the restriction filter")
                break
            end
        end
        reports["ghost_donors_rebuilt_level_$l"] = rebuilt
    end
    timings["links"] = time() - t0

    # ---- 5b. the special-direction masks of the step: boundary sources and cut links
    for l in 1:L
        Lv = levels[l]
        Lv.special = zeros(UInt32, 8, Lv.nocts)
        for o in 1:Lv.nocts
            Lv.real[o] || continue
            for k in 1:8
                Lv.kind[k, o] == K_FLUID || continue
                m = UInt32(0)
                for j in 2:Q
                    so = Lv.onbr[SRC_OFF[k, j], o]
                    (so == 0 || Lv.kind[SRC_CHILD[k, j], so] == K_BOUNDARY) && (m |= UInt32(1) << (j - 1))
                end
                ci = Lv.cutidx[k, o]
                if ci != 0
                    for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
                        m |= UInt32(1) << (OPP[Int(Lv.cut_dir[li])] - 1)
                    end
                end
                Lv.special[k, o] = m
            end
        end
    end

    # ---- 6. counts
    counts = Dict{String,Any}("levels" => L, "dx" => dxs, "domain_m" => collect(domain), "requested_domain_m" => collect(spec.domain),
        "root_cells" => collect(n1), "triangles" => length(tri_orig), "triangles_with_periodic_copies" => length(tri), "band_fine" => spec.band_fine, "band" => spec.band,
        "symmetry_boundary_refinement" => 1,
        "surface_level" => spec.surface_level, "wake_length" => spec.wake_length, "wake_level" => wake_level,
        "distance_queries" => sum(nquery), "timings_s" => timings, "reports" => reports, "seconds" => time() - t_start)
    per = Dict{String,Any}[]
    for l in 1:L
        kc = kind_counts(levels[l])
        push!(per, Dict("level" => l, "dx_m" => dxs[l], "octs" => levels[l].nocts, "real_octs" => count(levels[l].real), "fluid" => kc["fluid"], "solid" => kc["solid"],
                        "covered" => kc["covered"], "ghost" => kc["ghost"], "solid_ghost" => kc["solid_ghost"], "boundary" => kc["boundary"],
                        "wall_cells" => length(levels[l].cut_slot), "cut_links" => length(levels[l].cut_dir),
                        "ghost_listed" => length(levels[l].ghost_slot), "restricted" => length(levels[l].restrict_slot)))
    end
    counts["per_level"] = per
    counts["fluid_total"] = sum(p["fluid"] for p in per); counts["stored_slots"] = sum(8 * levels[l].nocts for l in 1:L)
    counts["work_W"] = sum(per[l]["fluid"] * 2^(l - 1) for l in 1:L)
    OctMesh(domain, offset, spec.periodic, levels, no_geometry ? Tri[] : tri_orig, config_sha, stl_sha, counts)
end

"Coarse-row buffer at nonperiodic faces, except symmetry faces with level-local reflection."
function face_margin_mask(size_l::NTuple{3,Int}, l::Int, periodic::NTuple{3,Bool},
                          faces::NTuple{6,Symbol}=(:inlet,:outlet,:slip,:slip,:slip,:slip))
    M = falses(size_l...)
    m = 2^(l - 1)
    for a in 1:3
        periodic[a] && continue
        n = size_l[a]
        for side in 1:2
            faces[2(a-1)+side]==:symmetry && continue
            for v in (side==1 ? (1:m) : ((n-m+1):n))
                a == 1 ? (M[v, :, :] .= true) : a == 2 ? (M[:, v, :] .= true) : (M[:, :, v] .= true)
            end
        end
    end
    M
end

"Wrap an oct origin into the level grid on periodic axes; returns (origin, inside)."
function wrap_origin(p::Tuple{Int,Int,Int}, n::NTuple{3,Int}, periodic::NTuple{3,Bool})
    q = [p[1], p[2], p[3]]; inside = true
    for a in 1:3
        if q[a] < 0 || q[a] >= n[a]
            periodic[a] ? (q[a] = mod(q[a], n[a])) : (inside = false)
        end
    end
    (q[1], q[2], q[3]), inside
end
"Wrap a cell on periodic axes; returns -1 components outside non-periodic faces."
function wrap_cell(c::Tuple{Int,Int,Int}, n::NTuple{3,Int}, periodic::NTuple{3,Bool})
    q = [c[1], c[2], c[3]]
    for a in 1:3
        if q[a] < 0 || q[a] >= n[a]
            periodic[a] ? (q[a] = mod(q[a], n[a])) : (q[a] = -1)
        end
    end
    (q[1], q[2], q[3])
end

"Slots whose centre is within `diag` cell diagonals of the surface (threaded)."
function near_surface_mask(Lv::OctLevel, bvh::BVH, stacks, diag::Float64)
    M = falses(8, Lv.nocts)
    nt = length(stacks); ranges = chunk_ranges(Lv.nocts, nt)
    bound = diag * sqrt(3) * Lv.dx
    @threads :static for t in 1:nt
        st = stacks[t]
        for o in ranges[t]
            # oct-level pre-test at the oct centre
            oc = ((Lv.origin[1, o] + 1.0) * Lv.dx, (Lv.origin[2, o] + 1.0) * Lv.dx, (Lv.origin[3, o] + 1.0) * Lv.dx)
            d = nearest_distance(bvh, oc, st; bound=bound + sqrt(3) * Lv.dx)
            d >= bound + sqrt(3) * Lv.dx && continue
            for k in 1:8
                Lv.kind[k, o] == K_BOUNDARY && continue
                M[k, o] = nearest_distance(bvh, slot_centre(Lv, slot_id(k, o)), st; bound=bound) < bound
            end
        end
    end
    M
end

"Push the fluid children of covered cell `c` (level l) on the face toward -e (the face adjacent to the source) onto the queue; sealed children stay out."
function push_near_children!(queue, fluid_reached, sealed, levels, l::Int, c::Tuple{Int,Int,Int}, e::Tuple{Int,Int,Int})
    l < length(levels) || return
    Lf = levels[l + 1]
    for dk in 0:1, dj in 0:1, di in 0:1
        # near face: the child offset along the axis of e is 0 when e is positive (the source lies at -e), 1 when negative
        (e[1] != 0 && di != (e[1] > 0 ? 0 : 1)) && continue
        (e[2] != 0 && dj != (e[2] > 0 ? 0 : 1)) && continue
        (e[3] != 0 && dk != (e[3] > 0 ? 0 : 1)) && continue
        cc = (2c[1] + di, 2c[2] + dj, 2c[3] + dk)
        cs = find_slot(Lf, cc...)
        cs == 0 && continue
        ck = Lf.kind[slot_k(cs), slot_oct(cs)]
        if ck == K_FLUID
            if !fluid_reached[l + 1][slot_k(cs), slot_oct(cs)] && !sealed[l + 1][slot_k(cs), slot_oct(cs)]
                fluid_reached[l + 1][slot_k(cs), slot_oct(cs)] = true; push!(queue, (l + 1, cs))
            end
        elseif ck == K_COVERED
            push_near_children!(queue, fluid_reached, sealed, levels, l + 1, cc, e)
        end
    end
end
