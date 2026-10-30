# FILE: src/domain_topology.jl
using Base.Threads
using StaticArrays

"""
DOMAIN_TOPOLOGY.JL - Block Connectivity and Hierarchy Management
"""

function get_active_blocks_for_level(mesh::Geometry.SolverMesh,
                                     dx::Float64,
                                     mesh_offset::SVector{3,Float64},
                                     bx_max::Int, by_max::Int, bz_max::Int)
    bs = BLOCK_SIZE
    block_span = bs * dx
    block_half = SVector(0.5 * block_span, 0.5 * block_span, 0.5 * block_span)
    search_margin = max(dx * 1.0e-6, eps(block_span))

    thread_sets = [Set{Tuple{Int, Int, Int}}() for _ in 1:Base.Threads.maxthreadid()]

    @threads for tri_idx in eachindex(mesh.triangles)
        tri = mesh.triangles[tri_idx]
        local_set = thread_sets[threadid()]
        v1, v2, v3, t_min, t_max = triangle_bounds_with_offset(tri, mesh_offset)
        min_bx, max_bx, min_by, max_by, min_bz, max_bz =
            block_index_bounds_from_aabb(t_min, t_max, block_span, bx_max, by_max, bz_max;
                                         margin=search_margin)

        for bz in min_bz:max_bz
            cz = (bz - 0.5) * block_span
            for by in min_by:max_by
                cy = (by - 0.5) * block_span
                for bx in min_bx:max_bx
                    center = SVector((bx - 0.5) * block_span, cy, cz)
                    if triangle_intersects_aabb(center, block_half, v1, v2, v3)
                        push!(local_set, (bx, by, bz))
                    end
                end
            end
        end
    end

    active_set = Set{Tuple{Int, Int, Int}}()
    for local_set in thread_sets
        union!(active_set, local_set)
    end

    return active_set
end

@inline function _add_child_family!(target::Set{Tuple{Int, Int, Int}},
                                    bx::Int, by::Int, bz::Int,
                                    bx_max::Int, by_max::Int, bz_max::Int)
    pbx = fld(bx + 1, 2)
    pby = fld(by + 1, 2)
    pbz = fld(bz + 1, 2)

    for dbz in 0:1, dby in 0:1, dbx in 0:1
        sbx = 2 * pbx - 1 + dbx
        sby = 2 * pby - 1 + dby
        sbz = 2 * pbz - 1 + dbz
        if sbx >= 1 && sbx <= bx_max && sby >= 1 && sby <= by_max && sbz >= 1 && sbz <= bz_max
            push!(target, (sbx, sby, sbz))
        end
    end
end

@inline function _parent_coord(bx::Int, by::Int, bz::Int)
    return (fld(bx + 1, 2), fld(by + 1, 2), fld(bz + 1, 2))
end

@inline function _has_active_parent(bx::Int, by::Int, bz::Int,
                                    parent_set::Set{Tuple{Int, Int, Int}})
    return _parent_coord(bx, by, bz) in parent_set
end

# Periodic-Y AMR (27 Oct 2026): wrap a block Y index on the periodic axis.
# Identity when periodic_y_amr is off (legacy clamp-and-skip semantics kept).
@inline _topo_wrap_by(by::Int, by_max::Int) = periodic_y_amr_active() ? mod1(by, by_max) : by

@inline function _axis_halo_metric(delta::Int, neg_layers::Int, pos_layers::Int)
    delta == 0 && return 0.0
    layers = delta < 0 ? neg_layers : pos_layers
    layers <= 0 && return Inf
    return (abs(delta) / layers)^2
end

function add_directional_halo_blocks_with_siblings!(active_set::Set{Tuple{Int, Int, Int}},
                                                    upstream_layers::Int,
                                                    downstream_layers::Int,
                                                    span_layers::Int,
                                                    up_layers::Int,
                                                    down_layers::Int,
                                                    bx_max::Int, by_max::Int, bz_max::Int;
                                                    parent_set::Union{Nothing, Set{Tuple{Int, Int, Int}}}=nothing)
    max(upstream_layers, downstream_layers, span_layers, up_layers, down_layers) <= 0 && return

    seed_blocks = collect(active_set)
    halo_blocks = Set{Tuple{Int, Int, Int}}()

    for dz in -down_layers:up_layers, dy in -span_layers:span_layers, dx in -upstream_layers:downstream_layers
        metric = _axis_halo_metric(dx, upstream_layers, downstream_layers) +
                 _axis_halo_metric(dy, span_layers, span_layers) +
                 _axis_halo_metric(dz, down_layers, up_layers)
        if metric == 0.0 || metric > 1.0
            continue
        end

        for (bx, by, bz) in seed_blocks
            nbx = bx + dx
            nby = _topo_wrap_by(by + dy, by_max)
            nbz = bz + dz
            if !(1 <= nbx <= bx_max && 1 <= nby <= by_max && 1 <= nbz <= bz_max)
                continue
            end
            if parent_set !== nothing && !_has_active_parent(nbx, nby, nbz, parent_set)
                continue
            end
            push!(halo_blocks, (nbx, nby, nbz))
        end
    end

    union!(active_set, halo_blocks)

    siblings_to_add = Set{Tuple{Int, Int, Int}}()
    for (bx, by, bz) in halo_blocks
        if parent_set !== nothing && !_has_active_parent(bx, by, bz, parent_set)
            continue
        end
        _add_child_family!(siblings_to_add, bx, by, bz, bx_max, by_max, bz_max)
    end

    union!(active_set, siblings_to_add)
end

function filter_orphan_blocks!(active_set::Set{Tuple{Int, Int, Int}},
                               parent_set::Set{Tuple{Int, Int, Int}})
    orphan_blocks = Tuple{Int, Int, Int}[]
    for (bx, by, bz) in active_set
        _has_active_parent(bx, by, bz, parent_set) || push!(orphan_blocks, (bx, by, bz))
    end
    isempty(orphan_blocks) && return 0
    for coord in orphan_blocks
        delete!(active_set, coord)
    end
    return length(orphan_blocks)
end

function add_halo_blocks_with_siblings!(active_set::Set{Tuple{Int, Int, Int}},
                                        layers::Int,
                                        bx_max::Int, by_max::Int, bz_max::Int)
    layers <= 0 && return

    seed_blocks = collect(active_set)
    halo_blocks = Set{Tuple{Int, Int, Int}}()
    radius_sq = layers * layers

    # Grow a rounded halo in block space instead of repeated 26-neighbor dilation.
    # This avoids turning every refinement frontier into a hard axis-aligned box.
    for dz in -layers:layers, dy in -layers:layers, dx in -layers:layers
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq == 0 || dist_sq > radius_sq
            continue
        end

        for (bx, by, bz) in seed_blocks
            nbx = bx + dx
            nby = _topo_wrap_by(by + dy, by_max)
            nbz = bz + dz
            if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                push!(halo_blocks, (nbx, nby, nbz))
            end
        end
    end

    union!(active_set, halo_blocks)

    siblings_to_add = Set{Tuple{Int, Int, Int}}()
    for (bx, by, bz) in halo_blocks
        _add_child_family!(siblings_to_add, bx, by, bz, bx_max, by_max, bz_max)
    end

    union!(active_set, siblings_to_add)
end

# ══════════════════════════════════════════════════════════════════════════
# Obstacle standoff (Fix, 2026-06-03): guaranteed shape-hugging separation
# between the obstacle surface and the FINEST AMR level boundary.
#
# Two complaints this addresses:
#   (A) the finest-level box edge touching/cutting the obstacle (no separation)
#   (B) the finest level being a wasteful loose box rather than hugging the shape
# (B) is handled by the wake cap (domain.jl WAKE_MAX_LEVEL); (A) here.
#
# The legacy directional halo (add_directional_halo_blocks_with_siblings!) gates
# each halo block on its PARENT existing (parent_set), so the finest margin
# silently collapses to ~0 on any side where the coarser level was clipped tight
# (the "rejected N orphan candidates" pathology). Here we INVERT that: the child
# DEMANDS its ancestors, force-cascading them up to L1, so the guaranteed shell
# can never be orphan-clipped.
# ══════════════════════════════════════════════════════════════════════════

"""
    dilate_blocks_isotropic(seed, n, bx_max, by_max, bz_max) -> Set

Euclidean (rounded) N-block dilation of `seed` in block space: returns
`seed ∪ {every block within distance n of any seed block}`, clamped to
[1,b*_max]. Same rounded metric as add_halo_blocks_with_siblings!
(dist_sq <= n^2), so it HUGS the seed shape instead of producing an
axis-aligned box, and the reach is identical on all six sides → guaranteed
n-block separation. Pure function (does not mutate `seed`).
"""
function dilate_blocks_isotropic(seed::Set{Tuple{Int,Int,Int}}, n::Int,
                                 bx_max::Int, by_max::Int, bz_max::Int)
    out = copy(seed)
    n <= 0 && return out
    r2 = n * n
    seed_list = collect(seed)
    for dz in -n:n, dy in -n:n, dx in -n:n
        d2 = dx*dx + dy*dy + dz*dz
        (d2 == 0 || d2 > r2) && continue
        for (bx, by, bz) in seed_list
            nbx = bx + dx; nby = _topo_wrap_by(by + dy, by_max); nbz = bz + dz
            (1 <= nbx <= bx_max && 1 <= nby <= by_max && 1 <= nbz <= bz_max) || continue
            push!(out, (nbx, nby, nbz))
        end
    end
    return out
end

"""
    detect_sharp_edge_triangles(mesh, angle_threshold_deg) -> Set{Int}

Feature-edge / sharp-corner detector (2026-06-24). Returns the set of triangle
indices that touch at least one SHARP STL edge — an edge shared by two triangles
whose face normals differ by more than `angle_threshold_deg` (the "angle between
adjacent triangles"). A wing leading/trailing edge, a box corner, or any crease
has a large normal-angle there; a smooth panel or gently-curved surface does not.

STL is a flat triangle soup with no connectivity, so this first builds an
edge→triangles adjacency map. Vertices are WELDED on a quantized grid (snap to
`weld` resolution) before edges are keyed, so the two triangles meeting at an edge
are recognized even when their shared vertices differ by float round-off (common in
exported STLs) — without welding, no edge would ever match and nothing would be
flagged. Boundary edges (touched by only one triangle, e.g. an open mesh) and
non-manifold edges (>2 triangles) are handled gracefully: a lone triangle is never
flagged from a boundary edge; for a >2 fan, every adjacent pair is tested.

CPU-only, one-time at domain build. O(N_tri) adjacency + O(N_edge) angle tests.
"""
function detect_sharp_edge_triangles(mesh::Geometry.SolverMesh,
                                     angle_threshold_deg::Float64)
    ntri = length(mesh.triangles)
    sharp = Set{Int}()
    (ntri == 0 || angle_threshold_deg <= 0.0) && return sharp

    # Weld resolution: a small fraction of the mesh diagonal so genuinely-distinct
    # vertices never collapse, but float-jittered duplicates do.
    diag = sqrt(
        (mesh.max_bounds[1] - mesh.min_bounds[1])^2 +
        (mesh.max_bounds[2] - mesh.min_bounds[2])^2 +
        (mesh.max_bounds[3] - mesh.min_bounds[3])^2)
    weld = max(diag * 1.0e-6, 1.0e-9)
    inv_weld = 1.0 / weld

    @inline keyv(v) = (round(Int, v[1]*inv_weld),
                       round(Int, v[2]*inv_weld),
                       round(Int, v[3]*inv_weld))
    # Undirected edge key: ordered pair of welded vertex keys (smaller first).
    @inline edgekey(a, b) = a <= b ? (a, b) : (b, a)

    # edge → list of triangle indices touching it
    edge_map = Dict{Tuple{NTuple{3,Int},NTuple{3,Int}}, Vector{Int}}()
    for ti in 1:ntri
        tri = mesh.triangles[ti]
        k1 = keyv(tri[1]); k2 = keyv(tri[2]); k3 = keyv(tri[3])
        for e in (edgekey(k1, k2), edgekey(k2, k3), edgekey(k3, k1))
            push!(get!(edge_map, e, Int[]), ti)
        end
    end

    cos_thresh = cos(deg2rad(angle_threshold_deg))
    for (_, tris) in edge_map
        length(tris) < 2 && continue           # boundary edge → not a fold
        # Test every adjacent pair sharing this edge (handles non-manifold fans).
        for ia in 1:length(tris)-1, ib in ia+1:length(tris)
            ta = tris[ia]; tb = tris[ib]
            na = mesh.normals[ta]; nb = mesh.normals[tb]
            # angle between face normals; flag if it EXCEEDS the threshold
            # (cos decreases as the angle grows ⇒ dotcap < cos_thresh = sharp).
            dotcap = clamp(na[1]*nb[1] + na[2]*nb[2] + na[3]*nb[3], -1.0, 1.0)
            if dotcap < cos_thresh
                push!(sharp, ta); push!(sharp, tb)
            end
        end
    end
    return sharp
end

"""
    get_sharp_edge_blocks_for_level(mesh, sharp_tris, dx, mesh_offset,
                                    bx_max, by_max, bz_max) -> Set

Finest-level block footprint of ONLY the sharp-edge triangles. Same
triangle-AABB intersection test as `get_active_blocks_for_level`, restricted to
`sharp_tris`. Used as the seed for the sharp-edge refinement dilation.
"""
function get_sharp_edge_blocks_for_level(mesh::Geometry.SolverMesh,
                                         sharp_tris::Set{Int},
                                         dx::Float64,
                                         mesh_offset::SVector{3,Float64},
                                         bx_max::Int, by_max::Int, bz_max::Int)
    bs = BLOCK_SIZE
    block_span = bs * dx
    block_half = SVector(0.5 * block_span, 0.5 * block_span, 0.5 * block_span)
    search_margin = max(dx * 1.0e-6, eps(block_span))
    out = Set{Tuple{Int,Int,Int}}()
    isempty(sharp_tris) && return out

    for tri_idx in sharp_tris
        tri = mesh.triangles[tri_idx]
        v1, v2, v3, t_min, t_max = triangle_bounds_with_offset(tri, mesh_offset)
        min_bx, max_bx, min_by, max_by, min_bz, max_bz =
            block_index_bounds_from_aabb(t_min, t_max, block_span, bx_max, by_max, bz_max;
                                         margin=search_margin)
        for bz in min_bz:max_bz
            cz = (bz - 0.5) * block_span
            for by in min_by:max_by
                cy = (by - 0.5) * block_span
                for bx in min_bx:max_bx
                    center = SVector((bx - 0.5) * block_span, cy, cz)
                    if triangle_intersects_aabb(center, block_half, v1, v2, v3)
                        push!(out, (bx, by, bz))
                    end
                end
            end
        end
    end
    return out
end

"""
    enforce_sharp_edge_refinement!(level_sets, bx_maxes, by_maxes, bz_maxes,
                                   sharp_fine; margin_blocks=0) -> Int

Guarantee the finest level covers the sharp-edge footprint plus an N-block hugging
margin, force-cascading ancestors to L1 — structurally identical to
`enforce_obstacle_standoff!` but seeded from the SHARP-EDGE blocks only. This is
what gives a wing LE/TE EXTRA finest-level cells beyond the uniform body shell:
the body shell hugs the whole surface uniformly, this concentrates additional
finest blocks specifically around the high-curvature feature edges. Idempotent.
`sharp_fine` is the finest-level footprint of the sharp triangles (captured ONCE).
"""
function enforce_sharp_edge_refinement!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                        bx_maxes::Vector{Int},
                                        by_maxes::Vector{Int},
                                        bz_maxes::Vector{Int},
                                        sharp_fine::Set{Tuple{Int,Int,Int}};
                                        margin_blocks::Int=0)
    isempty(sharp_fine) && return 0
    num_levels = length(level_sets)
    num_levels < 2 && return 0
    fine = num_levels
    fx, fy, fz = bx_maxes[fine], by_maxes[fine], bz_maxes[fine]

    required = dilate_blocks_isotropic(sharp_fine, max(margin_blocks, 0), fx, fy, fz)
    missing_fine = setdiff(required, level_sets[fine])
    isempty(missing_fine) && return 0

    union!(level_sets[fine], missing_fine)
    ensure_complete_parent_coverage!(level_sets[fine], fx, fy, fz)
    added = length(missing_fine)

    for lvl in fine:-1:2
        plvl = lvl - 1
        pset = level_sets[plvl]
        pbx, pby, pbz = bx_maxes[plvl], by_maxes[plvl], bz_maxes[plvl]
        before = length(pset)
        for (bx, by, bz) in level_sets[lvl]
            push!(pset, _parent_coord(bx, by, bz))
        end
        ensure_complete_parent_coverage!(pset, pbx, pby, pbz)
        added += length(pset) - before
    end
    return added
end

"""
    enforce_obstacle_standoff!(level_sets, bx_maxes, by_maxes, bz_maxes,
                               surface_fine; standoff_blocks=0) -> Int

Guarantee a >= standoff_blocks separation between every finest-level surface
block and the finest-level refinement boundary on ALL six sides, hugging the
shape. Dilates the finest surface footprint (`surface_fine`, captured ONCE
before the closure loop) by N blocks, adds the missing finest blocks, and
force-cascades the required ancestors UP to L1 (parent = fld(b+1,2)) so no
added child is ever an orphan. Returns blocks added across all levels (0 if
the guarantee already held this pass). Idempotent: once satisfied it adds 0.
"""
function enforce_obstacle_standoff!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                    bx_maxes::Vector{Int},
                                    by_maxes::Vector{Int},
                                    bz_maxes::Vector{Int},
                                    surface_fine::Set{Tuple{Int,Int,Int}};
                                    standoff_blocks::Int=0)
    standoff_blocks <= 0 && return 0
    isempty(surface_fine) && return 0
    num_levels = length(level_sets)
    num_levels < 2 && return 0
    fine = num_levels
    fx, fy, fz = bx_maxes[fine], by_maxes[fine], bz_maxes[fine]

    # (1) Required finest shell = N-ball dilation of the surface footprint.
    #     Fixed seed (surface_fine) ⇒ a fixed finite target ⇒ no runaway.
    required = dilate_blocks_isotropic(surface_fine, standoff_blocks, fx, fy, fz)
    missing_fine = setdiff(required, level_sets[fine])
    isempty(missing_fine) && return 0          # guarantee already holds

    # Loud warning if the shell reaches the farfield wall buffer: that is a
    # domain-too-small condition, NOT something to satisfy silently. We still
    # add the blocks (standoff is the stated guarantee) and they survive
    # because strip_wall_adjacent_refinement! runs only ONCE, before this loop.
    wall_hits = count(bc -> is_wall_adjacent(bc[1], bc[2], bc[3], fx, fy, fz;
                                             skip_y_min=HALF_Y_DOMAIN), missing_fine)
    if wall_hits > 0
        @warn "[Standoff] $wall_hits required shell blocks fall in the farfield wall buffer " *
              "(WALL_BUFFER_BLOCKS=$WALL_BUFFER_BLOCKS) — domain is too small for " *
              "body_shell_blocks=$standoff_blocks. Enlarge domain.upstream/downstream/" *
              "lateral/height or reduce body_shell_blocks."
    end

    # (2) Add the missing finest blocks, sibling-complete.
    union!(level_sets[fine], missing_fine)
    ensure_complete_parent_coverage!(level_sets[fine], fx, fy, fz)
    added = length(missing_fine)

    # (3) Cascade ancestors UP (coarse-after-fine within one sweep closes the
    #     whole chain to L1). Every child added above gets its direct parent;
    #     a parent added at level k gets ITS parent at k-1 in the same sweep.
    for lvl in fine:-1:2
        plvl = lvl - 1
        pset = level_sets[plvl]
        pbx, pby, pbz = bx_maxes[plvl], by_maxes[plvl], bz_maxes[plvl]
        before = length(pset)
        for (bx, by, bz) in level_sets[lvl]
            push!(pset, _parent_coord(bx, by, bz))
        end
        ensure_complete_parent_coverage!(pset, pbx, pby, pbz)
        added += length(pset) - before
    end
    return added
end

"""
    enforce_y_mirror_symmetry!(level_sets, bx_maxes, by_maxes, bz_maxes) -> Int

Y-axis sibling of `enforce_z_mirror_symmetry!` (2026-06-23). The AABB→block
rounding in `block_index_bounds_from_aabb` (intersection_utils.jl) uses `floor`
on BOTH the −side and +side of the body's bounding box, so when the body
straddles a block boundary the refinement box picks up one extra block-row on
one y hemisphere and not its mirror — a ~1-block-row TOP/BOTTOM asymmetry. Until
2026-06-23 only the z hemisphere was force-symmetrized (z-mirror pass); y was left
raw, producing a persistent top-vs-bottom asymmetry visible on a perfectly
symmetric, zero-AoA sphere (raw |u| top half ~faster than bottom by a measurable
margin, plus a one-sided AMR block-row "rail" and stepped/notched box edges in
schlieren). This pass closes the y hemisphere the same way the z pass does. Opt-in
(`enforce_y_symmetry`); only valid when geometry+BCs are mirror-symmetric about
the y-midplane (NOT for ground/no-slip-y_min cases).
"""
function enforce_y_mirror_symmetry!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                    bx_maxes::Vector{Int},
                                    by_maxes::Vector{Int},
                                    bz_maxes::Vector{Int};
                                    start_level::Int=2)
    num_levels = length(level_sets)
    num_levels < start_level && return 0

    total_added = 0
    first_level = max(1, start_level)

    for _pass in 1:8
        before_total = sum(length(s) for s in level_sets)

        for lvl in first_level:num_levels
            s = level_sets[lvl]
            isempty(s) && continue

            bxmax = bx_maxes[lvl]
            bymax = by_maxes[lvl]
            bzmax = bz_maxes[lvl]
            additions = Set{Tuple{Int,Int,Int}}()

            for (bx, by, bz) in s
                mby = bymax - by + 1
                (1 <= mby <= bymax) || continue
                mcoord = (bx, mby, bz)
                if !(mcoord in s)
                    push!(additions, mcoord)
                end
            end

            if !isempty(additions)
                union!(s, additions)
                ensure_complete_parent_coverage!(s, bxmax, bymax, bzmax)
            end
        end

        # Mirrored fine blocks need their parents (then sibling completion, then
        # possibly another y pass) — same cascade as the z mirror.
        for lvl in num_levels:-1:2
            parent_set = level_sets[lvl - 1]
            pbxmax = bx_maxes[lvl - 1]
            pbymax = by_maxes[lvl - 1]
            pbzmax = bz_maxes[lvl - 1]
            for (bx, by, bz) in level_sets[lvl]
                push!(parent_set, _parent_coord(bx, by, bz))
            end
            ensure_complete_parent_coverage!(parent_set, pbxmax, pbymax, pbzmax)
        end

        after_total = sum(length(s) for s in level_sets)
        total_added += after_total - before_total
        after_total == before_total && break
    end

    return total_added
end

"""
    enforce_z_mirror_symmetry!(level_sets, bx_maxes, by_maxes, bz_maxes) -> Int

Opt-in topology closure for cases whose geometry and boundary conditions are
mirror-symmetric about the domain z-midplane. STL/AABB surface seeding can miss
a few finest blocks on one z hemisphere because triangles sit exactly on block
boundaries. Those missing partners show up at step 0 as one-sided AMR corner
quads, then later as asymmetric schlieren seams. This pass adds the z-mirrored
block partner, keeps child families complete, and cascades required ancestors.
"""
function enforce_z_mirror_symmetry!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                    bx_maxes::Vector{Int},
                                    by_maxes::Vector{Int},
                                    bz_maxes::Vector{Int};
                                    start_level::Int=2)
    num_levels = length(level_sets)
    num_levels < start_level && return 0

    total_added = 0
    first_level = max(1, start_level)

    for _pass in 1:8
        before_total = sum(length(s) for s in level_sets)

        for lvl in first_level:num_levels
            s = level_sets[lvl]
            isempty(s) && continue

            bxmax = bx_maxes[lvl]
            bymax = by_maxes[lvl]
            bzmax = bz_maxes[lvl]
            additions = Set{Tuple{Int,Int,Int}}()

            for (bx, by, bz) in s
                mbz = bzmax - bz + 1
                (1 <= mbz <= bzmax) || continue
                mcoord = (bx, by, mbz)
                if !(mcoord in s)
                    push!(additions, mcoord)
                end
            end

            if !isempty(additions)
                union!(s, additions)
                ensure_complete_parent_coverage!(s, bxmax, bymax, bzmax)
            end
        end

        # New mirrored fine blocks require their direct parents. Adding those
        # parents can in turn require sibling completion and another z pass.
        for lvl in num_levels:-1:2
            parent_set = level_sets[lvl - 1]
            pbxmax = bx_maxes[lvl - 1]
            pbymax = by_maxes[lvl - 1]
            pbzmax = bz_maxes[lvl - 1]
            for (bx, by, bz) in level_sets[lvl]
                push!(parent_set, _parent_coord(bx, by, bz))
            end
            ensure_complete_parent_coverage!(parent_set, pbxmax, pbymax, pbzmax)
        end

        after_total = sum(length(s) for s in level_sets)
        total_added += after_total - before_total
        after_total == before_total && break
    end

    return total_added
end

"""
    enforce_span_uniform_levels!(level_sets, bx_maxes, by_maxes, bz_maxes) -> Int

Periodic-Y AMR option `advanced.refinement.span_uniform_levels` (27 Oct 2026).
Extrude every refined level's (bx, bz) footprint over the WHOLE Y period, then
cascade the ancestors (the parent of a full Y column is a full Y column, so the
result is again span-uniform). Guarantees that a Y-uniform (extruded) body gets
Y-uniform refinement zones even where the triangle/block SAT test rounds
differently from one span station to the next, and that every refined level
spans the period (no AMR interface normal to Y anywhere). Called inside the
closure loop of setup_multilevel_domain; returns the number of blocks added.
"""
function enforce_span_uniform_levels!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                      bx_maxes::Vector{Int},
                                      by_maxes::Vector{Int},
                                      bz_maxes::Vector{Int})
    num_levels = length(level_sets)
    num_levels < 2 && return 0
    before_total = sum(length(s) for s in level_sets)
    for lvl in 2:num_levels
        s = level_sets[lvl]
        isempty(s) && continue
        bym = by_maxes[lvl]
        cols = Set{Tuple{Int,Int}}()
        for (bx, _, bz) in s
            push!(cols, (bx, bz))
        end
        for (bx, bz) in cols, by in 1:bym
            push!(s, (bx, by, bz))
        end
    end
    for lvl in num_levels:-1:2
        pset = level_sets[lvl - 1]
        for (bx, by, bz) in level_sets[lvl]
            push!(pset, _parent_coord(bx, by, bz))
        end
        ensure_complete_parent_coverage!(pset, bx_maxes[lvl - 1], by_maxes[lvl - 1], bz_maxes[lvl - 1])
    end
    return sum(length(s) for s in level_sets) - before_total
end

function ensure_complete_parent_coverage!(active_set::Set{Tuple{Int, Int, Int}},
                                          bx_max::Int, by_max::Int, bz_max::Int)
    added = true
    iterations = 0
    max_iterations = 10

    while added && iterations < max_iterations
        added = false
        iterations += 1
        siblings_to_add = Set{Tuple{Int, Int, Int}}()

        for (bx, by, bz) in active_set
            before = length(siblings_to_add)
            _add_child_family!(siblings_to_add, bx, by, bz, bx_max, by_max, bz_max)
            added = added || (length(siblings_to_add) > before)
        end

        union!(active_set, siblings_to_add)
    end
end

"""
    ensure_prolongation_donor_halos!(level_sets, bx_maxes, by_maxes, bz_maxes; layers=2)

Add a coarse donor halo around every refined child footprint.  The fine-grid
prolongation uses a four-point parent stencil; if the parent level ends exactly
at the child footprint, the stencil is forced into one-sided fallbacks that
leave visible vorticity/schlieren seams.  Two parent-block layers give D3Q39
max-speed links and cubic stencils real coarse data on both sides of the
refinement interface.
"""
function ensure_prolongation_donor_halos!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                          bx_maxes::Vector{Int},
                                          by_maxes::Vector{Int},
                                          bz_maxes::Vector{Int};
                                          layers::Int=2)
    num_levels = length(level_sets)
    num_levels < 2 && return 0

    total_added = 0

    for child_lvl in num_levels:-1:2
        parent_lvl = child_lvl - 1
        child_set = level_sets[child_lvl]
        parent_set = level_sets[parent_lvl]
        parent_bx_max = bx_maxes[parent_lvl]
        parent_by_max = by_maxes[parent_lvl]
        parent_bz_max = bz_maxes[parent_lvl]

        new_parent_blocks = Set{Tuple{Int,Int,Int}}()

        for (fbx, fby, fbz) in child_set
            pbx = fld(fbx + 1, 2)
            pby = fld(fby + 1, 2)
            pbz = fld(fbz + 1, 2)

            for dz in -layers:layers, dy in -layers:layers, dx in -layers:layers
                nbx = pbx + dx
                nby = _topo_wrap_by(pby + dy, parent_by_max)
                nbz = pbz + dz

                if nbx < 1 || nbx > parent_bx_max ||
                   nby < 1 || nby > parent_by_max ||
                   nbz < 1 || nbz > parent_bz_max
                    continue
                end

                if parent_lvl > 1 &&
                   is_wall_adjacent(nbx, nby, nbz,
                                    parent_bx_max, parent_by_max, parent_bz_max;
                                    skip_y_min=HALF_Y_DOMAIN)
                    continue
                end

                if (nbx, nby, nbz) ∉ parent_set
                    push!(new_parent_blocks, (nbx, nby, nbz))
                end
            end
        end

        if !isempty(new_parent_blocks)
            before = length(parent_set)
            union!(parent_set, new_parent_blocks)
            ensure_complete_parent_coverage!(parent_set, parent_bx_max, parent_by_max, parent_bz_max)
            total_added += length(parent_set) - before
        end
    end

    return total_added
end

"""
    ensure_level_grading!(level_sets, bx_maxes, by_maxes, bz_maxes)

Enforce the AMR grading constraint: adjacent blocks must differ by at most
one refinement level.  Without this, a level N+2 block can border a level N
block directly, creating a 4:1 resolution jump that causes density banding
and spurious reflections.

Algorithm — propagate from finest to coarsest:
  For every block at level L (L ≥ 3), find its parent at level L-1.
  For each of the 26 neighbours of that parent block, if the neighbour
  exists at level L-2 but has NO children at level L-1, create children
  at level L-1 to bridge the gap.

The function modifies `level_sets` in-place and iterates until no new
blocks are added.
"""
# Wall-buffer width in block indices at each level (matches domain.jl:352-364).
# Blocks at indices 1, 2, N-1, N on any axis are considered wall-adjacent.
const WALL_BUFFER_BLOCKS = 4

@inline function is_wall_adjacent(bx::Int, by::Int, bz::Int,
                                  bx_max::Int, by_max::Int, bz_max::Int;
                                  skip_y_min::Bool=false)
    # Per-axis "room check": only apply the wall buffer on an axis if that
    # axis has at least 2*WALL_BUFFER_BLOCKS + 1 blocks. Otherwise the
    # buffer would cover the entire axis and strip every block — which
    # empties the level on coarse grids with many refinement levels and
    # crashes downstream code (flood fill on an empty block list).
    needs_x = bx_max >= 2 * WALL_BUFFER_BLOCKS + 1
    needs_y = by_max >= 2 * WALL_BUFFER_BLOCKS + 1
    needs_z = bz_max >= 2 * WALL_BUFFER_BLOCKS + 1

    # In half-domain mode the y=0 face is a physical boundary (slip symmetry
    # plane or no-slip ground), not a farfield outer wall. Refinement MUST be
    # allowed to extend down to y=0 there — otherwise the wing root / ground
    # boundary layer is rendered at the coarsest level and the geometry never
    # gets a Bouzidi cell at the symmetry plane.
    in_x = needs_x && (bx <= WALL_BUFFER_BLOCKS || bx >= bx_max - (WALL_BUFFER_BLOCKS - 1))
    # Periodic-Y AMR (27 Oct 2026): the Y faces are not walls — no Y buffer.
    in_y = !periodic_y_amr_active() &&
           needs_y && (((!skip_y_min) && by <= WALL_BUFFER_BLOCKS) || by >= by_max - (WALL_BUFFER_BLOCKS - 1))
    in_z = needs_z && (bz <= WALL_BUFFER_BLOCKS || bz >= bz_max - (WALL_BUFFER_BLOCKS - 1))
    return in_x || in_y || in_z
end

"""
Strip wall-adjacent blocks from every refined level (>1) and cascade the
removal to finer levels (drop orphans whose parent no longer exists).

This guarantees that only the coarsest level (level 1) touches the outer
domain walls, so every face of the domain sees identical physics. Called
after `ensure_level_grading!` to undo any wall-adjacent blocks grading may
have reintroduced. Because removing refined blocks only ever makes the mesh
coarser near the wall, grading constraints are preserved automatically.

Returns the total number of blocks removed across all levels.
"""
function strip_wall_adjacent_refinement!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                         bx_maxes::Vector{Int},
                                         by_maxes::Vector{Int},
                                         bz_maxes::Vector{Int})
    num_levels = length(level_sets)
    num_levels < 2 && return 0

    total_removed = 0

    # Pass 1: strip wall-adjacent blocks at every refined level.
    for lvl in 2:num_levels
        s = level_sets[lvl]
        bxm, bym, bzm = bx_maxes[lvl], by_maxes[lvl], bz_maxes[lvl]
        before = length(s)
        filter!(bc -> !is_wall_adjacent(bc[1], bc[2], bc[3], bxm, bym, bzm;
                                        skip_y_min=HALF_Y_DOMAIN), s)
        total_removed += before - length(s)
    end

    # Pass 2: cascade orphan removal from coarse to fine. After a level has
    # lost wall-adjacent blocks, any child at the next level whose parent no
    # longer exists becomes an orphan and must be removed too.
    for lvl in 2:num_levels
        parent_set = level_sets[lvl - 1]
        s = level_sets[lvl]
        before = length(s)
        filter!(s) do bc
            pbx = fld(bc[1] + 1, 2)
            pby = fld(bc[2] + 1, 2)
            pbz = fld(bc[3] + 1, 2)
            (pbx, pby, pbz) in parent_set
        end
        total_removed += before - length(s)
    end

    return total_removed
end

function ensure_level_grading!(level_sets::Vector{Set{Tuple{Int,Int,Int}}},
                                bx_maxes::Vector{Int},
                                by_maxes::Vector{Int},
                                bz_maxes::Vector{Int})
    num_levels = length(level_sets)
    num_levels < 3 && return 0

    total_added = 0
    changed = true
    pass = 0

    while changed
        changed = false
        pass += 1
        pass > 20 && break   # safety valve

        for lvl in num_levels:-1:3
            fine_set = level_sets[lvl]
            mid_set = level_sets[lvl - 1]
            coarse_set = level_sets[lvl - 2]
            mid_bx = bx_maxes[lvl - 1]
            mid_by = by_maxes[lvl - 1]
            mid_bz = bz_maxes[lvl - 1]

            new_mid_blocks = Set{Tuple{Int,Int,Int}}()

            for (fbx, fby, fbz) in fine_set
                # Parent at level lvl-1
                pbx = fld(fbx + 1, 2)
                pby = fld(fby + 1, 2)
                pbz = fld(fbz + 1, 2)

                # Grandparent at level lvl-2
                gpbx = fld(pbx + 1, 2)
                gpby = fld(pby + 1, 2)
                gpbz = fld(pbz + 1, 2)

                # Check 26 neighbours of grandparent at level lvl-2
                for dz in -1:1, dy in -1:1, dx in -1:1
                    (dx == 0 && dy == 0 && dz == 0) && continue
                    ngpx = gpbx + dx
                    ngpy = _topo_wrap_by(gpby + dy, by_maxes[lvl - 2])
                    ngpz = gpbz + dz
                    # Neighbour grandparent must exist at coarse level
                    (ngpx, ngpy, ngpz) in coarse_set || continue

                    # Check whether this coarse neighbour already has children at mid level
                    has_mid_child = false
                    for dbz in 0:1, dby in 0:1, dbx in 0:1
                        mcx = 2 * ngpx - 1 + dbx
                        mcy = 2 * ngpy - 1 + dby
                        mcz = 2 * ngpz - 1 + dbz
                        if (mcx, mcy, mcz) in mid_set || (mcx, mcy, mcz) in new_mid_blocks
                            has_mid_child = true
                            break
                        end
                    end
                    has_mid_child && continue

                    # No mid-level coverage → create all 8 children at mid level
                    for dbz in 0:1, dby in 0:1, dbx in 0:1
                        mcx = 2 * ngpx - 1 + dbx
                        mcy = 2 * ngpy - 1 + dby
                        mcz = 2 * ngpz - 1 + dbz
                        if mcx >= 1 && mcx <= mid_bx && mcy >= 1 && mcy <= mid_by && mcz >= 1 && mcz <= mid_bz
                            # Honour the wall buffer: grading must never
                            # introduce refined blocks adjacent to the outer
                            # domain walls, otherwise two different levels
                            # would touch the boundary in different regions
                            # and break wall symmetry.
                            # Grading is non-negotiable: a 2:1 jump produces density
                            # banding and spurious reflections, far worse than letting
                            # a refined block sit one row closer to a farfield wall.
                            # The wall buffer is enforced separately and won't override
                            # grading-required blocks.
                            if (mcx, mcy, mcz) ∉ mid_set
                                push!(new_mid_blocks, (mcx, mcy, mcz))
                            end
                        end
                    end
                end
            end

            if !isempty(new_mid_blocks)
                union!(mid_set, new_mid_blocks)
                total_added += length(new_mid_blocks)
                changed = true
            end
        end
    end

    return total_added
end

function build_neighbor_table(active_coords::Vector{Tuple{Int, Int, Int}},
                              bx_max::Int, by_max::Int, bz_max::Int;
                              periodic_y::Bool=false)
    n_blocks = length(active_coords)
    table = zeros(Int32, n_blocks, 27)

    temp_ptr = zeros(Int32, bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        if bx >= 1 && bx <= bx_max && by >= 1 && by <= by_max && bz >= 1 && bz <= bz_max
            temp_ptr[bx, by, bz] = Int32(i)
        end
    end

    # Periodic-Y AMR (27 Oct 2026): a one-block period is its own Y neighbour (the
    # legacy single-level rule `by_max > 1` left such a level without any Y wrap).
    wrap_one_block = periodic_y_amr_active()
    for i in 1:n_blocks
        (bx, by, bz) = active_coords[i]
        for dz in -1:1, dy in -1:1, dx in -1:1
            dir_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9 + 1
            nbx = bx + dx
            nby_raw = by + dy
            nby = if periodic_y && (by_max > 1 || wrap_one_block)
                nby_raw < 1 ? by_max : (nby_raw > by_max ? 1 : nby_raw)
            else
                nby_raw
            end
            nbz = bz + dz
            if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                table[i, dir_idx] = temp_ptr[nbx, nby, nbz]
            else
                table[i, dir_idx] = 0
            end
        end
    end
    return table
end
