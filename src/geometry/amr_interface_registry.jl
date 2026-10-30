"""
AMR_INTERFACE_REGISTRY.JL

CPU-side registry of coarse/fine interface topology.

This is the first OpenLB/Palabos-inspired AMR abstraction layer: a compact list
of interface entities classified as faces, edges, or corners. It is diagnostic
only for now. Later transfer kernels can use the same entity model for staged
decomposed C2F/F2C coupling.
"""

struct AMRInterfaceEntity
    topology::Int8      # 1=face, 2=edge, 3=corner
    sx::Int8
    sy::Int8
    sz::Int8
    child_block_index::Int32
    child_bx::Int32
    child_by::Int32
    child_bz::Int32
    parent_bx::Int32
    parent_by::Int32
    parent_bz::Int32
    exterior_parent_bx::Int32
    exterior_parent_by::Int32
    exterior_parent_bz::Int32
    parent_inside_active::Bool
    parent_exterior_active::Bool
    tangent_parent_count::Int16
    tangent_parent_total::Int16
    cubic_parent_count::Int16
    cubic_parent_total::Int16
end

struct AMRInterfaceRegistry
    parent_level_id::Int
    child_level_id::Int
    entities::Vector{AMRInterfaceEntity}
end

@inline function _amr_registry_parent_coord(child_b::Int)
    return fld(child_b + 1, 2)
end

@inline function _amr_registry_active(ptr::AbstractArray{Int32,3}, bx::Int, by::Int, bz::Int)
    if bx < 1 || by < 1 || bz < 1 ||
       bx > size(ptr, 1) || by > size(ptr, 2) || bz > size(ptr, 3)
        return false
    end
    return ptr[bx, by, bz] > Int32(0)
end

@inline function _amr_registry_nnz3(sx::Int, sy::Int, sz::Int)
    return (sx == 0 ? 0 : 1) + (sy == 0 ? 0 : 1) + (sz == 0 ? 0 : 1)
end

function _amr_registry_tangent_support(parent_ptr::AbstractArray{Int32,3},
                                       pbx::Int, pby::Int, pbz::Int,
                                       sx::Int, sy::Int, sz::Int)
    oxs = sx == 0 ? (-1:1) : (0:0)
    oys = sy == 0 ? (-1:1) : (0:0)
    ozs = sz == 0 ? (-1:1) : (0:0)
    n_total = 0
    n_active = 0
    @inbounds for oz in ozs, oy in oys, ox in oxs
        n_total += 1
        n_active += _amr_registry_active(parent_ptr, pbx + ox, pby + oy, pbz + oz) ? 1 : 0
    end
    return Int16(n_active), Int16(n_total)
end

function _amr_registry_cubic_block_support(parent_ptr::AbstractArray{Int32,3},
                                           pbx::Int, pby::Int, pbz::Int)
    n_total = 0
    n_active = 0
    @inbounds for oz in -1:2, oy in -1:2, ox in -1:2
        n_total += 1
        n_active += _amr_registry_active(parent_ptr, pbx + ox, pby + oy, pbz + oz) ? 1 : 0
    end
    return Int16(n_active), Int16(n_total)
end

"""
    build_amr_interface_registry(parent_level, child_level)

Build a compact block-topology registry for the child frontier relative to its
parent level. Each missing 26-neighbor of an active child block becomes one
interface entity:

- one missing axis: face;
- two missing axes: edge;
- three missing axes: corner.

The registry also records coarse donor support around the child-side parent
block. This is not yet a transfer operator; it is the stable metadata layer that
future decomposed AMR coupling can consume.
"""
function build_amr_interface_registry(parent_level::BlockLevel, child_level::BlockLevel)
    parent_ptr = Array(parent_level.block_pointer)
    child_ptr = Array(child_level.block_pointer)

    entities = AMRInterfaceEntity[]
    sizehint!(entities, max(0, length(child_level.active_block_coords) * 6))

    @inbounds for (child_idx, (cbx, cby, cbz)) in enumerate(child_level.active_block_coords)
        for sz in -1:1, sy in -1:1, sx in -1:1
            if sx == 0 && sy == 0 && sz == 0
                continue
            end

            nbx = cbx + sx
            nby = cby + sy
            nbz = cbz + sz
            _amr_registry_active(child_ptr, nbx, nby, nbz) && continue

            topology = _amr_registry_nnz3(sx, sy, sz)
            pbx = _amr_registry_parent_coord(cbx)
            pby = _amr_registry_parent_coord(cby)
            pbz = _amr_registry_parent_coord(cbz)
            epbx = _amr_registry_parent_coord(nbx)
            epby = _amr_registry_parent_coord(nby)
            epbz = _amr_registry_parent_coord(nbz)

            parent_inside_active = _amr_registry_active(parent_ptr, pbx, pby, pbz)
            parent_exterior_active = _amr_registry_active(parent_ptr, epbx, epby, epbz)
            tangent_count, tangent_total =
                _amr_registry_tangent_support(parent_ptr, pbx, pby, pbz, sx, sy, sz)
            cubic_count, cubic_total =
                _amr_registry_cubic_block_support(parent_ptr, pbx, pby, pbz)

            push!(entities, AMRInterfaceEntity(
                Int8(topology),
                Int8(sx), Int8(sy), Int8(sz),
                Int32(child_idx),
                Int32(cbx), Int32(cby), Int32(cbz),
                Int32(pbx), Int32(pby), Int32(pbz),
                Int32(epbx), Int32(epby), Int32(epbz),
                parent_inside_active,
                parent_exterior_active,
                tangent_count,
                tangent_total,
                cubic_count,
                cubic_total
            ))
        end
    end

    return AMRInterfaceRegistry(parent_level.level_id, child_level.level_id, entities)
end

function summarize_amr_interface_registry(registry::AMRInterfaceRegistry)
    n_entities = length(registry.entities)
    n_faces = 0
    n_edges = 0
    n_corners = 0
    n_face_x_pos = 0
    n_face_x_neg = 0
    n_face_y_pos = 0
    n_face_y_neg = 0
    n_face_z_pos = 0
    n_face_z_neg = 0
    n_parent_inside_missing = 0
    n_parent_exterior_missing = 0
    n_tangent_incomplete = 0
    n_cubic_incomplete = 0
    tangent_fill_sum = 0.0
    cubic_fill_sum = 0.0

    @inbounds for entity in registry.entities
        if entity.topology == Int8(1)
            n_faces += 1
            if entity.sx > 0
                n_face_x_pos += 1
            elseif entity.sx < 0
                n_face_x_neg += 1
            elseif entity.sy > 0
                n_face_y_pos += 1
            elseif entity.sy < 0
                n_face_y_neg += 1
            elseif entity.sz > 0
                n_face_z_pos += 1
            elseif entity.sz < 0
                n_face_z_neg += 1
            end
        elseif entity.topology == Int8(2)
            n_edges += 1
        elseif entity.topology == Int8(3)
            n_corners += 1
        end

        n_parent_inside_missing += entity.parent_inside_active ? 0 : 1
        n_parent_exterior_missing += entity.parent_exterior_active ? 0 : 1
        n_tangent_incomplete += entity.tangent_parent_count < entity.tangent_parent_total ? 1 : 0
        n_cubic_incomplete += entity.cubic_parent_count < entity.cubic_parent_total ? 1 : 0

        tangent_fill_sum += entity.tangent_parent_total > 0 ?
            Float64(entity.tangent_parent_count) / Float64(entity.tangent_parent_total) : 1.0
        cubic_fill_sum += entity.cubic_parent_total > 0 ?
            Float64(entity.cubic_parent_count) / Float64(entity.cubic_parent_total) : 1.0
    end

    return (
        n_entities = n_entities,
        n_faces = n_faces,
        n_edges = n_edges,
        n_corners = n_corners,
        n_face_x_pos = n_face_x_pos,
        n_face_x_neg = n_face_x_neg,
        n_face_y_pos = n_face_y_pos,
        n_face_y_neg = n_face_y_neg,
        n_face_z_pos = n_face_z_pos,
        n_face_z_neg = n_face_z_neg,
        n_parent_inside_missing = n_parent_inside_missing,
        n_parent_exterior_missing = n_parent_exterior_missing,
        n_tangent_incomplete = n_tangent_incomplete,
        n_cubic_incomplete = n_cubic_incomplete,
        mean_tangent_fill = n_entities > 0 ? tangent_fill_sum / n_entities : 1.0,
        mean_cubic_fill = n_entities > 0 ? cubic_fill_sum / n_entities : 1.0
    )
end

function write_amr_interface_registry_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level," *
            "n_entities,n_faces,n_edges,n_corners," *
            "face_x_pos,face_x_neg,face_y_pos,face_y_neg,face_z_pos,face_z_neg," *
            "parent_inside_missing,parent_exterior_missing," *
            "tangent_incomplete,cubic_block_incomplete," *
            "mean_tangent_fill,mean_cubic_block_fill")
    end
end

function append_amr_interface_registry_csv_row(filepath::String, step::Int, parent_level::Int,
                                               child_level::Int, summary)
    if !isfile(filepath)
        write_amr_interface_registry_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e\n",
            step,
            parent_level,
            child_level,
            summary.n_entities,
            summary.n_faces,
            summary.n_edges,
            summary.n_corners,
            summary.n_face_x_pos,
            summary.n_face_x_neg,
            summary.n_face_y_pos,
            summary.n_face_y_neg,
            summary.n_face_z_pos,
            summary.n_face_z_neg,
            summary.n_parent_inside_missing,
            summary.n_parent_exterior_missing,
            summary.n_tangent_incomplete,
            summary.n_cubic_incomplete,
            summary.mean_tangent_fill,
            summary.mean_cubic_fill)
    end
end

function amr_interface_registry_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_interface_registry_history.csv")
end
