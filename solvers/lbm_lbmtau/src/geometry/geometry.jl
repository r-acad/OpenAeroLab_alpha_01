#
module Geometry

using LinearAlgebra
using Printf
using StaticArrays
using Base.Threads
using WriteVTK
using CUDA
using Adapt
using KernelAbstractions

include("triangle_subdivision.jl")
using .SurfaceSubdivision

export load_mesh, load_mesh_bytes, compute_mesh_bounds, GPUMesh, upload_mesh_to_gpu
export SolverMesh, subdivide_mesh, write_stl_binary, write_stl_binary_exclusive,
       rotate_mesh, build_rotation_matrix, auto_slice_half_y

function _stl_bytes_are_binary(bytes::AbstractVector{UInt8})
    length(bytes) < 84 && return false
    if startswith(lowercase(String(bytes[1:5])), "solid")
        io = IOBuffer(bytes; read=true, write=false)
        seek(io, 80)
        count = read(io, UInt32)
        expected_size = UInt128(84) + UInt128(count) * UInt128(50)
        return expected_size == UInt128(length(bytes))
    end
    return true
end

struct SolverMesh
    triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}
    min_bounds::Tuple{Float64, Float64, Float64}
    max_bounds::Tuple{Float64, Float64, Float64}
    
    normals::Vector{SVector{3, Float64}}
    areas::Vector{Float64}
    centers::Vector{SVector{3, Float64}}
end

struct GPUMesh{A}
    n_triangles::Int
    centers_x::A
    centers_y::A
    centers_z::A
    normals_x::A
    normals_y::A
    normals_z::A
    areas::A
end

function Adapt.adapt_structure(to, mesh::GPUMesh)
    GPUMesh(
        mesh.n_triangles,
        adapt(to, mesh.centers_x),
        adapt(to, mesh.centers_y),
        adapt(to, mesh.centers_z),
        adapt(to, mesh.normals_x),
        adapt(to, mesh.normals_y),
        adapt(to, mesh.normals_z),
        adapt(to, mesh.areas)
    )
end

function upload_mesh_to_gpu(mesh::SolverMesh, backend)
    n = length(mesh.triangles)
    
    cx = Float32[c[1] for c in mesh.centers]
    cy = Float32[c[2] for c in mesh.centers]
    cz = Float32[c[3] for c in mesh.centers]
    
    nx = Float32[n[1] for n in mesh.normals]
    ny = Float32[n[2] for n in mesh.normals]
    nz = Float32[n[3] for n in mesh.normals]
    
    areas = Float32[a for a in mesh.areas]
    
    return GPUMesh(
        n,
        adapt(backend, cx),
        adapt(backend, cy),
        adapt(backend, cz),
        adapt(backend, nx),
        adapt(backend, ny),
        adapt(backend, nz),
        adapt(backend, areas)
    )
end

function compute_geometry_properties(triangles)
    n = length(triangles)
    normals = Vector{SVector{3, Float64}}(undef, n)
    areas = Vector{Float64}(undef, n)
    centers = Vector{SVector{3, Float64}}(undef, n)

    @threads for i in 1:n
        v1 = SVector(triangles[i][1])
        v2 = SVector(triangles[i][2])
        v3 = SVector(triangles[i][3])

        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_prod = cross(edge1, edge2)
        
        area = 0.5 * norm(cross_prod)
        if area > 1e-12
            normal = cross_prod / (2.0 * area)
        else
            normal = SVector(0.0, 0.0, 0.0)
        end
        
        normals[i] = normal
        areas[i] = area
        centers[i] = (v1 + v2 + v3) / 3.0
    end

    return normals, areas, centers
end

function compute_triangle_bounds(triangles)
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf

    for tri in triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    return (min_x, min_y, min_z), (max_x, max_y, max_z)
end

function flip_triangle_winding(triangles)
    flipped = similar(triangles)
    @threads for i in eachindex(triangles)
        v1, v2, v3 = triangles[i]
        flipped[i] = (v1, v3, v2)
    end
    return flipped
end

function orientation_centroid_score(normals, areas, centers)
    total_area = sum(areas)
    total_area <= 0.0 && return 0.0, SVector(0.0, 0.0, 0.0)

    centroid = SVector(0.0, 0.0, 0.0)
    score = 0.0

    for i in eachindex(centers)
        centroid += centers[i] * areas[i]
    end
    centroid /= total_area

    for i in eachindex(centers)
        score += dot(normals[i], centers[i] - centroid) * areas[i]
    end

    return score / total_area, centroid
end

function ensure_outward_orientation(triangles)
    bounds_min, bounds_max = compute_triangle_bounds(triangles)
    normals, areas, centers = compute_geometry_properties(triangles)
    score, centroid = orientation_centroid_score(normals, areas, centers)

    diag = norm(SVector(bounds_max...) - SVector(bounds_min...))
    tol = max(1.0e-10, 1.0e-6 * diag)

    if score < -tol
        triangles = flip_triangle_winding(triangles)
        normals, areas, centers = compute_geometry_properties(triangles)
        score_flipped, _ = orientation_centroid_score(normals, areas, centers)
        @printf("[Geometry] Flipped global triangle winding to enforce outward normals (score %.6e -> %.6e)\n",
                score, score_flipped)
    elseif abs(score) <= tol
        @printf("[Geometry] Global normal orientation is ambiguous (score %.6e, tol %.6e); keeping input winding\n",
                score, tol)
    end

    return triangles, normals, areas, centers, bounds_min, bounds_max, centroid
end

# ──────────────────────────────────────────────────────────────────────────────
#  Triangle subdivision (longest-edge bisection, iterative)
# ──────────────────────────────────────────────────────────────────────────────

"""
    subdivide_triangle(tri, max_edge) -> Vector of triangles

Bisect the longest edge until no edge exceeds `max_edge`. Child winding and
planar geometry are preserved. The bounded visitor prevents a single enormous
facet from allocating an unbounded temporary child array.
"""
function subdivide_triangle(tri::NTuple{3,NTuple{3,Float64}}, max_edge::Float64)
    children=typeof(tri)[]
    count=visit_subtriangles(t -> push!(children,t),tri,max_edge;limit=50_000_000)
    count <= 50_000_000 || throw(ArgumentError("Surface subdivision exceeds 50 million triangles; increase the maximum triangle edge length"))
    children
end

"""
    subdivide_mesh(mesh, max_edge) -> SolverMesh

Refine the actual geometry used by wall links and per-face force/pressure
sampling. Geometry, winding and parent normal are preserved; child areas and
sampling centers are recomputed. Count first, then allocate the exact bounded
output instead of using a global edge estimate that undercounts large facets.
"""
function subdivide_mesh(mesh::SolverMesh, max_edge::Float64)
    isfinite(max_edge) || throw(ArgumentError("Maximum triangle edge must be finite"))
    max_edge <= 0 && return mesh
    max_triangles=50_000_000
    n_orig=length(mesh.triangles)
    n_new=subdivision_count(mesh.triangles,max_edge;limit=max_triangles)
    n_new <= max_triangles || throw(ArgumentError(
        "Surface subdivision exceeds $max_triangles triangles; increase the maximum triangle edge length"))
    n_new==n_orig && return mesh
    @printf("[Geometry] Surface subdivision: %d -> %d triangles; maximum edge %.6f m\n", n_orig,n_new,max_edge)
    flush(stdout)
    new_triangles=Vector{eltype(mesh.triangles)}(undef,n_new)
    new_normals=Vector{SVector{3,Float64}}(undef,n_new)
    cursor=0
    for (i,tri) in enumerate(mesh.triangles)
        normal=mesh.normals[i]
        emitted=visit_subtriangles(tri,max_edge;limit=n_new-cursor) do child
            cursor+=1
            new_triangles[cursor]=child
            new_normals[cursor]=normal
        end
        emitted <= n_new || error("Surface subdivision count changed unexpectedly")
    end
    cursor==n_new || error("Surface subdivision count changed unexpectedly")
    new_areas=Vector{Float64}(undef,n_new)
    new_centers=Vector{SVector{3,Float64}}(undef,n_new)
    @threads for i in 1:n_new
        a,b,c=SVector.(new_triangles[i])
        new_areas[i]=0.5*norm(cross(b-a,c-a))
        new_centers[i]=(a+b+c)/3
    end
    println("[Geometry] Surface sampling geometry ready: $n_new triangles")
    flush(stdout)
    SolverMesh(new_triangles,mesh.min_bounds,mesh.max_bounds,new_normals,new_areas,new_centers)
end

function build_rotation_matrix(alpha_deg::Float64, beta_deg::Float64; ground_mode::Bool=false,
                               ground_up_axis::Symbol=:z)
    alpha = deg2rad(alpha_deg)
    beta  = deg2rad(beta_deg)

    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta),  sin(beta)

    # Ry(alpha) — rotation around Y
    Ry = SMatrix{3,3,Float64}(
         ca,  0.0, sa,
         0.0, 1.0, 0.0,
        -sa,  0.0, ca
    )

    # Rz(beta)  — rotation around Z
    Rz = SMatrix{3,3,Float64}(
        cb, -sb, 0.0,
        sb,  cb, 0.0,
        0.0, 0.0, 1.0
    )

    ground_align = ground_up_axis == :z ? SMatrix{3,3,Float64}(
                        1.0, 0.0,  0.0,
                        0.0, 0.0,  1.0,
                        0.0, -1.0, 0.0
                   ) :
                   ground_up_axis == :y ? SMatrix{3,3,Float64}(
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0
                   ) :
                   ground_up_axis == :x ? SMatrix{3,3,Float64}(
                        0.0, -1.0, 0.0,
                        1.0,  0.0, 0.0,
                        0.0,  0.0, 1.0
                   ) :
                   error("Unsupported ground_up_axis=$ground_up_axis")

    return ground_mode ? (Rz * Ry * ground_align) : (Rz * Ry)
end

"""
    rotate_mesh(mesh, alpha_deg, beta_deg) -> SolverMesh

Rotate the mesh: first by `alpha_deg` around the Y axis, then by `beta_deg`
around the Z axis. Both angles are in degrees. The rotation is applied about
the mesh centroid so that the model stays centred. When `ground_mode=true`,
the solver first aligns the configured STL up-axis with the solver y-axis so
the model base lands on the solver half-domain plane y=0.

Returns a new SolverMesh with rotated triangles, normals, centers, and updated bounds.
"""
function rotate_mesh(mesh::SolverMesh, alpha_deg::Float64, beta_deg::Float64;
                     ground_mode::Bool=false, ground_up_axis::Symbol=:z)
    if !ground_mode && alpha_deg == 0.0 && beta_deg == 0.0
        return mesh
    end

    R = build_rotation_matrix(alpha_deg, beta_deg; ground_mode=ground_mode,
                              ground_up_axis=ground_up_axis)

    # Rotation centre = mesh centroid
    cx = (mesh.min_bounds[1] + mesh.max_bounds[1]) / 2.0
    cy = (mesh.min_bounds[2] + mesh.max_bounds[2]) / 2.0
    cz = (mesh.min_bounds[3] + mesh.max_bounds[3]) / 2.0
    origin = SVector(cx, cy, cz)

    # Helper: rotate a point about origin
    @inline rot_pt(p) = Tuple(R * (SVector(p) - origin) + origin)

    n_tri = length(mesh.triangles)
    new_tris    = Vector{eltype(mesh.triangles)}(undef, n_tri)
    new_normals = Vector{SVector{3, Float64}}(undef, n_tri)
    new_centers = Vector{SVector{3, Float64}}(undef, n_tri)
    new_areas   = Vector{Float64}(undef, n_tri)

    @threads for i in 1:n_tri
        v1r = rot_pt(mesh.triangles[i][1])
        v2r = rot_pt(mesh.triangles[i][2])
        v3r = rot_pt(mesh.triangles[i][3])
        new_tris[i] = (v1r, v2r, v3r)

        # Rotate normal (pure direction — no translation)
        new_normals[i] = R * mesh.normals[i]

        # Recompute area & centre from rotated vertices
        sv1 = SVector(v1r); sv2 = SVector(v2r); sv3 = SVector(v3r)
        new_areas[i]   = 0.5 * norm(cross(sv2 - sv1, sv3 - sv1))
        new_centers[i] = (sv1 + sv2 + sv3) / 3.0
    end

    # Recompute bounds
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in new_tris
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Rotated mesh: α=%.2f° (Y), β=%.2f° (Z)\n", alpha_deg, beta_deg)
    if ground_mode
        println("[Geometry] Ground mode: aligned STL $(uppercase(string(ground_up_axis)))-up axis with solver Y axis")
    end
    @printf("[Geometry] New bounds: [%.3f, %.3f] x [%.3f, %.3f] x [%.3f, %.3f]\n",
            min_x, max_x, min_y, max_y, min_z, max_z)
    flush(stdout)

    return SolverMesh(new_tris,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      new_normals, new_areas, new_centers)
end

"""
    auto_slice_half_y(mesh::SolverMesh) -> SolverMesh

Clip a full (Y-symmetric) mesh at its mid-Y plane and shift so the plane lies at y=0.
Triangles fully below the mid-plane are dropped; triangles crossing it are cut
at the plane (edges re-welded on y=0); triangles fully above are kept.

Use when `symmetric_analysis: true` but the STL contains both semi-spans.
"""
function auto_slice_half_y(mesh::SolverMesh)
    y_mid = (mesh.min_bounds[2] + mesh.max_bounds[2]) / 2.0
    tol_y = max(1.0e-10, (mesh.max_bounds[2] - mesh.min_bounds[2]) * 1.0e-9)

    # Classification admits a small roundoff band. Weld those vertices to the
    # actual plane and never extrapolate a near-coplanar edge outside its parent.
    @inline shift_pt(p) = (p[1], abs(p[2]-y_mid)<=tol_y ? 0.0 : p[2]-y_mid, p[3])
    @inline interp_to_plane(A, B) = begin
        # Linear interpolation to y = y_mid; inputs guaranteed to straddle
        t = clamp((y_mid - A[2]) / (B[2] - A[2]),0.0,1.0)
        (A[1] + t * (B[1] - A[1]), y_mid, A[3] + t * (B[3] - A[3]))
    end

    new_tris    = Vector{eltype(mesh.triangles)}()
    new_normals = Vector{SVector{3, Float64}}()
    new_areas   = Vector{Float64}()
    new_centers = Vector{SVector{3, Float64}}()

    n_kept = 0
    n_clipped = 0
    n_dropped = 0

    for i in 1:length(mesh.triangles)
        v1, v2, v3 = mesh.triangles[i]
        nrm = mesh.normals[i]

        # Classify vertices: above (true) if y >= y_mid - tol
        a1 = v1[2] >= y_mid - tol_y
        a2 = v2[2] >= y_mid - tol_y
        a3 = v3[2] >= y_mid - tol_y
        n_above = (a1 ? 1 : 0) + (a2 ? 1 : 0) + (a3 ? 1 : 0)

        if n_above == 3
            # Fully above — keep (shifted)
            p1 = shift_pt(v1); p2 = shift_pt(v2); p3 = shift_pt(v3)
            sv1 = SVector(p1); sv2 = SVector(p2); sv3 = SVector(p3)
            push!(new_tris, (p1, p2, p3))
            push!(new_normals, nrm)
            push!(new_areas, 0.5 * norm(cross(sv2 - sv1, sv3 - sv1)))
            push!(new_centers, (sv1 + sv2 + sv3) / 3.0)
            n_kept += 1
        elseif n_above == 0
            # Fully below — drop
            n_dropped += 1
        else
            # Rotate so the odd-one-out is v3, preserving winding order.
            # We want to handle the two canonical cases by ordering vertices
            # such that: (verts_above..., verts_below...) follows CCW.
            if n_above == 1
                # Exactly one vertex above — call it vA; the other two below (vB, vC)
                if a1
                    vA, vB, vC = v1, v2, v3
                elseif a2
                    vA, vB, vC = v2, v3, v1
                else
                    vA, vB, vC = v3, v1, v2
                end
                # Clip edges vA→vB and vA→vC
                iAB = interp_to_plane(vA, vB)
                iAC = interp_to_plane(vA, vC)
                # Emit one triangle (vA, iAB, iAC) preserving winding
                p1 = shift_pt(vA); p2 = shift_pt(iAB); p3 = shift_pt(iAC)
                sv1 = SVector(p1); sv2 = SVector(p2); sv3 = SVector(p3)
                area = 0.5 * norm(cross(sv2 - sv1, sv3 - sv1))
                if area > 1.0e-20
                    push!(new_tris, (p1, p2, p3))
                    push!(new_normals, nrm)
                    push!(new_areas, area)
                    push!(new_centers, (sv1 + sv2 + sv3) / 3.0)
                    n_clipped += 1
                end
            else  # n_above == 2
                # Two vertices above (vA, vB), one below (vC)
                if !a1
                    vA, vB, vC = v2, v3, v1
                elseif !a2
                    vA, vB, vC = v3, v1, v2
                else
                    vA, vB, vC = v1, v2, v3
                end
                # Clip edges vB→vC and vA→vC
                iBC = interp_to_plane(vB, vC)
                iAC = interp_to_plane(vA, vC)
                # Emit two triangles (vA, vB, iBC) and (vA, iBC, iAC)
                for (q1, q2, q3) in ((vA, vB, iBC), (vA, iBC, iAC))
                    p1 = shift_pt(q1); p2 = shift_pt(q2); p3 = shift_pt(q3)
                    sv1 = SVector(p1); sv2 = SVector(p2); sv3 = SVector(p3)
                    area = 0.5 * norm(cross(sv2 - sv1, sv3 - sv1))
                    if area > 1.0e-20
                        push!(new_tris, (p1, p2, p3))
                        push!(new_normals, nrm)
                        push!(new_areas, area)
                        push!(new_centers, (sv1 + sv2 + sv3) / 3.0)
                        n_clipped += 1
                    end
                end
            end
        end
    end

    # Recompute bounds
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in new_tris
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Auto-slice at y_mid=%.6f → shifted to y=0 (kept=%d, clipped=%d, dropped=%d, out=%d tris)\n",
            y_mid, n_kept, n_clipped, n_dropped, length(new_tris))
    @printf("[Geometry] Sliced bounds: [%.3f, %.3f] x [%.3f, %.3f] x [%.3f, %.3f]\n",
            min_x, max_x, min_y, max_y, min_z, max_z)
    flush(stdout)

    return SolverMesh(new_tris,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      new_normals, new_areas, new_centers)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Binary STL writer
# ──────────────────────────────────────────────────────────────────────────────

"""
    write_stl_binary(filename, mesh)

Write the mesh to a binary STL file.
"""
function _write_stl_binary(io::IO, mesh::SolverMesh)
    # 80-byte header
    header = zeros(UInt8, 80)
    msg = Vector{UInt8}(codeunits("LBM Solver - Subdivided STL"))
    header[1:length(msg)] .= msg
    write(io, header)

    # Triangle count
    write(io, UInt32(length(mesh.triangles)))

    for (i, tri) in enumerate(mesh.triangles)
        n = mesh.normals[i]
        # Normal
        write(io, Float32(n[1])); write(io, Float32(n[2])); write(io, Float32(n[3]))
        # Vertices
        for v in tri
            write(io, Float32(v[1])); write(io, Float32(v[2])); write(io, Float32(v[3]))
        end
        # Attribute byte count
        write(io, UInt16(0))
    end
    return nothing
end

function write_stl_binary(filename::String, mesh::SolverMesh)
    open(filename, "w") do io
        _write_stl_binary(io, mesh)
    end
    println("[Geometry] Wrote subdivided STL: $filename ($(length(mesh.triangles)) triangles)")
end

"""Create a binary STL only if no directory entry already owns `filename`."""
function write_stl_binary_exclusive(filename::String, mesh::SolverMesh)
    flags = Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT |
            Base.Filesystem.JL_O_EXCL
    io = Base.Filesystem.open(filename, flags, 0o600)
    try
        _write_stl_binary(io, mesh)
    finally
        close(io)
    end
    println("[Geometry] Wrote subdivided STL: $filename ($(length(mesh.triangles)) triangles)")
end

# ──────────────────────────────────────────────────────────────────────────────
#  STL loaders (winding-corrected using STL facet normals)
# ──────────────────────────────────────────────────────────────────────────────

function parse_binary_stl(io::IO, scale::Float64)
    skip(io, 80)
    count = read(io, UInt32)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}(undef, count)
    n_flipped = 0
    for i in 1:count
        # ── Read the facet normal stored in the STL file ──
        fnx = Float64(read(io, Float32))
        fny = Float64(read(io, Float32))
        fnz = Float64(read(io, Float32))

        x1 = Float64(read(io, Float32)) * scale
        y1 = Float64(read(io, Float32)) * scale
        z1 = Float64(read(io, Float32)) * scale
        x2 = Float64(read(io, Float32)) * scale
        y2 = Float64(read(io, Float32)) * scale
        z2 = Float64(read(io, Float32)) * scale
        x3 = Float64(read(io, Float32)) * scale
        y3 = Float64(read(io, Float32)) * scale
        z3 = Float64(read(io, Float32)) * scale
        skip(io, 2)

        # ── Check if vertex winding agrees with the stored normal ──
        # cross(v2-v1, v3-v1)  should point in the same direction as (fnx,fny,fnz)
        e1x = x2-x1; e1y = y2-y1; e1z = z2-z1
        e2x = x3-x1; e2y = y3-y1; e2z = z3-z1
        cx = e1y*e2z - e1z*e2y
        cy = e1z*e2x - e1x*e2z
        cz = e1x*e2y - e1y*e2x

        # dot(cross, stl_normal) < 0  →  winding is flipped  →  swap v2 ↔ v3
        if (cx*fnx + cy*fny + cz*fnz) < 0.0
            triangles[i] = ((x1,y1,z1), (x3,y3,z3), (x2,y2,z2))
            n_flipped += 1
        else
            triangles[i] = ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
        end
    end
    if n_flipped > 0
        @printf("[Geometry] Fixed vertex winding on %d / %d triangles (normals now consistent)\n", n_flipped, count)
    end
    return triangles
end

function parse_ascii_stl(io::IO, scale::Float64)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}()
    current_tri = Vector{Tuple{Float64,Float64,Float64}}()
    for line in eachline(io)
        s = strip(line)
        if startswith(s, "vertex")
            parts = split(s)
            if length(parts) >= 4
                x = parse(Float64, parts[2]) * scale
                y = parse(Float64, parts[3]) * scale
                z = parse(Float64, parts[4]) * scale
                push!(current_tri, (x, y, z))
            end
        elseif startswith(s, "endloop")
            if length(current_tri) == 3
                push!(triangles, (current_tri[1], current_tri[2], current_tri[3]))
            end
            empty!(current_tri)
        end
    end
    return triangles
end

"""
    load_mesh_bytes(bytes; source_name="captured STL", scale=1.0,
                    min_facet_size=0.0)

Parse one immutable STL byte generation entirely from an `IOBuffer`. This
function never opens a pathname or writes an artifact.
"""
function load_mesh_bytes(bytes::AbstractVector{UInt8};
                         source_name::AbstractString="captured STL",
                         scale=1.0, min_facet_size=0.0)
    isempty(bytes) && error("STL byte snapshot is empty: $source_name")
    scale64 = Float64(scale)
    min_facet64 = Float64(min_facet_size)
    println("[Geometry] Loading captured STL bytes from: $source_name")

    is_binary = _stl_bytes_are_binary(bytes)

    io = IOBuffer(bytes; read=true, write=false)
    triangles = if is_binary
        parsed = parse_binary_stl(io, scale64)
        println("[Geometry] Format: Binary STL")
        parsed
    else
        parsed = parse_ascii_stl(io, scale64)
        println("[Geometry] Format: ASCII STL")
        parsed
    end
    isempty(triangles) && error("No triangles loaded.")

    triangles, normals, areas, centers, bounds_min, bounds_max, centroid =
        ensure_outward_orientation(triangles)
    min_x, min_y, min_z = bounds_min
    max_x, max_y, max_z = bounds_max
    println("[Geometry] Loaded $(length(triangles)) triangles.")
    println("[Geometry] Bounds: [$(round(min_x,digits=3)), $(round(max_x,digits=3))] x [$(round(min_y,digits=3)), $(round(max_y,digits=3))] x [$(round(min_z,digits=3)), $(round(max_z,digits=3))]")
    @printf("[Geometry] Area-weighted centroid: (%.6f, %.6f, %.6f)\n", centroid...)
    flush(stdout)

    mesh = SolverMesh(triangles, (min_x, min_y, min_z), (max_x, max_y, max_z),
                      normals, areas, centers)
    if min_facet64 > 0.0
        mesh = subdivide_mesh(mesh, min_facet64)
    end
    return mesh
end

"""
    load_mesh(filename; scale=1.0, min_facet_size=0.0)

Load an STL file. If `min_facet_size > 0`, subdivide triangles so that no edge
exceeds that length. A subdivided artifact is written only when the caller
supplies an explicit `subdivision_output` path.
"""
function load_mesh(filename::String; scale=1.0, min_facet_size=0.0,
                   subdivision_output::Union{Nothing,AbstractString}=nothing)
    isfile(filename) || error("STL file not found: $filename")
    mesh = load_mesh_bytes(read(filename); source_name=filename, scale=scale,
                           min_facet_size=min_facet_size)
    subdivision_output === nothing ||
        write_stl_binary(String(subdivision_output), mesh)
    return mesh
end

function compute_mesh_bounds(mesh::SolverMesh)
    return (min_bounds=mesh.min_bounds, max_bounds=mesh.max_bounds)
end

"""
    merge_meshes(meshes::Vector{SolverMesh}) -> SolverMesh

Merge multiple SolverMesh objects into a single unified mesh.
Triangles are concatenated in order: mesh 1 first, then mesh 2, etc.
Bounding box is recomputed from the combined geometry.
"""
function merge_meshes(meshes::Vector{SolverMesh})
    if length(meshes) == 1
        return meshes[1]
    end

    total_n = sum(length(m.triangles) for m in meshes)

    all_triangles = Vector{eltype(meshes[1].triangles)}()
    all_normals   = Vector{SVector{3, Float64}}()
    all_areas     = Vector{Float64}()
    all_centers   = Vector{SVector{3, Float64}}()

    sizehint!(all_triangles, total_n)
    sizehint!(all_normals, total_n)
    sizehint!(all_areas, total_n)
    sizehint!(all_centers, total_n)

    for m in meshes
        append!(all_triangles, m.triangles)
        append!(all_normals, m.normals)
        append!(all_areas, m.areas)
        append!(all_centers, m.centers)
    end

    # Recompute global bounding box
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in all_triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Merged %d meshes → %d total triangles\n", length(meshes), total_n)
    @printf("[Geometry] Combined bounds: [%.3f, %.3f] x [%.3f, %.3f] x [%.3f, %.3f]\n",
            min_x, max_x, min_y, max_y, min_z, max_z)

    return SolverMesh(all_triangles,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      all_normals, all_areas, all_centers)
end

end
