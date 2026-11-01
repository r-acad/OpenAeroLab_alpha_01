# FILE: src/forces/io.jl
using LinearAlgebra
using Printf
using WriteVTK

function write_force_csv_header(filepath::String)
    open(filepath, "w") do io
        # Per-step columns:
        #   Step timing, u_inlet_lattice
        #   Fx/y/z total, pressure, viscous (N)
        #   Mx/y/z (N·m)
        #   Cd/Cl/Cs (total), Cdp/Cdv (pressure/friction split), Clp/Clv, Csp/Csv
        #   Cmx/y/z
        #   Time-averaged coefficients: N_avg, Cd_mean/Cl_mean/..., Cdp_mean etc.
        println(io, "Step,Time_phys_s,u_inlet_lattice,Fx_N,Fy_N,Fz_N,Fx_p,Fy_p,Fz_p,Fx_v,Fy_v,Fz_v,Mx,My,Mz,Cd,Cl,Cs,Cdp,Cdv,Clp,Clv,Csp,Csv,Cmx,Cmy,Cmz,N_avg,Cd_mean,Cl_mean,Cs_mean,Cmx_mean,Cmy_mean,Cmz_mean")
    end
end

function append_force_csv(filepath::String, step::Int, time_phys::Float64,
                          fd::ForceData, u_inlet::Float32)
    # Decompose Cd/Cl/Cs into pressure and viscous (friction) parts.
    # Wave-drag separation would require an additional momentum-flux control
    # surface (post-processing via far-field integral). Here "p" is the normal-
    # force integral of p·n, "v" is the tangential shear integral of σ_tan.
    q_inf = 0.5 * fd.rho_ref * fd.u_ref^2
    F_ref = q_inf * fd.area_ref
    inv_Fref = F_ref > 1e-10 ? 1.0 / F_ref : 0.0
    Cdp = fd.Fx_pressure * inv_Fref
    Cdv = fd.Fx_viscous  * inv_Fref
    Clp = fd.Fz_pressure * inv_Fref
    Clv = fd.Fz_viscous  * inv_Fref
    Csp = fd.Fy_pressure * inv_Fref
    Csv = fd.Fy_viscous  * inv_Fref

    open(filepath, "a") do io
        @printf(io, "%d,%.6e,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                step, time_phys, u_inlet,
                fd.Fx, fd.Fy, fd.Fz,
                fd.Fx_pressure, fd.Fy_pressure, fd.Fz_pressure,
                fd.Fx_viscous, fd.Fy_viscous, fd.Fz_viscous,
                fd.Mx, fd.My, fd.Mz,
                fd.Cd, fd.Cl, fd.Cs,
                Cdp, Cdv, Clp, Clv, Csp, Csv,
                fd.Cmx, fd.Cmy, fd.Cmz,
                fd.avg_n,
                fd.Cd_avg, fd.Cl_avg, fd.Cs_avg,
                fd.Cmx_avg, fd.Cmy_avg, fd.Cmz_avg)
    end
end

function print_force_summary(fd::ForceData; re_number::Float64=0.0,
                              ma_physical::Float64=0.0, ma_lattice::Float64=0.0,
                              compressibility_correction::Bool=false)
    q_inf = 0.5 * fd.rho_ref * fd.u_ref^2
    F_ref = q_inf * fd.area_ref

    Cdp = F_ref > 1e-10 ? fd.Fx_pressure / F_ref : 0.0
    Cdv = F_ref > 1e-10 ? fd.Fx_viscous  / F_ref : 0.0

    println()
    println("============================================================")
    println("          FINAL AERODYNAMIC FORCES")
    if compressibility_correction
        println("          (Compressible solver: O(u⁴) equilibrium + DDF)")
    end
    println("============================================================")
    println()
    println("Reference Values:")
    @printf("  Re     = %.2e\n", re_number)
    @printf("  Ma     = %.4f (physical), %.4f (lattice)\n", ma_physical, ma_lattice)
    @printf("  ρ_ref  = %.4f kg/m³\n", fd.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", fd.u_ref)
    @printf("  A_ref  = %.4f m²\n", fd.area_ref)
    @printf("  L_ref  = %.4f m\n", fd.chord_ref)
    @printf("  q_∞    = %.4f Pa\n", q_inf)
    println()
    println("Forces [N]:")
    @printf("  Fx (drag)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fx, fd.Fx_pressure, fd.Fx_viscous)
    @printf("  Fy (side)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fy, fd.Fy_pressure, fd.Fy_viscous)
    @printf("  Fz (lift)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fz, fd.Fz_pressure, fd.Fz_viscous)
    println()
    println("Moments [N·m]:")
    @printf("  Mx (roll)  = %+.4e\n", fd.Mx)
    @printf("  My (pitch) = %+.4e\n", fd.My)
    @printf("  Mz (yaw)   = %+.4e\n", fd.Mz)
    println()
    println("Coefficients:")
    @printf("  Cd  = %+.6f  (Cdp = %+.6f, Cdv = %+.6f)\n", fd.Cd, Cdp, Cdv)
    @printf("  Cl  = %+.6f\n", fd.Cl)
    @printf("  Cs  = %+.6f\n", fd.Cs)
    @printf("  Cmx = %+.6f\n", fd.Cmx)
    @printf("  Cmy = %+.6f\n", fd.Cmy)
    @printf("  Cmz = %+.6f\n", fd.Cmz)
    println("============================================================")
end

# Modified to accept params and apply offset
function save_surface_vtk(filename::String, force_data::ForceData,
                          mesh::Geometry.SolverMesh, params;
                          object_names=nothing)

    n_tri_mesh = length(mesh.triangles)
    n_tri_force = length(force_data.pressure_map)
    n_tri = min(n_tri_mesh, n_tri_force)
    if n_tri != n_tri_mesh || n_tri != n_tri_force
        @warn "Surface VTK triangle-count mismatch; exporting common prefix" mesh_triangles=n_tri_mesh force_triangles=n_tri_force exported=n_tri
    end
    ox = params.mesh_offset[1]
    oy = params.mesh_offset[2]
    oz = params.mesh_offset[3]

    n_pts = 3 * n_tri
    points = Matrix{Float64}(undef, 3, n_pts)
    cells  = Vector{MeshCell}(undef, n_tri)

    for i in 1:n_tri
        tri = mesh.triangles[i]
        for (vi, v) in enumerate(tri)
            pidx = 3 * (i - 1) + vi
            # Apply domain offset here
            points[1, pidx] = v[1] + ox
            points[2, pidx] = v[2] + oy
            points[3, pidx] = v[3] + oz
        end
        base = 3 * (i - 1)
        cells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE,
                             (base + 1, base + 2, base + 3))
    end

    p_cpu  = Array(force_data.pressure_map)[1:n_tri]
    sx_cpu = Array(force_data.shear_x_map)[1:n_tri]
    sy_cpu = Array(force_data.shear_y_map)[1:n_tri]
    sz_cpu = Array(force_data.shear_z_map)[1:n_tri]

    # Two-sided probing (useful for thin-surface meshes — for a closed body the
    # minus-side probe is always inside solid so p_minus ≈ 0 there).
    p_minus_cpu  = Array(force_data.pressure_map_minus)[1:n_tri]
    sx_minus_cpu = Array(force_data.shear_x_map_minus)[1:n_tri]
    sy_minus_cpu = Array(force_data.shear_y_map_minus)[1:n_tri]
    sz_minus_cpu = Array(force_data.shear_z_map_minus)[1:n_tri]

    shear_mag = sqrt.(sx_cpu .^ 2 .+ sy_cpu .^ 2 .+ sz_cpu .^ 2)
    shear_mag_minus = sqrt.(sx_minus_cpu .^ 2 .+ sy_minus_cpu .^ 2 .+ sz_minus_cpu .^ 2)

    # Net pressure (outer - inner). For a closed body this equals p_cpu (since
    # the minus side is in solid and reads ~0). For a thin surface this is the
    # real aerodynamic net pressure jump across the sheet.
    p_net_cpu = p_cpu .- p_minus_cpu

    # Wall y⁺ = u_τ · d / ν,  u_τ = √(|τ_wall| / ρ).  Uses physical units.
    # nu_phys is pulled from the global config — it is the molecular kinematic
    # viscosity used to build τ_fine. ρ_ref is the freestream density (a
    # reasonable near-wall approximation for subsonic/transonic).
    wall_dist_cpu = Array(force_data.wall_dist_map)[1:n_tri]
    nu_phys       = Float32(FLUID_KINEMATIC_VISCOSITY)
    rho_phys      = Float32(force_data.rho_ref)
    tau_mag_phys  = sqrt.(sx_cpu .^ 2 .+ sy_cpu .^ 2 .+ sz_cpu .^ 2)
    u_tau         = sqrt.(max.(tau_mag_phys, Float32(0)) ./ rho_phys)
    y_plus_cpu    = nu_phys > Float32(1e-12) ?
                    u_tau .* wall_dist_cpu ./ nu_phys :
                    zeros(Float32, n_tri)

    normal_signs = Array(force_data.normal_sign_map)
    normals = Matrix{Float64}(undef, 3, n_tri)
    for i in 1:n_tri
        normals[1, i] = mesh.normals[i][1]*normal_signs[i]
        normals[2, i] = mesh.normals[i][2]*normal_signs[i]
        normals[3, i] = mesh.normals[i][3]*normal_signs[i]
    end

    shear_vec = Matrix{Float32}(undef, 3, n_tri)
    shear_vec[1, :] .= sx_cpu
    shear_vec[2, :] .= sy_cpu
    shear_vec[3, :] .= sz_cpu

    # Pressure coefficient Cp = (p − p_∞) / (½ ρ_∞ U_∞²).
    # Surface pressure is stored in gauge form (vanishes at freestream), so
    # numerator = p_cpu and we only divide by the dynamic pressure.
    q_inf = Float32(0.5 * force_data.rho_ref * force_data.u_ref^2)
    inv_qinf = q_inf > Float32(1.0e-10) ? Float32(1) / q_inf : Float32(0)
    cp_cpu       = p_cpu       .* inv_qinf
    cp_minus_cpu = p_minus_cpu .* inv_qinf
    cp_net_cpu   = p_net_cpu   .* inv_qinf

    shear_vec_minus = Matrix{Float32}(undef, 3, n_tri)
    shear_vec_minus[1, :] .= sx_minus_cpu
    shear_vec_minus[2, :] .= sy_minus_cpu
    shear_vec_minus[3, :] .= sz_minus_cpu

    try
        vtk_grid(filename, points, cells; compress=true, append=false) do vtk
            # Outer-side (+n) fields — primary, consistent with force integration
            # convention. For a closed body this is the only meaningful side.
            vtk["Pressure"]         = p_cpu
            vtk["Cp"]               = cp_cpu
            vtk["ShearStress"]      = shear_vec
            vtk["ShearMag"]         = shear_mag
            # Inner-side (−n) fields — for thin-surface meshes only.
            vtk["Pressure_minus"]   = p_minus_cpu
            vtk["Cp_minus"]         = cp_minus_cpu
            vtk["ShearStress_minus"] = shear_vec_minus
            vtk["ShearMag_minus"]   = shear_mag_minus
            # Net fields (outer minus inner). Equal to the outer fields for a
            # closed STL; equal to the real aerodynamic net for a thin sheet.
            vtk["Pressure_net"]     = p_net_cpu
            vtk["Cp_net"]           = cp_net_cpu
            # Wall-model diagnostics (Phase 1.7). y⁺ shows the near-wall
            # resolution regime; for the wall-law WMLES we configured in this
            # case, ParaView coloring by y⁺ lets the user verify the target
            # operating range (≈30–300 for a wall-function model).
            vtk["WallDistance"]     = wall_dist_cpu
            vtk["Yplus"]            = y_plus_cpu
            # Time-averaged fields (Tier A-3). Accumulated only post-ramp so
            # these are meaningful post-transient means. `avg_n` counts how many
            # samples are in the mean; when 0, the averaged arrays are still zero.
            if force_data.avg_n > 0
                vtk["Pressure_mean"] = Array(force_data.pressure_map_avg)[1:n_tri]
                vtk["Cp_mean"]       = Array(force_data.cp_map_avg)[1:n_tri]
                vtk["Nsamples"]      = fill(Int32(force_data.avg_n), n_tri)
            end
            vtk["Normal"]           = normals
            vtk["Area"]             = Float32.(mesh.areas[1:n_tri])
            if object_names !== nothing
                # Map unique object names to integer IDs for ParaView filtering
                unique_names = unique(object_names[1:n_tri])
                name_to_id = Dict(n => Int32(i) for (i, n) in enumerate(unique_names))
                obj_ids = Int32[name_to_id[object_names[j]] for j in 1:n_tri]
                vtk["ObjectID"] = obj_ids
            end
        end
    catch e
        println("[Error] Surface VTK write failed: $e")
    end
end

function export_net_triangle_forces_csv(filename::String,
                                        net_Fx::Vector{Float32},
                                        net_Fy::Vector{Float32},
                                        net_Fz::Vector{Float32},
                                        gpu_mesh, mesh_offset;
                                        unrotated_centers=nothing,
                                        alpha_deg::Float64=0.0,
                                        beta_deg::Float64=0.0,
                                        ground_mode::Bool=false,
                                        ground_up_axis::Symbol=:z,
                                        object_names=nothing)

    n_tri = length(net_Fx)

    # Use unrotated (subdivided-but-not-rotated) centroids when available.
    # These are in original STL coordinates — no domain offset applied.
    # When unrotated coords are used, force vectors must also be inverse-rotated
    # back to the original STL frame so positions and forces are consistent.
    # Otherwise fall back to the rotated GPU mesh centroids + domain offset.
    use_unrotated = unrotated_centers !== nothing

    if use_unrotated
        cx_cpu = Float32[c[1] for c in unrotated_centers]
        cy_cpu = Float32[c[2] for c in unrotated_centers]
        cz_cpu = Float32[c[3] for c in unrotated_centers]
    else
        cx_cpu = Array(gpu_mesh.centers_x)
        cy_cpu = Array(gpu_mesh.centers_y)
        cz_cpu = Array(gpu_mesh.centers_z)
    end

    ox = use_unrotated ? 0.0f0 : Float32(mesh_offset[1])
    oy = use_unrotated ? 0.0f0 : Float32(mesh_offset[2])
    oz = use_unrotated ? 0.0f0 : Float32(mesh_offset[3])

    need_inv_rotate = use_unrotated && (ground_mode || alpha_deg != 0.0 || beta_deg != 0.0)

    if need_inv_rotate
        R_inv = transpose(Geometry.build_rotation_matrix(alpha_deg, beta_deg;
                                                         ground_mode=ground_mode,
                                                         ground_up_axis=ground_up_axis))
        r11 = Float32(R_inv[1, 1]); r12 = Float32(R_inv[1, 2]); r13 = Float32(R_inv[1, 3])
        r21 = Float32(R_inv[2, 1]); r22 = Float32(R_inv[2, 2]); r23 = Float32(R_inv[2, 3])
        r31 = Float32(R_inv[3, 1]); r32 = Float32(R_inv[3, 2]); r33 = Float32(R_inv[3, 3])
    end

    out_dir = dirname(filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    open(filename, "w") do f
        println(f, "triangle_id,object_name,cx,cy,cz,Fx,Fy,Fz")

        for i in 1:n_tri
            fx = net_Fx[i]
            fy = net_Fy[i]
            fz = net_Fz[i]

            if need_inv_rotate
                # Apply R^{-1} to force vector
                fx_rot = r11*fx + r12*fy + r13*fz
                fy_rot = r21*fx + r22*fy + r23*fz
                fz_rot = r31*fx + r32*fy + r33*fz
                fx = fx_rot
                fy = fy_rot
                fz = fz_rot
            end

            obj_name = object_names !== nothing ? object_names[i] : "default"

            @printf(f, "%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    i, obj_name,
                    cx_cpu[i] + ox,
                    cy_cpu[i] + oy,
                    cz_cpu[i] + oz,
                    fx, fy, fz)
        end
    end

    println("[NetForce] Exported $(n_tri) triangle forces to $(filename)")
end

function _surface_atomic_write(writer::Function,path::AbstractString)
    target=abspath(String(path));mkpath(dirname(target))
    temporary,stream=mktemp(dirname(target);cleanup=false)
    try
        writer(stream);close(stream)
        for attempt in 1:6
            try
                Base.Filesystem.rename(temporary,target);return target
            catch error
                error isa Union{Base.IOError,SystemError} && attempt<6 || rethrow()
                sleep(.02*attempt)
            end
        end
    finally
        isopen(stream) && close(stream)
        isfile(temporary) && rm(temporary;force=true)
    end
end

"""
    export_surface_pressure_bin(filename, force_data, mesh, params; object_names=nothing)

Export surface pressure data as a compact binary file for web visualization.
The file contains the full triangle mesh with per-face pressure and shear magnitude,
readable by postv8.html (BabylonJS-based pressure viewer).
The mesh must be the exact solver geometry used by the per-face arrays.
An adjacent `.lbmp.toml` records the geometry frame, offset and triangle order;
unmatched geometry/field counts are rejected instead of truncating either.

Binary format (little-endian):
  [Header]
    4 bytes  : magic "LBMP"
    4 bytes  : version (UInt32 = 1)
    4 bytes  : n_triangles (UInt32)
    4 bytes  : n_objects (UInt32)
  [Object name table]  — for each object:
    4 bytes  : name length L (UInt32)
    L bytes  : UTF-8 name string
  [Per-triangle data]  — all arrays are contiguous Float32 / Int32:
    n_tri × 9 Float32 : vertices (v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z)
    n_tri × 3 Float32 : normals  (nx, ny, nz)
    n_tri     Float32 : pressure (gauge, physical Pa)
    n_tri     Float32 : shear magnitude (physical Pa)
    n_tri     Float32 : Cp (pressure coefficient)
    n_tri     Int32   : object ID (1-based index into name table)
"""
function export_surface_pressure_bin(filename::String, force_data::ForceData,
                                      mesh::Geometry.SolverMesh, params;
                                      object_names=nothing,
                                      use_domain_offset::Bool=true)

    n_tri_mesh = length(mesh.triangles)
    n_tri_force = length(force_data.pressure_map)
    n_tri_mesh>0 && n_tri_mesh==n_tri_force ||
        throw(ArgumentError("Surface geometry has $n_tri_mesh triangles but pressure has $n_tri_force faces; export requires the exact solver geometry, without prefix truncation."))
    n_tri=n_tri_mesh
    length(mesh.normals)==n_tri || throw(ArgumentError("Surface normal count does not match the solver geometry."))
    for values in (force_data.shear_x_map,force_data.shear_y_map,force_data.shear_z_map)
        length(values)==n_tri || throw(ArgumentError("Surface shear count does not match the solver geometry."))
    end
    labels=object_names
    if labels!==nothing && length(labels)!=n_tri
        # A single named source keeps the same identity after native half clipping.
        # Multiple sources require an explicit retained-face label map.
        !isempty(labels) && all(==(first(labels)),labels) ||
            throw(ArgumentError("Surface object labels do not match the clipped solver geometry."))
        labels=fill(first(labels),n_tri)
    end
    ox = use_domain_offset ? Float32(params.mesh_offset[1]) : 0.0f0
    oy = use_domain_offset ? Float32(params.mesh_offset[2]) : 0.0f0
    oz = use_domain_offset ? Float32(params.mesh_offset[3]) : 0.0f0

    # Build object name table
    if labels !== nothing
        unique_names = unique(labels)
    else
        unique_names = ["default"]
    end
    n_objects = length(unique_names)
    name_to_id = Dict(n => Int32(i) for (i, n) in enumerate(unique_names))

    # Copy GPU data to CPU
    p_cpu  = Array(force_data.pressure_map)
    sx_cpu = Array(force_data.shear_x_map)
    sy_cpu = Array(force_data.shear_y_map)
    sz_cpu = Array(force_data.shear_z_map)
    shear_mag = sqrt.(sx_cpu .^ 2 .+ sy_cpu .^ 2 .+ sz_cpu .^ 2)

    normal_signs = Array(force_data.normal_sign_map)
    # Compute Cp = p_gauge / q_inf
    q_inf = Float32(0.5 * force_data.rho_ref * force_data.u_ref^2)
    Cp = q_inf > Float32(1e-10) ? p_cpu ./ q_inf : zeros(Float32, n_tri)

    out_dir = dirname(filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    _surface_atomic_write(filename) do io
        # Header
        write(io, UInt8.([0x4C, 0x42, 0x4D, 0x50]))  # "LBMP"
        write(io, UInt32(1))                            # version
        write(io, UInt32(n_tri))
        write(io, UInt32(n_objects))

        # Object name table
        for name in unique_names
            name_bytes = Vector{UInt8}(codeunits(name))
            write(io, UInt32(length(name_bytes)))
            write(io, name_bytes)
        end

        # Vertices: n_tri × 9 Float32 (with domain offset applied)
        for i in 1:n_tri
            tri = mesh.triangles[i]
            for v in tri
                write(io, Float32(v[1]) + ox)
                write(io, Float32(v[2]) + oy)
                write(io, Float32(v[3]) + oz)
            end
        end

        # Normals: n_tri × 3 Float32
        for i in 1:n_tri
            write(io, Float32(mesh.normals[i][1]*normal_signs[i]))
            write(io, Float32(mesh.normals[i][2]*normal_signs[i]))
            write(io, Float32(mesh.normals[i][3]*normal_signs[i]))
        end

        # Pressure: n_tri Float32
        for i in 1:n_tri
            write(io, p_cpu[i])
        end

        # Shear magnitude: n_tri Float32
        for i in 1:n_tri
            write(io, shear_mag[i])
        end

        # Cp: n_tri Float32
        for i in 1:n_tri
            write(io, Cp[i])
        end

        # Object IDs: n_tri Int32
        for i in 1:n_tri
            obj_name = labels !== nothing ? labels[i] : "default"
            write(io, name_to_id[obj_name])
        end
    end

    metadata=Dict("sampling_revision"=>2,"missing_samples"=>count(x->!isfinite(x),p_cpu),"reversed_normal_samples"=>count(<(0),normal_signs),"moment_origin_m"=>collect(force_data.moment_center),"reference_area_m2"=>force_data.area_ref,"reference_chord_m"=>force_data.chord_ref,"format"=>"LBMP","format_version"=>1,"triangles"=>n_tri,
        "geometry_frame"=>use_domain_offset ? "domain" : "solver_local",
        "triangle_order"=>"solver_geometry","mesh_offset_m"=>Float64.(collect(params.mesh_offset)))
    _surface_atomic_write(filename*".toml") do io
        TOML.print(io,metadata;sorted=true)
    end

    file_size_mb = filesize(filename) / (1024^2)
    @printf("[Surface] Exported binary pressure map: %s (%.1f MB, %d triangles)\n",
            filename, file_size_mb, n_tri)
end

function export_net_triangle_forces_vtk(filename::String,
                                        net_Fx::Vector{Float32},
                                        net_Fy::Vector{Float32},
                                        net_Fz::Vector{Float32},
                                        gpu_mesh, mesh_offset)

    n_tri = length(net_Fx)

    cx_cpu = Array(gpu_mesh.centers_x)
    cy_cpu = Array(gpu_mesh.centers_y)
    cz_cpu = Array(gpu_mesh.centers_z)
    nx_cpu = Array(gpu_mesh.normals_x)
    ny_cpu = Array(gpu_mesh.normals_y)
    nz_cpu = Array(gpu_mesh.normals_z)
    areas_cpu = Array(gpu_mesh.areas)

    ox = Float64(mesh_offset[1])
    oy = Float64(mesh_offset[2])
    oz = Float64(mesh_offset[3])

    vtp_filename = filename * ".vtp"

    out_dir = dirname(vtp_filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    open(vtp_filename, "w") do f
        println(f, "<?xml version=\"1.0\"?>")
        println(f, "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">")
        println(f, "  <PolyData>")
        println(f, "    <Piece NumberOfPoints=\"$(n_tri)\" NumberOfVerts=\"$(n_tri)\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">")

        println(f, "      <Points>")
        println(f, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.8e %.8e %.8e\n",
                    Float64(cx_cpu[i]) + ox,
                    Float64(cy_cpu[i]) + oy,
                    Float64(cz_cpu[i]) + oz)
        end
        println(f, "        </DataArray>")
        println(f, "      </Points>")

        println(f, "      <Verts>")
        println(f, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
        for i in 0:(n_tri - 1)
            @printf(f, "          %d\n", i)
        end
        println(f, "        </DataArray>")
        println(f, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %d\n", i)
        end
        println(f, "        </DataArray>")
        println(f, "      </Verts>")

        println(f, "      <PointData Vectors=\"NetForce\">")

        println(f, "        <DataArray type=\"Float32\" Name=\"NetForce\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e %.6e %.6e\n", net_Fx[i], net_Fy[i], net_Fz[i])
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"NetForceMagnitude\" format=\"ascii\">")
        for i in 1:n_tri
            mag = sqrt(net_Fx[i]^2 + net_Fy[i]^2 + net_Fz[i]^2)
            @printf(f, "          %.6e\n", mag)
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"Normal\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e %.6e %.6e\n", nx_cpu[i], ny_cpu[i], nz_cpu[i])
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"Area\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e\n", areas_cpu[i])
        end
        println(f, "        </DataArray>")

        println(f, "      </PointData>")

        println(f, "    </Piece>")
        println(f, "  </PolyData>")
        println(f, "</VTKFile>")
    end

    println("[NetForce] Exported VTK PolyData to $(vtp_filename)")
end
