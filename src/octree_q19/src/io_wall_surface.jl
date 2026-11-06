# Fine surface diagnostics, independent of the solver's original-facet force integration.
# The static geometry stores one mean link intersection per (fluid wall cell, original facet).
# All links in such a group use the SAME cell density: coalescing loses no pressure information.
# No extrapolation to unsampled triangles, no mirrored half and no fabricated local wall shear.
const WALL_GEOMETRY_MAGIC = "LBMWGEO\0"
const WALL_FIELD_MAGIC = "LBMWFLD\0"
const WALL_GEOMETRY_RECORD_BYTES = 36
const WALL_FIELD_HEADER_BYTES = 128
const WALL_GEOMETRY_CACHE = Dict{Tuple{UInt,String},Dict{String,Any}}()

"Visit supported wall-cell/facet groups in deterministic level, cut-cell, first-facet order."
function foreach_wall_group(f::F, mesh::OctMesh) where F
    nt = length(mesh.triangles)
    normals = [tri_normal_area(T)[1] for T in mesh.triangles]
    valid = [all(isfinite, n) && vnorm(n) > 0.5 for n in normals]
    ids = zeros(Int, Q); links = zeros(Int, Q); sums = zeros(Float64, 3, Q)
    samples=0; contributing=0; unmapped=0; invalid=0; outside=0; ambiguous=0
    perlevel=zeros(Int,nlevels(mesh))
    for (l, Lv) in enumerate(mesh.levels)
        for (ci, s32) in enumerate(Lv.cut_slot)
            s = Int(s32)
            Lv.kind[s] == K_FLUID || continue
            c = slot_centre(Lv, s); ng = 0
            for li in Lv.cut_off[ci]:Lv.cut_off[ci+1]-1
                t = Int(Lv.cut_tri[li])
                if t == 0
                    unmapped += 1; continue
                end
                i = Int(Lv.cut_dir[li]); q = Float64(Lv.cut_q[li])
                if !(1 <= t <= nt && valid[t] && 2 <= i <= Q && isfinite(q) && 0 <= q <= 1)
                    invalid += 1; continue
                end
                p = (c[1]+q*Lv.dx*CX[i], c[2]+q*Lv.dx*CY[i], c[3]+q*Lv.dx*CZ[i])
                if !all(a -> -1e-6Lv.dx <= p[a] <= mesh.domain[a]+1e-6Lv.dx, 1:3)
                    outside += 1; continue
                end
                g = 1
                while g <= ng && ids[g] != t; g += 1; end
                if g > ng
                    ng += 1; ids[g] = t; links[g] = 0
                    sums[1,g] = sums[2,g] = sums[3,g] = 0.0
                end
                links[g] += 1
                for a in 1:3; sums[a,g] += p[a]; end
            end
            for g in 1:ng
                t = ids[g]; nlinks = links[g]
                p = (sums[1,g]/nlinks, sums[2,g]/nlinks, sums[3,g]/nlinks)
                n = normals[t]; side = vdot(n, vsub(c,p))
                side < 0 && (n = vmul(n, -1.0))
                flags = UInt16(1) # supported geometric intersection
                if abs(side) <= 1e-10Lv.dx
                    flags |= UInt16(2); ambiguous += 1
                end
                f(l, ci, s, t, p, n, Lv.dx, nlinks, flags)
                samples += 1; contributing += nlinks; perlevel[l] += 1
            end
        end
    end
    Dict{String,Any}("samples"=>samples,"contributing_links"=>contributing,"unmapped_links"=>unmapped,
        "invalid_links"=>invalid,"outside_domain_links"=>outside,"ambiguous_normal_samples"=>ambiguous,
        "samples_per_level"=>perlevel)
end

function _wall_flush!(io, buffer::IOBuffer; force=false)
    (force || position(buffer) >= 1_048_576) && write(io, take!(buffer))
    nothing
end

_wall_sha(path) = open(io -> bytes2hex(sha256(io)), path)
_wall_tmp(path) = path*".tmp."*string(getpid())*"."*string(rand(UInt64);base=16)

function _wall_atomic_toml(path, metadata)
    tmp = _wall_tmp(path)
    try
        open(tmp, "w") do io; TOML.print(io, metadata); end
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp)
    end
end

"Publish immutable scientific data; a different existing frame is never silently overwritten."
function _wall_publish(tmp, path)
    if isfile(path)
        filesize(tmp) == filesize(path) && _wall_sha(tmp) == _wall_sha(path) ||
            error("Refusing to replace different existing wall-surface data: $path")
        rm(tmp)
    else
        mv(tmp, path)
    end
    path
end

function wall_surface_geometry(mesh::OctMesh, out::AbstractString)
    vtk = abspath(joinpath(out,"VTK")); mkpath(vtk)
    key = (objectid(mesh.levels), vtk)
    if haskey(WALL_GEOMETRY_CACHE, key)
        cached = WALL_GEOMETRY_CACHE[key]
        if isfile(cached["path"])
            filesize(cached["path"]) == 24+36cached["samples"] && _wall_sha(cached["path"]) == cached["geometry_sha256"] ||
                error("Stored wall geometry changed or is corrupt: $(cached["path"])")
            return cached
        end
    end
    tmp = _wall_tmp(joinpath(vtk,"surface_wall_geometry.lbmwg"))
    try
        stats = open(tmp,"w+") do io
            write(io,codeunits(WALL_GEOMETRY_MAGIC),UInt32(1),UInt32(WALL_GEOMETRY_RECORD_BYTES),UInt64(0))
            buffer = IOBuffer()
            summary = foreach_wall_group(mesh) do l,ci,s,t,p,n,dx,nlinks,flags
                write(buffer,Float32(p[1]),Float32(p[2]),Float32(p[3]),Float32(n[1]),Float32(n[2]),Float32(n[3]),
                    Float32(dx),UInt32(t),UInt8(l),UInt8(nlinks),flags)
                _wall_flush!(io,buffer)
            end
            _wall_flush!(io,buffer;force=true)
            seek(io,16); write(io,UInt64(summary["samples"]))
            summary
        end
        sha = _wall_sha(tmp)
        path = joinpath(vtk,"surface_wall_geometry_"*sha*".lbmwg")
        _wall_publish(tmp,path)
        metadata = merge(stats,Dict{String,Any}("format"=>"LBMWGEO","format_version"=>1,"record_bytes"=>36,
            "geometry_sha256"=>sha,"geometry_frame"=>"domain","representation"=>"wall_cell_facet_intersections",
            "domain_size_m"=>collect(mesh.domain),"mesh_offset_m"=>collect(mesh.offset),
            "config_sha256"=>mesh.config_sha,"stl_sha256"=>mesh.stl_sha,"original_triangles"=>length(mesh.triangles),
            "sample_order"=>"level, cut-cell order, first occurrence of original facet in cell links",
            "position_note"=>"Mean of actual cut-link intersections sharing one fluid cell and one original facet; no interpolated or mirrored samples.",
            "normal_note"=>"Unit geometric facet normal oriented toward the contributing fluid cell, independent of STL winding.",
            "flags_note"=>"bit 0: supported intersection; bit 1: grazing/ambiguous fluid-facing orientation",
            "parent_facet_index_base"=>1,"byte_order"=>"little_endian"))
        _wall_atomic_toml(path*".toml",metadata)
        info = merge(metadata,Dict{String,Any}("path"=>path))
        WALL_GEOMETRY_CACHE[key] = info
        println("[WallSurface] static geometry: $(stats["samples"]) native cell/facet samples, $(stats["contributing_links"]) links, $(stats["unmapped_links"]) unmapped links omitted")
        flush(stdout)
        info
    finally
        isfile(tmp) && rm(tmp)
    end
end

"Write local pressure from rho(l, cut_cell_index, slot); buffers stay bounded independently of sample count."
function write_wall_surface!(mesh::OctMesh, cfg::CaseConfig, rho::F, out::AbstractString, step::Int;
                             source::AbstractDict=Dict{String,Any}()) where F
    ENDIAN_BOM == 0x04030201 || error("Wall-surface v1 currently requires a little-endian host")
    step >= 0 || error("Wall-surface step must be nonnegative")
    geometry = wall_surface_geometry(mesh,out)
    sc = lattice_scaling(cfg); qref = 0.5cfg.flow.density*cfg.flow.velocity^2
    cs2rho = sc.velocity_scale^2/3*cfg.flow.density
    all(isfinite,(qref,cs2rho)) && qref > 0 || error("Wall pressure requires a positive finite reference dynamic pressure")
    path = joinpath(out,"VTK",@sprintf("surface_%06d.lbmw",step)); tmp = _wall_tmp(path)
    pmin = fill(Inf,length(mesh.triangles)); pmax = fill(-Inf,length(mesh.triangles))
    pressure_stats = [0.0,Inf,-Inf]
    try
        stats = open(tmp,"w") do io
            write(io,codeunits(WALL_FIELD_MAGIC),UInt32(1),UInt32(1),UInt64(geometry["samples"]),
                codeunits(geometry["geometry_sha256"]),Int64(step),Float64(step*sc.dt_finest*2.0^(nlevels(mesh)-1)),
                Float64(cfg.flow.density),Float64(cfg.flow.velocity),Float64(qref))
            position(io) == WALL_FIELD_HEADER_BYTES || error("Internal wall-field header size mismatch")
            buffer = IOBuffer()
            summary = foreach_wall_group(mesh) do l,ci,s,t,p,n,dx,nlinks,flags
                r = Float64(rho(l,ci,s)); pressure = Float32(isfinite(r) && r > 0 ? (r-1)*cs2rho : NaN)
                if isfinite(pressure)
                    pressure_stats[1] += 1; pressure_stats[2] = min(pressure_stats[2],pressure); pressure_stats[3] = max(pressure_stats[3],pressure)
                    pmin[t] = min(pmin[t],pressure); pmax[t] = max(pmax[t],pressure)
                end
                write(buffer,pressure); _wall_flush!(io,buffer)
            end
            _wall_flush!(io,buffer;force=true)
            summary
        end
        stats["samples"] == geometry["samples"] || error("Wall geometry and field sample counts differ")
        filesize(tmp) == 128+4stats["samples"] || error("Wall field size mismatch")
        finite_count=Int(pressure_stats[1]); minp=pressure_stats[2]; maxp=pressure_stats[3]
        varying = count(t -> isfinite(pmin[t]) && pmax[t]-pmin[t] > max(1e-5,1e-6qref),eachindex(pmin))
        metadata = Dict{String,Any}("format"=>"LBMWFLD","format_version"=>1,"geometry_file"=>basename(geometry["path"]),
            "geometry_sha256"=>geometry["geometry_sha256"],"geometry_frame"=>"domain","representation"=>"wall_cell_facet_intersections",
            "samples"=>stats["samples"],"finite_samples"=>finite_count,"step"=>step,
            "config_sha256"=>mesh.config_sha,"stl_sha256"=>mesh.stl_sha,"source"=>Dict{String,Any}(source),
            "physical_time_s"=>step*sc.dt_finest*2.0^(nlevels(mesh)-1),"reference_density_kg_m3"=>cfg.flow.density,
            "reference_speed_m_s"=>cfg.flow.velocity,"reference_dynamic_pressure_pa"=>qref,
            "pressure_min_pa"=>finite_count>0 ? minp : NaN,"pressure_max_pa"=>finite_count>0 ? maxp : NaN,
            "original_facets_with_pressure_variation"=>varying,"pressure_unit"=>"Pa","cp_reference"=>"Cp=pressure/reference_dynamic_pressure_pa",
            "pressure_note"=>"Local adjacent fluid-cell density: rho0*(Uref/u_lattice)^2/3*(rho_lattice-1). No facet averaging or wall extrapolation.",
            "coverage_note"=>"Only actual supported intersections in the solved domain. Unmapped/fallback links and unsampled or excluded geometry are not filled.",
            "shear_available"=>false,"convergence_assessed"=>false,"byte_order"=>"little_endian")
        # Geometry and metadata precede the atomic frame publication; readers only discover complete frames.
        exists = isfile(path)
        if exists
            filesize(tmp)==filesize(path) && _wall_sha(tmp)==_wall_sha(path) || error("Refusing to replace existing wall frame $path")
        end
        if exists && isfile(path*".toml")
            previous=TOML.parsefile(path*".toml")
            get(previous,"geometry_sha256",nothing)==geometry["geometry_sha256"] &&
                get(previous,"samples",nothing)==stats["samples"] && get(previous,"step",nothing)==step ||
                error("Existing wall frame metadata does not match its data: $path")
            # Idempotent export must not erase an earlier checkpoint/source audit trail.
        else
            _wall_atomic_toml(path*".toml",metadata)
        end
        _wall_publish(tmp,path)
        println("[WallSurface] step $step: $(stats["samples"]) samples, $finite_count finite, $varying original facets have independently varying pressure; $path")
        flush(stdout)
        path
    finally
        isfile(tmp) && rm(tmp)
    end
end

write_wall_surface!(st::State,out::AbstractString,step::Int=st.step) =
    write_wall_surface!(st.mesh,st.cfg,(l,ci,s)->st.levels[l].rho[s],out,step;
        source=Dict("kind"=>"live_solver_state","step"=>step,
            "collision"=>String(st.cfg.numerics.collision),
            "hrr_sigma"=>st.cfg.numerics.collision==:hrr ? st.cfg.numerics.hrr_sigma : 1.0))

"Read only wall-cell densities from a checkpoint; never allocate populations, advance, repair or remesh."
function checkpoint_wall_densities(path::AbstractString, mesh::OctMesh, cfg::CaseConfig)
    checkpoint_bytes = filesize(path)
    open(path,"r") do io
        String(read(io,length(CHECKPOINT_MAGIC)))==CHECKPOINT_MAGIC || error("Not an octree Q19 checkpoint")
        read(io,Int64)==1 || error("Unsupported checkpoint version")
        bytes = read(io,Int64); bytes in (4,8) || error("Unsupported checkpoint precision")
        bytes == sizeof(cfg.precision) || error("Checkpoint precision differs from the case")
        T = bytes==4 ? Float32 : Float64
        csha=String(read(io,64));ssha=String(read(io,64))
        strip(csha)==mesh.config_sha==cfg.config_sha || error("Checkpoint/config/mesh physical setup identity differs")
        strip(ssha)==mesh.stl_sha==cfg.stl_sha || error("Checkpoint/config/mesh STL identity differs")
        step=Int(read(io,Int64));read(io,Float64);read(io,Float64);read(io,Int64)
        skip(io,9sizeof(Float64)) # running force coefficient sums, unchanged
        read(io,Int64)==nlevels(mesh) || error("Checkpoint level count differs from mesh")
        densities=Vector{Vector{T}}(undef,nlevels(mesh))
        for (l,Lv) in enumerate(mesh.levels)
            no=read(io,Int64);no==Lv.nocts || error("Checkpoint level $l oct count differs from mesh")
            slots=8no
            # f,g,u,v: 2Q populations and 6 velocities per stored slot.
            skip(io,slots*(2Q+6)*sizeof(T))
            position(io)+slots*sizeof(T) <= checkpoint_bytes || error("Truncated checkpoint density block at level $l")
            full=Vector{T}(undef,slots);read!(io,full)
            densities[l]=T[full[Int(s)] for s in Lv.cut_slot]
            full=nothing
            skip(io,slots*sizeof(T)) # rhob
            nw=read(io,Int64);nw==length(Lv.cut_slot) || error("Checkpoint wall-cell count differs at level $l")
            skip(io,nw*sizeof(T)) # tau_w
            position(io)<=checkpoint_bytes || error("Truncated checkpoint at level $l")
        end
        position(io)==checkpoint_bytes || error("Checkpoint has unexpected trailing data")
        step,densities
    end
end

"Export a stored checkpoint without a solver run. Existing mesh and checkpoint are read-only."
function export_checkpoint_wall_surface(case_dir::AbstractString, checkpoint::AbstractString;
        mesh_path::AbstractString=joinpath(case_dir,"mesh.oct"),out::AbstractString=joinpath(case_dir,"RESULTS"))
    isfile(mesh_path) || error("Existing mesh required; this command never creates a mesh: $mesh_path")
    isfile(checkpoint) || error("Checkpoint not found: $checkpoint")
    cfg=load_case(case_dir;build_only=false)
    before=stat(checkpoint)
    println("[WallSurface] Read-only checkpoint export: loading existing mesh; no simulation will advance.");flush(stdout)
    mesh=read_mesh(mesh_path)
    mesh.config_sha==cfg.config_sha && mesh.stl_sha==cfg.stl_sha || error("Mesh does not match this case")
    step,rhos=checkpoint_wall_densities(checkpoint,mesh,cfg)
    println("[WallSurface] checkpoint step $step read; verifying checkpoint SHA256.");flush(stdout)
    sha=_wall_sha(checkpoint)
    after=stat(checkpoint)
    before.size==after.size && before.mtime==after.mtime || error("Checkpoint changed during read-only export; retry after stopping its writer")
    meshsha=open(mesh_path) do io;seekend(io);skip(io,-32);bytes2hex(read(io,32));end
    write_wall_surface!(mesh,cfg,(l,ci,s)->rhos[l][ci],out,step;
        source=Dict("kind"=>"checkpoint_read_only","checkpoint_file"=>abspath(checkpoint),
            "checkpoint_sha256"=>sha,"checkpoint_bytes"=>before.size,"mesh_file"=>abspath(mesh_path),
            "mesh_sha256"=>meshsha,"simulation_advanced"=>false))
end
