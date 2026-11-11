# Restart on a separately built, finer octree. This is a startup aid, not a
# flux-conservative dynamic AMR method. The old checkpoint is never modified.

"Validate physical equivalence before transferring a checkpoint between meshes."
function check_refinement_start(src::State{T}, dst::State{T}) where T
    a, b = src.cfg, dst.cfg
    all(isapprox.(src.mesh.domain,dst.mesh.domain;rtol=0,atol=1e-9)) || error("Refinement restart requires the same snapped domain")
    all(isapprox.(src.mesh.offset,dst.mesh.offset;rtol=0,atol=1e-9)) || error("Refinement restart requires the same model placement")
    src.mesh.periodic == dst.mesh.periodic && a.mesh.faces == b.mesh.faces || error("Refinement restart requires identical boundary conditions")
    a.stl_sha == b.stl_sha && a.stl_scale == b.stl_scale || error("Refinement restart requires the same geometry")
    a.flow == b.flow || error("Refinement restart requires identical flow conditions")
    a.numerics == b.numerics && a.walls == b.walls || error("Refinement restart requires identical numerical and wall settings")
    a.sponge_thickness == b.sponge_thickness && a.sponge_strength == b.sponge_strength || error("Refinement restart requires identical outlet sponge settings")
    a.reference_length == b.reference_length && a.mass_control == b.mass_control || error("Refinement restart requires the same meshing reference length and mass-control setting")
    a.reference_chord == b.reference_chord && a.reference_area == b.reference_area && a.moment_center == b.moment_center || error("Refinement restart requires the same force reference dimensions")
    b.outputs.ramp_steps == a.outputs.ramp_steps || error("Refinement restart requires the same inlet ramp")
    nlevels(dst.mesh) > nlevels(src.mesh) || error("The target must have additional refinement levels")
    src.mesh.levels[1].dx == dst.mesh.levels[1].dx || error("Refinement restart requires the same coarse cell size")
    src.scaling.velocity_scale == dst.scaling.velocity_scale || error("Refinement restart requires the same velocity scale")
    isapprox(src.scaling.dt_finest * 2.0^(nlevels(src.mesh)-1), dst.scaling.dt_finest * 2.0^(nlevels(dst.mesh)-1); rtol=1e-12) || error("Refinement restart requires the same coarse physical time step")
    dst.cfg.outputs.steps > src.step || error("Target step limit must exceed the source checkpoint step")
    nothing
end

"Find a nearby source fluid cell without crossing the actual triangle surface."
function refinement_visible_donor(src::State, p::V3, h::Float64, bvh::BVH, stack::Vector{Int32}; radius::Int=2)
    bestlevel = 0; bestslot = 0; bestdistance = Inf
    for dz in -radius:radius, dy in -radius:radius, dx in -radius:radius
        dx == dy == dz == 0 && continue
        q = (p[1]+dx*h, p[2]+dy*h, p[3]+dz*h)
        l,s = locate_leaf(src.mesh,q); l == 0 && continue
        lv = src.mesh.levels[l]; k = slot_k(s); o = slot_oct(s)
        lv.kind[k,o] == K_FLUID || continue
        centre = slot_centre(lv,s)
        distance = sqrt(sum((p[i]-centre[i])^2 for i in 1:3))
        distance < bestdistance || continue
        _,tri = first_crossing(bvh,p,centre,stack)
        tri == 0 || continue
        bestlevel,bestslot,bestdistance = l,s,distance
    end
    bestlevel == 0 ? nothing : (bestlevel,bestslot,bestdistance)
end

"Reject nonfinite or negative physical populations before a transferred state can be used."
@inline function valid_refinement_populations(fp::NTuple{Q,T}) where T
    @inbounds for i in 1:Q
        isfinite(fp[i]) && fp[i]+T(WT[i])>=zero(T) || return false
    end
    true
end

"Seed a newly exposed near-wall cell only when the real surface is sufficiently close."
function refinement_wall_seed!(S::LevelState{T},k::Int,o::Int,p::V3,dx::Float64,bvh::BVH,stack::Vector{Int32}) where T
    distance=nearest_distance(bvh,p,stack)
    isfinite(distance) && distance <= 0.25dx || error("No visible source donor within eight cells, and target is not near a real surface: $p")
    @inbounds begin
        for i in 1:Q;S.f[k,i,o]=zero(T);S.g[k,i,o]=zero(T);end
        for a in 1:3;S.u[k,a,o]=zero(T);S.v[k,a,o]=zero(T);end
        S.rho[k,o]=one(T);S.rhob[k,o]=one(T)
    end
    distance
end

"""Transfer a completed coarse state to a newly allocated finer state.

The target is private to the caller until all checks succeed. Coarse-cell
moments are sampled at each new fluid cell, with a bounded visibility-checked
fluid donor for newly exposed cells. Non-equilibrium stress is rescaled using
the solver's existing AMR moment convention. Geometric fluid volume may change:
the report records it; no global density rescaling is silently imposed.
"""
function transfer_refined!(dst::State{T}, src::State{T}) where T
    check_refinement_start(src,dst)
    initialise_uniform!(dst,1.0,src.u_inlet,0.0,0.0)
    bvh = isempty(src.mesh.triangles) ? nothing : build_bvh(src.mesh.triangles)
    fallback_counts = zeros(Int,Threads.nthreads())
    crossed_counts = zeros(Int,Threads.nthreads())
    expanded_counts = zeros(Int,Threads.nthreads())
    fallback_distances = zeros(Float64,Threads.nthreads())
    wall_seed_points = [V3[] for _ in 1:Threads.nthreads()]
    wall_seed_distances = zeros(Float64,Threads.nthreads())
    for (l,L) in enumerate(dst.mesh.levels)
        S = dst.levels[l]
        @threads :static for t in eachindex(S.chunks)
            stack = bvh === nothing ? Int32[] : new_stack()
            local_count = 0; local_crossed = 0; local_expanded = 0; local_max = 0.0
            for oi in S.chunks[t]
                o = Int(S.real_octs[oi])
                for k in 1:8
                    L.kind[k,o] == K_FLUID || continue
                    p = slot_centre(L,slot_id(k,o))
                    sl,ss = locate_leaf(src.mesh,p)
                    sl != 0 || error("No source leaf at target point $p")
                    Ls = src.mesh.levels[sl]; sk = slot_k(ss); so = slot_oct(ss)
                    fallback = false
                    needs_donor = Ls.kind[sk,so] != K_FLUID
                    if !needs_donor && bvh !== nothing && Ls.dx > L.dx
                        # A coarse fluid voxel may straddle a thin sheet. Its
                        # centre is not necessarily on the target's side.
                        _,tri = first_crossing(bvh,p,slot_centre(Ls,ss),stack)
                        if tri != 0; needs_donor=true;local_crossed+=1; end
                    end
                    if needs_donor
                        bvh === nothing && error("Target fluid cell has no source fluid donor at $p")
                        donor=nothing;radius_used=0
                        for radius in (2,4,8)
                            donor=refinement_visible_donor(src,p,Ls.dx,bvh,stack;radius)
                            if donor !== nothing;radius_used=radius;break;end
                        end
                        if donor === nothing
                            # A few newly exposed cells can be enclosed by an
                            # open/nonmanifold source's solid classification.
                            # Only directly next to a real surface, start them
                            # at reference-density no-slip equilibrium. This is
                            # a recorded initial guess, not a measured pressure.
                            distance=refinement_wall_seed!(S,k,o,p,L.dx,bvh,stack)
                            push!(wall_seed_points[t],p)
                            wall_seed_distances[t]=max(wall_seed_distances[t],distance)
                            continue
                        end
                        sl,ss,distance = donor
                        Ls = src.mesh.levels[sl]; sk = slot_k(ss); so = slot_oct(ss)
                        local_count += 1; local_max = max(local_max,distance); fallback = true
                        radius_used>2 && (local_expanded+=1)
                    end
                    ratio = Ls.dx/L.dx
                    ratio >= 1-1e-12 || error("Refinement restart would coarsen a source fluid cell")
                    Ss = src.levels[sl]
                    rho,ux,uy,uz,pn = slot_moments(Ss.f,sk,so)
                    all(isfinite,(rho,ux,uy,uz,pn...)) && T(0.2) < rho < T(5) && ux^2+uy^2+uz^2 < T(0.25) || error("Source donor has an unstable state at $p")
                    # Preserve the complete population tuple in unchanged cells;
                    # moment reconstruction is needed only for a finer or
                    # visibility-extrapolated donor.
                    fp = if !fallback && ratio == 1 && slot_centre(Ls,ss) == p
                        ntuple(i->(@inbounds Ss.f[sk,i,so]),Val(Q))
                    else
                        scale = (inv(S.omega)/(T(ratio)*inv(Ss.omega))) * pre_collision_factor(Ss.omega)
                        rebuild(rho,ux,uy,uz,ntuple(i->scale*pn[i],6),S.omega)
                    end
                    valid_refinement_populations(fp) || error("Refinement produced a nonfinite or negative population at $p")
                    @inbounds for i in 1:Q; S.f[k,i,o]=fp[i]; S.g[k,i,o]=fp[i]; end
                    @inbounds begin
                        S.u[k,1,o]=S.v[k,1,o]=ux; S.u[k,2,o]=S.v[k,2,o]=uy; S.u[k,3,o]=S.v[k,3,o]=uz
                        S.rho[k,o]=S.rhob[k,o]=rho
                    end
                end
            end
            fallback_counts[t] += local_count
            crossed_counts[t] += local_crossed
            expanded_counts[t] += local_expanded
            fallback_distances[t] = max(fallback_distances[t],local_max)
        end
    end
    seeds=sum(length,wall_seed_points)
    seed_limit=max(64,ceil(Int,1e-4*dst.mesh.counts["fluid_total"]))
    seeds <= seed_limit || error("Refinement would initialize $seeds cells without a donor, exceeding the limit $seed_limit")
    nlevels(dst.mesh)>1 && couple_initial!(dst)
    dst.step=src.step; dst.u_inlet=src.u_inlet
    dst.mass0=total_mass(dst)
    rmin,rmax,umax=state_extrema(dst)
    isfinite(rmin) && isfinite(rmax) && isfinite(umax) && rmin>0.2 && umax<0.5 || error("Transferred state fails the solver stability gate")
    src_volume=sum(Float64(p["fluid"])*src.mesh.levels[p["level"]].dx^3 for p in src.mesh.counts["per_level"])
    dst_volume=sum(Float64(p["fluid"])*dst.mesh.levels[p["level"]].dx^3 for p in dst.mesh.counts["per_level"])
    source_mass=total_mass(src)*src.scaling.dx_finest^3
    target_mass=total_mass(dst)*dst.scaling.dx_finest^3
    source_momentum=total_momentum(src).*src.scaling.dx_finest^3
    target_momentum=total_momentum(dst).*dst.scaling.dx_finest^3
    Dict{String,Any}("source_step"=>src.step,"source_levels"=>nlevels(src.mesh),"target_levels"=>nlevels(dst.mesh),
        "source_fluid_volume_m3"=>src_volume,"target_fluid_volume_m3"=>dst_volume,
        "source_mass_lattice_m3"=>source_mass,"target_mass_lattice_m3"=>target_mass,
        "source_density_anomaly_integral_m3"=>source_mass-src_volume,"target_density_anomaly_integral_m3"=>target_mass-dst_volume,
        "source_momentum_lattice_m3"=>source_momentum,"target_momentum_lattice_m3"=>target_momentum,
        "visible_fallback_cells"=>sum(fallback_counts),"expanded_search_cells"=>sum(expanded_counts),"cross_surface_donors_rejected"=>sum(crossed_counts),"maximum_fallback_distance_m"=>maximum(fallback_distances),
        "wall_equilibrium_seed_cells"=>seeds,"wall_equilibrium_seed_max_surface_distance_m"=>maximum(wall_seed_distances),
        "wall_equilibrium_seed_positions_m"=>[collect(p) for points in wall_seed_points for p in points],
        "rho_min"=>rmin,"rho_max"=>rmax,"u_max_lattice"=>umax,
        "note"=>"Startup transfer is not exactly conservative. Geometry voxelization changes fluid volume; no density correction was applied.",
        "history_note"=>"Both population buffers start from the current source state. Prior half-time history is not available on the new mesh; a restart transient is expected.",
        "wall_seed_note"=>"Only cells within one quarter of a target cell from a real surface and with no visible donor within eight source cells may start at no-slip reference-density equilibrium; their coordinates are recorded. These values are initial guesses.")
end

"Load an already built source mesh without creating or replacing files."
function refinement_source_mesh(cfg::CaseConfig;role::AbstractString="source")
    for path in (joinpath(cfg.dir,"mesh.oct"),joinpath(dirname(cfg.dir),"mesh","mesh.oct"))
        isfile(path) || continue
        mesh=read_mesh(path)
        _require_current_cavity_policy(cfg,mesh)
        mesh.config_sha == cfg.config_sha && mesh.stl_sha == cfg.stl_sha || continue
        return mesh,path
    end
    error("No existing $role mesh matches the $role case; it will not be rebuilt or overwritten")
end

"Load a completed source stage and transfer its final checkpoint into a new target state."
function refinement_start(cfg::CaseConfig, targetmesh::OctMesh, source_dir::AbstractString;
                          expected_checkpoint::Union{Nothing,String}=nothing,
                          expected_bytes::Union{Nothing,Int}=nothing,
                          expected_mtime::Union{Nothing,Float64}=nothing)
    sourcecfg=load_case(source_dir;build_only=false)
    realpath(sourcecfg.dir) != realpath(cfg.dir) || error("Refinement source and target must be different case directories")
    sourcecfg.precision == cfg.precision || error("Refinement source and target precisions must match")
    checkpoint_dir=joinpath(sourcecfg.dir,"RESULTS","checkpoints")
    isdir(checkpoint_dir) || error("Source stage has no checkpoint directory")
    files=sort(filter(f->startswith(f,"checkpoint_") && endswith(f,".bin"),readdir(checkpoint_dir)))
    isempty(files) && error("Source stage has no completed checkpoint")
    checkpoint=joinpath(checkpoint_dir,files[end])
    if expected_checkpoint !== nothing
        isfile(expected_checkpoint) && !islink(expected_checkpoint) || error("The admitted source checkpoint is unavailable or not a regular file")
        realpath(expected_checkpoint)==realpath(checkpoint) || error("The source checkpoint changed after admission")
    end
    before=stat(checkpoint)
    expected_bytes === nothing || before.size==expected_bytes || error("The source checkpoint size changed after admission")
    expected_mtime === nothing || before.mtime==expected_mtime || error("The source checkpoint modification time changed after admission")
    meshpaths=(joinpath(sourcecfg.dir,"mesh.oct"),joinpath(dirname(sourcecfg.dir),"mesh","mesh.oct"))
    existing=filter(isfile,meshpaths);isempty(existing) && error("Source mesh is missing")
    # A mesh file can expand substantially while being read. Keep a real RAM
    # reserve before loading it; the stricter combined check follows below.
    reserve=max(8_000_000_000,round(Int,.15*Sys.total_memory()))
    Sys.free_memory() > 3filesize(first(existing))+reserve || error("Not enough free RAM to load the source mesh with the required reserve")
    sourcemesh,source_mesh_path=refinement_source_mesh(sourcecfg)
    required=estimated_run_bytes(sourcecfg,sourcemesh)+estimated_run_bytes(cfg,targetmesh)
    # Both meshes are already resident. Their included size makes this a
    # conservative check, rather than underestimating the simultaneous states.
    Sys.free_memory() > required+reserve || error("Insufficient free RAM for both refinement-stage states plus reserve; use a coarser target")
    src=State(sourcemesh,sourcecfg,sourcecfg.precision)
    read_checkpoint!(src,checkpoint,nothing)
    after=stat(checkpoint)
    before.size == after.size && before.mtime == after.mtime || error("Source checkpoint changed while it was being read")
    src.step == sourcecfg.outputs.steps || error("The source stage must finish before a finer stage starts (checkpoint step $(src.step), configured $(sourcecfg.outputs.steps))")
    dst=State(targetmesh,cfg,cfg.precision)
    report=transfer_refined!(dst,src)
    report["source_case"]=abspath(sourcecfg.dir);report["source_mesh"]=abspath(source_mesh_path)
    report["source_checkpoint"]=abspath(checkpoint);report["source_checkpoint_bytes"]=before.size
    report["source_checkpoint_mtime"]=before.mtime
    report["source_physical_time_s"]=src.step*src.scaling.dt_finest*2.0^(nlevels(src.mesh)-1)
    report["written_utc"]=Dates.format(Dates.now(Dates.UTC),"yyyy-mm-ddTHH:MM:SS")*"Z"
    # Release the old level arrays before the new solve allocates output work.
    src=nothing;sourcemesh=nothing;GC.gc(true)
    dst,report
end
