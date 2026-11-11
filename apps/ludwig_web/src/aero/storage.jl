# Estimates are intentionally explicit about what they include. Disk free space
# can change between this check and writes; it is not an allocation guarantee.
const Q19_OUTPUT_RESERVE_BYTES = 15_000_000_000
function _available_output_bytes(folder::AbstractString)
    isempty(strip(folder)) && return nothing
    path=abspath(expanduser(folder))
    while !isdir(path)
        parent=dirname(path); parent==path && return nothing; path=parent
    end
    try Int64(Base.Filesystem.diskstat(path).available) catch; nothing end
end

function _q19_storage_forecast(case,actual;mesh_bytes=0,available_bytes=nothing)
    outputs=get(case,"outputs",OUTPUT_DEFAULTS); octree=get(actual,"octree",Dict())
    reports=get(octree,"reports",Dict()); count=get(reports,"wall_sample_count",nothing)
    exact=count isa Integer && !(count isa Bool) && count>=0
    wallcells=get(reports,"wall_cells",nothing)
    samples=exact ? Int64(count) : Int64(ceil(1.15*max(0,wallcells isa Integer ? wallcells : 0)))
    steps=Int64(case["steps"]);every=max(1,min(steps,Int64(get(case,"effective_display_every",get(case,"display_every",100)))))
    frames=cld(steps,every)
    # Native wall data is retained by the solver even when its typed export is off.
    wall_samples=get(get(case,"octree",Dict()),"wall_samples",true)
    vtk=get(outputs,"surface_vtk",true) && wall_samples
    native=wall_samples && (get(outputs,"surface_vtk",true) || get(outputs,"surface_lbmp",true))
    per_frame=(vtk ? 39samples+2048 : 0)+(native ? 4samples+128 : 0)
    static_bytes=native ? 36samples+24 : 0
    core=Int64(mesh_bytes)+static_bytes+frames*per_frame
    # The checkpoint stores both population buffers and both velocity/density
    # buffers for *all allocated slots*, including ghosts and covered parents.
    # total_cells excludes some of those slots, so it is not a checkpoint count.
    # Aero's D3Q19 cases use f32 and always write a final checkpoint (30 min
    # periodic interval); admission must allow space for at least that file.
    stored=get(octree,"stored_slots",nothing)
    stored_exact=stored isa Integer && !(stored isa Bool) && stored>=0
    wall_exact=wallcells isa Integer && !(wallcells isa Bool) && wallcells>=0
    precision=get(octree,"precision",nothing)
    element_bytes=precision in ("Float64","f64") ? 8 : precision in ("Float32","f32") ? 4 : 0
    # File header is 264 bytes plus two Int64 size fields per level.
    levels=get(actual,"num_levels",nothing); levels_exact=levels isa Integer && !(levels isa Bool) && levels>0
    checkpoint_exact=stored_exact && wall_exact && levels_exact && element_bytes>0
    levels=levels_exact ? Int64(levels) : 0
    checkpoint_one=stored_exact && element_bytes>0 ? 46element_bytes*Int64(stored)+element_bytes*(wall_exact ? Int64(wallcells) : 0)+264+16levels : 0
    keep=Int64(get(get(case,"octree",Dict()),"checkpoint_keep",2))
    checkpoint=(keep+1)*checkpoint_one # atomic replacement may coexist with retained checkpoints
    estimate=core+checkpoint+Q19_OUTPUT_RESERVE_BYTES
    surfaces_exact=exact || !native
    minimum=core+(checkpoint_exact ? checkpoint_one : 0)+Q19_OUTPUT_RESERVE_BYTES
    blocked=surfaces_exact && available_bytes!==nothing && minimum>available_bytes
    Dict("samples"=>samples,"sample_count_exact"=>exact,"frames"=>frames,"every"=>every,
         "vtk_bytes_per_frame"=>(vtk ? 39samples+2048 : 0),"native_bytes_per_frame"=>(native ? 4samples+128 : 0),
         "core_bytes"=>core,"checkpoint_bytes"=>checkpoint_one,"checkpoint_size_exact"=>checkpoint_exact,
         "checkpoint_peak_bytes"=>checkpoint,"minimum_bytes"=>minimum,"reserve_bytes"=>Q19_OUTPUT_RESERVE_BYTES,
         "forecast_bytes"=>estimate,"available_bytes"=>available_bytes,"blocked"=>blocked,
         "volume_vtk"=>get(outputs,"volume_vtk",false))
end

function _q19_storage_for_job(id,case,actual)
    mesh=joinpath(_case_dir(id,"mesh"),"mesh.oct")
    mesh_bytes=isfile(mesh) ? filesize(mesh) : 0
    folder=get(get(case,"outputs",Dict()),"folder","")
    _q19_storage_forecast(case,actual;mesh_bytes,available_bytes=_available_output_bytes(folder))
end

function output_storage_estimate(id,req)
    updates=_body(req);state=_load_job(id)
    _apply_run_output_updates!(state,updates)
    actual=build_summary(id;lattice=_memory_lattice(state["case"]))
    if !_q19_case(state["case"]) || actual===nothing
        return _response(200,Dict("ok"=>true,"available"=>false))
    end
    _response(200,merge(Dict("ok"=>true,"available"=>true),_q19_storage_for_job(id,state["case"],actual)))
end
