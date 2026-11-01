# App orchestration for the conservative two-block thermal-gas/SST worker.
# Included inside CFDBackend; no GPU package is loaded in the HTTP process.

function _auto_ch_selected(case,path,flow)
    String(_get(flow,"cfd_te_topology","aplus"))=="auto" || return false
    source=_get(case,"airfoil",Dict())
    af=_get(source,"kind","")=="naca4" ?
        CMeshGen.naca4_points(parse(Int,String(_get(source,"id","")))) :
        CMeshGen.read_airfoil_dat(path)
    conditioned=CMeshGen.condition_airfoil(af;
        trailing_edge_thickness=_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05))
    f=conditioned.airfoil
    hypot(f.x[1]-f.x[end],f.y[1]-f.y[end])>1e-8
end

function _ch_transonic_config(case,path,flow,alpha,dir)
    mode=String(_get(flow,"cfd_mesh","aplus_nested_refined"))
    mode in ("aplus_nested_refined","cavitation_aplus_farfield") || throw(ArgumentError(
        "Auto C+H preserves the finite trailing edge. Select the shared mesh profile to run it."))
    time_mode=String(_get(flow,"cfd_time_mode","steady"))
    time_mode=="steady" || throw(ArgumentError(
        "C+H currently supports steady local pseudo-time. Select Steady pseudo-time in the flow controls."))
    viscosity=String(_get(flow,"cfd_viscosity_model","constant_kinematic"))
    viscosity=="constant_kinematic" || throw(ArgumentError(
        "C+H currently supports constant kinematic air viscosity. Select Constant kinematic in the flow controls."))
    !_boolean(flow,"cfd_sfd_enabled",false) || throw(ArgumentError(
        "C+H uses conservative coupled SSPRK3 without an SFD filter."))
    resolution=String(_get(flow,"cfd_resolution","standard"))
    resolution in ("standard","fine","very_fine") || throw(ArgumentError("invalid C+H resolution"))
    cfg=Dict{String,Any}(
        "geometry"=>String(path),"source_geometry"=>String(path),"outdir"=>String(dir),
        "mesh_mode"=>"conformal_C_H_v1","resolution"=>resolution,
        "mach"=>_number(flow,"mach",.1;lo=.02,hi=.98),
        "reynolds"=>_number(flow,"reynolds",300000.;lo=1e3,hi=1e9),
        "alpha_deg"=>Float64(alpha),"cfd_te_thickness"=>_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05),
        # Must match _build_ch_mesh/Create Mesh, including nonzero alpha sweeps.
        "cfd_farfield_angle_deg"=>_number(flow,"cfd_farfield_angle_deg",0.;lo=-45,hi=45),
        "steps"=>_integer(flow,"cfd_steps",500000;lo=1,hi=2000000),
        "history_stride"=>_integer(flow,"cfd_history_stride",25;lo=1,hi=1000000),
        "display_every"=>_integer(flow,"cfd_display_every",0;lo=0,hi=1000000),
        "gpu"=>_boolean(flow,"cfd_gpu",true),"time_mode"=>time_mode,
        "viscosity_model"=>viscosity,"cfl"=>_number(flow,"cfd_cfl",.1;lo=.01,hi=.5),
        "max_pseudo_dt"=>_number(flow,"cfd_max_pseudo_dt",.01;lo=1e-10,hi=1.),
        "auto_stop"=>_boolean(flow,"cfd_auto_stop",true),
        "mass_ceiling"=>_number(flow,"cfd_mass_ceiling",.1;lo=1e-6,hi=1000.),
        "mass_net_limit"=>_number(flow,"cfd_mass_net_limit",.001;lo=1e-8,hi=1.),
        "rk_weight_closure"=>true,"sst_wall_metric_correction"=>true,
        "sfd_enabled"=>false,"trip_upper"=>0.,"trip_lower"=>0.,
        "transition_model"=>"fully_turbulent",
        "requested_trip_upper"=>_number(flow,"trip_upper",0.;lo=0,hi=1),
        "requested_trip_lower"=>_number(flow,"trip_lower",0.;lo=0,hi=1))
    source=_get(case,"airfoil",Dict())
    if _get(source,"kind","")=="naca4"
        cfg["naca4_code"]=parse(Int,String(_get(source,"id","")))
    end
    _get(flow,"cfd_ch_h_cross",nothing)===nothing || (cfg["cfd_ch_h_cross"]=String(_get(flow,"cfd_ch_h_cross","default")))
    # Opt-in stall-study keys (21 Oct 2026): written only when requested, so a default job's input is unchanged.
    _boolean(flow,"cfd_irsj",false) && (cfg["irsj"]=true)
    _boolean(flow,"cfd_stall_layer",false) && (cfg["cfd_stall_layer"]=true)
    _get(flow,"cfd_le_ds_scale",nothing)===nothing ||
        (cfg["cfd_le_ds_scale"]=_number(flow,"cfd_le_ds_scale",1.;lo=.1,hi=1.))
    _get(flow,"cfd_ch_pair_factor",nothing)===nothing ||
        (cfg["cfd_ch_pair_factor"]=_integer(flow,"cfd_ch_pair_factor",8;lo=1,hi=8))
    _get(flow,"cfd_ch_te_multiple",nothing)===nothing ||
        (cfg["cfd_ch_te_multiple"]=_number(flow,"cfd_ch_te_multiple",8.;lo=1.,hi=8.))
    _get(flow,"cfd_ch_foil_cells",nothing)===nothing ||
        (cfg["cfd_ch_foil_cells"]=_integer(flow,"cfd_ch_foil_cells",512;lo=128,hi=4096))
    _get(flow,"cfd_ch_surface_distribution",nothing)===nothing ||
        (cfg["cfd_ch_surface_distribution"]=String(_get(flow,"cfd_ch_surface_distribution","legacy")))
    # Streamline alignment (opt-in, development, 24 Oct 2026): the hashed JXfoil path file (written in the point folder by
    # the server) and the morph table; the worker applies it as Create mesh does (ch_streamline.jl). Absent otherwise.
    if streamline_enabled(flow)
        cfg["streamline"]=streamline_worker_cfg(source,flow,alpha,dir)
    end
    cfg
end

# The job response contains a mesh both in each point and in its per-angle
# lookup. Keep those previews bounded for a 31-angle binary response. Every
# full mesh/field artifact remains unchanged and retains all solver values.
function _ch_transport_slots(n,step;first_rows=0)
    sort!(unique(vcat(collect(1:step:n),collect(1:min(first_rows,n)),[n])))
end

function _ch_transport_block(block,ii,jj;mesh=false)
    b=copy(block);ni=Int(block["ni"])
    take(values)=[values[i+ni*(j-1)] for j in jj for i in ii]
    if mesh
        for key in ("x","y");b[key]=take(block[key]);end
    else
        b["samples"]=Dict(k=>take(v) for (k,v) in block["samples"])
        b["sample_shape"]=[length(ii),length(jj)]
    end
    for (key,selected) in (("i_indices",ii),("j_indices",jj))
        b[key]=[block[key][k] for k in selected]
    end
    b["ni"]=length(ii);b["nj"]=length(jj)
    di=length(ii)>1 ? maximum(diff(Int.(b["i_indices"]))) : 1
    dj=length(jj)>1 ? maximum(diff(Int.(b["j_indices"]))) : 1
    if mesh
        b["stride_i"]=di;b["stride_j"]=dj
    else
        b["sample_stride"]=max(di,dj)
        b["sample_stride_i"]=di;b["sample_stride_j"]=dj
    end
    b["transport_preview"]=true
    b
end

function _ch_transport_geometry(metadata,full_mesh)
    value=copy(metadata)
    for key in ("conditioned_x","conditioned_y")
        if haskey(value,key)
            value[key*"_count"]=length(value[key]);delete!(value,key)
        end
    end
    value["conditioned_geometry_artifact"]=full_mesh
    value["conditioned_geometry_scope"]="Complete conditioned support coordinates remain in the full mesh artifact; exact solver wall coordinates remain in wall_x/y"
    value
end

function _ch_transport_mesh(mesh,step;full_mesh=nothing,full_mesh_loader=nothing)
    value=copy(mesh);c,h=mesh["blocks"]
    hi=_ch_transport_slots(Int(h["ni"]),step)
    # Both displayed interface polylines must use the same original abscissas.
    # Selecting the two blocks independently can leave a misleading preview
    # mismatch when the parent wake count is not divisible by the display step.
    nw=Int(mesh["wake_cells_per_side"]);nf=Int(mesh["foil_surface_cells"])
    lookup=Dict(Int(v)=>k for (k,v) in enumerate(c["i_indices"]))
    stream=Int.(h["i_indices"][hi])
    native=vcat(nw .+ 2 .- stream,nw .+ nf .+ stream)
    surface=findall(v->nw+1<=v<=nw+nf+1,c["i_indices"])
    le=nw+argmin(mesh["wall_x"])
    append!(native,Int.(c["i_indices"][surface[_ch_transport_slots(length(surface),step)]]))
    append!(native,[nw+1,nw+nf+1,le]);sort!(unique!(native))
    cj=_ch_transport_slots(Int(c["nj"]),step;first_rows=16)
    hj=_ch_transport_slots(Int(h["nj"]),step;first_rows=16)
    compact_c=if all(v->haskey(lookup,v),native)
        _ch_transport_block(c,[lookup[v] for v in native],cj;mesh=true)
    else
        # Adaptive wake counts need not share the original C/H preview stride.
        # Read missing junction/LE nodes from the unchanged full artifact rather
        # than interpolating them or dropping valid geometry from the preview.
        full_mesh===nothing && full_mesh_loader!==nothing && (full_mesh=full_mesh_loader())
        full_mesh===nothing && throw(ArgumentError("Full C+H mesh is required to preserve adaptive preview junctions"))
        source=only(b for b in full_mesh["blocks"] if b["id"]=="C")
        full_i=Dict(Int(v)=>k for (k,v) in enumerate(source["i_indices"]))
        full_j=Dict(Int(v)=>k for (k,v) in enumerate(source["j_indices"]))
        _ch_transport_block(source,[full_i[v] for v in native],
            [full_j[Int(c["j_indices"][k])] for k in cj];mesh=true)
    end
    value["blocks"]=[compact_c,_ch_transport_block(h,hi,hj;mesh=true)]
    value["ni"]=compact_c["ni"];value["nj"]=length(cj)
    value["transport_preview"]=true
    value["transport_preview_scope"]="Indexed display subset; exact wall and complete solver mesh remain available"
    value
end

_ch_transport_wire_value(x::AbstractDict)=Dict(string(k)=>_ch_transport_wire_value(v) for (k,v) in x)
_ch_transport_wire_value(x::AbstractVector{UInt8})=x
_ch_transport_wire_value(x::AbstractVector)=[_ch_transport_wire_value(v) for v in x]
_ch_transport_wire_value(x::Tuple)=[_ch_transport_wire_value(v) for v in x]
_ch_transport_wire_value(x::AbstractFloat)=isfinite(x) ? Float64(x) : nothing
_ch_transport_wire_value(x)=x

function _compact_ch_transport(point;max_point_bytes=700*1024)
    # Geometry support arrays were repeated in three metadata dictionaries.
    # Their full, immutable copy is already embedded in mesh_full.msgpack.
    base=copy(point);full_mesh=point["mesh"]["artifact"]
    base["geometry"]=_ch_transport_geometry(point["geometry"],full_mesh)
    diagnostics=copy(point["diagnostics"])
    haskey(diagnostics,"geometry") && (diagnostics["geometry"]=_ch_transport_geometry(diagnostics["geometry"],full_mesh))
    base["diagnostics"]=diagnostics
    original_mesh=copy(point["mesh"])
    original_mesh["geometry_metadata"]=_ch_transport_geometry(original_mesh["geometry_metadata"],full_mesh)
    full_cache=Ref{Any}(nothing)
    full_loader()=begin
        full_cache[]===nothing && (full_cache[]=AirfoilWire.decode(read(full_mesh),:msgpack))
        full_cache[]
    end
    for step in (4,5,6,8,10,12,16,24,32,48,64,128)
        value=copy(base);value["mesh"]=_ch_transport_mesh(original_mesh,step;full_mesh_loader=full_loader)
        fields=copy(point["fields"])
        fields["blocks"]=[_ch_transport_block(b,_ch_transport_slots(Int(b["ni"]),step),
            _ch_transport_slots(Int(b["nj"]),step)) for b in fields["blocks"]]
        fields["transport_preview"]=true
        fields["transport_preview_scope"]="Exact indexed cell subset; all final cells and quantities remain in full_artifact"
        value["fields"]=fields
        value["transport_preview"]=Dict("version"=>1,"selection_stride"=>step,
            "maximum_point_bytes"=>max_point_bytes,"formats"=>["msgpack","bson"],
            "scope"=>"Display previews only; native mesh, full fields, coefficients and residual evidence are unchanged")
        # Match the HTTP layer's JSON-like representation. Optional unavailable
        # extrema in native TOML diagnostics are Inf and become wire nulls.
        value=_ch_transport_wire_value(value)
        all(fmt->length(AirfoilWire.encode(value,fmt))<=max_point_bytes,(:msgpack,:bson)) && return value
    end
    throw(ArgumentError("C+H diagnostic metadata exceeds the binary job preview budget; full artifacts are retained"))
end

function _collect_ch_transonic_output(dir,alpha)
    meta=TOML.parsefile(joinpath(dir,"result.toml"))
    mesh=AirfoilWire.decode(read(joinpath(dir,"mesh_preview.msgpack")),:msgpack)
    mesh["artifact"]=joinpath(dir,"mesh_full.msgpack")
    mesh["preview_artifact"]=joinpath(dir,"mesh_preview.msgpack")
    fields=AirfoilWire.decode(read(joinpath(dir,meta["fields_blocks_msgpack"])),:msgpack)
    get(fields,"kind",nothing)=="structured_blocks" || error("C+H worker did not return block fields")
    blocks=get(fields,"blocks",Any[])
    [get(b,"id",nothing) for b in blocks]==["C","H"] || error("C+H fields must retain both blocks")
    actual_mesh_hash=bytes2hex(sha256(read(joinpath(dir,"mesh_full.msgpack"))))
    fields["mesh_sha256"]==actual_mesh_hash || error("C+H field/mesh provenance differs")
    for b in blocks
        shape=get(b,"sample_shape",[get(b,"ni",0),get(b,"nj",0)])
        length(shape)==2 && all(>(1),shape) || error("Invalid C+H field sample shape")
        count=prod(shape)
        all(k->haskey(b["samples"],k),FullFieldArtifact.COLUMNS) || error("C+H field quantities missing")
        all(a->length(a)==count && all(isfinite,a),values(b["samples"])) || error("C+H field data invalid")
        haskey(b,"full_artifact") && (b["full_artifact"]=joinpath(dir,b["full_artifact"]))
    end
    full_name=get(fields,"full_artifact","fields_blocks_full.msgpack")
    fields["full_artifact"]=joinpath(dir,full_name)
    fields["artifact"]=joinpath(dir,meta["fields_blocks_msgpack"])
    fields["full_cell_count"]=get(fields,"total_cells",get(meta,"mesh_cells",0))
    fields["full_iteration"]=meta["steps"]
    cp=Dict{String,Any}("upper"=>Any[],"lower"=>Any[])
    for row in _csv_rows(joinpath(dir,meta["cp_csv"]))
        push!(cp[row["surface"]],[_float(row["x"]),_float(row["cp"])])
    end
    history=Any[]
    rows=_csv_rows(joinpath(dir,meta["iteration_csv"]))
    # Streaming keeps the complete live history; bound the final JSON preview.
    stride=max(1,cld(length(rows),1200))
    for k in unique(vcat(collect(1:stride:length(rows)),length(rows)))
        row=rows[k]
        sample=Dict{String,Any}("iteration"=>parse(Int,row["iteration"]))
        for key in ("residual_scaled","residual_linf","residual_l2","cl","cd","cm","mass_net_frac")
            haskey(row,key) || continue
            value=_float(row[key]);value===nothing || (sample[key]=value)
        end
        sample["residual"]=get(sample,"residual_scaled",nothing)
        push!(history,sample)
    end
    guard=get(meta,"convergence_guard",Dict{String,Any}("status"=>"max_steps","reason"=>"Iteration cap reached; inspect all six equation residuals"))
    converged=get(meta,"aerodynamic_converged",false) && get(guard,"all_six_equations_confirmed",false) && get(mesh,"quality_passed",false)
    state=converged ? "completed" : get(guard,"status","")=="stalled_stop" ? "stopped_without_convergence" : "not_converged"
    diagnostics=Dict{String,Any}(string(k)=>v for (k,v) in meta)
    merge!(diagnostics,Dict("mesh_topology"=>"conformal_C_H_v1","mesh_cells"=>fields["full_cell_count"],
        "time_mode"=>"steady","clock"=>"Unequal local pseudo-time; no elapsed physical time",
        "convergence_guard"=>guard,"mesh_quality_passed"=>get(mesh,"quality_passed",false),
        "residual_scope"=>"All six native conservative gas/energy/SST equations at the final accepted state",
        "physical_time"=>false,"validated"=>false,"aerodynamic_parity_established"=>false))
    notes=String[
        "Conservative two-block compressible gas/energy and SST on the same C+H mesh as Create Mesh.",
        "Fully turbulent SST with constant kinematic viscosity. Natural transition and prescribed XFOIL trips are different models.",
        "Local pseudo-time is a steady numerical relaxation, not a physical transient. Inspect all six residuals before using coefficients."]
    converged || push!(notes,"The final mesh, fields and coefficients are saved diagnostics; this point is not a converged polar point.")
    get(mesh,"quality_passed",false) || push!(notes,"The mesh has failed air geometry limits; inspect the mesh quality diagnostics.")
    point=Dict{String,Any}("backend"=>"transonic","alpha_deg"=>alpha,"status"=>state,
        "cl"=>meta["cl"],"cd"=>meta["cd"],"cm"=>meta["cm"],
        "convergence"=>get(guard,"reason","Inspect all six native equation residuals"),
        "validation"=>"experimental coupled C+H; aerodynamic parity unvalidated",
        "reference_compatible"=>get(meta,"requested_trip_upper",0.)==0 && get(meta,"requested_trip_lower",0.)==0,
        "geometry"=>get(meta,"geometry",get(mesh,"geometry_metadata",Dict())),
        "mesh"=>mesh,"fields"=>fields,"cp"=>cp,"iteration_history"=>history,
        "diagnostics"=>diagnostics,"notes"=>notes,
        "artifacts"=>Dict("mesh_full"=>joinpath(dir,"mesh_full.msgpack"),
            "mesh_preview"=>joinpath(dir,"mesh_preview.msgpack"),
            "field_preview"=>fields["artifact"],"field_full"=>fields["full_artifact"],
            "cp_csv"=>joinpath(dir,meta["cp_csv"]),"iteration_csv"=>joinpath(dir,meta["iteration_csv"]),
            "result_toml"=>joinpath(dir,"result.toml"),"log"=>joinpath(dir,"solver.log")))
    _compact_ch_transport(point)
end

function _ch_transonic_point(case,path,flow,alpha,dir,progress,log)
    cfg=_ch_transonic_config(case,path,flow,alpha,dir)
    _deliver_log(log,"Starting coupled C+H compressible gas/energy and fully turbulent SST; steady, constant kinematic viscosity, $(cfg["gpu"] ? "GPU" : "CPU").")
    if cfg["requested_trip_upper"]>0 || cfg["requested_trip_lower"]>0
        _deliver_log(log,"C+H uses fully turbulent SST; requested trip locations remain reference/XFOIL settings and are recorded separately.")
    end
    cfg_path=joinpath(dir,"input.toml")
    open(cfg_path,"w") do io;TOML.print(io,cfg);end
    julia=joinpath(Sys.BINDIR,Base.julia_exename())
    worker=joinpath(@__DIR__,"cfd_backend.jl")
    cmd=`$julia --project=$SRC --startup-file=no -t auto $worker --ch-transonic-worker $cfg_path`
    ok=_log_process(cmd,joinpath(dir,"solver.log"),progress,log,"transonic",Float64(alpha);
        residual_kind="convective continuity L-infinity; all six equations retained")
    ok || throw(ErrorException("Coupled C+H worker failed; see solver.log"))
    point=_collect_ch_transonic_output(dir,Float64(alpha))
    haskey(cfg,"streamline") && append!(point["notes"],streamline_point_notes(get(point["mesh"]["geometry_metadata"],"streamline_alignment",nothing)))
    point
end
