# Host-side conforming finite-base C+H geometry for the real app preview.
# Included inside CFDBackend. There is no fallback to an A-plus construction.

function _ch_cross_count(gap,dn1;growth=1.15,minimum_count=4)
    half=0.;n=0
    while half<gap/2
        half+=dn1*growth^n;n+=1
        n<=512 || throw(ArgumentError("finite base needs more than 1024 cross cells at this wall spacing"))
    end
    max(minimum_count,2n)
end

# Match the native Q-A capped geometric wake rule before allocating the mesh.
# The surface solve meets te_ds to 1e-9 relative; use its conservative lower
# bound so a thin base cannot exhaust a fixed interactive profile's wake count.
function _ch_wake_count(profile_count,te_ds,foil,ch,quality)
    corner_xi(i)=ch.xi_ref+(foil.x[i]-ch.ref[1])*ch.streamwise[1]+
        (foil.y[i]-ch.ref[2])*ch.streamwise[2]
    wake_length=ch.xi_out-min(corner_xi(1),corner_xi(length(foil.x)))
    isfinite(wake_length) && wake_length>0 || throw(ArgumentError("C+H outlet must lie downstream of both trailing-edge corners"))
    interval=max(te_ds*(1-2e-9),1e-6)
    distance=0.;required=0
    while distance<wake_length
        distance+=min(interval,quality.wake_ds_max)
        interval*=quality.far_growth
        required+=1
        required<=2048 || throw(ArgumentError("C+H wake needs more than 2048 parent cells; select a less demanding interactive mesh"))
    end
    (;count=max(profile_count,required),required,wake_length)
end

function _build_ch_mesh(af,flow;resolution="standard",solver_backend="transonic")
    G=CMeshGen;Q=G.CMeshQualityMetrics
    conditioned=G.condition_airfoil(af;trailing_edge_thickness=_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05))
    foil=conditioned.airfoil
    gap=hypot(foil.x[1]-foil.x[end],foil.y[1]-foil.y[end])
    gap>1e-8 || throw(ArgumentError("a zero-thickness TE uses the sharp C mesh, not a zero-area H block"))
    re=_number(flow,"reynolds",300000.;lo=1e3,hi=1e9)
    dn1=.7/(re*sqrt((.026re^(-1/7))/2))
    nf,nw,nn=get(Dict("standard"=>(512,320,160),"fine"=>(512,384,176),
        "very_fine"=>(640,448,192)),resolution,(512,320,160))
    factor=8
    # Optional stall-mesh knobs (21 Oct 2026); absent, the mesh uses this
    # resolution's standard geometry and curvature-aware surface spacing.
    # cfd_ch_pair_factor: the paired subdivision of each parent column (the parent's trailing-edge spacing
    # stays factor*dn1, so the child corner width becomes factor/pair dn1); cfd_le_ds_scale scales the
    # parent nose target; cfd_stall_layer rebuilds the C block's columns with the stall layer
    # (stall_layer_mesh.jl: wall row, interface rows and outer boundary unchanged, so H is unchanged).
    # cfd_ch_te_multiple: the parent's trailing-edge spacing in wall spacings (default factor); with pair 1 it
    # is the corner cell's width, which the corner gate G2l bounds by twice the base's first cross spacing
    # (the wall spacing). cfd_ch_foil_cells overrides the profile's parent surface cells. The H block's
    # cross count stays the default parent's (it is rebuilt graded from both interfaces).
    pair=_integer(flow,"cfd_ch_pair_factor",factor;lo=1,hi=factor)
    kte=_number(flow,"cfd_ch_te_multiple",Float64(factor);lo=1.,hi=Float64(factor))
    nf=_integer(flow,"cfd_ch_foil_cells",nf;lo=128,hi=4096)
    le_scale=_number(flow,"cfd_le_ds_scale",1.;lo=.1,hi=1.)
    stall=_boolean(flow,"cfd_stall_layer",false)
    # Standard and finer meshes use the curvature-weighted arc-length law so
    # surface columns follow the rapid turning around the leading edge.
    sdist=Symbol(_get(flow,"cfd_ch_surface_distribution","curvature"))
    sdist in (:legacy,:curvature) || throw(ArgumentError("cfd_ch_surface_distribution must be legacy or curvature"))
    nb=max(2,ceil(Int,.75gap/(kte*dn1)))
    nb<=256 || throw(ArgumentError("the requested Reynolds number needs a larger finite-base mesh than this interactive profile supports"))
    nb_cross=max(2,ceil(Int,.75gap/(factor*dn1)))
    wall_dn1=min(dn1,gap/16,.9gap/(2nb))
    te_ds=clamp(kte*wall_dn1,gap/(2nb)*(1+1e-8),gap/nb*(1-1e-8))
    # cfd_ch_h_cross (opt-in, 23 Oct 2026, the Gurney flap): "graded" rebuilds the H block's cross distribution with the
    # geometric count alone (growth 1.15 from the wall spacing at both legs, at least 8 cells) instead of the parent's
    # near-uniform 2 x 0.75h/(8 dn1); absent, the count is the default one bitwise.
    hcross=String(_get(flow,"cfd_ch_h_cross","default"))
    hcross in ("default","graded") || throw(ArgumentError("cfd_ch_h_cross must be default or graded"))
    ns=hcross=="graded" ? _ch_cross_count(gap,wall_dn1;minimum_count=8) : _ch_cross_count(gap,wall_dn1;minimum_count=max(8,2nb_cross))
    edge=max(wall_dn1,gap/ns)
    angle=_number(flow,"cfd_farfield_angle_deg",0.;lo=-45,hi=45)
    ch=G.ChannelSpec(height=60.,xi_in=-29.5,xi_out=30.5,angle_deg=angle)
    q=G.mesh_quality_v1(:strip;wake=(.11,1.15),strip=(1.15,min(4,nb-1)),gates=false)
    requested_nw=nw
    wake=_ch_wake_count(nw,te_ds,foil,ch,q)
    nw=wake.count
    le_kappa=maximum(abs(conditioned.curvature[i]) for i in eachindex(foil.x) if foil.x[i]<=minimum(foil.x)+.02)
    le_target=clamp(le_scale*.03/max(le_kappa,1.),4wall_dn1,.0006)
    # Try curvature-resolved leading-edge stations first. Every profile is
    # measured; a refused geometry is returned for inspection with its failures.
    best=nothing;errors=String[]
    for distribution in (sdist===:curvature ? (:curvature,:legacy) : (:legacy,))
      for (le,fillet) in ((le_target,.09),(0.,.08))
        try
            parent=G.generate_cmesh(foil;n_foil=nf,n_wake=nw,n_normal=nn,dn1=wall_dn1,
                growth_max=1.15,smooth_iters=80,bl_freeze=12,channel=ch,
                te_topology=:strip,n_base=nb,te_ds,le_ds=le,quality=q,
                ray_anchor_fillet=fillet,strip_leg_edge_spacing=edge,
                surface_evaluator=:natural_c2,surface_distribution=distribution,
                surface_curvature_weight=3.0)
            m,meta=G.paired_uniform_strip_refine(parent,foil;factor=pair,te_margin=pair==1 ? 1. : 1.05,
                cross_count=ns,cross_growth_max=1.15,cap=gap/8,check_air_gates=false)
            (m.NI*m.NJ+m.strip.NI*m.strip.NJ)<=3_000_000 ||
                throw(ArgumentError("requested C+H refinement exceeds three million interactive cells"))
            patch=_nested_outlet_patch!(m;ncols=4pair,sweeps=20,
                start_row=round(Int,68nn/128),full_row=round(Int,108nn/128))
            stall_info=nothing
            if stall
                sl=_stall_layer_mesh(m;common=true);m=sl.mesh;stall_info=sl.info
                # The remap moves the outer rows along their columns, undoing part of the outlet patch's
                # orthogonality (G2h): re-apply the same patch on the outer section's rows.
                o=sl.info["n_inner"]+sl.info["n_plateau"]
                _nested_outlet_patch!(m;ncols=4pair,sweeps=20,start_row=o+4,full_row=o+40)
                stall_info["outlet_repatch_rows"]=[o+4,o+40]
            end
            measured=G._paired_strip_quality_report(m;check_air_gates=false)
            score=(length(measured.failed),measured.summary.theta_max)
            if best===nothing || score<best.score
                best=(;mesh=m,refinement=meta,measured,score,le,fillet,patch,stall_info,distribution)
            end
            measured.passed && break
        catch err
            push!(errors,sprint(showerror,err))
        end
      end
      best!==nothing && best.measured.passed && break
    end
    best===nothing && throw(ArgumentError("C+H mesh construction refused: "*join(errors," | ")))
    m=best.mesh;s=best.measured.summary
    cavity=Q.quality_gates(s;cavitating=true,strip_grade=true)
    final_refinement=copy(best.refinement)
    final_refinement["air_gates_passed"]=best.measured.passed
    final_refinement["air_gate_failures"]=best.measured.failed
    final_refinement["quality_check_requested"]=true
    final_refinement["admitted"]=best.measured.passed
    final_refinement["admission_scope"]="final post-outlet-patch air geometry gates; no solver qualification"
    metadata=Dict{String,Any}("mesh_family"=>"conforming C + finite-base H",
        "mesh_topology"=>"conformal_C_H_v1","topology_requested"=>"auto",
        "geometry_pipeline_revision"=>2,"surface_evaluator"=>"natural_c2",
        "surface_distribution"=>String(best.distribution),
        "quality_passed"=>best.measured.passed,
        "quality_status"=>best.measured.passed ? "passed_air_geometry" : "refused_air_geometry",
        "quality_failures"=>best.measured.failed,
        "solver_supported"=>solver_backend=="transonic",
        "solver_backends"=>["transonic"],
        "solver_support_message"=>solver_backend=="transonic" ?
            "Compressible C+H Run: conservative gas/energy and fully turbulent SST, steady pseudo-time, constant kinematic viscosity. Numerical convergence and aerodynamic validation are separate." :
            "C+H Run is available for the compressible backend. The Species/cavitation app driver currently requires Legacy A-plus.",
        "quality_scope"=>"geometry and air gates only; not CFD convergence or cavitation validation",
        "cavitation_quality_passed"=>all(r[5] for r in cavity),
        "cavitation_quality_failures"=>Q.gate_failures(cavity),
        "source_te_gap"=>hypot(af.x[1]-af.x[end],af.y[1]-af.y[end]),"cfd_te_gap"=>gap,
        "wall_dn1"=>wall_dn1,"n_base"=>ns÷2,"wall_surface_cells"=>m.NIF,
        "conditioned_x"=>foil.x,"conditioned_y"=>foil.y,"conditioning"=>conditioned.metadata,
        "leading_edge_target_spacing"=>best.le,"leading_edge_max_curvature"=>le_kappa,
        "parent_wake_cells_requested"=>requested_nw,"parent_wake_cells"=>nw,
        "parent_wake_minimum_cells"=>wake.required,"wake_length_max"=>wake.wake_length,
        "wake_count_adjustment"=>nw>requested_nw ? "increased to satisfy unchanged Q-A wake spacing and growth limits" : "profile count already satisfies Q-A wake limits",
        "ray_anchor_fillet"=>best.fillet,"pair_refinement_factor"=>pair,
        "refinement"=>final_refinement,"maximum_face_nonorthogonality_deg"=>s.theta_max,
        "quality_report"=>Dict("air_reference_gates"=>Dict("passed"=>best.measured.passed,"failed"=>best.measured.failed),
            "geometric_gate"=>Dict("passed"=>true),"cells"=>s.cells,
            "global"=>Dict("maximum_face_nonorthogonality_deg"=>s.theta_max,"minimum_corner_angle_deg"=>s.minang,
                "maximum_aspect_ratio"=>s.ar_max,"maximum_neighbor_volume_ratio"=>s.vr_max),
            "interfaces"=>Dict("maximum_volume_ratio"=>s.vr_junction,
                "maximum_nonorthogonality_deg"=>s.theta_junction,"maximum_skew_over_distance"=>s.skew_junction)))
    if stall || pair!=factor || le_scale!=1 || kte!=factor || _get(flow,"cfd_ch_foil_cells",nothing)!==nothing || sdist!==:legacy
        metadata["stall_mesh"]=Dict{String,Any}("pair_refinement_factor"=>pair,"le_ds_scale"=>le_scale,
            "surface_distribution"=>String(best.distribution),
            "surface_curvature_weight"=>best.distribution===:curvature ? 3.0 : 0.0,
            "parent_te_spacing_over_dn1"=>kte,"child_te_corner_width_over_dn1"=>kte/pair,
            "parent_foil_cells"=>nf,"parent_n_base"=>nb,"h_cross_cells"=>ns,
            "stall_layer"=>best.stall_info===nothing ? "off" : best.stall_info)
        stall && (metadata["mesh_family"]*=" + stall layer")
    end
    hcross=="graded" && (metadata["h_cross_rule"]=Dict{String,Any}("rule"=>"graded","h_cross_cells"=>ns,
        "default_h_cross_cells"=>_ch_cross_count(gap,wall_dn1;minimum_count=max(8,2nb_cross)),
        "scope"=>"opt-in cfd_ch_h_cross=graded: the H block's cross cells grow by 1.15 from the wall spacing at both legs (cap h/8)"))
    if sdist===:curvature && best.distribution===:legacy
        metadata["surface_distribution_fallback"]="curvature distribution refused by geometry or quality checks; legacy distribution used"
        metadata["surface_distribution_refusals"]=errors
    end
    (;mesh=m,metadata)
end

function _block_mesh_payload(X,Y;id,full=false,preview_stride_i=8,nose_range=nothing)
    ni,nj=size(X)
    ii=if full
        collect(1:ni)
    elseif id=="C" && nose_range!==nothing
        # Keep the overview light, but do not hide the actual nose columns in
        # the zoomed preview. The solver always receives every column.
        xmin,chord=nose_range
        dense=[i for i in 1:ni if 0 <= (X[i,1]-xmin)/chord < .01 && (i-1)%2==0]
        sort!(unique(vcat(collect(1:preview_stride_i:ni),dense,[ni])))
    else
        sort!(unique(vcat(collect(1:preview_stride_i:ni),[ni])))
    end
    jj=full ? collect(1:nj) : sort!(unique(vcat(collect(1:min(16,nj)),collect(1:max(1,cld(nj,81)):nj),[nj])))
    Dict{String,Any}("id"=>id,"kind"=>"structured","ni"=>length(ii),"nj"=>length(jj),
        "full_ni"=>ni,"full_nj"=>nj,"i_indices"=>ii,"j_indices"=>jj,
        "x"=>[Float64(X[i,j]) for j in jj for i in ii],"y"=>[Float64(Y[i,j]) for j in jj for i in ii],
        "stride_i"=>full ? 1 : preview_stride_i,"stride_j"=>length(jj)>1 ? maximum(diff(jj)) : 1,"node_order"=>"j-major, i-fastest")
end

function _write_ch_mesh_artifacts(m,metadata,outdir,resolution;streamlines=nothing)
    # Optional quality regions can have no cells; their extrema are +/-Inf.
    # Preserve that as unavailable metadata, never as an invalid wire number.
    safe(x::AbstractDict)=Dict(string(k)=>safe(v) for (k,v) in pairs(x))
    safe(x::AbstractVector)=[safe(v) for v in x]
    safe(x::Tuple)=[safe(v) for v in x]
    safe(x::AbstractFloat)=isfinite(x) ? x : nothing
    safe(x)=x
    metadata=safe(metadata)
    function payload(full)
        surface_x=@view m.X[m.NIW+1:m.NIW+m.NIF+1,1]
        nose_range=(minimum(surface_x),maximum(surface_x)-minimum(surface_x))
        blocks=[_block_mesh_payload(m.X,m.Y;id="C",full,nose_range),
                _block_mesh_payload(m.strip.X,m.strip.Y;id="H",full)]
        p=Dict{String,Any}("kind"=>"structured_blocks","ni"=>blocks[1]["ni"],"nj"=>blocks[1]["nj"],
            "full_ni"=>m.NI+1,"full_nj"=>m.NJ+1,"blocks"=>blocks,"resolution"=>resolution,
            "wall_x"=>collect(m.X[m.NIW+1:m.NIW+m.NIF+1,1]),
            "wall_y"=>collect(m.Y[m.NIW+1:m.NIW+m.NIF+1,1]),
            "topology"=>metadata["mesh_topology"],"mesh_topology"=>metadata["mesh_topology"],
            "quality"=>metadata["quality_report"],"geometry_metadata"=>metadata,
            "quality_passed"=>metadata["quality_passed"],"quality_status"=>metadata["quality_status"],
            "quality_failures"=>metadata["quality_failures"],"solver_supported"=>metadata["solver_supported"],
            "solver_support_message"=>metadata["solver_support_message"],
            "total_cells"=>m.NI*m.NJ+m.strip.NI*m.strip.NJ,"normal_cells"=>m.NJ,
            "wake_cells_per_side"=>m.NIW,"foil_surface_cells"=>m.NIF,
            "base_faces_per_side"=>m.strip.n_base,"wall_dn1"=>m.dn1,
            "preview"=>!full,"full_resolution_available"=>true,
            "interfaces"=>[Dict("from"=>"C","to"=>"H","side"=>"lower","conforming"=>true),
                Dict("from"=>"C","to"=>"H","side"=>"upper","conforming"=>true)])
        # a streamline-aligned mesh (opt-in, development) carries the JXfoil paths it follows; absent otherwise
        streamlines===nothing || (p["streamlines"]=safe(streamlines))
        p
    end
    preview=payload(false)
    write(joinpath(outdir,"mesh_preview.msgpack"),AirfoilWire.encode(preview,:msgpack))
    full=payload(true)
    bytes=AirfoilWire.encode(full,:msgpack)
    length(bytes)<=AirfoilWire.MAX_BYTES || throw(ArgumentError("full C+H artifact exceeds transport limit; select a smaller interactive resolution"))
    write(joinpath(outdir,"mesh_full.msgpack"),bytes)
    preview
end
