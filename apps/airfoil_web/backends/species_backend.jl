# Conservative Species integration for external air and water. Included in
# CFDBackend; no separate surrogate solver is used here.

const UNDERWATER_UNAVAILABLE = "Underwater free-surface CFD is unavailable: the current conservative Species model has water/vapour transport but no gravity-balanced noncondensable air/water interface or surface-wave boundary. A depth pressure offset does not model a free surface."

function cavitation_configuration(flow)
    material=lowercase(String(_get(flow,"cav_fluid","water")))
    domain=lowercase(String(_get(flow,"cav_domain","tunnel")))
    material in ("air","water") || throw(ArgumentError("cav_fluid must be air or water"))
    domain in ("tunnel","free_fluid","underwater") || throw(ArgumentError("cav_domain must be tunnel, free_fluid or underwater"))
    domain == "underwater" && material!="water" && throw(ArgumentError("underwater requires water with a noncondensable air phase"))
    material=="air" && _number(flow,"mach",0.1;lo=0.02,hi=0.3)
    String(_get(flow,"cav_air_mode","steady")) in ("steady","physical") ||
        throw(ArgumentError("cav_air_mode must be steady or physical"))
    material,domain
end

function cavitation_capabilities()
    Dict("fluids"=>["water","air"],"domains"=>[
        Dict("id"=>"tunnel","available"=>true,"description"=>"Slip tunnel walls, characteristic inlet and fixed static-pressure outlet"),
        Dict("id"=>"free_fluid","available"=>true,"description"=>"Shared external mesh; characteristic outer contour with reflecting fixed-static-pressure downstream ends"),
        Dict("id"=>"underwater","available"=>true,"experimental"=>true,
            "description"=>"Physical-time nonlinear water/vapor/air interface with gravity and an impermeable airfoil; inviscid startup only, no phase change or stall validation")],
        "air_model"=>"Isothermal ideal gas; conservative Species AUSM+-up momentum, mass and SST; no total-energy equation",
        "air_mach_range"=>[0.02,0.3])
end

"Validated app-to-Species settings; defaults are selected by the existing route."
function _species_model_settings(flow;alpha=_number(flow,"alpha_deg",0.),
        default_nut="none",default_init="uniform",default_sst_limit=.5,
        default_eos="double",default_reconstruction=true,default_omega_init="freestream")
    xin=_number(flow,"cav_channel_xi_in",-4/3;lo=-1e4,hi=1e4)
    xout=_number(flow,"cav_channel_xi_out",5.;lo=-1e4,hi=1e4)
    xref=_number(flow,"cav_channel_xi_ref",.5;lo=-1e4,hi=1e4)
    xin<xref<xout || throw(ArgumentError("cav_channel_xi_in < cav_channel_xi_ref < cav_channel_xi_out is required"))
    nut=lowercase(String(_get(flow,"cav_nut_density",default_nut)))
    nut in ("none","reboud_v1","reboud_v2","reboud_v3") || throw(ArgumentError("cav_nut_density must be none, reboud_v1, reboud_v2, or reboud_v3"))
    rn=_number(flow,"cav_reboud_n",10.;lo=1.,hi=1000.)
    rn>1 || throw(ArgumentError("cav_reboud_n must be greater than one"))
    init=lowercase(String(_get(flow,"cav_init",default_init)))
    init in ("uniform","panel") || throw(ArgumentError("cav_init must be uniform or panel"))
    eos=lowercase(String(_get(flow,"cav_eos_precision",default_eos)))
    eos in ("single","double") || throw(ArgumentError("cav_eos_precision must be single or double"))
    omega=lowercase(String(_get(flow,"cav_sst_omega_init",default_omega_init)))
    omega in ("freestream","wall") || throw(ArgumentError("cav_sst_omega_init must be freestream or wall"))
    climit=_number(flow,"cav_sst_correction_limit",default_sst_limit;lo=-1,hi=1)
    reconstruction=_boolean(flow,"cav_species_reconstruction",default_reconstruction)
    climit>=0 && !reconstruction && throw(ArgumentError("A nonnegative cav_sst_correction_limit requires cav_species_reconstruction=true"))
    material=lowercase(String(_get(flow,"cav_fluid","water")))
    upper=_number(flow,"trip_upper",.05;lo=0,hi=1);lower=_number(flow,"trip_lower",.05;lo=0,hi=1)
    legacy_trip=material=="air" && upper==lower ? upper : 0.
    trip=_number(flow,"cav_trip_x",legacy_trip;lo=0,hi=.5)
    trip<.5 || throw(ArgumentError("cav_trip_x must be zero (off) or lie strictly between zero and 0.5"))
    width=_number(flow,"cav_trip_width",.005;lo=eps(Float64),hi=.1)
    density_ratio=_number(flow,"cav_density_ratio",43197.;lo=1.,hi=1e9)
    density_ratio>1 || throw(ArgumentError("cav_density_ratio must be greater than one"))
    Dict{String,Any}("cav_channel_height"=>_number(flow,"cav_channel_height",1.28;lo=.2,hi=100),
        "cav_channel_xi_in"=>xin,"cav_channel_xi_out"=>xout,"cav_channel_xi_ref"=>xref,
        "cav_channel_angle_deg"=>_number(flow,"cav_channel_angle_deg",alpha;lo=-90,hi=90),
        "cav_a2l"=>_number(flow,"cav_a2l",879.8;lo=eps(Float64),hi=1e9),
        "cav_density_ratio"=>density_ratio,
        "cav_epspv"=>_number(flow,"cav_epspv",.1;lo=eps(Float64),hi=10.),
        "cav_cprod"=>_number(flow,"cav_cprod",100.;lo=0,hi=1e6),
        "cav_cdest"=>_number(flow,"cav_cdest",5.;lo=0,hi=1e6),
        "cav_nut_density"=>nut,"cav_reboud_n"=>rn,"cav_trip_x"=>trip,"cav_trip_width"=>width,
        "cav_sst_correction_limit"=>climit,"cav_species_reconstruction"=>reconstruction,
        "cav_init"=>init,"cav_eos_precision"=>eos,"cav_sst_omega_init"=>omega)
end

function _species_channel_metadata(channel)
    Dict{String,Any}("height"=>channel.height,"xi_in"=>channel.xi_in,"xi_out"=>channel.xi_out,
        "xi_ref"=>channel.xi_ref,"reference_x"=>channel.ref[1],"reference_y"=>channel.ref[2],
        "angle_deg"=>atand(channel.streamwise[2],channel.streamwise[1]),
        "streamwise"=>collect(channel.streamwise),"scope"=>"actual ChannelSpec used by the mesh generator")
end

"""The production driver's mesh call. A finite base with Auto topology uses the
driver's strip construction: the conforming C block around foil plus strip and
the H block behind the base (`TOPO_MESH_TE_TOPOLOGY=strip`). The keywords are the
ones `tools/cmesh_cav_movie.jl` passes for the same environment, so the preview
and the Run build the same nodes."""
function _generate_production_tunnel_mesh(af,p,channel)
    extra=(;)
    haskey(p,"wake_near_ds") && (extra=(;wake_near_ds=p["wake_near_ds"],wake_near_len=p["wake_near_len"],
        wake_near_growth=p["wake_near_growth"]))
    get(p,"te_topology","aplus")=="strip" && (extra=merge(extra,(te_topology=:strip,)))
    quality=get(p,"mesh_quality","none")=="v1" ?
        (quality=CMeshGen.mesh_quality_v1(get(p,"te_topology","aplus")=="strip" ? :strip : :sharp),) : (;)
    CMeshGen.generate_cmesh(af;n_foil=p["n_foil"],n_wake=p["n_wake"],n_normal=p["n_normal"],
        dn1=p["dn1"],growth_max=p["growth"],R_ff=25.,L_wake=25.,smooth_iters=p["smooth_iters"],bl_freeze=6,
        le_ds=p["le_ds"],channel,n_base=p["n_base"],te_ds=p["te_ds"],
        surface_evaluator=Symbol(p["surface_evaluator"]),surface_distribution=Symbol(p["surface_distribution"]),
        extra...,quality...)
end

"One geometry/configuration plan for the production tunnel preview and driver."
function _production_tunnel_settings(path,flow,alpha)
    topology=String(_get(flow,"cfd_te_topology","aplus"))
    topology in ("auto","aplus") || throw(ArgumentError(
        "The production water tunnel accepts cfd_te_topology=auto (conforming C+H for a finite trailing edge, C for a sharp one) or aplus (Legacy A-plus)."))
    source_af=CMeshGen.read_airfoil_dat(path)
    te=_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05)
    af=te>0 ? CMeshGen.condition_airfoil(source_af;trailing_edge_thickness=te).airfoil : source_af
    gap=hypot(af.x[1]-af.x[end],af.y[1]-af.y[end]);finite=gap>1e-8
    # Auto with a finite base selects the production driver's strip topology:
    # the C block wraps foil plus strip, and the H block extrudes the base cells.
    strip=topology=="auto" && finite
    model=_species_model_settings(flow;alpha,default_nut="reboud_v3",default_eos="single",
        default_init=finite ? "panel" : "uniform",default_sst_limit=finite ? .5 : -1.,
        default_reconstruction=finite,default_omega_init=finite ? "wall" : "freestream")
    model["cav_channel_xi_ref"]==.5 || throw(ArgumentError(
        "The production water-tunnel driver requires cav_channel_xi_ref=0.5; other reference frames are not implemented."))
    abs(model["cav_channel_angle_deg"]-Float64(alpha))<=1e-10 || throw(ArgumentError(
        "The production water-tunnel driver rotates the channel with alpha. Omit cav_channel_angle_deg or set it equal to alpha; independent channel angles are not implemented."))
    channel=CMeshGen.ChannelSpec(height=model["cav_channel_height"],xi_in=model["cav_channel_xi_in"],
        xi_out=model["cav_channel_xi_out"],xi_ref=.5,ref=(.5,0.),angle_deg=Float64(alpha))
    resolution=String(_get(flow,"cfd_resolution","standard"))
    sizes=get(Dict("standard"=>(256,64,96),"fine"=>(384,96,144),"very_fine"=>(512,128,176)),resolution,nothing)
    sizes===nothing && throw(ArgumentError("invalid cfd_resolution"))
    # Strip default: 8 faces per half-base, the validated N60R/V1 level (BB8).
    nb=finite ? (strip ? _integer(flow,"cfd_n_base",8;lo=2,hi=16) : _integer(flow,"cfd_n_base",4;lo=1,hi=8)) : 0
    requested_te_ds=_number(flow,"cav_mesh_te_ds",0.;lo=0,hi=.099)
    te_ds=finite && requested_te_ds==0 ? 1.5gap/(2nb) : requested_te_ds
    nfoil=_integer(flow,"cfd_n_foil",sizes[1];lo=32,hi=2048)
    # The strip's near wake is held at h/n_base over 10 h (BB arms, 2 Oct 2026),
    # so its wake needs more columns than the A-plus default: 136 at n_base 8.
    strip_wake=strip ? max(sizes[2],ceil(Int,136*nb/8),ceil(Int,1.7*10nb)) : sizes[2]
    quality=lowercase(String(_get(flow,"cav_mesh_quality","none")))
    quality in ("none","v1") || throw(ArgumentError("cav_mesh_quality must be none or v1 (CMeshQualityV1)"))
    quality=="v1" && finite && !strip && throw(ArgumentError(
        "cav_mesh_quality=v1 does not support the Legacy A-plus construction; select Auto (C+H)"))
    dn1=_number(flow,"cav_mesh_dn1",2e-4;lo=1e-8,hi=.01)
    evaluator=String(_get(flow,"cav_mesh_surface_evaluator","natural_c2"))
    distribution=String(_get(flow,"cav_mesh_surface_distribution","curvature"))
    evaluator in ("legacy_catmull_rom","natural_c2") || throw(ArgumentError("invalid cav_mesh_surface_evaluator"))
    distribution in ("legacy","curvature") || throw(ArgumentError("invalid cav_mesh_surface_distribution"))
    distribution=="curvature" && evaluator!="natural_c2" && throw(ArgumentError("curvature mesh distribution requires natural_c2 geometry"))
    conditioned=CMeshGen.condition_airfoil(af)
    xmin=minimum(conditioned.airfoil.x)
    le_curvature=maximum(abs(conditioned.curvature[k]) for k in eachindex(conditioned.airfoil.x)
        if conditioned.airfoil.x[k]<=xmin+.03)
    default_le_ds=distribution=="curvature" ? max(1.05dn1,.03/max(le_curvature,1.)*256/nfoil) : 0.
    mesh=Dict{String,Any}("n_foil"=>nfoil,
        "n_wake"=>_integer(flow,"cfd_n_wake",strip_wake;lo=8,hi=512),
        "n_normal"=>_integer(flow,"cfd_n_normal",sizes[3];lo=16,hi=512),
        "dn1"=>dn1,
        "growth"=>_number(flow,"cav_mesh_growth",1.20;lo=1,hi=1.5),
        "smooth_iters"=>_integer(flow,"cav_mesh_smooth",80;lo=0,hi=2000),
        "n_base"=>nb,"te_ds"=>te_ds,"surface_evaluator"=>evaluator,"surface_distribution"=>distribution,
        "leading_edge_max_curvature"=>le_curvature,
        "le_ds"=>_number(flow,"cav_mesh_le_ds",default_le_ds;lo=0,hi=.099),
        "te_topology"=>strip ? "strip" : (finite ? "aplus" : "sharp"),"mesh_quality"=>quality)
    # The near wake is the strip's default; a sharp or A-plus tunnel takes it
    # only on request (the BS arm: the strip family's sharp control).
    if strip || _get(flow,"cav_mesh_wake_near_ds",nothing)!==nothing
        mesh["wake_near_ds"]=_number(flow,"cav_mesh_wake_near_ds",gap/max(nb,1);lo=1e-6,hi=.1)
        mesh["wake_near_len"]=_number(flow,"cav_mesh_wake_near_len",10gap;lo=1e-4,hi=5.)
        mesh["wake_near_growth"]=_number(flow,"cav_mesh_wake_near_growth",1.05;lo=1,hi=1.5)
    end
    prepared_mesh=nothing
    # Curvature redistribution can move a finite-base mesh's outer-column
    # transition. Admit that actual geometry with the existing unrelaxed
    # corner-sine gate before fixing the production input. Retain 80 sweeps
    # whenever it passes; only a named corner-sine refusal triggers more
    # smoothing. Explicit user requests are never silently changed.
    # (The strip's C block is the sharp construction; the corner-sine retry is A-plus's.)
    if finite && !strip && distribution=="curvature" && _get(flow,"cav_mesh_smooth",nothing)===nothing
        attempts=Int[]
        for sweeps in (80,400,800)
            mesh["smooth_iters"]=sweeps;push!(attempts,sweeps)
            try
                prepared_mesh=_generate_production_tunnel_mesh(af,mesh,channel)
                break
            catch err
                err isa ArgumentError && occursin("minimum corner sine",sprint(showerror,err)) && sweeps<800 || rethrow()
            end
        end
        mesh["automatic_smoothing_attempts"]=attempts
        mesh["automatic_smoothing_scope"]="same geometry and wall stations; first passing unchanged finite-base corner-sine gate"
    end
    (;source_af,af,finite,gap,te,model,channel,mesh,resolution,conditioned,prepared_mesh)
end

function _production_cavitation_environment(plan,path,flow,alpha,cav_out,frames,reynolds,mach)
    model=plan.model;mesh=plan.mesh
    env=Dict{String,String}(
        "TOPO_SPECIES"=>"1","TOPO_CAV"=>"4","TOPO_CAV_WARM"=>"4","TOPO_SPECIES_LEDGER"=>"1",
        "TOPO_OUTLET_PRESSURE"=>"ambient","TOPO_CHANNEL_H"=>string(plan.channel.height),
        "TOPO_CHANNEL_XMIN"=>string(plan.channel.xi_in),"TOPO_CHANNEL_XMAX"=>string(plan.channel.xi_out),
        "TOPO_FOIL"=>String(path),"TOPO_RE"=>string(reynolds),"TOPO_MACH"=>string(mach),
        "TOPO_ALPHA"=>string(alpha),"TOPO_MARCH_OMEGA"=>"1.0",
        "TOPO_A2L"=>string(model["cav_a2l"]),"TOPO_RATIO"=>string(model["cav_density_ratio"]),
        "TOPO_EPSPV"=>string(model["cav_epspv"]),"TOPO_CPROD"=>string(model["cav_cprod"]),"TOPO_CDEST"=>string(model["cav_cdest"]),
        "TOPO_SPECIES_NUT_DENSITY"=>model["cav_nut_density"],"TOPO_REBOUD_N"=>string(model["cav_reboud_n"]),
        "TOPO_SPECIES_REBOUD_PRODUCTION"=>"0","TOPO_SPECIES_SOURCE_MODEL"=>"merkle","TOPO_SPECIES_SOURCE_WEIGHT"=>"alpha",
        "TOPO_SPECIES_EOS"=>model["cav_eos_precision"],"TOPO_SPECIES_RECON"=>model["cav_species_reconstruction"] ? "1" : "0",
        "TOPO_SPECIES_SST_CORRECTION_LIMIT"=>string(model["cav_sst_correction_limit"]),
        "TOPO_SPECIES_TRIP"=>string(model["cav_trip_x"]),"TOPO_SPECIES_TRIP_WIDTH"=>string(model["cav_trip_width"]),
        "TOPO_INIT"=>model["cav_init"],"TOPO_SST_OMEGA_INIT"=>model["cav_sst_omega_init"],
        "TOPO_MESH"=>join((mesh["n_foil"],mesh["n_normal"],mesh["dn1"],mesh["growth"],mesh["smooth_iters"],mesh["n_wake"]),','),
        "TOPO_MESH_TE_BASE"=>string(mesh["n_base"]),"TOPO_MESH_TE_DS"=>string(mesh["te_ds"]),"TOPO_MESH_LE_DS"=>string(mesh["le_ds"]),
        "TOPO_MESH_SURFACE_EVALUATOR"=>mesh["surface_evaluator"],"TOPO_MESH_SURFACE_DISTRIBUTION"=>mesh["surface_distribution"],
        "CAV_MOVIE_OUT_DIR"=>String(cav_out),"CAV_MOVIE_TG"=>"0",
        "CAV_MOVIE_MAKIE"=>_boolean(flow,"cav_render_makie",true) ? "1" : "0","CAV_MOVIE_MAKIE_EVERY"=>string(frames),
        "CAV_MOVIE_BACKEND"=>_boolean(flow,"cfd_gpu",true) ? "cuda" : "cpu")
    gpu=_boolean(flow,"cfd_gpu",true)
    # CFL 0.9 is the N60R/V1 cavitating setting (its time-step error is unmeasured);
    # the ceiling knob admits it explicitly. The app default stays 0.3.
    cfl=_number(flow,"cfd_cfl",.3;lo=.001,hi=.9)
    env["TOPO_CFL"]=string(cfl)
    env["TOPO_SPECIES_CFL_MAX"]=string(max(.5,cfl))
    # The production execution recipe of the V1 launcher (deferred admission,
    # CUDA graph, tiled admission reduction); every knob is named, none is left
    # to a driver default that differs from the validated runs.
    env["TOPO_SPECIES_ADMISSION"]="deferred"
    env["TOPO_SPECIES_GRAPH"]=gpu ? "1" : "0"
    env["TOPO_SPECIES_REDUCTION_TILE"]="1024"
    env["TOPO_SPECIES_CLOCK_MARGIN"]="0.9";env["TOPO_SPECIES_REPLAY"]="fallback"
    env["TOPO_SPECIES_FLUX"]="ausm";env["TOPO_SPECIES_WALL_FUNCTION"]="none"
    env["TOPO_SPECIES_STEADY"]="0";env["TOPO_SPECIES_IMEX"]="0";env["TOPO_SPECIES_FREE_GAS"]="0"
    env["TOPO_SST"]="1";env["TOPO_GLOBALDT"]="1";env["TOPO_RK_INCREMENTAL"]="1"
    # The V1 launcher's explicit values of the legacy/optional physics knobs.
    for k in ("RP","RP_S","RP_PG0","TVP","SLIP","SLIPTAU","CLOUD","POLY","HK","WALLNUC","BTKE",
              "PVTURB","PVVORT","TFREE","EMIX","LATENT","KEPS","WALLFUN","WF_PG","ARECON")
        env["TOPO_"*k]="0"
    end
    env["TOPO_TMIN"]="1.2";env["TOPO_CVV"]="-1";env["TOPO_RP_KAPPA"]="1.4";env["TEM_P3_DIAG"]="0"
    retreat=lowercase(String(_get(flow,"cav_recon_retreat","continuous")))
    retreat in ("continuous","halving") || throw(ArgumentError("cav_recon_retreat must be continuous (option H, the v1 solver) or halving"))
    env["TOPO_SPECIES_RECON_RETREAT"]=retreat
    # The stage-1 clock margin: the finite-base default since BB8A, and on any mesh whose
    # flow names it (a sharp edge's thin wake-cut cells bind the SST clock as well).
    if plan.finite || _get(flow,"cav_stage_clock_margin",nothing)!==nothing
        env["TOPO_SPECIES_STAGE_CLOCK_MARGIN"]=string(_number(flow,"cav_stage_clock_margin",.99;lo=.5,hi=1.))
    end
    plan.finite && (env["TOPO_BASE_PROBE_EVERY"]="100")
    haskey(mesh,"wake_near_ds") &&
        (env["TOPO_MESH_WAKE_NEAR"]=join((mesh["wake_near_ds"],mesh["wake_near_len"],mesh["wake_near_growth"]),','))
    if mesh["te_topology"]=="strip"
        env["TOPO_MESH_TE_TOPOLOGY"]="strip"
        # The driver's Makie pixmap renders one block; its Plots frames render both.
        env["CAV_MOVIE_MAKIE"]="0"
    end
    mesh["mesh_quality"]=="v1" && (env["TOPO_MESH_QUALITY"]="v1")
    window=_get(flow,"cav_frame_window",nothing)
    window===nothing || (env["CAV_MOVIE_WIN"]=join(Float64.(window),','))
    env
end

"""Two-block (C + H strip) quality report for the cavitation tunnel mesh.

Convexity is checked on both blocks; the face metrics include both C/H
interfaces and the base. Gates are the cavitating CMeshQualityV1 limits."""
function _tunnel_strip_quality_report(mesh)
    Q=CMeshGen.CMeshQualityMetrics
    for (label,X,Y,NI,NJ) in (("C",mesh.X,mesh.Y,mesh.NI,mesh.NJ),("H",mesh.strip.X,mesh.strip.Y,mesh.strip.NI,mesh.strip.NJ))
        defects,worst,location=CMeshGen._convexity_defects(X,Y,NI,1:NJ)
        defects==0 && worst>0 || error("C+H tunnel mesh block $label has $defects non-convex cells; worst corner Jacobian $worst at $location")
    end
    s=Q.quality_summary(mesh.X,mesh.Y,mesh.NIW,mesh.NIF;strip=mesh.strip,band_j=min(40,mesh.NJ),nb=min(3,mesh.NJ))
    gates=Q.quality_gates(s;cavitating=true)
    finite_or_nothing(x)=x isa Real && !isfinite(x) ? nothing : x
    Dict{String,Any}("cells"=>s.cells,"geometric_gate"=>Dict("passed"=>true,"scope"=>"positive convex cells in both blocks"),
        "cavitation_reference_gates"=>Dict("passed"=>all(g[5] for g in gates),"failed"=>Q.gate_failures(gates),
            "gates"=>[Dict("id"=>g[1],"quantity"=>g[2],"value"=>finite_or_nothing(g[3]),"limit"=>g[4],"passed"=>g[5]) for g in gates],
            "scope"=>"CMeshQualityV1 cavitating limits; geometry only, not CFD convergence"),
        "global"=>Dict("maximum_face_nonorthogonality_deg"=>s.theta_max,"minimum_corner_angle_deg"=>s.minang,
            "maximum_aspect_ratio"=>s.ar_max,"maximum_outlet_aspect_ratio"=>s.ar_out3,
            "maximum_neighbor_volume_ratio"=>s.vr_max,"maximum_skew_over_distance"=>s.skew_max),
        "interfaces"=>Dict("maximum_volume_ratio"=>finite_or_nothing(s.vr_junction),
            "maximum_nonorthogonality_deg"=>finite_or_nothing(s.theta_junction),"maximum_skew_over_distance"=>finite_or_nothing(s.skew_junction)))
end

function _preview_production_tunnel(path,flow,outdir)
    alpha=_number(flow,"alpha_deg",0.);plan=_production_tunnel_settings(path,flow,alpha);p=plan.mesh
    mesh=plan.prepared_mesh===nothing ? _generate_production_tunnel_mesh(plan.af,p,plan.channel) : plan.prepared_mesh
    strip=p["te_topology"]=="strip"
    metadata=Dict{String,Any}("mesh_family"=>"production Species water tunnel",
        "mesh_topology"=>strip ? "conformal_C_H_v1" : (plan.finite ? "aplus" : "sharp"),
        "topology_requested"=>String(_get(flow,"cfd_te_topology","aplus")),
        "n_base"=>p["n_base"],"wall_dn1"=>p["dn1"],"source_te_gap"=>hypot(plan.source_af.x[1]-plan.source_af.x[end],plan.source_af.y[1]-plan.source_af.y[end]),
        "cfd_te_gap"=>plan.gap,"transform"=>plan.te>0 ? "explicit trailing-edge thickness" : "none",
        "quality_report"=>strip ? _tunnel_strip_quality_report(mesh) : _transonic_quality_report(mesh),"channel"=>_species_channel_metadata(mesh.channel),
        "species_requested_model_settings"=>plan.model,"production_mesh_settings"=>p,
        "geometry_pipeline_revision"=>3,"surface_evaluator"=>p["surface_evaluator"],"surface_distribution"=>p["surface_distribution"],
        "surface_distribution_version"=>p["surface_distribution"]=="curvature" ? "curvature_v1" : "legacy",
        "leading_edge_target_spacing"=>p["le_ds"],"leading_edge_max_curvature"=>p["leading_edge_max_curvature"],
        "conditioned_x"=>plan.conditioned.airfoil.x,"conditioned_y"=>plan.conditioned.airfoil.y,
        "conditioning"=>plan.conditioned.metadata,
        "outer_boundary_condition"=>"slip tunnel walls, characteristic inlet, pressure outlet",
        "species_boundary_conditions"=>_species_boundary_conditions("tunnel"),
        "geometry_support"=>p["surface_distribution"]=="curvature" ?
            "the same source-interpolating smooth curve, curvature-distributed wall stations and generator settings as the production tunnel driver" :
            "the same loaded source coordinates and explicitly selected surface evaluator/distribution as the production tunnel driver")
    metadata["mesh_coordinate_sha256"]=Dict("x"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.X)))),
        "y"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.Y)))),
        (strip ? ("strip_x"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.strip.X)))),
                  "strip_y"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.strip.Y))))) : ())...,
        "scope"=>"the driver prints the same hashes in its MESH COORDINATE RECEIPT")
    if strip
        q=metadata["quality_report"]["cavitation_reference_gates"]
        merge!(metadata,Dict{String,Any}("quality_passed"=>q["passed"],
            "quality_status"=>q["passed"] ? "passed_cavitation_geometry" : "cavitation_geometry_gate_failures_recorded",
            "quality_failures"=>q["failed"],"solver_supported"=>true,"solver_backends"=>["cavitation"],
            "solver_support_message"=>"Species water-tunnel C+H Run: the production cavitation driver's strip topology (C block around foil plus strip, H block behind the base). Geometry gates are recorded, not a convergence or validation certificate."))
        payload=_write_ch_mesh_artifacts(mesh,metadata,outdir,plan.resolution)
    else
        payload=_write_mesh_artifacts(mesh,metadata,outdir,plan.resolution)
    end
    (;mesh=payload,metadata)
end

"Realized production Species boundary operators, independent of mesh provenance."
function _species_boundary_conditions(domain)
    Dict{String,Any}("outer_j"=>domain=="free_fluid" ?
        "mixture characteristic farfield" : "slip tunnel walls and characteristic inlet",
        "downstream_i_ends"=>"fixed static pressure on both ends",
        "downstream_acoustically_reflecting"=>true,
        "scope"=>"current production configuration; no nonreflecting downstream boundary claim")
end

"Explicit boundary provenance; cell geometry remains bitwise the shared metric geometry."
function _species_external_metrics(met)
    raw=NamedTuple{fieldnames(typeof(met))}(Tuple(getfield(met,k) for k in fieldnames(typeof(met))))
    merge(raw,(outer_face_kind=fill(UInt8(1),met.NI),channel_provenance=false,
               free_fluid_provenance=true))
end

"Trapezoidal force means on the latter half of the actual recorded physical time."
_species_time_statistics(history;blocks::Int=10,kwargs...)=
    CFDTimeStatistics.time_statistics(history;min_blocks=blocks,kwargs...)

function _species_moment(s,built,pressure)
    g=s.g;indices=collect(built.foil_indices);range=(first(indices)+g):(last(indices)+g)
    rho,u,v,p=map(a->Array(view(a,range,g+1)),(s.rho,s.ux,s.uy,pressure))
    q=0.5Float64(s.rho_inf)*(Float64(s.uxi)^2+Float64(s.uyi)^2);cm=0.0
    for (k,i) in enumerate(indices)
        sx=Float64(s.Sjx1_h[i]);sy=Float64(s.Sjy1_h[i]);sf=hypot(sx,sy)
        d1=0.5Float64(s.vol1_h[i])/sf+1e-30
        mu=Float64(rho[k])*Float64(s.nu_lam)*sf/d1
        tx,ty=CMeshSolverKA._wall_stokes_traction(Float64(u[k]),Float64(v[k]),sx/sf,sy/sf,mu)
        fx=-Float64(p[k])*sx+tx;fy=-Float64(p[k])*sy+ty
        cm+=_aerodynamic_cm(0.5(built.mesh.X[i,1]+built.mesh.X[i+1,1]),
            0.5(built.mesh.Y[i,1]+built.mesh.Y[i+1,1]),fx,fy,q)
    end
    cm
end

"""Pure noncondensing-gas limit of the production local-pressure mixture EOS.

With alpha=0 and lq=1 the specific volume is p_ref/(rho_ref*p),
apart from a liquid-compliance term smaller than 1e-30 in solver units.
Consequently p=c_air^2*rho and dp/drho=c_air^2 to stored precision.
Rates are exactly zero, and alpha remains exactly zero. This is isothermal
air, with no thermodynamic energy or adiabatic shock claim.
"""
function _species_air_tem!(s,tem,sound_speed::Real)
    c2=Float32(sound_speed^2)
    tem.p_v=c2*s.rho_inf;tem.rho_l_sat=s.rho_inf
    tem.rho_v_sat=0.001f0*s.rho_inf;tem.a2l=1.0f30
    tem.lq=1.0f0;tem.psm=0.0f0;tem.am_floor_frac=0.0f0
    tem.kprod=0.0f0;tem.kdest=0.0f0
    tem.omega=1.0f0;tem.src_jac=Int32(0)
    fill!(tem.alpha,0.0f0)
    c2
end

function _species_mesh(cfg)
    settings=_species_model_settings(cfg;alpha=cfg["alpha_deg"])
    af=haskey(cfg,"naca4_code") ? CMeshGen.naca4_points(cfg["naca4_code"]) : CMeshGen.read_airfoil_dat(cfg["geometry"])
    re=cfg["reynolds"];cf=0.026re^(-1/7);dn1=0.7/(re*sqrt(cf/2))
    if cfg["cav_domain"]=="free_fluid"
        cfg["mesh_mode"] in ("cavitation_aplus_farfield","aplus_nested_refined") ||
            throw(ArgumentError("external Species requires the shared A-plus mesh"))
        built=_build_transonic_mesh(af;n_foil=cfg["n_foil"],n_wake=cfg["n_wake"],
            n_normal=cfg["n_normal"],dn1,alpha_deg=cfg["alpha_deg"],
            trip_upper=cfg["trip_upper"],trip_lower=cfg["trip_lower"],
            n_base=cfg["n_base"],cfd_te_thickness=cfg["cfd_te_thickness"],
            mesh_refinement=cfg["mesh_mode"]=="aplus_nested_refined" ? :aplus_nested_refined : :none,
            wake_first_spacing_growth=cfg["wake_first_spacing_growth"],
            wake_first_spacing_max=cfg["wake_first_spacing_max"],
            wake_first_spacing_delay_cells=cfg["wake_first_spacing_delay_cells"],
            farfield_angle_deg=cfg["cfd_farfield_angle_deg"])
        metadata=copy(built.metadata)
        metadata["outer_boundary_condition"]="mixture characteristic outer contour; fixed-static-pressure downstream i ends (acoustically reflecting)"
        metadata["species_boundary_conditions"]=_species_boundary_conditions("free_fluid")
        return (;af,mesh=built.mesh,met=_species_external_metrics(built.metrics),
            metadata,foil_indices=built.foil_indices,surface_indices=built.surface_indices)
    end
    conditioned=CMeshGen.condition_airfoil(af;trailing_edge_thickness=cfg["cfd_te_thickness"])
    channel=CMeshGen.ChannelSpec(;height=settings["cav_channel_height"],xi_in=settings["cav_channel_xi_in"],xi_out=settings["cav_channel_xi_out"],
        xi_ref=settings["cav_channel_xi_ref"],ref=(0.5,0.0),angle_deg=settings["cav_channel_angle_deg"])
    gap=hypot(conditioned.airfoil.x[1]-conditioned.airfoil.x[end],conditioned.airfoil.y[1]-conditioned.airfoil.y[end])
    nb=gap>1e-8 ? cfg["n_base"] : 0
    mesh=CMeshGen.generate_cmesh(conditioned.airfoil;n_foil=cfg["n_foil"],n_wake=cfg["n_wake"],
        n_normal=cfg["n_normal"],dn1,growth_max=1.20,R_ff=25.0,L_wake=25.0,
        smooth_iters=400,bl_freeze=6,channel,
        (nb>0 ? (n_base=nb,te_ds=1.5gap/(2nb)) : (;))...)
    met=CMeshMetrics.build_metrics(mesh;trip_upper=cfg["trip_upper"],trip_lower=cfg["trip_lower"])
    metadata=Dict{String,Any}("mesh_family"=>"CMesh tunnel","mesh_topology"=>nb>0 ? "aplus" : "sharp",
        "n_base"=>nb,"wall_dn1"=>dn1,"source_te_gap"=>gap,"cfd_te_gap"=>gap,"transform"=>"none",
        "outer_boundary_condition"=>"slip tunnel walls, characteristic inlet, pressure outlet",
        "quality_report"=>_transonic_quality_report(mesh))
    metadata["species_boundary_conditions"]=_species_boundary_conditions("tunnel")
    metadata["channel"]=_species_channel_metadata(channel)
    metadata["species_requested_model_settings"]=settings
    foil_indices=met.NIW+1:met.NIW+met.NIF
    surface_indices=met.NIW+nb+1:met.NIW+met.NIF-nb
    (;af,mesh,met,metadata,foil_indices,surface_indices)
end

"Build the selected Species domain without starting a flow worker or loading CUDA."
function preview_species_mesh(path,flow,outdir;source=Dict())
    material,domain=cavitation_configuration(flow)
    gurney_enabled(flow) && gurney_check_request(flow,["cavitation"])   # Gurney flap: refused on every Species domain
    domain=="underwater" && return preview_underwater_mesh(path,flow,outdir;source)
    material=="water" && domain=="tunnel" && return _preview_production_tunnel(path,flow,outdir)
    if String(_get(flow,"cfd_te_topology","aplus"))=="auto"
        domain=="free_fluid" && return preview_mesh(path,flow,outdir;source,solver_backend="cavitation")
        throw(ArgumentError("Auto C+H is currently an external-domain mesh. For the established tunnel validation mesh select Legacy A-plus explicitly; no far-boundary mesh will be substituted for tunnel walls."))
    end
    resolution=String(_get(flow,"cfd_resolution","standard"))
    sizes=get(Dict("standard"=>(256,64,112),"fine"=>(384,96,144),"very_fine"=>(512,128,176)),resolution,nothing)
    sizes===nothing && throw(ArgumentError("invalid cfd_resolution"))
    cfg=Dict{String,Any}("geometry"=>String(path),"cav_fluid"=>material,"cav_domain"=>domain,
        "mesh_mode"=>String(_get(flow,"cfd_mesh","cavitation_aplus_farfield")),
        "cav_channel_height"=>_number(flow,"cav_channel_height",1.28;lo=0.2,hi=100),
        "resolution"=>resolution,"reynolds"=>_number(flow,"reynolds",3e6;lo=1e3,hi=1e9),
        "alpha_deg"=>_number(flow,"alpha_deg",0.0),"cfd_te_thickness"=>_number(flow,"cfd_te_thickness",0.0;lo=0,hi=0.05),
        "n_foil"=>_integer(flow,"cfd_n_foil",sizes[1];lo=32,hi=2048),
        "n_wake"=>_integer(flow,"cfd_n_wake",sizes[2];lo=8,hi=512),
        "n_normal"=>_integer(flow,"cfd_n_normal",sizes[3];lo=16,hi=512),
        "n_base"=>_integer(flow,"cfd_n_base",4;lo=1,hi=8),
        "trip_upper"=>_number(flow,"trip_upper",0.03;lo=0,hi=1),"trip_lower"=>_number(flow,"trip_lower",0.05;lo=0,hi=1),
        "wake_first_spacing_growth"=>_number(flow,"cfd_wake_first_spacing_growth",1.0;lo=1,hi=2),
        "wake_first_spacing_max"=>_number(flow,"cfd_wake_first_spacing_max",0.01;lo=0,hi=0.1),
        "wake_first_spacing_delay_cells"=>_integer(flow,"cfd_wake_first_spacing_delay_cells",0;lo=0,hi=511),
        "cfd_farfield_angle_deg"=>_number(flow,"cfd_farfield_angle_deg",_number(flow,"alpha_deg",0.0);lo=-90,hi=90))
    merge!(cfg,_species_model_settings(flow;alpha=cfg["alpha_deg"]))
    _get(source,"kind","")=="naca4" && (cfg["naca4_code"]=parse(Int,String(_get(source,"id",""))))
    built=_species_mesh(cfg)
    payload=_write_mesh_artifacts(built.mesh,built.metadata,outdir,resolution)
    (;mesh=payload,metadata=built.metadata)
end

function _species_worker(cfg_path)
    cfg=TOML.parsefile(cfg_path);cavitation_configuration(cfg)
    settings=_species_model_settings(cfg;alpha=cfg["alpha_deg"])
    settings["cav_init"]=="uniform" || throw(ArgumentError("The shared Species worker currently supports cav_init=uniform; panel initialization is available in the production tunnel driver"))
    settings["cav_sst_omega_init"]=="freestream" || throw(ArgumentError("The shared Species worker currently supports cav_sst_omega_init=freestream"))
    material=cfg["cav_fluid"];domain=cfg["cav_domain"]
    built=_species_mesh(cfg);(;mesh,met)=built
    out=cfg["outdir"];metadata=built.metadata
    metadata["geometry_support"]=haskey(cfg,"naca4_code") ? "NACA four-digit equations evaluated at 4097 cosine-spaced support points" : "C2 interpolation of source coordinates"
    _write_mesh_artifacts(mesh,metadata,out,cfg["resolution"])
    xhash=bytes2hex(sha256(reinterpret(UInt8,vec(mesh.X))))
    yhash=bytes2hex(sha256(reinterpret(UInt8,vec(mesh.Y))))
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    backend=cfg["gpu"] ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    cx,cy,cz,w,_,_=build_d3q39_static()
    s=CMeshSolverKA.build_sol_ka(backend,met;drot=2.0f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    s.wall_flux=true;s.recon=Int32(1);s.fullvisc=Int32(1)
    CMeshSolverKA.set_freestream_ka!(s,cfg["mach"],cfg["alpha_deg"];pg_correction=false)
    speed=hypot(Float64(s.uxi),Float64(s.uyi));s.nu_lam=Float32(speed/cfg["reynolds"])
    CMeshSolverKA.set_sst_ka!(s;tu_inf=0.001,mu_t_ratio=0.1)
    s.cav=Int32(4)
    tem=if material=="water"
        # Same material/source calibration as the production tunnel driver.
        CMeshCavTEM.build_tem(s;sigma=cfg["sigma"],Cprod=Float64(settings["cav_cprod"]),Cdest=Float64(settings["cav_cdest"]),
            ratio=Float64(settings["cav_density_ratio"]),a2l_frac=Float64(settings["cav_a2l"]),eps_pv=Float64(settings["cav_epspv"]),Tmin=1.2f0,
            rate_ref_pv=true,vfloor=false,eos=1,wallis=1,vaploc=1.0,exact_reference=true)
    else
        CMeshCavTEM.build_tem(s;sigma=cfg["sigma"],ratio=1000.0,vfloor=false,
            eos=1,wallis=1,vaploc=1.0,exact_reference=true)
    end
    tem.omega=1.0f0;tem.src_jac=Int32(0)
    p_ref,a2m=if material=="air"
        c2=_species_air_tem!(s,tem,speed/cfg["mach"])
        (Float64(c2*s.rho_inf),Float64(c2))
    else
        CMeshCavTEM.tem_ambient(tem,s)
    end
    CMeshSolverKA.set_farfield_mixture!(s;p_ref,a2m,Tmin=tem.Tmin,Tmax=tem.Tmax)
    fill!(s.T,clamp(Float32(p_ref)/(s.rho_inf*CMeshSolverKA.CS2),tem.Tmin,tem.Tmax))
    s.pexc .= Float32(p_ref).-s.rho_inf*CMeshSolverKA.CS2.*s.T
    fill!(s.am,sqrt(Float32(a2m)))
    CMeshSolverKA.set_pressure_outlet!(s,tem;p_out=p_ref,checked_mixture=true)
    CMeshSolverKA.GLOBAL_DT[]=1
    CMeshSolverKA.seed_channel_reference_ka!(s)
    CMeshCavSpecies.LEDGER_ENABLED[]=false
    # Species owns a calibrated k-floor trip, not the legacy carrier's
    # intermittency multiplication. Name the actual treatment in the receipt.
    trip_x=settings["cav_trip_x"]
    material=="air" && cfg["trip_upper"]!=cfg["trip_lower"] && trip_x==0 && println(
        "Species asymmetric trip locations unsupported; geometry trip mask retained as metadata only, no k-floor trip applied.")
    c=CMeshCavSpecies.build_species_ka(s,tem;wallmode=Int32(2),reconstruction=settings["cav_species_reconstruction"],
        eos_precision=Symbol(settings["cav_eos_precision"]),nut_density=Symbol(settings["cav_nut_density"]),
        reboud_n=settings["cav_reboud_n"],sst_correction_limit=settings["cav_sst_correction_limit"],trip_x,trip_width=settings["cav_trip_width"])
    steady=material=="air" && cfg["cav_air_mode"]=="steady"
    workspace=steady ? CMeshCavSpecies.build_species_steady(c) : nothing
    @printf("Conservative Species initialized on %s: %s, %s; %d cells; %s\n",
        cfg["gpu"] ? "GPU" : "CPU",material,domain,met.NI*met.NJ,steady ? "steady pseudo-time (no physical clock)" : "physical time")
    println("BOUNDARY: ",metadata["outer_boundary_condition"])
    domain=="tunnel" && println("REALISED CHANNEL: ",metadata["channel"])
    println("SPECIES MODEL SETTINGS: ",settings,"; actual Reboud form=",c.reboud_form," exponent=",c.reboud_n," trip=",c.trip," SST limit=",c.sst_climit);flush(stdout)
    material=="air" && println("AIR EOS: isothermal p=c_air^2*rho, no phase change, no total-energy equation")
    println("MESH COORDINATE RECEIPT: x_sha256=$xhash y_sha256=$yhash");flush(stdout)
    monitor=CFDConvergence.Monitor(CFDConvergence.Config(sfd_start=0,max_steps=cfg["steps"],
        residual_scaled_limit=cfg["mass_ceiling"],mass_net_limit=cfg["mass_net_limit"]))
    history=Dict{String,Any}[];dt_elapsed=0.0;start_time=time();guard=Dict{String,Any}()
    last_raw=Inf;last_scaled=Inf;last_l2=Inf;mass_net=Inf;cl=0.;cd=0.;actual=0
    history_path=joinpath(out,"iteration_history.csv")
    open(history_path,"w") do io
        println(io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,cm,mass_net_frac,t_ctu")
        for n in 1:cfg["steps"]
            if steady
                CMeshCavSpecies.step_species_steady_ka!(c,workspace,Float32(cfg["cfl"]))
            else
                CMeshCavSpecies.step_species_ka!(c,Float32(cfg["cfl"]);admission=:deferred)
                dt_elapsed=c.elapsed*speed
            end
            actual=n
            if n==1 || n%cfg["cav_history_stride"]==0 || n==cfg["steps"] || (!steady && dt_elapsed>=cfg["cav_duration_ctu"])
                # Refresh the final committed state's real flux divergence.
                # The Species stage kernel deliberately does not populate dc_*.
                CMeshCavSpecies._stage_residual!(c,Int32(2))
                CMeshSolverKA._k_residual!(s.backend)(s.dc_r,s.dc_mx,s.dc_my,s.dc_e,
                    s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e,s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,
                    Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
                dc=Array(s.dc_r);vol=Array(s.vol);scaled=Float64.(dc)./Float64.(vol)
                last_raw=maximum(abs,dc);last_scaled=maximum(abs,scaled);last_l2=sqrt(sum(abs2,scaled)/length(scaled))
                mass_net=sum(Float64,dc)/(Float64(s.rho_inf)*speed)
                cl,cd=CMeshSolverKA.loads_ka(s;mixture_pressure=c.pressure,wallmode=Int32(2))
                cm=_species_moment(s,built,c.pressure)
                row=Dict{String,Any}("iteration"=>n,"residual_linf"=>last_raw,"residual_scaled"=>last_scaled,
                    "residual"=>last_scaled,"residual_l2"=>last_l2,"cl"=>Float64(cl),"cd"=>Float64(cd),
                    "cm"=>cm,"mass_net_frac"=>mass_net,"t_ctu"=>dt_elapsed)
                push!(history,row)
                println(io,join((n,last_raw,last_scaled,last_l2,cl,cd,cm,mass_net,dt_elapsed),','));flush(io)
                println("ITERATION,",join((n,cfg["steps"],last_scaled,last_raw,last_l2,cl,cd,mass_net),','));flush(stdout)
                if !steady
                    @printf("PHYSICAL_TIME,%d,%.12g,%.12g,%.12g\n",n,dt_elapsed,cfg["cav_duration_ctu"],Float64(s.last_global_dt)*speed)
                    flush(stdout)
                end
                decision=CFDConvergence.observe!(monitor,(iteration=n,residual_scaled=last_scaled,
                    residual_l2=last_l2,cl=Float64(cl),cd=Float64(cd),mass_net_frac=mass_net))
                decision.abort && error("Species sustained divergence: $(decision.reason)")
                steady && cfg["auto_stop"] && decision.stop && break
                !steady && dt_elapsed>=cfg["cav_duration_ctu"] && break
            end
        end
    end
    guard=CFDConvergence.convergence_report(monitor)
    exported=_export_species_fields(c,built,cfg,actual,p_ref,speed)
    time_stats=steady ? Dict{String,Any}("available"=>false,"reason"=>"steady pseudo-time has no physical averaging clock") :
        _species_time_statistics(history;requested_duration_ctu=cfg["cav_duration_ctu"],
            burn_in_ctu=get(cfg,"burnin_ctu",nothing),block_duration_ctu=get(cfg,"block_ctu",5.0),
            time_step_refined=get(cfg,"physical_time_refined",false))
    averaged=!steady && get(time_stats,"available",false)
    meta=Dict{String,Any}("backend"=>"cavitation","solver"=>"Conservative Species AUSM+-up",
        "cl"=>averaged ? time_stats["cl_mean"] : Float64(cl),
        "cd"=>averaged ? time_stats["cd_mean"] : Float64(cd),
        "cm"=>averaged ? time_stats["cm_mean"] : exported.cm,
        "cl_last"=>Float64(cl),"cd_last"=>Float64(cd),"cm_last"=>exported.cm,
        "coefficient_method"=>averaged ? "physical-time mean over latter half of record" : "final iterate",
        "time_statistics"=>time_stats,"field_time_scope"=>"instantaneous final state",
        "steps"=>actual,"max_steps"=>cfg["steps"],
        "residual_linf"=>last_raw,"residual_scaled"=>last_scaled,"residual_l2"=>last_l2,
        "mass_net_frac"=>mass_net,"convergence_guard"=>guard,"geometry"=>metadata,
        "fluid"=>material,"domain"=>domain,"boundary_conditions"=>_species_boundary_conditions(domain),
        "gpu"=>cfg["gpu"],"reynolds"=>cfg["reynolds"],
        "mach"=>cfg["mach"],"realized_acoustic_mach"=>speed/sqrt(a2m),
        "trip_upper"=>cfg["trip_upper"],"trip_lower"=>cfg["trip_lower"],
        "applied_species_trip_x"=>trip_x,
        "requested_model_settings"=>settings,
        "realized_model"=>Dict("nut_density"=>settings["cav_nut_density"],"reboud_form"=>Int(c.reboud_form),
            "reboud_n"=>Float64(c.reboud_n),"reboud_production"=>c.reboud_prod==1,
            "trip_x"=>Float64(c.trip[1]),"trip_width"=>Float64(c.trip[2]),"sst_correction_limit"=>Float64(c.sst_climit),
            "eos_precision"=>c.eos_single ? "single" : "double","phase_change"=>material=="water",
            "source_model"=>c.source_model==0 ? "merkle" : "schnerr_sauer","source_weight"=>c.source_beta ? "beta" : "alpha"),
        "trip_model"=>trip_x>0 ? "Species near-wall turbulence k-floor on both surfaces" : "no Species k-floor trip; carrier gammat is metadata only",
        "eos"=>Dict("p_v"=>Float64(tem.p_v),"rho_l_sat"=>Float64(tem.rho_l_sat),
            "rho_v_sat"=>Float64(tem.rho_v_sat),"a2l"=>Float64(tem.a2l),"lq"=>Float64(tem.lq),
            "kprod"=>Float64(tem.kprod),"kdest"=>Float64(tem.kdest)),
        "clock"=>steady ? "steady pseudo-time" : "physical time", "elapsed_ctu"=>dt_elapsed,
        "wall_seconds"=>time()-start_time,"mesh_node_x_sha256"=>xhash,"mesh_node_y_sha256"=>yhash,
        "mesh_cells"=>met.NI*met.NJ,"full_field_mesh_sha256"=>exported.field["mesh_sha256"],
        "energy_model"=>material=="air" ? "isothermal air, no total-energy transport" : "barotropic liquid/vapour",
        "validation"=>"experimental shared-mesh Species backend; aerodynamic and stall parity unestablished")
    open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,meta);end
    println("Conservative Species worker complete");flush(stdout)
end

function _export_species_fields(c,built,cfg,iteration,p_inf,speed)
    s=c.solver;mesh=built.mesh;met=built.met;g=s.g;ni=met.NI;nj=met.NJ
    rho,u,v,T,nut,p,am=map(Array,(s.rho,s.ux,s.uy,s.T,s.nut,c.pressure,s.am))
    xc,yc=map(Array,(s.xc,s.yc));q=0.5Float64(s.rho_inf)*speed^2
    cp(i,j)=(Float64(p[i+g,j+g])-p_inf)/q
    function vort(i,j)
        im=max(1,i-1);ip=min(ni,i+1);jm=max(1,j-1);jp=min(nj,j+1)
        xi=xc[ip,j]-xc[im,j];yi=yc[ip,j]-yc[im,j];xj=xc[i,jp]-xc[i,jm];yj=yc[i,jp]-yc[i,jm]
        ui=u[ip+g,j+g]-u[im+g,j+g];uj=u[i+g,jp+g]-u[i+g,jm+g]
        vi=v[ip+g,j+g]-v[im+g,j+g];vj=v[i+g,jp+g]-v[i+g,jm+g]
        det=xi*yj-xj*yi
        abs(det)<1e-20 ? 0.0 : (vi*yj-vj*yi+ui*xj-uj*xi)/det
    end
    at(i,j)=(xc[i,j],yc[i,j],hypot(u[i+g,j+g],v[i+g,j+g])/am[i+g,j+g],cp(i,j),
        p[i+g,j+g],rho[i+g,j+g],u[i+g,j+g],v[i+g,j+g],T[i+g,j+g],vort(i,j),nut[i+g,j+g])
    stride=max(1,ceil(Int,sqrt(ni*nj/18000)));ii=unique(vcat(collect(1:stride:ni),ni));jj=unique(vcat(collect(1:stride:nj),nj))
    open(joinpath(cfg["outdir"],"field_samples.csv"),"w") do io
        println(io,join(FullFieldArtifact.COLUMNS,','))
        for j in jj,i in ii;println(io,join(at(i,j),','));end
    end
    field=FullFieldArtifact.write_full_field(at,joinpath(cfg["outdir"],"field_full.msgpack"),ni,nj,
        joinpath(cfg["outdir"],"mesh_full.msgpack"),iteration)
    surface=collect(built.surface_indices);ile=surface[argmin(xc[surface,1])]
    open(joinpath(cfg["outdir"],"surface_cp.csv"),"w") do io
        println(io,"surface,x,cp")
        for i in surface;println(io,i>ile ? "upper" : "lower",",",xc[i,1],",",cp(i,1));end
    end
    open(joinpath(cfg["outdir"],"field_info.toml"),"w") do io
        TOML.print(io,Dict("sampled_ni"=>length(ii),"sampled_nj"=>length(jj),"sample_stride"=>stride,
            "full_field_cells"=>ni*nj,"full_field_mesh_sha256"=>field["mesh_sha256"]))
    end
    cm=0.0
    for i in built.foil_indices
        sx=Float64(s.Sjx1_h[i]);sy=Float64(s.Sjy1_h[i]);sf=hypot(sx,sy)
        d1=0.5Float64(s.vol1_h[i])/sf+1e-30
        mu=Float64(rho[i+g,g+1])*Float64(s.nu_lam)*sf/d1
        tx,ty=CMeshSolverKA._wall_stokes_traction(Float64(u[i+g,g+1]),Float64(v[i+g,g+1]),sx/sf,sy/sf,mu)
        fx=-Float64(p[i+g,g+1])*sx+tx;fy=-Float64(p[i+g,g+1])*sy+ty
        cm+=_aerodynamic_cm(0.5(mesh.X[i,1]+mesh.X[i+1,1]),0.5(mesh.Y[i,1]+mesh.Y[i+1,1]),fx,fy,q)
    end
    (;cm,field)
end

function _collect_species_output(dir,alpha)
    meta=TOML.parsefile(joinpath(dir,"result.toml"));info=TOML.parsefile(joinpath(dir,"field_info.toml"))
    mesh=AirfoilWire.decode(read(joinpath(dir,"mesh_preview.msgpack")),:msgpack)
    mesh["artifact"]=joinpath(dir,"mesh_full.msgpack");mesh["preview_artifact"]=joinpath(dir,"mesh_preview.msgpack")
    cp=Dict("upper"=>Any[],"lower"=>Any[])
    for r in _csv_rows(joinpath(dir,"surface_cp.csv"));push!(cp[r["surface"]],[_float(r["x"]),_float(r["cp"])]);end
    samples=Dict(k=>Float64[] for k in FullFieldArtifact.COLUMNS)
    for r in _csv_rows(joinpath(dir,"field_samples.csv")),k in keys(samples);push!(samples[k],parse(Float64,r[k]));end
    hist=[Dict{String,Any}(k=>parse(k=="iteration" ? Int : Float64,v) for (k,v) in r) for r in _csv_rows(joinpath(dir,"iteration_history.csv"))]
    for h in hist;h["residual"]=h["residual_scaled"];end
    guard=meta["convergence_guard"];status=get(guard,"status","")
    inadequate=meta["clock"]=="physical time" && !get(meta["time_statistics"],"sampling_adequate",false)
    unqualified=meta["clock"]=="physical time" && !get(meta["time_statistics"],"physical_mean_eligible",false)
    Dict{String,Any}("backend"=>"cavitation","alpha_deg"=>alpha,"status"=>inadequate ? "insufficient_sampling" :
        (unqualified ? "unqualified_physical_mean" : (status=="stalled_stop" ? "stopped_without_convergence" :
            (status=="early_stop_eligible" ? "completed" : "not_converged"))),
        "cl"=>meta["cl"],"cd"=>meta["cd"],"cm"=>meta["cm"],"geometry"=>meta["geometry"],"mesh"=>mesh,"cp"=>cp,
        "iteration_history"=>hist,"convergence"=>meta["clock"]=="physical time" ?
            "physical time; sampling status: $(get(meta["time_statistics"],"status","insufficient_sampling"))" : "steady pseudo-time; guard: $status", "validation"=>meta["validation"],
        "fields"=>Dict("kind"=>"samples","samples"=>samples,"sample_shape"=>[info["sampled_ni"],info["sampled_nj"]],
            "sample_stride"=>info["sample_stride"],"artifact"=>joinpath(dir,"field_samples.csv"),
            "full_artifact"=>joinpath(dir,"field_full.msgpack"),"full_cell_count"=>info["full_field_cells"],
            "mesh_sha256"=>info["full_field_mesh_sha256"],"full_iteration"=>meta["steps"]),
        "diagnostics"=>meta,"artifacts"=>Dict("cp_csv"=>joinpath(dir,"surface_cp.csv"),"field_full"=>joinpath(dir,"field_full.msgpack"),
            "mesh_full"=>joinpath(dir,"mesh_full.msgpack"),"iteration_csv"=>joinpath(dir,"iteration_history.csv"),"log"=>joinpath(dir,"solver.log")),
        "notes"=>[meta["energy_model"],meta["trip_model"],"SST closure; no transition/stall validation.",
            "The carrier temperature field is an EOS pedestal, not a thermodynamic temperature."])
end
