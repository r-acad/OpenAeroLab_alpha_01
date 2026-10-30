module AirfoilCavitationCases

using SHA
export catalog, preset, geometry_path, geometry_coordinates

const ROOT=normpath(get(ENV,"LUDWIG_PRIVATE_ROOT",
    joinpath(@__DIR__,"..","..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const CALTECH_URL="https://authors.library.caltech.edu/records/g98f0-2kd42"
const LEROUX_URL="https://doi.org/10.1063/1.1865692"
const CALTECH_GEOMETRY="tools/naca66mod_t09_f020_a08_blunt.dat"
const LEROUX_GEOMETRY="memory/project_docs/_records/2026-09-08_primary_foil_mesh/leroux2003_section7_raw_scale.dat"
const GEOMETRIES=Dict(
    "caltech"=>(CALTECH_GEOMETRY,"9af147ed8476bbdc12bb69c44a8c0bf8203a90c9dfc848f5fbcec7f2f14c6922"),
    "leroux"=>(LEROUX_GEOMETRY,"4a4d0a4a854e2f1d4990b1d8d61e8304a3fa007c52c1158156b848b00a4a120d"))
const TAP_X=[.03,.06,.11,.16,.26,.33,.45,.56,.68,.81,.90,.95]

"Real-water liquid stiffness at the reference speed; recorded, not used by the presets."
_water_stiffness(speed)=1.4*.2^2*(1490.0/speed)^2

# The presets run the validated version-1 production configuration: N60R's
# (memory/project_docs/2026-10-09_N60R_REGISTRATION.md) model liquid and rates on the conforming
# C+H strip mesh, with option H. The real-water stiffness of the earlier catalog
# (density ratio 43197, a2l from 1490 m/s) is not runnable on the explicit
# density-based step: its bounded replay marched dt = 8.4e-9 CTU, about 2.6e9
# steps for 22 CTU (TEMP_DELETEABLE/build/airfoil_web_cache/leroux_runtime_20261020).
const MODEL_LIQUID=(a2l=1.4,density_ratio=100.,cprod=300.,cdest=5.,eps_pv=.1)
# Reboud: N60R (none) matched run 132's loads and closure bracket; the V3
# default over-predicted them (V1-N60, 16 Oct: C_L +10%, C_D +75%, l_p 0.77 c).
# Run 132 is the setting-selection point, so the presets carry none.
const PRESET_NUT="none"

"""Strip-mesh literals: N60R-Q, N60R's registered settings with the CMeshQualityV1
option (Q-A/Q-B/Q-C; 164 wake columns, 65,696 cells, every cavitating gate passing;
memory/project_docs/_records/2026-10-15_mesh_quality/implement/README.md). N60R's own mesh
fails those gates (82-degree faces, outlet aspect ratio 4,983); with
cav_mesh_quality=none and cfd_n_wake=136 the app rebuilds N60R node for node.
Other sections use the same rules from their base height h: te_ds = 1.5h/(2 n_base),
a near wake of h/n_base over 10h (BB arms)."""
function _strip_mesh(gap;literal=nothing,evaluator="legacy_catmull_rom",distribution="legacy")
    nb=8
    te_ds,ds,len=literal===nothing ? (round(1.5gap/(2nb);sigdigits=4),round(gap/nb;sigdigits=5),round(10gap;sigdigits=4)) : literal
    Dict{String,Any}("cfd_te_topology"=>"auto","cfd_te_thickness"=>0.,"cfd_resolution"=>"standard",
        "cfd_n_foil"=>256,"cfd_n_normal"=>108,"cfd_n_wake"=>164,"cfd_n_base"=>nb,
        "cav_mesh_dn1"=>5e-5,"cav_mesh_growth"=>1.2,"cav_mesh_smooth"=>80,"cav_mesh_le_ds"=>3e-4,
        "cav_mesh_te_ds"=>te_ds,"cav_mesh_wake_near_ds"=>ds,"cav_mesh_wake_near_len"=>len,
        "cav_mesh_surface_evaluator"=>evaluator,"cav_mesh_surface_distribution"=>distribution,
        "cav_mesh_quality"=>"v1")
end

function _flow(alpha,re,sigma,height,xi_in,xi_out;mesh,duration=12.)
    merge(Dict{String,Any}("mach"=>.2,"reynolds"=>re,"alpha_deg"=>alpha,"sigma"=>sigma,
        "cav_fluid"=>"water","cav_domain"=>"tunnel","cfd_gpu"=>true,"cfd_auto_stop"=>false,
        "cav_channel_height"=>height,"cav_channel_xi_in"=>xi_in,
        "cav_channel_xi_out"=>xi_out,"cav_channel_xi_ref"=>.5,
        "cav_a2l"=>MODEL_LIQUID.a2l,"cav_density_ratio"=>MODEL_LIQUID.density_ratio,"cav_epspv"=>MODEL_LIQUID.eps_pv,
        "cav_cprod"=>MODEL_LIQUID.cprod,"cav_cdest"=>MODEL_LIQUID.cdest,"cav_nut_density"=>PRESET_NUT,
        "cav_reboud_n"=>10.,"cav_trip_x"=>0.,"cav_trip_width"=>.005,
        "cav_sst_correction_limit"=>.5,"cav_species_reconstruction"=>true,
        "cav_init"=>"uniform","cav_sst_omega_init"=>"freestream",
        "cav_eos_precision"=>"double","cav_recon_retreat"=>"continuous","cav_stage_clock_margin"=>.99,
        "cfd_cfl"=>.9,"cav_render_makie"=>false,"cav_warm_steps"=>0,
        "cav_duration_ctu"=>duration,"cav_frames"=>round(Int,10duration),"cfd_steps"=>500000),mesh)
end

function _boundary(height,xi_in,xi_out,alpha;experimental_xi_in=xi_in)
    Dict{String,Any}("domain"=>"tunnel","channel_height_chords"=>height,
        "xi_in"=>xi_in,"xi_out"=>xi_out,"xi_ref"=>.5,"reference_point"=>[.5,0.],
        "tunnel_angle_deg"=>alpha,"experimental_xi_in"=>experimental_xi_in,
        "foil"=>"impermeable no-slip wall",
        "inlet"=>"mixture characteristic inlet",
        "upper_lower"=>"impermeable slip tunnel walls",
        "outlet"=>"fixed ambient static pressure; acoustically reflecting",
        "summary"=>"H/c=$height; inlet xi=$xi_in, outlet xi=$xi_out; characteristic inlet, slip tunnel walls, fixed-static-pressure outlet")
end

function _numerical(speed;mesh_note)
    Dict{String,Any}("scope"=>"The validated version-1 production configuration (N60R's model and rates, option H) on the conforming C+H strip mesh; not a real-water calculation",
        "model"=>"Conservative isothermal Species AUSM+-up with SST and Merkle phase change",
        "carrier_mach"=>.2,"carrier_mach_is_physical_water_mach"=>false,
        "liquid_stiffness_a2l"=>MODEL_LIQUID.a2l,"saturation_density_ratio"=>MODEL_LIQUID.density_ratio,
        "model_liquid"=>"soft model liquid (a2l 1.4, density ratio 100): a recorded physical limitation of every cavitating claim",
        "real_water_a2l_at_reference_speed"=>_water_stiffness(speed),"real_water_density_ratio"=>43197.,
        "real_water_runnable"=>false,
        "real_water_note"=>"On the explicit density-based step, real-water stiffness marched dt = 8.4e-9 CTU (25,000-step bounded replay, 20 Oct 2026): about 2.6e9 steps for 22 CTU. Real water is the pressure-based version-2 solver's scope.",
        "cprod"=>MODEL_LIQUID.cprod,"cdest"=>MODEL_LIQUID.cdest,"eps_pv"=>MODEL_LIQUID.eps_pv,
        "eddy_viscosity_correction"=>"none (selected on run 132: N60R matched with none; Reboud V3 over-predicted loads and cavity, V1-N60 16 Oct 2026)",
        "reconstruction_retreat"=>"continuous (option H)","cfl"=>.9,
        "cfl_note"=>"CFL 0.9 is N60R's and V1-N60's cavitating setting; its time-step error is not measured",
        "mesh_topology"=>"conforming C+H (driver strip topology): C block around foil plus strip, H block behind the finite base",
        "mesh"=>mesh_note,"initial_state"=>"uniform",
        "eos_precision"=>"double","physical_horizon_ctu"=>12.,"output_frames"=>120,
        "steady_convergence_claim"=>false,"reference_parity_established"=>false,
        "step_limit_scope"=>"cfd_steps=500000 is a generic app control; it does not cap the production tunnel movie driver's physical-time march",
        "timing_note"=>"12 CTU is the September series' registered length, not an established settling time: N60R's cavity swings with a period of about 10 passes. Cost on an RTX 3080: about 0.9 GPU h per CTU (V1-N60: 22 CTU in 19.1 h).")
end

function _cp(upper,lower)
    Dict{String,Any}("upper"=>[Dict("x"=>x,"cp"=>p) for (x,p) in zip(TAP_X,upper)],
        "lower"=>[Dict("x"=>.09,"cp"=>lower)])
end

# N60R's literals (0.005994-c base): te_ds, near-wake ds, near-wake length.
const CALTECH_MESH=_strip_mesh(.005994;literal=(5.62e-4,7.4925e-4,.05994))

function _caltech(run,alpha,sigma,v_fts,p_psia,cl,cd,cm,nominal,closure,cp;flag="")
    id="caltech_1142_"*string(run)
    reference=Dict{String,Any}("name"=>"Caltech HSWT 1142 run $run",
        "kind"=>"experimental transcribed pressure taps and force coefficients",
        "source_url"=>CALTECH_URL,"alpha_deg"=>alpha,"sigma"=>sigma,"cl"=>cl,"cd"=>cd,
        "cm"=>nothing,"cm_mid_chord_as_printed"=>cm,"moment_reference_xc"=>.5,
        "moment_overlay_available"=>false,"cp"=>cp,
        "nominal_cavity_label"=>nominal,"closure_bracket_xc"=>closure,
        "cavity_label_is_measured_steady_length"=>false,"measurement_flag"=>flag,
        "sigma_status"=>alpha==6. ? "Derived from run-specific pressure and 24 C water state; approximately 2.5% uncertainty" :
            "Derived from published static pressure and dynamic head; water/vapor-pressure assumptions are documented in the transcription",
        "points"=>[Dict{String,Any}("alpha_deg"=>alpha,"cl"=>cl,"cd"=>cd,"cm"=>nothing,"cp"=>cp)])
    physical=Dict{String,Any}("report"=>"GALCIT HSWT 1142","run"=>run,
        "chord_m"=>.1524,"speed_ft_s"=>v_fts,"speed_m_s"=>v_fts*.3048,
        "static_pressure_psiA"=>p_psia,"alpha_deg"=>alpha,"sigma"=>sigma,
        "reynolds_nominal"=>2.05e6,"reynolds_status"=>"Neither Reynolds number nor viscosity is printed in HSWT 1142; 2.05e6 is the registered rounded solver condition",
        "reynolds_derived"=>v_fts*.5/.95871e-5,
        "assumed_kinematic_viscosity_ft2_s"=>.95871e-5,
        "viscosity_assumption_source"=>"HSWT 1139 constant at 77.2 F; not a run-specific HSWT 1142 measurement",
        "physical_water_mach"=>nothing,"mach_note"=>"The app's Mach 0.20 is an internal carrier setting, not the physical water Mach number",
        "physical_water_mach_assuming_1490_m_s"=>v_fts*.3048/1490.,
        "experimental_channel_height_chords"=>5.,"experimental_working_section_xi"=>[-1.5,6.83],
        "water_temperature_c"=>alpha==6. ? 24. : nothing,
        "water_temperature_note"=>alpha==6. ? "24.00 C printed in this six-degree series" : "Most individual four-degree files have no entered temperature; report-wide range 24.0–25.8 C",
        "roughness"=>Dict("condition"=>"rough leading edge on both surfaces",
            "material"=>"glass spheres","height_in"=>.004,"height_m"=>.0001016,
            "extent_chord_fraction"=>.015,"modeled_as_exact_roughness_law"=>false))
    notes=String[
        "Published flow and tunnel references are separate from the numerical model; the model liquid (a2l 1.4, density ratio 100) is soft, a recorded limitation of every cavitating claim.",
        "The registered inlet correction extends xi_in from the experimental -1.5 to -5.0; H/c=5 and the 6.83-c outlet are retained.",
        "The tabulated 9%-thick section retains its approximately 0.006-c finite trailing edge on the conforming C+H mesh (strip topology, 8 faces per half-base).",
        "Mesh and model are N60R's registered configuration (256 x 108 C block, dn1 5e-5, near wake h/8 over 10h; Merkle 300/5, uniform start, double EOS) plus option H, on the CMeshQualityV1 mesh (164 wake columns): N60R's own mesh fails the cavitating quality gates. With cav_mesh_quality=none and cfd_n_wake=136 the 4-degree mesh is N60R's node for node.",
        "Reboud's eddy-viscosity correction is off: at run 132, the setting-selection point, N60R (off) matched the loads and closure bracket while the V3 default over-predicted them. Runs 134 and Leroux are held out and inform no setting.",
        "No leading-edge roughness model: the glass-sphere band of the experiment is absent from the smooth solver wall.",
        "Published moments are about mid-chord; they are retained as metadata and are not overlaid on the app's quarter-chord moments.",
        "Cavity percentages are nominal photographic setpoints; use the pressure-tap closure brackets for comparisons."]
    run==130 && push!(notes,"This preset uses run 130's sigma=0.994. The historical U25 sigma=1.011 averages two different-speed runs and does not exactly match run 130's pressure taps.")
    !isempty(flag) && push!(notes,flag)
    Dict{String,Any}("id"=>id,"name"=>"Caltech NACA 66(MOD) · run $run · $(Int(alpha))° · σ=$sigma",
        "description"=>"HSWT 1142 $nominal reference at Re≈2.05 million; H/c=5. Internal carrier Mach 0.20 is not the physical water Mach.",
        "geometry_id"=>"caltech","geometry_name"=>"Caltech NACA 66(MOD), 9% thickness, 2% camber, a=0.8, finite TE",
        "backends"=>["cavitation"],"sweep"=>Dict("enabled"=>false),
        "flow"=>_flow(alpha,2.05e6,sigma,5.,-5.,6.83;mesh=CALTECH_MESH),
        "boundary_conditions"=>_boundary(5.,-5.,6.83,alpha;experimental_xi_in=-1.5),
        "physical_conditions"=>physical,"reference_conditions"=>physical,
        "numerical_settings"=>_numerical(v_fts*.3048;mesh_note="N60R-Q: N60R's registered strip mesh settings with the CMeshQualityV1 option (164 wake columns, 65,696 cells, every cavitating gate passing; legacy Catmull-Rom surface, 0.006-c base)"),
        "reference"=>reference,"notes"=>notes,
        "source_url"=>CALTECH_URL,"provenance"=>Dict("source_url"=>CALTECH_URL,
            "geometry_source"=>"tools/naca66mod_section.jl; Brockett tabulated thickness with analytic a=0.8 camber",
            "geometry_sha256"=>GEOMETRIES["caltech"][2]),
        "sources"=>["memory/project_docs/2026-09-14_TIER1C_CALTECH_TABLES.md",
            alpha==4. ? "memory/project_docs/2026-09-21_CALTECH_CAVITATING_CP_SERIES.md" : "memory/project_docs/2026-09-21_CALTECH_SIX_DEGREE_CAVITATING_SERIES.md",
            "memory/project_docs/2026-10-14_V1_N60_REGISTRATION.md"])
end

# The author table is a closed contour (its traversal starts and ends at x/c 1.01924), so
# Auto selects the sharp C mesh. It takes the strip family's sharp control (BS, 2 Oct 2026):
# N60R's wall, nose and near-wake settings with no base.
# The near-wake setting is refused here (minimum corner sine 0.08 in the H/c 1.28 channel, as in
# the strip stage-1 stress set), and CMeshQualityV1 cannot meet G2a (68.8 against 65 degrees),
# so Leroux takes the plain C mesh: N60R's wall and nose, 136 wake columns, worst face 70.8
# degrees, largest aspect ratio 2,133 (measured 20 Oct 2026).
# Wall spacing at the Caltech meshes' y+: dn1 scales as 1/(Re sqrt(cf/2)), 5e-5 at Re 2.05e6 -> 1.2e-4 at 0.8e6.
# At 5e-5 the wake-cut first-row cells (as thick as dn1) bound the SST clock once the wake turned turbulent:
# dt fell fivefold at 0.6 CTU with one stage-2 rejection per accepted step (LRX, 20 Oct 2026, stopped).
const LEROUX_MESH=let m=_strip_mesh(0.;literal=(0.,0.,0.),evaluator="natural_c2")
    foreach(k->delete!(m,k),("cfd_n_base","cav_mesh_te_ds","cav_mesh_wake_near_ds","cav_mesh_wake_near_len"))
    m["cfd_n_wake"]=136;m["cav_mesh_quality"]="none";m["cav_mesh_dn1"]=1.2e-4;m
end

function _leroux()
    physical=Dict{String,Any}("source"=>"Leroux, Coutier-Delgosha and Astolfi, Physics of Fluids 17, 052101 (2005)",
        "chord_m"=>.150,"speed_m_s"=>5.33,"reynolds"=>.8e6,"alpha_deg"=>8.,
        "sigma_nominal"=>1.25,"sigma_reported_range"=>[1.25,1.28],
        "tunnel_height_m"=>.192,"tunnel_width_m"=>.192,"tunnel_height_chords"=>1.28,
        "physical_water_mach"=>nothing,"physical_water_mach_assuming_1490_m_s"=>5.33/1490.,"roughness"=>nothing,
        "roughness_note"=>"No verified roughness/trip specification was recovered in the local reference audit; the Caltech roughness is not substituted",
        "geometry_note"=>"Author table keeps the original published x/c units and closes at x/c=1.019240; its mapping to the nominal 0.150-m manufactured chord remains unresolved")
    numerical=_numerical(5.33;mesh_note="sharp C mesh (the author table is closed): N60R's nose settings with the wall spacing at the Caltech meshes' y+ (dn1 1.2e-4 at Re 0.8e6), 136 wake columns, no near-wake hold (refused in this channel); natural C2 surface evaluator (the Catmull-Rom sawtooth reaches a 1.53 spacing ratio on this section). CMeshQualityV1 is not met here (worst face 70.8 against 65 degrees).")
    numerical["mesh_topology"]="C (sharp trailing edge; Auto selects C+H only for a finite base)"
    numerical["held_out"]="Leroux is a held-out case: its settings are the Caltech series', fixed at run 132"
    reference=Dict{String,Any}("name"=>"Leroux 8-degree cloud-cavitation reference",
        "source_url"=>LEROUX_URL,"kind"=>"published frequency and maximum attached cavity length",
        "alpha_deg"=>8.,"sigma"=>1.25,"cl"=>nothing,"cd"=>nothing,"cm"=>nothing,
        "cp"=>Dict("upper"=>Any[],"lower"=>Any[]),"frequency_hz"=>18.,
        "maximum_attached_cavity_length_xc"=>.60,"strouhal_maximum_cavity"=>.304,
        "strouhal_chord"=>.507,"period_ctu"=>1.974,
        "frequency_measurement"=>"wall-pressure spectrum, not force oscillation or video-frame frequency",
        "no_measured_force_overlay"=>true,"points"=>Any[])
    Dict{String,Any}("id"=>"leroux_2005_a8_s125",
        "name"=>"Leroux N1.1 NACA 66(MOD) · 8° · σ=1.25",
        "description"=>"Published Re=0.8 million, U=5.33 m/s, H/c=1.28. Internal carrier Mach 0.20 and the current calibration are separate numerical settings.",
        "geometry_id"=>"leroux","geometry_name"=>"Author-table N1.1-mod. NACA 66(mod.)-312, a=0.8",
        "backends"=>["cavitation"],"sweep"=>Dict("enabled"=>false),
        "flow"=>_flow(8.,.8e6,1.25,1.28,-4/3,5.;mesh=LEROUX_MESH),
        "boundary_conditions"=>_boundary(1.28,-4/3,5.,8.),
        "physical_conditions"=>physical,"reference_conditions"=>physical,
        "numerical_settings"=>numerical,"reference"=>reference,
        "source_url"=>LEROUX_URL,"provenance"=>Dict("source_url"=>LEROUX_URL,
            "coordinate_source_url"=>"https://actesjh.ec-nantes.fr/images/9JH/Annexe/07-02-03.pdf",
            "geometry_sha256"=>GEOMETRIES["leroux"][2]),
        "notes"=>[
            "The exact 101-point author table is whitelisted. The older tools/naca66mod312_a08.dat omits the N=1.1 forebody modification and is not substituted.",
            "Published coordinate scale is preserved; the extended-tail/chord convention remains unresolved. This source is not an admitted manufacturing reconstruction.",
            "The author table closes at x/c 1.01924, so Auto meshes it as a sharp C (N60R's nose settings, the wall spacing at the Caltech meshes' y+, 136 wake columns) with the natural C2 surface evaluator; the near-wake hold and CMeshQualityV1 cannot be met in this H/c 1.28 channel.",
            "Model and rates are the Caltech series' version-1 configuration (soft model liquid a2l 1.4 and density ratio 100, Merkle 300/5, no Reboud correction, option H, CFL 0.9); Leroux is held out and informs no setting. Published reference parity is not established.",
            "The 18-Hz and St=0.304 targets use maximum attached cavity length and wall-pressure spectra. No measured CL/CD/CM or digitized Cp values are invented.",
            "The 12-CTU horizon holds about six published shedding periods (1.97 CTU each); it is not a claim that the solution settles or that shedding is reproduced."],
        "sources"=>["memory/project_docs/2026-09-04_LEROUX_CONVENTION_MATCH.md",
            "memory/project_docs/2026-09-08_LEROUX_REFERENCE_GEOMETRY_AUDIT.md",
            "memory/project_docs/2026-09-12_LEROUX_GEOMETRY_VALIDATION_PLAN.md"])
end

const PRESETS=[
    _caltech(130,4.,.994,39.39,10.82,.664,.0164,.108,"25% cavity",[.16,.26],
        _cp([-.992,-1.017,-.999,-.997,-.472,-.495,-.533,-.507,-.433,-.305,-.117,-.008],.297)),
    _caltech(132,4.,.842,39.37,9.22,.713,.0198,.110,"60% cavity",[.45,.68],
        _cp([-.815,-.877,-.865,-.847,-.843,-.837,-.832,-.743,-.323,-.245,-.094,.002],.301);
        flag="The published run-132 row is flagged: particle stuck between model and wall. Its repeat and a lower-speed repeat are documented separately."),
    _caltech(134,4.,.765,39.53,8.49,.712,.0314,.088,"80% cavity",[.56,.81],
        _cp([-.728,-.775,-.777,-.796,-.793,-.797,-.787,-.721,-.604,-.406,-.178,-.106],.287)),
    _caltech(123,6.,1.464,39.38,15.92,.820,.0310,.153,"25% cavity",[.16,.26],
        _cp([-1.43585,-1.45885,-1.46212,-1.38086,-.53700,-.56994,-.57318,-.53497,-.44863,-.30340,-.12193,-.02091],.42210)),
    _caltech(125,6.,1.118,39.49,12.33,.884,.0447,.152,"60% cavity",[.56,.68],
        _cp([-1.13403,-1.13010,-1.13213,-1.12869,-1.05209,-1.03803,-1.01685,-.80851,-.27002,-.20810,-.10423,-.03962],.41557)),
    _leroux()]

"Return independent public preset dictionaries; callers cannot mutate the catalog."
catalog()=deepcopy(PRESETS)

function preset(id::AbstractString)
    index=findfirst(p->p["id"]==id,PRESETS)
    index===nothing && throw(ArgumentError("Unknown cavitation reference case: $id"))
    deepcopy(PRESETS[index])
end

"Resolve only the two internal, checksum-bound coordinate sources in this catalog."
function geometry_path(id::AbstractString)
    selected=preset(id);relative,expected=GEOMETRIES[selected["geometry_id"]]
    path=normpath(joinpath(ROOT,relative))
    isfile(path) || throw(ArgumentError("Cavitation reference geometry is missing: $relative"))
    bytes2hex(sha256(read(path)))==expected || throw(ArgumentError("Cavitation reference geometry checksum changed: $relative"))
    path
end

"""
Return exact source coordinates in one contiguous airfoil traversal.

The Caltech table stores upper LE-to-TE and lower LE-to-TE rows without a blank
separator. Reorder those two surfaces into upper TE-to-LE-to-lower TE, removing
only the duplicated shared LE. The Leroux source is already a Selig traversal;
retain every row in its original order. No fitting or coordinate changes occur.
"""
function geometry_coordinates(id::AbstractString)
    selected=preset(id)
    rows=Tuple{Float64,Float64}[]
    for line in eachline(geometry_path(id))
        fields=split(strip(line))
        length(fields)>=2 || continue
        startswith(fields[1],"#") && continue
        x=tryparse(Float64,fields[1]);y=tryparse(Float64,fields[2])
        (isnothing(x) || isnothing(y)) && continue
        isfinite(x) && isfinite(y) || throw(ArgumentError("Nonfinite reference coordinate for $id"))
        push!(rows,(x,y))
    end
    length(rows)>=20 || throw(ArgumentError("Too few reference coordinates for $id"))
    if selected["geometry_id"]=="caltech"
        restarts=[k for k in 3:length(rows)-2 if rows[k][1]<.05 && rows[k-1][1]>.5]
        length(restarts)==1 || throw(ArgumentError("Caltech reference must contain exactly two LE-to-TE surfaces"))
        split_at=only(restarts)
        upper=rows[1:split_at-1];lower=rows[split_at:end]
        upper[1]==lower[1] || throw(ArgumentError("Caltech reference surfaces must share the exact leading-edge node"))
        rows=vcat(reverse(upper),lower[2:end])
    end
    (x=first.(rows),y=last.(rows))
end

end # module AirfoilCavitationCases
