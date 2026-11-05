include(joinpath(PUBLIC_ROOT,"src","solver","grid_quantization.jl"))
include(joinpath(PUBLIC_ROOT,"src","geometry","triangle_subdivision.jl"))

const MAX_WEB_SURFACE_TRIANGLES = 5_000_000

function _triangle_surface_area(a,b,c)
    u=ntuple(i->b[i]-a[i],3);v=ntuple(i->c[i]-a[i],3)
    .5sqrt((u[2]*v[3]-u[3]*v[2])^2+(u[3]*v[1]-u[1]*v[3])^2+(u[1]*v[2]-u[2]*v[1])^2)
end

"Exact area of source facets on the solved side of a placed Y=0 plane; no symmetry cap is invented."
function _positive_y_surface_area(triangles,shift_y)
    area=0.
    for tri in triangles
        inside=map(p->p[2]+shift_y>=0,tri)
        all(inside) && (area+=_triangle_surface_area(tri...);continue)
        any(inside) || continue
        polygon=NTuple{3,Float64}[]
        for i in 1:3
            a=tri[i];b=tri[mod1(i+1,3)]
            inside[i] && push!(polygon,a)
            if inside[i]!=inside[mod1(i+1,3)]
                f=(-shift_y-a[2])/(b[2]-a[2])
                push!(polygon,ntuple(k->a[k]+f*(b[k]-a[k]),3))
            end
        end
        for i in 2:length(polygon)-1
            area+=_triangle_surface_area(polygon[1],polygon[i],polygon[i+1])
        end
    end
    area
end

function _surface_subdivision_preview(triangles,max_edge)
    count=SurfaceSubdivision.subdivision_count(triangles,max_edge;limit=MAX_WEB_SURFACE_TRIANGLES)
    count <= MAX_WEB_SURFACE_TRIANGLES || throw(ArgumentError(
        "The requested maximum surface triangle edge would create over 5 million triangles. Increase Model → Maximum surface triangle edge (currently $(round(1000max_edge,sigdigits=5)) mm). The fluid-cell size is a separate setting."))
    original_max=maximum(SurfaceSubdivision.triangle_max_edge,triangles)
    Dict{String,Any}("maximum_edge_m"=>max_edge,"source_triangles"=>length(triangles),
        "refined_triangles_before_symmetry"=>count,"source_maximum_edge_m"=>original_max,
        "gpu_surface_allowance_gb"=>128count/1e9,
        "note"=>"Actual solver sampling triangles are subdivided after model scaling; symmetry clipping can reduce this count. Refinement preserves the original surface shape and does not add fluid resolution.")
end

function _symmetry_placement(domain)
    value=get(domain,"symmetry_placement",get(domain,"symmetry_center_y",false) ? "center" : "wall")
    value isa AbstractString && value in ("none","wall","center") ||
        throw(ArgumentError("domain.symmetry_placement must be none, wall, or center"))
    String(value)
end

_fixed_symmetry(domain)=get(domain,"boundary","")=="symmetry" && _symmetry_placement(domain)=="none"

function _domain_lengths(domain::AbstractDict, extent)
    boundary = domain["boundary"]
    e = boundary == "ground" ? [extent[1],extent[3],extent[2]] : extent
    [e[1] + domain["upstream"] + domain["downstream"],
     (_fixed_symmetry(domain) ? domain["symmetry_body_y_max"] : boundary == "symmetry" && _symmetry_placement(domain)=="center" ? e[2]/2 : e[2]) + (boundary == "free_air" ? 2*domain["lateral"] :
             boundary == "ground" ? domain["height"] : domain["lateral"]),
     e[3] + 2*(boundary == "ground" ? domain["lateral"] : domain["height"])]
end

function _resolved_body_extent(domain, extent)
    e = domain["boundary"] == "ground" ? [extent[1],extent[3],extent[2]] : collect(extent)
    if _fixed_symmetry(domain)
        e[2]=domain["symmetry_body_y_max"]
    elseif domain["boundary"] == "symmetry" && _symmetry_placement(domain)=="center"
        e[2] /= 2
    end
    e
end

function _domain_sizing(value)
    value isa AbstractString && value in ("minimum","fixed","proportions") ||
        throw(ArgumentError("domain.sizing must be minimum, fixed, or proportions"))
    String(value)
end

function _domain_proportions(value)
    value isa AbstractVector && length(value)==3 ||
        throw(ArgumentError("domain.proportions must contain three positive finite X, Y, Z ratios"))
    all(x->x isa Real && !(x isa Bool) && isfinite(x) && x>0,value) ||
        throw(ArgumentError("domain.proportions must contain three positive finite X, Y, Z ratios"))
    ratios=Float64.(value)
    ratios ./= maximum(ratios)
    all(x->isfinite(x)&&x>0,ratios) || throw(ArgumentError("Domain proportions are too extreme; choose ratios closer together"))
    ratios
end

function _effective_domain_proportions(domain)
    ratios=_domain_proportions(get(domain,"proportions",[1.,1.,1.]))
    domain["boundary"]=="symmetry" && (ratios[2]=.5*min(ratios[1],ratios[3]))
    ratios ./= maximum(ratios)
    ratios
end

function _grow_domain_to_lengths!(domain,before,target)
    boundary=domain["boundary"]
    extra=max.(0.,target.-before)
    domain["upstream"] += extra[1]/2
    domain["downstream"] += extra[1]/2
    domain[boundary=="ground" ? "height" : "lateral"] += extra[2]/(boundary=="free_air" ? 2 : 1)
    domain[boundary=="ground" ? "lateral" : "height"] += extra[3]/2
    domain
end

function _quantize_proportioned_domain(target,coarse_dx,boundary;octree=false)
    block=(octree ? 2 : 8)*coarse_dx
    if boundary=="symmetry"
        nx=max(octree ? 2 : 8,2*_stable_block_ceil(target[1]/(2block)))
        nz=max(octree ? 2 : 8,2*_stable_block_ceil(target[3]/(2block)))
        return block.*[nx,min(nx,nz)÷2,nz]
    end
    block.*[max(octree ? 1 : 4,_stable_block_ceil(length/block)) for length in target]
end

"Treat physical clearances as minima while preserving a ground or symmetry plane at Y=0."
function _expand_minimum_domain!(domain::AbstractDict, extent, dx::Real, levels::Integer;
                                 translation_x::Real=0.,octree::Bool=false)
    requested = deepcopy(domain)
    before = _domain_lengths(domain,extent)
    x_offset=_number(get(domain,"x_offset_m",0.),"domain.x_offset_m";lo=-10_000,hi=10_000)
    relative_x=Float64(translation_x)-x_offset
    required=copy(before)
    # X clearances grow equally about the shifted domain centre. This preserves
    # the requested minima and uses any AMR/proportion padding before growing
    # the final envelope further; the geometry itself is never translated here.
    required[1]+=2abs(relative_x)
    coarse_dx = dx * 2.0^(levels-1)
    # The native healthy-pyramid rule needs four 8-cell coarse blocks per axis.
    # When an axis needs expansion, leave one additional coarse block so YAML
    # reference-unit conversion cannot round the dimension just below that rule.
    minimum_axis = (octree ? 2 : 32)*coarse_dx
    expanded_axis = (octree ? 2 : 40)*coarse_dx
    fit_floor = minimum_axis*(octree ? 1 : 1+1e-10)
    boundary=domain["boundary"]
    sizing=_domain_sizing(get(domain,"sizing","minimum"))
    if sizing=="fixed" && !(domain["upstream"]+relative_x>0 && domain["downstream"]-relative_x>0)
        throw(ArgumentError("The domain X offset places the model on or outside the inlet/outlet of this fixed closed section. Reduce the offset, increase the upstream/downstream distances, or use minimum domain sizing; the model is not moved automatically."))
    end
    proportion_target=nothing;continuous_target=nothing;effective_ratios=nothing
    if sizing == "minimum" && boundary!="symmetry"
        target=[value<fit_floor ? expanded_axis : value for value in required]
        _grow_domain_to_lengths!(domain,before,target)
    elseif sizing=="minimum" && boundary=="symmetry"
        safe_floor=minimum_axis
        min_y=max(required[2],safe_floor)
        target_x=max(required[1],safe_floor,2min_y)
        target_z=max(required[3],safe_floor,2min_y)
        continuous_target=[target_x,.5*min(target_x,target_z),target_z]
        effective_ratios=continuous_target./maximum(continuous_target)
        proportion_target=_quantize_proportioned_domain(continuous_target,coarse_dx,boundary;octree)
        _grow_domain_to_lengths!(domain,before,proportion_target)
    elseif sizing == "proportions"
        effective_ratios=_effective_domain_proportions(domain)
        # Retain the solver's four-coarse-block safety floor on every axis.
        # In symmetry mode X/Z grow sufficiently for Y=half(shorter X/Z);
        # the interpolation buffer at the mirror plane is never weakened.
        safe_floor=minimum_axis
        scale=maximum(max.(required,safe_floor)./effective_ratios)
        continuous_target=effective_ratios.*scale
        all(isfinite,continuous_target) || throw(ArgumentError("Domain proportions are too extreme; choose ratios closer together"))
        proportion_target=_quantize_proportioned_domain(continuous_target,coarse_dx,boundary;octree)
        _grow_domain_to_lengths!(domain,before,proportion_target)
    end
    after = _domain_lengths(domain,extent)
    changed = any(after .> before)
    note=proportion_target!==nothing ?
        "$(sizing=="proportions" ? "Desired domain proportions are retained approximately after native coarse-block snapping. " : "")$(boundary=="symmetry" ? "The symmetry Y span targets half the shorter X or Z span; X/Z may grow to retain the AMR safety floor. Native block snapping can change these proportions slightly. " : "")Requested clearances remain minima and all $levels AMR levels are retained." :
        changed ? "$(boundary=="free_air" ? "Free-air" : boundary=="ground" ? "Ground" : "Symmetry") domain expanded automatically to retain $levels AMR levels at the requested surface cell size. Requested clearances are minima; $(boundary=="free_air" ? "far faces grew" : "the Y=0 plane and model clearance remain anchored")." : ""
    adjustment=Dict{String,Any}("expanded"=>changed,"requested_domain"=>requested,
        "requested_size_m"=>before,"resolved_size_m"=>after,
        "minimum_axis_size_m"=>minimum_axis,"sizing"=>sizing,
        "anchored_plane"=>boundary=="ground" ? "ground_y0" : boundary=="symmetry" ? "symmetry_y0" : nothing,
        "message"=>note)
    adjustment["x_offset_m"]=x_offset
    adjustment["model_min_x_m"]=domain["upstream"]+relative_x
    adjustment["model_center_x_m"]=domain["upstream"]+extent[1]/2+relative_x
    adjustment["streamwise_clearances_m"]=[domain["upstream"]+relative_x,domain["downstream"]-relative_x]
    if !iszero(x_offset)
        adjustment["message"] *= (isempty(note) ? "" : " ") *
            "Domain X offset $(round(x_offset,sigdigits=6)) m shifts the domain $(x_offset>0 ? "downstream" : "upstream") relative to the unchanged model. $(sizing=="fixed" ? "The fixed section keeps its requested span." : "Upstream/downstream clearances remain minima; the domain grows when needed.")"
    end
    if proportion_target!==nothing
        sizing=="proportions" && (adjustment["requested_proportions"]=_domain_proportions(get(requested,"proportions",[1.,1.,1.])))
        adjustment["effective_proportions"]=effective_ratios
        adjustment["proportion_target_size_m"]=proportion_target
        adjustment["proportion_continuous_size_m"]=continuous_target
        adjustment["resolved_proportions"]=after./maximum(after)
        adjustment["coarse_block_size_m"]=(octree ? 2 : 8)*coarse_dx
        adjustment["symmetry_y_rule"]=boundary=="symmetry" ? "half_shorter_xz" : nothing
    end
    adjustment
end

# Compatibility with callers that used the original free-air-only helper name.
_expand_free_air_domain!(domain::AbstractDict,extent,dx::Real,levels::Integer)=
    _expand_minimum_domain!(domain,extent,dx,levels)

"Solver admission policy; wall intersections in both native backends already use both triangle sides."
function _geometry_handling(value)
    value isa AbstractString && value in ("closed_solid", "two_sided_open") ||
        throw(ArgumentError("geometry_handling must be closed_solid or two_sided_open"))
    String(value)
end

function _validate_case(raw::AbstractDict; enforce_mesh_limits::Bool=true)
    model_id = _id(get(raw, "model_id", ""))
    source, source_bounds = _load_model(model_id)
    t = get(raw, "transform", Dict{String,Any}())
    t isa AbstractDict || throw(ArgumentError("transform must be an object"))
    scale = _vec3(get(t, "scale", [1,1,1]), "transform.scale"; lo=1e-5, hi=1e5)
    rotation = _vec3(get(t, "rotation_deg", [0,0,0]), "transform.rotation_deg"; lo=-360, hi=360)
    translation = _vec3(get(t, "translation", [0,0,0]), "transform.translation"; lo=-1e4, hi=1e4)
    f_input=get(raw,"flow",Dict{String,Any}()); f_input isa AbstractDict || throw(ArgumentError("flow must be an object"))
    alpha=_number(get(f_input,"alpha",0.),"flow.alpha";lo=-85,hi=85)
    beta=_number(get(f_input,"beta",0.),"flow.beta";lo=-85,hi=85)
    transformed, bounds = _transform(source, source_bounds, scale, rotation)
    if alpha!=0 || beta!=0
        transformed=Triangle[ntuple(i->_flow_rotation(tri[i],alpha,beta),3) for tri in transformed]
        bounds=_bounds(transformed)
    end
    triangle_edge=_number(get(raw,"surface_triangle_max_edge_m",0.05),"surface_triangle_max_edge_m";lo=1e-4,hi=100)
    subdivision=_surface_subdivision_preview(transformed,triangle_edge)
    surface_area = 0.0
    for (a,b,c) in transformed
        u = ntuple(i -> b[i]-a[i], 3); v = ntuple(i -> c[i]-a[i], 3)
        cross = (u[2]*v[3]-u[3]*v[2], u[3]*v[1]-u[1]*v[3], u[1]*v[2]-u[2]*v[1])
        surface_area += 0.5*sqrt(sum(x*x for x in cross))
    end
    isfinite(surface_area) && surface_area > 0 || throw(ArgumentError("Model surface area must be positive and finite"))
    d = get(raw, "domain", nothing)
    d isa AbstractDict || throw(ArgumentError("domain must be an object"))
    domain = Dict{String,Any}()
    for key in ("upstream", "downstream", "lateral", "height")
        domain[key] = _number(get(d, key, nothing), "domain.$key"; lo=0.02, hi=10_000)
    end
    boundary = string(get(d, "boundary", "free_air"))
    boundary in ("free_air", "ground", "symmetry") ||
        throw(ArgumentError("domain.boundary must be free_air, ground, or symmetry"))
    domain["boundary"] = boundary
    domain["x_offset_m"]=_number(get(d,"x_offset_m",0.),"domain.x_offset_m";lo=-10_000,hi=10_000)
    boundary=="symmetry" && beta!=0 && throw(ArgumentError("Symmetry requires beta = 0"))
    reference=haskey(raw,"reference") ? _reference_dimensions(raw["reference"],boundary) : nothing
    sizing=_domain_sizing(get(d,"sizing","minimum"))
    haskey(d,"sizing") && (domain["sizing"]=sizing)
    ratios=_domain_proportions(get(d,"proportions",[1.,1.,1.]))
    (haskey(d,"proportions") || sizing=="proportions") && (domain["proportions"]=ratios)
    center_y = get(d,"symmetry_center_y",false)
    center_y isa Bool || throw(ArgumentError("domain.symmetry_center_y must be true or false"))
    placement=_symmetry_placement(d);center_y=placement=="center"
    (haskey(d,"symmetry_center_y") || haskey(d,"symmetry_placement")) && (domain["symmetry_center_y"] = center_y)
    haskey(d,"symmetry_placement") && (domain["symmetry_placement"]=placement)
    if placement=="none"
        domain["symmetry_offset_y_m"]=_number(get(d,"symmetry_offset_y_m",0.),"domain.symmetry_offset_y_m";lo=-10_000,hi=10_000)
        if boundary=="symmetry"
            # The frozen offset is relative to the transformed construction origin.
            # It is not recomputed when a saved study is opened or a new mesh is built.
            domain["symmetry_body_y_max"]=Float64(bounds["max"][2])+domain["symmetry_offset_y_m"]+translation[2]
            domain["symmetry_body_y_max"]>0 || throw(ArgumentError("The model is entirely outside the positive-Y symmetry domain. Adjust its Y offset or select automatic symmetry alignment."))
        end
    end
    dx = _number(get(raw, "surface_dx", nothing), "surface_dx"; lo=1e-5, hi=100)
    # The CPU build creates the coarse lattice explicitly. Reject combinations
    # that cannot be built interactively before computing integer grid counts.
    extent = [bounds["max"][i] - bounds["min"][i] for i in 1:3]
    resolved = _resolved_body_extent(domain,extent)
    lengths = _domain_lengths(domain,extent)
    all(isfinite, lengths) || throw(ArgumentError("Transformed model or domain is too large"))
    solver = string(get(raw, "solver", "main"))
    solver in ("main", "lbmtau", "pond_thermal", "q19_octree") || throw(ArgumentError("Unknown Aero Lab solver backend"))
    lattice = string(get(raw, "lattice", "d3q27"))
    lattice in ("d3q27", "d3q39", "d3q19") || throw(ArgumentError("Aero Lab lattice must be d3q27, d3q39 or d3q19"))
    lattice == "d3q39" && solver != "main" && throw(ArgumentError("D3Q39 in Aero Lab requires the main LUDWIG solver"))
    lattice == "d3q19" && solver != "q19_octree" && throw(ArgumentError("D3Q19 is the lattice of the CPU octree solver"))
    solver == "q19_octree" && lattice != "d3q19" && throw(ArgumentError("The CPU octree solver uses the D3Q19 lattice"))
    (solver == "q19_octree" ? _q19_available() : isfile(joinpath(solver == "lbmtau" ? PINNED_SOLVER : MAIN_SOLVER, "src", "main.jl"))) ||
        throw(ArgumentError("Selected solver is unavailable"))
    if solver == "lbmtau" && any(x -> !iszero(x), translation)
        throw(ArgumentError("The pinned lbmtau solver cannot place a model at an offset; select the main solver"))
    end
    solver=="lbmtau" && _fixed_symmetry(domain) && throw(ArgumentError("The pinned lbmtau backend cannot retain manual symmetry placement. Select the main, thermal or D3Q19 backend, or choose automatic alignment."))
    if boundary == "symmetry" && placement!="none" && !iszero(translation[2])
        throw(ArgumentError("Symmetry requires the model anchored at the Y=0 plane; set lateral translation to zero"))
    end
    _fixed_symmetry(domain) || abs(translation[2]) <= domain["lateral"] ||
        throw(ArgumentError("Lateral translation moves the model outside the domain"))
    if boundary == "ground"
        0 <= translation[3] <= domain["height"] ||
            throw(ArgumentError("Ground mode requires model clearance from 0 to domain.height"))
    else
        abs(translation[3]) <= domain["height"] ||
            throw(ArgumentError("Vertical translation moves the model outside the domain"))
    end
    levels = _integer(get(raw, "levels", 3), "levels"; lo=1, hi=solver == "q19_octree" ? Q19_MAX_LEVELS : 7)
    domain_adjustment = _expand_minimum_domain!(domain,extent,dx,levels;translation_x=translation[1],octree=solver=="q19_octree")
    lengths = domain_adjustment["resolved_size_m"]
    target_raw = get(raw, "target_vram_gb", nothing)
    target_vram_gb = target_raw === nothing ? nothing : _number(target_raw, "target_vram_gb"; lo=0.5, hi=24)
    percent_raw = get(raw, "target_vram_percent", nothing)
    target_vram_percent = percent_raw === nothing ? nothing : _number(percent_raw, "target_vram_percent"; lo=1, hi=100)
    target_requested = target_vram_gb !== nothing || target_vram_percent !== nothing
    mesh_sizing = String(get(raw,"mesh_sizing",target_requested ? "vram" : "manual"))
    mesh_sizing in ("manual","vram") || throw(ArgumentError("mesh_sizing must be manual or vram"))
    fitting = mesh_sizing == "vram"
    fitting && !target_requested && throw(ArgumentError("VRAM sizing requires a target percentage or GB value"))
    fitting && solver == "q19_octree" && throw(ArgumentError("VRAM sizing is not available for the CPU octree solver; choose a manual surface cell size"))
    coarse_dx = dx * 2.0^(levels - 1)
    if enforce_mesh_limits && !fitting && solver!="q19_octree" && levels > 1 &&
            any(_roundoff_snap_grid_integer(length/(32coarse_dx))<1 for length in lengths)
        throw(ArgumentError("The fixed domain is too small for $levels AMR levels at the requested surface cell size. Use minimum domain sizing or enlarge the closed section; requested levels are not reduced automatically."))
    end
    coarse_cells = prod(lengths ./ coarse_dx)
    !enforce_mesh_limits || fitting || coarse_cells <= 30_000_000 || throw(ArgumentError(
        "The requested domain has over 30 million coarse cells; increase surface cell size or levels, or reduce the domain"))
    fine_envelope = prod(max.(resolved .+ 16*dx, 16*dx)) / dx^3
    !enforce_mesh_limits || fitting || solver=="q19_octree" || levels == 1 || fine_envelope <= 30_000_000 || throw(ArgumentError(
        "The near-body fine-cell envelope is too large; increase surface cell size or reduce model size"))
    steps = _integer(get(raw, "steps", 800), "steps"; lo=8, hi=1_000_000)
    frames = _integer(get(raw, "frames", 12), "frames"; lo=2, hi=40)
    default_every = max(8, 8*cld(cld(steps,frames),8))
    display_every = _integer(get(raw,"display_every",default_every), "display_every"; lo=1, hi=1_000_000)
    outputs = _validate_outputs(get(raw,"outputs",Dict{String,Any}()))
    f = get(raw, "flow", Dict{String,Any}())
    f isa AbstractDict || throw(ArgumentError("flow must be an object"))
    flow = Dict{String,Any}("alpha"=>alpha,"beta"=>beta,
        "velocity" => _number(get(f, "velocity", 20.0), "flow.velocity"; lo=0.01, hi=solver=="pond_thermal" ? 10000 : 1000),
        "density" => _number(get(f, "density", 1.225), "flow.density"; lo=0.01, hi=1e4),
        "kinematic_viscosity" => _number(get(f, "kinematic_viscosity", 1.5e-5), "flow.kinematic_viscosity"; lo=1e-10, hi=1),
        "speed_of_sound" => _number(get(f, "speed_of_sound", 343.0), "flow.speed_of_sound"; lo=1, hi=1e5))
    solver=="pond_thermal" && _pond_flow!(flow,f)
    mach = flow["velocity"] / flow["speed_of_sound"]
    max_mach=solver=="pond_thermal" ? 1.5 : solver=="q19_octree" ? Q19_MAX_MACH : lattice=="d3q39" ? 1.6 : 0.6
    mach <= max_mach || throw(ArgumentError("Selected Aero Lab backend requires Mach <= $max_mach"))
    cuts_raw = get(raw, "cuts", Any[])
    cuts_raw isa AbstractVector && length(cuts_raw) <= 6 ||
        throw(ArgumentError("cuts must be a list of at most six planes"))
    cuts = Dict{String,Any}[]
    for (i, cut) in enumerate(cuts_raw)
        cut isa AbstractDict || throw(ArgumentError("cuts[$i] must be an object"))
        axis = string(get(cut, "axis", ""))
        axis in ("x", "y", "z") || throw(ArgumentError("cuts[$i].axis must be x, y, or z"))
        position = _number(get(cut, "position", nothing), "cuts[$i].position"; lo=0, hi=1e5)
        axis_index = axis == "x" ? 1 : axis == "y" ? 2 : 3
        limit = (solver=="q19_octree" ? _q19_coarse_axis(lengths[axis_index],coarse_dx) : 8*ceil(lengths[axis_index] / (8*coarse_dx)))*coarse_dx
        position <= limit || throw(ArgumentError("cuts[$i] lies outside the resolved domain on axis $axis"))
        push!(cuts, Dict("axis" => axis, "position" => position))
    end
    geometry_handling = _geometry_handling(get(raw,"geometry_handling","closed_solid"))
    geometry_audit = _model_geometry_audit(model_id)
    symmetry_shift=placement=="none" ? get(domain,"symmetry_offset_y_m",0.)+translation[2] : center_y ? -(bounds["min"][2]+bounds["max"][2])/2 : -bounds["min"][2]
    if geometry_audit isa AbstractDict && boundary == "symmetry" && !get(geometry_audit,"outward_orientation_verified",false)
        geometry_audit["symmetry_boundary_verified"] = _open_edges_match_symmetry(transformed,bounds;shift_y=symmetry_shift)
    end
    solved_surface_area=surface_area
    if solver=="q19_octree" && boundary=="symmetry"
        solved_surface_area=_positive_y_surface_area(transformed,symmetry_shift)
        solved_surface_area>0 || throw(ArgumentError("No model surface intersects the positive-Y domain at this placement"))
    end
    Dict{String,Any}("reference"=>reference,"model_id" => model_id,
        "transform" => Dict("scale" => collect(scale), "rotation_deg" => collect(rotation), "translation" => collect(translation)),
        "domain" => domain, "requested_domain" => domain_adjustment["requested_domain"],
        "domain_adjustment" => domain_adjustment,
        "surface_dx" => dx, "solver" => solver, "lattice" => lattice,
        "surface_triangle_max_edge_m"=>triangle_edge,"surface_subdivision"=>subdivision,
        "mesh_sizing" => mesh_sizing,
        "levels" => levels, "steps" => steps, "frames" => frames, "flow" => flow,
        "target_vram_gb" => target_vram_gb, "target_vram_percent" => target_vram_percent,
        "display_every" => display_every, "effective_display_every" => min(display_every,steps),
        "outputs" => outputs,
        "cuts" => cuts, "triangles" => length(transformed),
        "surface_area_m2" => surface_area,"solved_surface_area_m2"=>solved_surface_area, "geometry_audit" => geometry_audit,
        "geometry_handling" => geometry_handling,
        "symmetry_adjustment" => Dict("center_y"=>boundary == "symmetry" && center_y,"placement"=>placement,
            "message"=>_fixed_symmetry(domain) ? "The saved model Y position is retained without automatic alignment. Only the positive-Y domain is solved." : boundary == "symmetry" && center_y ?
                "The full model is centered on Y=0. The native mesher clips its positive-Y half for the symmetry domain; the original geometry remains unchanged. Symmetry assumes the geometry and flow are mirrored across this plane." : ""),
        "octree" => solver == "q19_octree" ? _q19_settings(raw) : nothing,
        "bounds" => bounds), transformed
end

function _estimate(case; gpu=_gpu_memory())
    b = case["bounds"]
    ext = Float64[b["max"][i] - b["min"][i] for i in 1:3]
    boundary = case["domain"]["boundary"]
    e = _resolved_body_extent(case["domain"],ext)
    d = case["domain"]; dx = case["surface_dx"]; L = case["levels"]
    lx,ly,lz = _domain_lengths(d,ext)
    h0 = dx * 2.0^(L-1)
    dims = [_q19_case(case) ? _q19_coarse_axis(v,h0) : 8*_stable_grid_block_count(v,h0,8;minimum_blocks=4) for v in (lx,ly,lz)]
    coarse = prod(dims)
    # A conservative shape-envelope preview; the solver's build-only summary replaces this.
    fine = L == 1 ? 0 : ceil(Int, prod(max.(e .+ 16*dx, 16*dx)) / dx^3)
    intermediate = L <= 2 ? 0 : ceil(Int, prod(max.(e .+ 32*dx, 32*dx)) / (2*dx)^3)
    cells = coarse + fine + intermediate
    lattice_factor = get(case, "lattice", "d3q27") == "d3q39" ? 3.0 : 1.0
    memory = lattice_factor*(coarse*462 + intermediate*462 + fine*700)/1e9 +
             (lattice_factor > 1 ? 1.6 : 0.7)
    _pond_case(case) && (memory=max(memory,1800cells/1e9+.7))
    # Uploaded centers/normals/areas, stress maps, averages and temporary force
    # buffers scale with surface facets independently of allocated fluid cells.
    surface_memory=get(get(case,"surface_subdivision",Dict()),"gpu_surface_allowance_gb",0.0)
    memory += surface_memory
    q19=_q19_case(case)
    if q19
        # the octree keeps surface bands instead of block shells: a thin-band envelope in RAM, no GPU budget
        cells=_q19_cells_estimate(case,coarse,_mesh_surface_area(case),dx,L)
        memory=Q19_BYTES_PER_SLOT*1.25*cells/1e9+0.3
    end
    result = Dict{String,Any}("cells" => cells, "memory_gb" => memory, "budget_gb" => q19 ? nothing : gpu["budget_gb"],
         "fits" => q19 ? nothing : memory <= gpu["budget_gb"],
         "basis" => q19 ? "octree surface-band envelope in RAM at $(round(Int,Q19_BYTES_PER_SLOT)) bytes per stored slot; replaced by the generator's counts" : "shape envelope; replaced by solver build-only counts",
         "domain_size_m" => [lx,ly,lz], "coarse_grid_cells" => dims,
         "gpu" => gpu,"measured"=>false)
    result["surface_geometry_allowance_gb"]=surface_memory
    if haskey(case,"domain_adjustment")
        result["domain_adjustment"] = case["domain_adjustment"]
        result["requested_domain_size_m"] = case["domain_adjustment"]["requested_size_m"]
    end
    result
end

"Keep an explicitly supplied half model above Y=0, or let native clipping cut a centered full model."
function _solver_model_triangles(case, triangles)
    boundary=case["domain"]["boundary"]
    boundary == "ground" && return _ground_solver_triangles(triangles)
    if boundary == "symmetry"
        # Rotation about the source bounding-box centre need not leave the
        # transformed bounding box centred at zero (an asymmetric tetrahedron
        # can lie entirely on one side). Stage an explicit Y placement so the
        # native min-Y test reliably selects its full-model clipping path.
        bounds=case["bounds"]
        shift=_fixed_symmetry(case["domain"]) ? Float64(get(case["domain"],"symmetry_offset_y_m",0.))+case["transform"]["translation"][2] : _symmetry_placement(case["domain"])=="center" ?
            -(Float64(bounds["min"][2])+Float64(bounds["max"][2]))/2 :
            -Float64(bounds["min"][2])
        return Triangle[ntuple(i->(tri[i][1],tri[i][2]+shift,tri[i][3]),3) for tri in triangles]
    end
    triangles
end

function _yaml_case(case, transformed_bounds; build_only::Bool)
    d = case["domain"]; f = case["flow"]; b = transformed_bounds
    lattice = get(case, "lattice", "d3q27")
    d3q39 = lattice == "d3q39"
    q19 = _q19_case(case)
    dx = case["surface_dx"]; levels = case["levels"]
    triangle_edge=Float64(get(case,"surface_triangle_max_edge_m",0.0))
    boundary = d["boundary"]
    ext = Float64[b["max"][i]-b["min"][i] for i in 1:3]
    e = boundary == "ground" ? [ext[1],ext[3],ext[2]] : ext
    mesh_chord = max(e[1], dx)
    reference=get(case,"reference",nothing)
    chord=reference===nothing || get(reference,"chord",nothing)===nothing ? mesh_chord : reference["chord"]
    reference_area=reference===nothing || get(reference,"area",nothing)===nothing ? max(e[2]*e[3],dx^2) : reference["area"]
    moment_setting=_reference_moment_setting(case,b,chord)
    mesh_reference = max(1, round(Int, mesh_chord/dx)) * dx
    chord_factor = max(mesh_reference, e[1]) / (mesh_reference * chord)
    every = min(case["steps"],get(case,"display_every",max(8,8*cld(cld(case["steps"],case["frames"]),8))))
    outputs = get(case,"outputs",OUTPUT_DEFAULTS)
    volume_every = outputs["volume_vtk"] ? every : case["steps"]+1
    symmetry = boundary == "ground" ? "\"ground\"" : boundary == "symmetry" ? "true" : "false"
    offset = case["transform"]["translation"]
    # User offsets are in source X(stream)/Y(lateral)/Z(up). Ground aligns source Z-up to solver Y-up.
    so = boundary == "ground" ? [offset[1], offset[3], -offset[2]] : offset
    lx,ly,lz = _estimate(case)["domain_size_m"]
    coarse_dx = dx * 2.0^(levels - 1)
    snapped_x = (q19 ? _q19_coarse_axis(lx,coarse_dx) : 8*_stable_grid_block_count(lx,coarse_dx,8))*coarse_dx
    snapped_y = (q19 ? _q19_coarse_axis(ly,coarse_dx) : 8*_stable_grid_block_count(ly,coarse_dx,8))*coarse_dx
    snapped_z = (q19 ? _q19_coarse_axis(lz,coarse_dx) : 8*_stable_grid_block_count(lz,coarse_dx,8))*coarse_dx
    # The app resolves physical dimensions and AMR levels. Native legacy domain
    # keys are dimensionless; its ground branch already applies height in Y and
    # lateral clearance in Z, so these physical clearances must not be swapped.
    solver_lateral = d["lateral"]
    solver_height = d["height"]
    dom = Dict("upstream" => d["upstream"], "downstream" => d["downstream"],
               "lateral" => solver_lateral, "height" => solver_height)
    placement = String[]
    relative_x=so[1]-get(d,"x_offset_m",0.)
    if case["solver"] in ("main","pond_thermal","q19_octree")
        !iszero(relative_x) && push!(placement, "    object_center_x_m: $(@sprintf("%.12g", d["upstream"] + e[1]/2 + relative_x))")
        !_fixed_symmetry(d) && !iszero(so[2]) && push!(placement, "    object_center_y_m: $(@sprintf("%.12g", (boundary == "free_air" ? snapped_y/2 : e[2]/2) + so[2]))")
        !iszero(so[3]) && push!(placement, "    object_center_z_m: $(@sprintf("%.12g", snapped_z/2 + so[3]))")
    elseif case["solver"]=="lbmtau" && !iszero(relative_x)
        # The pinned solver predates metre-based placement, but its fraction
        # option uses the same final block-snapped streamwise domain length.
        center_fraction=(d["upstream"]+e[1]/2+relative_x)/snapped_x
        push!(placement,"    object_center_x_fraction: $(@sprintf("%.17g",center_fraction))")
    end
    plane_lines = String[]
    if isempty(case["cuts"])
        append!(plane_lines, ["      - \"xz_mid\"", "      - \"xy_mid\"", "      - \"yz_mid\""])
    else
        for cut in case["cuts"]
            axis = cut["axis"]; position = @sprintf("%.12g", cut["position"])
            plane = axis == "x" ? "yz" : axis == "y" ? "xz" : "xy"
            push!(plane_lines, "      - {plane: \"$plane\", $axis: $position}")
        end
    end
    lines = String[
        "# Aero Lab case; generated from the uploaded STL. User clearances are encoded in solver reference units; min_dx is metres.",
        "# Geometry handling: $(get(case,"geometry_handling","closed_solid")). Native STL wall intersections are two-sided; faces are not duplicated.",
        "basic:", "  stl_file: [\"model.stl\"]", "  stl_scale: 1.0", "  alpha: 0.0", "  beta: 0.0",
        "  ground_up_axis: \"$(boundary == "ground" ? "y" : "z")\"",
        "  surface_resolution: $(max(1,round(Int,mesh_reference/dx)))", "  num_levels: $levels",
        "  reference_chord: $(@sprintf("%.12g", chord))", "  reference_length_for_meshing: $(@sprintf("%.12g", mesh_reference))",
        "  reference_area_of_full_model: $(@sprintf("%.12g", reference_area))",
        "  reference_dimension: \"x\"", "  minimum_facet_size: $triangle_edge", "  subdivision: $(triangle_edge>0)",
        "  fluid:", "    density: $(f["density"])", "    kinematic_viscosity: $(f["kinematic_viscosity"])",
        "    speed_of_sound: $(f["speed_of_sound"])", "  flow:", "    velocity: $(f["velocity"])",
        "  simulation:", "    steps: $(case["steps"])", "    ramp_steps: $(q19 ? something(get(get(case,"octree",Dict()),"ramp_steps",nothing),min(100,div(case["steps"],10))) : min(100,div(case["steps"],10)))",
        "    output_freq: $volume_every", "    slice_output_freq: $every",
        "    output_startup_slices: true", "    output_startup_volume: false",
        "    build_only: $build_only", "    output_dir: \"RESULTS\"",
        "    output_fields:", "      density: true", "      velocity: true", "      velocity_magnitude: true",
        "      pressure: true", "      mach: $(d3q39 || _pond_case(case))", "      temperature: $(d3q39 || _pond_case(case))",
        "      obstacle: true", "      level: true",
        # A zero min_dx bypasses the native VRAM/domain auto-sizer, which can
        # shrink clearances or drop levels. The explicit reference length and
        # surface resolution above retain exactly the requested physical dx.
        "mesh:", "  min_dx: 0", "  num_levels: $levels",
        # The app gates memory using measured active-block counts.
        "  vram_budget_gb: 1000000000", "  safety_factor: 1.15",
        "advanced:", "  solver_preset: \"vanilla\"", "  numerics:", "    lattice: \"$lattice\"",
        "    collision_operator: \"$(d3q39 ? "entropic" : q19 ? "rr" : "cumulant")\"",
        "    compressible_mode: $d3q39", "    u_lattice: $(q19 ? get(get(case,"octree",Dict()),"u_lattice",.08) : .08)", "    tau_min: $(d3q39 ? 0.505 : 0.501)",
        (_pond_case(case) ? ["    pond_thermal:","      enabled: true",
            "      cfl: $(f["cfl"])","      gamma: $(f["gamma"])",
            "      temperature_k: $(f["temperature_k"])","      gas_constant: $(f["gas_constant"])",
            "      wall_model: \"$(f["wall_model"])\"",
            "      export_surface_vtk: $(outputs["surface_vtk"])",
            "      export_surface_lbmp: $(outputs["surface_lbmp"])"] : String[])...,
        (d3q39 ? ["    eos_thermodynamic: true", "    shifted_lattice:",
                  "      auto: false", "      shift_x: $(f["velocity"] / f["speed_of_sound"] >= 1.2 ? 1 : 0)",
                  "      shift_y: 0", "      shift_z: 0"] : String[])...,
        "    hrr_sigma_per_level: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
        "  thermal:", "    enabled: $d3q39",
        (d3q39 ? ["    feedback_to_flow: true", "    gamma: 1.4", "    cv: 2.5",
                  "    wall_bc: \"adiabatic\"", "  initial_condition:", "    type: \"uniform\"",
                  "    uniform:", "      use_freestream_velocity: true", "      rho: 1.0"] : String[])...,
        "  domain:",
        (_fixed_symmetry(d) ? ["    preserve_symmetry_y: true"] : String[])...,
        "    upstream: $(@sprintf("%.12g", dom["upstream"]/mesh_reference))",
        "    downstream: $(@sprintf("%.12g", dom["downstream"]/mesh_reference))",
        "    lateral: $(@sprintf("%.12g", dom["lateral"]/mesh_reference))",
        "    height: $(@sprintf("%.12g", dom["height"]/mesh_reference))",
        "    upstream_chords: $(@sprintf("%.12g", dom["upstream"]*chord_factor))",
        "    downstream_chords: $(@sprintf("%.12g", dom["downstream"]*chord_factor))",
        "    lateral_chords: $(@sprintf("%.12g", dom["lateral"]*chord_factor))",
        "    height_chords: $(@sprintf("%.12g", dom["height"]*chord_factor))",
        "    sponge_thickness: 0.20", "    sponge_transverse_strength: 0.0",
        placement...,
        "  refinement:", "    block_size: 8", "    symmetric_analysis: $symmetry",
        # Exact triangle/block surface seeds plus the native one-block safety
        # floor hug the body. Parent/donor/2:1 closure remains unchanged.
        "    strategy: \"geometry_first\"", "    body_shell_blocks: 1",
        "  boundary:", "    method: \"bouzidi\"", "    bc_scheme: \"grad\"",
        "    q_geometry: \"stl_ray\"", "    outlet_mode: \"extrapolate\"",
        "    inlet_density: \"local\"", "    transverse_freestream: $(boundary == "free_air")",
        "  gpu:", "    async_depth: $(gcd(8,every))",
        "  diagnostics:", "    freq: $(q19 ? gcd(10,every) : every)",
        "  forces:", "    enabled: true", "    moment_center: [$(join(moment_setting, ", "))]", "    output_freq: $(q19 ? gcd(10,every) : every)",
        "  slice:", "    mode: \"production\"", "    export_data: true", "    png: $(outputs["slice_png"])",
        "    fields: [\"velocity\", \"pressure\", \"vorticity\"]",
        "    export_fields: [\"density\", \"pressure\", \"u\", \"v\", \"w\", \"velocity_magnitude\", \"vorticity\"$(_pond_case(case) ? ", \"mach\", \"temperature\"" : "")]",
        "    planes:", plane_lines..., "    export_window: \"domain\"", "    export_level: \"finest\"",
        (q19 ? _q19_yaml_lines(case) : String[])...,
    ]
    join(lines, "\n") * "\n"
end
