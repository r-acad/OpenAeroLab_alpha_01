# The case configuration: the app's `basic:` / `advanced:` YAML as apps/ludwig_web/src/aero/case.jl writes it, plus
# the `octree_q19:` block of DESIGN.md section 10. Everything the run needs is derived here once: the mesh spec,
# the placement of the STL, the lattice scaling, the boundary faces, the outputs. Hashes of the config and STL bind
# the mesh file to the case.
using YAML

Base.@kwdef struct FlowSpec
    density::Float64 = 1.225
    nu::Float64 = 1.5e-5              # kinematic viscosity, m^2/s
    sound::Float64 = 343.0            # physical speed of sound (reported only; the lattice is athermal)
    velocity::Float64 = 30.0          # inlet speed, m/s
    alpha_deg::Float64 = 0.0
    beta_deg::Float64 = 0.0
end

Base.@kwdef struct Numerics
    collision::Symbol = :rr           # :rr | :hrr
    hrr_sigma::Float64 = 0.98
    sgs::Symbol = :wale               # :wale | :smagorinsky | :vreman | :none
    c_wale::Float64 = 0.5
    c_smagorinsky::Float64 = 0.1
    u_lattice::Float64 = 0.08
    temporal::Symbol = :linear        # :linear | :quadratic
    body_force::NTuple{3,Float64} = (0.0, 0.0, 0.0)   # uniform force per unit volume, lattice units of the finest level (channel tests)
end

Base.@kwdef struct WallSpec
    scheme::Symbol = :bouzidi         # :bouzidi | :bounce_back
    wall_function::Bool = false
    law::Symbol = :reichardt          # :reichardt | :spalding | :werner_wengle
    sampling::Symbol = :normal_1      # :cell | :normal_1 | :normal_2
    pressure_gradient::Bool = false
end

struct Plane
    normal::Symbol                    # :x, :y or :z
    position_m::Float64
    tag::String
end

Base.@kwdef struct OutputSpec
    steps::Int = 1000                 # coarse steps
    ramp_steps::Int = 100
    output_every::Int = 1000          # volume output cadence, coarse steps
    slice_every::Int = 100
    startup_slices::Bool = false     # optional step-zero initialized fields, never converged results
    forces_every::Int = 10
    diagnostics_every::Int = 100
    planes::Vector{Plane} = Plane[]
    export_fields::Vector{String} = ["density", "pressure", "u", "v", "w", "velocity_magnitude", "vorticity"]
    slice_png::Bool = false
    volume_vtk::Bool = false
    surface_lbmp::Bool = true
    wall_samples::Bool = true         # local wall-cell/facet samples; static geometry + pressure frames
    surface_vtk::Bool = false
    averaging_start_ctu::Float64 = 0.0
    checkpoint_minutes::Float64 = 30.0
    checkpoint_keep::Int = 2
end

struct CaseConfig
    dir::String
    name::String
    stl_path::String
    stl_scale::Float64
    stl_sha::String
    config_sha::String
    precision::DataType
    threads::Int
    pin_threads::Bool
    flow::FlowSpec
    reference_chord::Float64
    reference_area::Float64
    reference_length::Float64
    moment_center::V3                 # chord-relative [from leading edge X, mid-span Y, mid-height Z]
    mesh::MeshSpec
    offset::V3
    numerics::Numerics
    walls::WallSpec
    sponge_thickness::Float64         # fraction of the reference length, at the outlet
    sponge_strength::Float64
    mass_control::Bool
    outputs::OutputSpec
    build_only::Bool
    raw::Dict{String,Any}
end

_get(d, keys...; default=nothing) = begin
    cur = d
    for k in keys
        (cur isa AbstractDict && haskey(cur, k)) || return default
        cur = cur[k]
    end
    cur === nothing ? default : cur
end
_sym(v, default::Symbol) = v === nothing ? default : Symbol(lowercase(String(v)))
_f(v, default) = v === nothing ? Float64(default) : Float64(v)
_i(v, default) = v === nothing ? Int(default) : Int(v)
_b(v, default) = v === nothing ? Bool(default) : (v isa Bool ? v : lowercase(String(v)) == "true")

"STL bounds without reading the whole file into triangles (the generator reads it again with the offset)."
function stl_extent(path::AbstractString, scale::Float64)
    tri, _ = read_stl(path)
    lo, hi = bounds(tri)
    vmul(lo, scale), vmul(hi, scale), length(tri)
end

"""
Domain size and STL placement as the app's YAML means them: clearances `upstream`, `downstream`, `lateral` and
`height` in units of the reference length (the app writes them as `d / mesh_reference`); the body's min x sits at
`upstream` reference lengths from the inlet; y and z are centred unless `object_center_{x,y,z}_m` are given. The
domain is then rounded up to an even number of coarse cells by the generator (the placement stays).
"""
function domain_and_offset(raw, lo::V3, hi::V3, reference_length::Float64, levels::Int, dx_finest::Float64; half_y::Bool=false,ground::Bool=false)
    up = _f(_get(raw, "advanced", "domain", "upstream"), 4.0) * reference_length
    down = _f(_get(raw, "advanced", "domain", "downstream"), 8.0) * reference_length
    lat = _f(_get(raw, "advanced", "domain", "lateral"), 4.0) * reference_length
    hgt = _f(_get(raw, "advanced", "domain", "height"), 4.0) * reference_length
    ext = vsub(hi, lo)
    preserve_y=_get(raw,"advanced","domain","preserve_symmetry_y")
    preserve_y===nothing && (preserve_y=false)
    preserve_y isa Bool || error("advanced.domain.preserve_symmetry_y must be true or false")
    preserve_y && !half_y && error("preserve_symmetry_y requires a symmetry boundary")
    preserve_y && hi[2]<=0 && error("The preserved model lies entirely outside the positive-Y symmetry domain")
    dx1 = dx_finest * 2.0^(levels - 1)
    span = _get(raw, "advanced", "domain", "span_override")
    size_x = up + ext[1] + down
    # A staged full model straddles Y=0: retain its location and mesh only Y>=0.
    centered_symmetry=half_y && lo[2]<0<hi[2]
    size_y = span !== nothing ? Float64(span) : half_y ? (preserve_y || centered_symmetry ? hi[2] : ext[2]) + lat : ground ? ext[2]+hgt : ext[2]+2lat
    size_z = ext[3] + 2*(ground ? lat : hgt)
    # round each axis up to an even number of coarse cells (the generator requires it; periodic axes must be exact)
    n = ntuple(a -> (v = (size_x, size_y, size_z)[a] / dx1; c = ceil(Int, v - 1e-9); c + c % 2), 3)
    domain = ntuple(a -> n[a] * dx1, 3)
    cx = _get(raw, "advanced", "domain", "object_center_x_m"); cy = _get(raw, "advanced", "domain", "object_center_y_m"); cz = _get(raw, "advanced", "domain", "object_center_z_m")
    half_y && cy !== nothing && error("advanced.domain.object_center_y_m cannot move a body off the y = 0 symmetry plane; omit it")
    centre = (cx === nothing ? up + ext[1] / 2 : Float64(cx),
              cy === nothing ? (preserve_y || centered_symmetry ? (lo[2]+hi[2])/2 : half_y||ground ? ext[2]/2 : domain[2]/2) : Float64(cy),
              cz === nothing ? domain[3] / 2 : Float64(cz))
    offset = vsub(centre, vmul(vadd(lo, hi), 0.5))
    domain, offset
end

"Write a deterministic typed representation, preserving array order and exact numeric values."
function _write_config_identity(io, value)
    if value isa AbstractDict
        print(io,'{')
        for key in sort!(collect(keys(value));by=string)
            print(io,repr(String(key)),':');_write_config_identity(io,value[key])
        end
        print(io,'}')
    elseif value isa AbstractVector
        print(io,'[');foreach(v->_write_config_identity(io,v),value);print(io,']')
    elseif value isa Bool
        print(io,value ? "true;" : "false;")
    elseif value isa Integer
        print(io,"number:",value,"/1;")
    elseif value isa AbstractFloat && isfinite(value)
        # 1 and 1.0 are the same setting; nearby representable floats remain
        # distinct. Exact rationals avoid rounding large integers or floats.
        rational=rationalize(BigInt,value;tol=zero(value))
        print(io,"number:",numerator(rational),'/',denominator(rational),';')
    elseif value isa AbstractString
        print(io,"string:",repr(value),';')
    elseif value===nothing
        print(io,"null;")
    else
        print(io,string(typeof(value)),':',repr(value),';')
    end
end

"Physical setup identity; formatting and continuation/output controls do not change it."
function _configuration_fingerprint(raw::AbstractDict)
    setup=deepcopy(raw)
    simulation=_get(setup,"basic","simulation")
    if simulation isa AbstractDict
        for key in ("build_only","steps","output_freq","slice_output_freq","ramp_steps")
            delete!(simulation,key)
        end
    end
    forces=_get(setup,"advanced","forces")
    forces isa AbstractDict && delete!(forces,"output_freq")
    bytes2hex(sha256(sprint(io->_write_config_identity(io,setup))))
end

"Load `config.yaml` of a case directory into a CaseConfig; the STL is read for its bounds and hash."
function load_case(dir::AbstractString; build_only::Union{Nothing,Bool}=nothing)
    path = joinpath(dir, "config.yaml")
    isfile(path) || error("no config.yaml in $dir")
    # String(::Vector{UInt8}) may take ownership and empty its input. Keep the
    # immutable text so parsing cannot erase the bytes used by the fingerprint.
    text = read(path, String)
    raw = YAML.load(text; dicttype=Dict{String,Any})
    # The app can emit 1 versus 1.0 after a JSON round trip. Fingerprint parsed
    # values so equivalent build/run inputs reuse the same mesh and checkpoint.
    config_sha = _configuration_fingerprint(raw)
    q = _get(raw, "octree_q19"; default=Dict{String,Any}())
    q isa AbstractDict || (q = Dict{String,Any}())
    stl = _get(raw, "basic", "stl_file")
    stl isa AbstractVector && (stl = first(stl))
    stl === nothing && error("basic.stl_file is required")
    stl_path = isabspath(String(stl)) ? String(stl) : joinpath(dir, String(stl))
    isfile(stl_path) || error("STL not found: $stl_path")
    stl_scale = _f(_get(raw, "basic", "stl_scale"), 1.0)
    lo, hi, ntri = stl_extent(stl_path, stl_scale)
    # mesh resolution: octree_q19.mesh first, else the app's surface_resolution against the meshing reference length
    ref_len = _f(_get(raw, "basic", "reference_length_for_meshing"), hi[1] - lo[1])
    levels = _i(_get(q, "mesh", "levels"), _i(_get(raw, "basic", "num_levels"), 4))
    dxf = _get(q, "mesh", "finest_dx_m")
    dx_finest = dxf === nothing ? ref_len / max(1, _i(_get(raw, "basic", "surface_resolution"), 64)) : Float64(dxf)
    1 <= levels <= 12 || error("levels must be in 1..12")
    dx_finest > 0 || error("finest dx must be positive")
    sym = _get(raw, "advanced", "refinement", "symmetric_analysis")
    half_y = sym === true || (sym isa AbstractString && lowercase(sym) == "true")
    ground=sym isa AbstractString && lowercase(sym)=="ground"
    domain, offset = domain_and_offset(raw, lo, hi, ref_len, levels, dx_finest; half_y=half_y,ground=ground)
    per = _get(q, "mesh", "periodic")
    periodic = per isa AbstractVector && length(per) == 3 ? ntuple(a -> Bool(per[a]), 3) :
               (false, _b(_get(raw, "advanced", "boundary", "periodic_y"), false), false)
    # boundary faces: the app's flags, overridden by octree_q19.boundaries
    transverse = _b(_get(raw, "advanced", "boundary", "transverse_freestream"), false)
    side = transverse ? :freestream : :slip
    faces = [:inlet, :outlet, side, side, side, side]
    half_y && (faces[3] = :symmetry)
    (sym isa AbstractString && lowercase(sym) == "ground") && (faces[3] = :ground)
    zw = _get(raw, "advanced", "boundary", "z_walls")
    zw !== nothing && (faces[5] = faces[6] = lowercase(String(zw)) == "noslip" ? :noslip : :slip)
    bnd = _get(q, "boundaries"; default=Dict{String,Any}())
    for (i, key) in enumerate(("inlet", "outlet", "y_min", "y_max", "z_min", "z_max"))
        v = _get(bnd, key); v === nothing || (faces[i] = Symbol(lowercase(String(v))))
    end
    periodic[2] && (faces[3] = faces[4] = :periodic); periodic[3] && (faces[5] = faces[6] = :periodic); periodic[1] && (faces[1] = faces[2] = :periodic)
    for f in faces
        f in (:inlet, :outlet, :slip, :freestream, :noslip, :symmetry, :ground, :periodic) || error("unknown boundary face condition $f")
    end
    wake = _get(q, "mesh", "wake"; default=Dict{String,Any}())
    wake_default = any(faces .== :inlet) ? 1.0 : 0.0
    regions = Region[]
    for r in (_get(q, "mesh", "regions"; default=Any[]))
        kind = _sym(_get(r, "type"), :box); lvl = _i(_get(r, "level"), levels)
        if kind == :box
            push!(regions, Region(:box, lvl; lo=Tuple(Float64.(r["min"])), hi=Tuple(Float64.(r["max"]))))
        elseif kind == :sphere
            push!(regions, Region(:sphere, lvl; centre=Tuple(Float64.(r["center"])), radius=Float64(r["radius"])))
        elseif kind == :cylinder
            push!(regions, Region(:cylinder, lvl; lo=Tuple(Float64.(r["from"])), hi=Tuple(Float64.(r["to"])), radius=Float64(r["radius"])))
        else
            error("unknown region type $kind")
        end
    end
    surface_levels = [SurfaceLevel(_i(_get(s, "level"), levels), Tuple(Float64.(s["box"]["min"])), Tuple(Float64.(s["box"]["max"]))) for s in (_get(q, "mesh", "surface_levels"; default=Any[]))]
    mesh = MeshSpec(domain=domain, periodic=periodic, levels=levels, dx_finest=dx_finest,
                    surface_level=_i(_get(q, "mesh", "surface_level"), levels),
                    band_fine=max(3, _i(_get(q, "mesh", "band_fine"), 8)), band=max(3, _i(_get(q, "mesh", "band"), 6)),
                    wake_length=_f(_get(wake, "length"), wake_default), wake_level_offset=_i(_get(wake, "level_offset"), 1), wake_margin=_f(_get(wake, "margin"), 0.1),
                    regions=regions, surface_levels=surface_levels, faces=Tuple(faces),
                    seal_cavities=_b(_get(q, "mesh", "seal_cavities"), false))
    flow = FlowSpec(density=_f(_get(raw, "basic", "fluid", "density"), 1.225), nu=_f(_get(raw, "basic", "fluid", "kinematic_viscosity"), 1.5e-5),
                    sound=_f(_get(raw, "basic", "fluid", "speed_of_sound"), 343.0), velocity=_f(_get(raw, "basic", "flow", "velocity"), 30.0),
                    alpha_deg=_f(_get(raw, "basic", "alpha"), 0.0), beta_deg=_f(_get(raw, "basic", "beta"), 0.0))
    flow.velocity > 0 || error("basic.flow.velocity must be positive")
    numerics = Numerics(collision=_sym(_get(q, "numerics", "collision"), :rr), hrr_sigma=_f(_get(q, "numerics", "hrr_sigma"), 0.98),
                        sgs=_sym(_get(q, "numerics", "sgs"), :wale), c_wale=_f(_get(q, "numerics", "c_wale"), 0.5), c_smagorinsky=_f(_get(q, "numerics", "c_smagorinsky"), 0.1),
                        u_lattice=_f(_get(q, "numerics", "u_lattice"), _f(_get(raw, "advanced", "numerics", "u_lattice"), 0.08)),
                        temporal=_sym(_get(q, "numerics", "temporal_interpolation"), :linear))
    numerics.collision in (:rr, :hrr) || error("numerics.collision must be rr or hrr")
    numerics.sgs in (:wale, :smagorinsky, :vreman, :none) || error("numerics.sgs must be wale, smagorinsky, vreman or none")
    0 < numerics.u_lattice <= 0.12 || error("numerics.u_lattice must be in (0, 0.12]")
    wf = _get(q, "walls", "wall_function"; default=Dict{String,Any}())
    walls = WallSpec(scheme=_sym(_get(q, "walls", "scheme"), :bouzidi), wall_function=_b(_get(wf, "enabled"), false), law=_sym(_get(wf, "law"), :reichardt),
                     sampling=_sym(_get(wf, "sampling"), :normal_1), pressure_gradient=_b(_get(wf, "pressure_gradient"), false))
    steps = _i(_get(raw, "basic", "simulation", "steps"), 1000)
    planes = Plane[]
    for p in (_get(raw, "advanced", "slice", "planes"; default=Any[]))
        if p isa AbstractString
            tag = String(p)
            axis = tag == "xz_mid" ? :y : tag == "xy_mid" ? :z : tag == "yz_mid" ? :x : error("unknown plane $tag")
            pos = axis == :x ? domain[1] / 2 : axis == :y ? domain[2] / 2 : domain[3] / 2
            push!(planes, Plane(axis, pos, tag))
        elseif p isa AbstractDict
            pl = lowercase(String(p["plane"]))
            axis = pl == "yz" ? :x : pl == "xz" ? :y : :z
            pos = Float64(p[String(axis)])
            push!(planes, Plane(axis, pos, "$(pl)_$(replace(@sprintf("%.6g", pos), "." => "p"))"))
        end
    end
    outputs = OutputSpec(steps=steps, ramp_steps=_i(_get(raw, "basic", "simulation", "ramp_steps"), min(100, steps ÷ 10)),
                         output_every=_i(_get(raw, "basic", "simulation", "output_freq"), steps + 1), slice_every=_i(_get(raw, "basic", "simulation", "slice_output_freq"), max(1, steps ÷ 10)),
                         startup_slices=_b(_get(raw,"basic","simulation","output_startup_slices"),false),
                         forces_every=_i(_get(raw, "advanced", "forces", "output_freq"), 10), diagnostics_every=_i(_get(raw, "advanced", "diagnostics", "freq"), 100),
                         planes=planes, export_fields=String.(_get(raw, "advanced", "slice", "export_fields"; default=["density", "pressure", "u", "v", "w", "velocity_magnitude", "vorticity"])),
                         slice_png=_b(_get(raw, "advanced", "slice", "png"), false), volume_vtk=_i(_get(raw, "basic", "simulation", "output_freq"), steps + 1) <= steps,
                         surface_lbmp=_b(_get(q,"outputs","surface_lbmp"),true), wall_samples=_b(_get(q,"outputs","wall_samples"),true),
                         surface_vtk=_b(_get(q,"outputs","surface_vtk"),false),
                         averaging_start_ctu=_f(_get(q, "outputs", "averaging_start_ctu"), 0.0),
                         checkpoint_minutes=_f(_get(q, "checkpoint", "every_minutes"), 30.0), checkpoint_keep=_i(_get(q, "checkpoint", "keep"), 2))
    mc = _get(raw, "advanced", "forces", "moment_center")
    moment = mc isa AbstractVector && length(mc) == 3 ? Tuple(Float64.(mc)) : (0.25, 0.0, 0.0)
    precision = lowercase(String(something(_get(q, "precision"), "f32"))) == "f64" ? Float64 : Float32
    bo = build_only === nothing ? _b(_get(raw, "basic", "simulation", "build_only"), false) : build_only
    CaseConfig(String(dir), basename(normpath(dir)), stl_path, stl_scale, bytes2hex(sha256(read(stl_path))), config_sha,
               precision, _i(_get(q, "threads"), 0), _b(_get(q, "pin_threads"), false), flow,
               _f(_get(raw, "basic", "reference_chord"), ref_len), _f(_get(raw, "basic", "reference_area_of_full_model"), (hi[2] - lo[2]) * (hi[3] - lo[3])), ref_len, moment,
               mesh, offset, numerics, walls,
               _f(_get(raw, "advanced", "domain", "sponge_thickness"), 0.2), _f(_get(q, "sponge", "strength"), 0.05), _b(_get(q, "mass_control"), false),
               outputs, bo, raw)
end

"""
A configuration for tests and synthetic cases: the mesh spec, a lattice viscosity of the finest level (physical
values are derived so that `lattice_scaling` returns it), numerics and walls.
"""
function test_case_config(spec::MeshSpec; nu_lattice::Float64=0.01, u_lattice::Float64=0.08, velocity::Float64=1.0, collision::Symbol=:rr,
                          sgs::Symbol=:none, walls::WallSpec=WallSpec(), steps::Int=100, ramp_steps::Int=0, precision::DataType=Float32, name::String="test",
                          body_force::NTuple{3,Float64}=(0.0, 0.0, 0.0))
    nu_phys = nu_lattice * velocity * spec.dx_finest / u_lattice
    scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT"))),"TEMP_DELETEABLE","q19_tests");mkpath(scratch)
    CaseConfig(mktempdir(scratch), name, "", 1.0, "", "", precision, 0, false,
               FlowSpec(nu=nu_phys, velocity=velocity), spec.domain[1], spec.domain[2] * spec.domain[3], spec.domain[1], (0.0, 0.0, 0.0),
               spec, (0.0, 0.0, 0.0), Numerics(collision=collision, sgs=sgs, u_lattice=u_lattice, body_force=body_force), walls, 0.0, 0.0, false,
               OutputSpec(steps=steps, ramp_steps=ramp_steps), false, Dict{String,Any}())
end

"Lattice scaling of the finest level: dt, velocity scale, viscosity and tau per level."
struct LatticeScaling
    u_lattice::Float64
    velocity_scale::Float64           # m/s per lattice unit of velocity
    dt_finest::Float64                # s
    dx_finest::Float64
    nu_lattice::Vector{Float64}       # per level, lattice units of that level
    tau::Vector{Float64}
    reynolds::Float64
    mach_lattice::Float64
end
function lattice_scaling(cfg::CaseConfig)
    u = cfg.numerics.u_lattice; U = cfg.flow.velocity
    vs = U / u; dxf = cfg.mesh.dx_finest; dtf = dxf / vs
    L = cfg.mesh.levels
    nul = [cfg.flow.nu * (dtf * 2.0^(L - l)) / (dxf * 2.0^(L - l))^2 for l in 1:L]
    tau = [3nu + 0.5 for nu in nul]
    LatticeScaling(u, vs, dtf, dxf, nul, tau, U * cfg.reference_chord / cfg.flow.nu, u * sqrt(3))
end
