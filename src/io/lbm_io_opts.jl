# ==============================================================================
# lbm_io_opts.jl — opt-in input/output options of the LBM driver
# ==============================================================================
# lbm-io-opts (27 Oct 2026), for the web app's "LBM 3D (LUDWIG)" backend. Every key
# is optional; with none of them set the driver behaves exactly as before (the
# legacy branches in main.jl run unchanged). Reference, file format and tests:
# memory/project_docs/_records/2026-10-27_lbm_io_opts/README.md.
#
#   advanced.slice.export_data   Bool (false)  numeric plane files (.lbmslice) at every slice step
#   advanced.slice.planes        list          planes for PNGs and numeric files: "xz_mid", "xy_mid",
#                                              "yz_mid", or {plane: "xz", y: <m>} / {plane: "yz", x: <m>} /
#                                              {plane: "xy", z: <m>} (domain frame, metres)
#   advanced.slice.fields        list          PNG fields (renderer names, e.g. "velocity", "vorticity")
#   advanced.slice.png           Bool (true)   false: no slice PNG at all (startup, per step, final)
#   advanced.slice.export_fields list          numeric fields (default: all seven, see LBMIO_ALL_EXPORT_FIELDS)
#   advanced.slice.export_window "domain" | "finest" | [a1_min, a1_max, a2_min, a2_max] (m)
#   advanced.slice.export_level  "finest" | level number: the export spacing is that level's dx
#   advanced.slice.export_max_cells Int (4194304): above it the spacing doubles (next coarser level)
#   basic.simulation.build_only  Bool (false)  build the domain, write a summary, stop before step 1
#
# The case selector (LUDWIG_CASES_FILE / LUDWIG_CASE, --cases-file / --case) lives in
# case_id_contract.jl because it must run before any dependency is loaded.
# ==============================================================================

isdefined(@__MODULE__, :LBMSliceFormat) || include(joinpath(@__DIR__, "lbm_slice_format.jl"))
using .LBMSliceFormat
import TOML

# ── lbm-io-opts globals: ONE block; the defaults reproduce the legacy behaviour ──
const LBMIO_ALL_EXPORT_FIELDS = ("density", "pressure", "u", "v", "w", "velocity_magnitude", "vorticity")
const LBMIO_PNG_FIELD_NAMES = (:mach, :density, :velocity, :vorticity, :temperature, :schlieren,
                               :schlieren_calibrated, :pressure, :pressure_rel, :cp, :pi_neq,
                               :velocity_highpass, :velocity_highpass_calibrated, :local_shift,
                               :level, :sample_mode, :amr_edge, :wall_dist, :schlieren_support,
                               :amr_rho_jump, :amr_u_jump, :amr_x_u_jump, :amr_y_ux_jump)
global LBMIO_EXPORT_DATA = false
global LBMIO_SLICE_PLANES = nothing     # nothing ⇒ the legacy three PNG planes; else Vector{LBMIOPlane}
global LBMIO_SLICE_FIELDS = nothing     # nothing ⇒ the legacy field set; else Vector{Symbol}
global LBMIO_SLICE_PNG = true
global LBMIO_SLICE_CUSTOM = false       # planes, fields or png=false given ⇒ main.jl takes the new PNG branch
global LBMIO_EXPORT_FIELDS = collect(LBMIO_ALL_EXPORT_FIELDS)
global LBMIO_EXPORT_WINDOW = :domain    # :domain | :finest | NTuple{4,Float64}
global LBMIO_EXPORT_LEVEL = 0           # 0 = the finest level
global LBMIO_EXPORT_MAX_CELLS = 4_194_304
global LBMIO_BUILD_ONLY = false
# ── end of the lbm-io-opts globals ──

"""One slice plane: `tag` names the numeric file, `png_suffix` the PNG files (legacy names
for the legacy planes), `png_kwargs` the overrides passed to export_slice_image."""
struct LBMIOPlane
    tag::String
    plane::Symbol                       # :xz, :xy, :yz, or :default (legacy: :xy in ground mode, else :xz)
    coord::Union{Nothing,Float64}       # explicit normal coordinate (domain frame, m); nothing = through the STL centre
    png_suffix::String
    label::String
    png_kwargs::NamedTuple
end

_lbmio_fmt_coord(c::Real) = replace(@sprintf("%g", Float64(c)), "." => "p", "-" => "m", "+" => "")

"""The three planes the driver renders when `advanced.slice.planes` is absent (legacy calls)."""
lbmio_legacy_planes() = LBMIOPlane[
    LBMIOPlane("xz_mid", :default, nothing, "", "Slice", (;)),
    LBMIOPlane("xz_y0p25", :default, 0.25, "_Y0p25", "SliceY025", (y_override=0.25,)),
    LBMIOPlane("yz_mid", :yz, nothing, "_YZ", "SliceYZ", (plane_override=:yz,)),
]

function _lbmio_parse_plane(raw, i::Int)
    src = "advanced.slice.planes[$i]"
    if raw isa AbstractString
        s = lowercase(strip(String(raw)))
        s == "xz_mid" && return LBMIOPlane("xz_mid", :xz, nothing, "", "Slice", (plane_override=:xz,))
        s == "xy_mid" && return LBMIOPlane("xy_mid", :xy, nothing, "_XY", "SliceXY", (plane_override=:xy,))
        s == "yz_mid" && return LBMIOPlane("yz_mid", :yz, nothing, "_YZ", "SliceYZ", (plane_override=:yz,))
        error("$src = $(repr(raw)): valid strings are \"xz_mid\", \"xy_mid\", \"yz_mid\"; " *
              "an explicit plane is a mapping such as {plane: \"xz\", y: 0.25}")
    elseif raw isa AbstractDict
        d = Dict(lowercase(string(k)) => v for (k, v) in raw)
        haskey(d, "plane") || error("$src must name `plane` (\"xz\", \"xy\" or \"yz\")")
        p = lowercase(string(d["plane"]))
        normal = p == "xz" ? "y" : p == "yz" ? "x" : p == "xy" ? "z" :
                 error("$src plane = $(repr(d["plane"])) must be \"xz\", \"xy\" or \"yz\"")
        extra = setdiff(keys(d), Set(["plane", normal]))
        isempty(extra) || error("$src: unknown key(s) $(join(sort(collect(extra)), ", ")) (a $p plane takes `$normal`)")
        haskey(d, normal) || error("$src: a $p plane needs its normal coordinate `$normal` (metres, domain frame)")
        c = d[normal]
        (c isa Real && !(c isa Bool) && isfinite(c)) || error("$src.$normal must be a finite number")
        c = Float64(c)
        f = _lbmio_fmt_coord(c)
        if p == "xz"
            return LBMIOPlane("xz_y$f", :xz, c, "_Y$f", "SliceY$f", (plane_override=:xz, y_override=c, normal_clamp=:domain))
        elseif p == "yz"
            return LBMIOPlane("yz_x$f", :yz, c, "_YZ_X$f", "SliceYZX$f", (plane_override=:yz, x_override=c, normal_clamp=:domain))
        else
            return LBMIOPlane("xy_z$f", :xy, c, "_XY_Z$f", "SliceXYZ$f", (plane_override=:xy, z_override=c, normal_clamp=:domain))
        end
    end
    error("$src must be a string or a mapping (got $(typeof(raw)))")
end

function _lbmio_bool(cfg, keys...)
    config_has_path(cfg, keys...) || return nothing
    v = safe_get(cfg, keys...)
    v isa Bool || error("$(join(keys, ".")) must be true or false (got $(repr(v)))")
    return v
end

function _lbmio_list(cfg, keys...)
    config_has_path(cfg, keys...) || return nothing
    v = safe_get(cfg, keys...)
    v isa AbstractVector || error("$(join(keys, ".")) must be a list (got $(repr(v)))")
    isempty(v) && error("$(join(keys, ".")) must not be empty")
    return v
end

"""Parse the lbm-io-opts keys of one case config into the LBMIO_* globals (called by
config_loader.jl for every case; resets every global first, so cases do not leak)."""
function lbmio_load_options!(cfg)
    global LBMIO_EXPORT_DATA = false
    global LBMIO_SLICE_PLANES = nothing
    global LBMIO_SLICE_FIELDS = nothing
    global LBMIO_SLICE_PNG = true
    global LBMIO_EXPORT_FIELDS = collect(LBMIO_ALL_EXPORT_FIELDS)
    global LBMIO_EXPORT_WINDOW = :domain
    global LBMIO_EXPORT_LEVEL = 0
    global LBMIO_EXPORT_MAX_CELLS = 4_194_304
    global LBMIO_BUILD_ONLY = false
    empty!(_LBMIO_GEOM_CACHE)   # plane geometry is per case (objectids are reused after GC)

    v = _lbmio_bool(cfg, "advanced", "slice", "export_data")
    if v !== nothing; global LBMIO_EXPORT_DATA = v; end
    v = _lbmio_bool(cfg, "advanced", "slice", "png")
    if v !== nothing; global LBMIO_SLICE_PNG = v; end
    v = _lbmio_bool(cfg, "basic", "simulation", "build_only")
    if v !== nothing; global LBMIO_BUILD_ONLY = v; end

    raw = _lbmio_list(cfg, "advanced", "slice", "planes")
    if raw !== nothing
        planes = LBMIOPlane[_lbmio_parse_plane(r, i) for (i, r) in enumerate(raw)]
        tags = [p.tag for p in planes]
        length(unique(tags)) == length(tags) || error("advanced.slice.planes names a plane twice: $(join(tags, ", "))")
        global LBMIO_SLICE_PLANES = planes
    end
    raw = _lbmio_list(cfg, "advanced", "slice", "fields")
    if raw !== nothing
        fields = Symbol[]
        for (i, r) in enumerate(raw)
            r isa AbstractString || error("advanced.slice.fields[$i] must be a string")
            s = Symbol(lowercase(strip(String(r))))
            s in LBMIO_PNG_FIELD_NAMES ||
                error("advanced.slice.fields[$i] = $(repr(r)) is not a slice PNG field; valid: $(join(string.(LBMIO_PNG_FIELD_NAMES), ", "))")
            s in fields || push!(fields, s)
        end
        global LBMIO_SLICE_FIELDS = fields
    end
    raw = _lbmio_list(cfg, "advanced", "slice", "export_fields")
    if raw !== nothing
        fields = String[]
        for (i, r) in enumerate(raw)
            r isa AbstractString || error("advanced.slice.export_fields[$i] must be a string")
            s = lowercase(strip(String(r)))
            s in LBMIO_ALL_EXPORT_FIELDS ||
                error("advanced.slice.export_fields[$i] = $(repr(r)) is unknown; valid: $(join(LBMIO_ALL_EXPORT_FIELDS, ", "))")
            s in fields || push!(fields, s)
        end
        global LBMIO_EXPORT_FIELDS = [f for f in LBMIO_ALL_EXPORT_FIELDS if f in fields]   # canonical order
    end
    if config_has_path(cfg, "advanced", "slice", "export_window")
        w = safe_get(cfg, "advanced", "slice", "export_window")
        if w isa AbstractString && lowercase(w) in ("domain", "finest")
            global LBMIO_EXPORT_WINDOW = Symbol(lowercase(w))
        elseif w isa AbstractVector && length(w) == 4 && all(x -> x isa Real && !(x isa Bool) && isfinite(x), w)
            a = Float64.(w)
            (a[1] < a[2] && a[3] < a[4]) || error("advanced.slice.export_window [a1_min, a1_max, a2_min, a2_max] must be increasing")
            global LBMIO_EXPORT_WINDOW = (a[1], a[2], a[3], a[4])
        else
            error("advanced.slice.export_window must be \"domain\", \"finest\" or [a1_min, a1_max, a2_min, a2_max] in metres (got $(repr(w)))")
        end
    end
    if config_has_path(cfg, "advanced", "slice", "export_level")
        l = safe_get(cfg, "advanced", "slice", "export_level")
        if l isa AbstractString && lowercase(l) == "finest"
            global LBMIO_EXPORT_LEVEL = 0
        elseif l isa Integer && !(l isa Bool) && l >= 1
            global LBMIO_EXPORT_LEVEL = Int(l)
        else
            error("advanced.slice.export_level must be \"finest\" or a level number >= 1 (got $(repr(l)))")
        end
    end
    if config_has_path(cfg, "advanced", "slice", "export_max_cells")
        m = safe_get(cfg, "advanced", "slice", "export_max_cells")
        (m isa Integer && !(m isa Bool) && m >= 1) || error("advanced.slice.export_max_cells must be a positive integer")
        global LBMIO_EXPORT_MAX_CELLS = Int(m)
    end
    global LBMIO_SLICE_CUSTOM = LBMIO_SLICE_PLANES !== nothing || LBMIO_SLICE_FIELDS !== nothing || !LBMIO_SLICE_PNG

    any_set = LBMIO_EXPORT_DATA || LBMIO_SLICE_CUSTOM || LBMIO_BUILD_ONLY ||
              any(k -> config_has_path(cfg, "advanced", "slice", k),
                  ("export_fields", "export_window", "export_level", "export_max_cells"))
    if any_set
        println("[Init]    lbm-io-opts: export_data=$LBMIO_EXPORT_DATA png=$LBMIO_SLICE_PNG build_only=$LBMIO_BUILD_ONLY")
        println("[Init]       planes: ", LBMIO_SLICE_PLANES === nothing ? "legacy (xz_mid, xz_y0p25, yz_mid)" :
                                          join([p.tag for p in LBMIO_SLICE_PLANES], ", "))
        println("[Init]       PNG fields: ", LBMIO_SLICE_FIELDS === nothing ? "legacy ($(SLICE_FIELD_SET))" :
                                              join(string.(LBMIO_SLICE_FIELDS), ", "))
        LBMIO_EXPORT_DATA && println("[Init]       numeric fields: ", join(LBMIO_EXPORT_FIELDS, ", "),
                                     " | window=", LBMIO_EXPORT_WINDOW, " | level=",
                                     LBMIO_EXPORT_LEVEL == 0 ? "finest" : LBMIO_EXPORT_LEVEL)
    end
    return nothing
end

lbmio_png_planes() = LBMIO_SLICE_PLANES === nothing ? lbmio_legacy_planes() : LBMIO_SLICE_PLANES
lbmio_export_planes_list() = LBMIO_SLICE_PLANES === nothing ?
    LBMIOPlane[LBMIOPlane("xz_mid", :xz, nothing, "", "Slice", (plane_override=:xz,))] : LBMIO_SLICE_PLANES

# ──────────────────────────────────────────────────────────────────────────────
# PNG slices with plane / field selection (used only when LBMIO_SLICE_CUSTOM)
# ──────────────────────────────────────────────────────────────────────────────

"""Per-step PNGs: each selected plane, the selected fields (or `default_field`), each on a
background task through main.jl's spawn_slice_export! (as the legacy calls)."""
function lbmio_spawn_png_slices!(step::Int, grids, slices_dir::String, backend, default_field::Symbol;
                                 cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing)
    LBMIO_SLICE_PNG || return nothing
    for p in lbmio_png_planes()
        kw = p.png_kwargs
        spawn_slice_export!(p.label, step, () -> begin
            export_slice_image(step, grids, slices_dir, backend; field=default_field,
                               fields=LBMIO_SLICE_FIELDS,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                               fname_suffix=p.png_suffix, kw...)
        end)
    end
    return nothing
end

"""Synchronous PNGs (startup geometric set, final slice): one export stage per plane."""
function lbmio_sync_png_slices!(stage_prefix::String, step::Int, grids, slices_dir::String, backend,
                                default_field::Symbol, fields;
                                cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing)
    LBMIO_SLICE_PNG || return nothing
    for p in lbmio_png_planes()
        kw = p.png_kwargs
        run_export_stage_sync(stage_prefix * p.label, step, () -> begin
            lock(_SLICE_RENDER_LOCK) do
                export_slice_image(step, grids, slices_dir, backend; field=default_field, fields=fields,
                                   cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                   fname_suffix=p.png_suffix, kw...)
            end
        end)
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# Numeric plane export
# ──────────────────────────────────────────────────────────────────────────────
# Combination rule (multi-level): the plane is sampled on a uniform grid of cell
# centres with spacing h = dx of the export level (default: the finest). Each node
# takes the value of the FINEST level, among the levels whose dx >= h, that has an
# active block containing the node's point; the value is that level's own cell value
# at the cell containing the point (injection: no interpolation, no averaging). The
# plane's normal coordinate is a finest-level cell centre (the same plane the PNG of
# the same spec shows), so it is strictly inside one cell of every level. Single
# level: the nodes are exactly the level's cells. The `level` array records the
# level used (0 = no level covers the node: values NaN).

_lbmio_axes(plane::Symbol) = plane == :xz ? (1, 3, 2) : plane == :xy ? (1, 2, 3) :
                             plane == :yz ? (2, 3, 1) : error("bad plane $plane")
const _LBMIO_AXIS_NAME = ("x", "y", "z")

mutable struct LBMIOPlaneGeom
    key::UInt64
    tag::String
    plane::Symbol
    meta::Any
    n1::Int
    n2::Int
    h::Float64
    spacing_level::Int
    spacing_requested_level::Int
    origin::NTuple{2,Float64}
    window_mode::String
    node_level::Matrix{UInt8}
    node_lin::Matrix{Int}
    obstacle::Matrix{UInt8}
    nb_lin::NTuple{4,Matrix{Int}}      # owner-level neighbours: a1-, a1+, a2-, a2+ (0 = absent or solid)
    levels_used::Vector{Int}
end

const _LBMIO_GEOM_CACHE = Dict{String,LBMIOPlaneGeom}()

"""Derivative of the velocity component at offset `c` (0, s or 2s) at cell `lin` from its
neighbours `m` (minus side) and `p` (plus side), 0 = absent: central when both exist, else
one-sided, else NaN. The same Float32 expressions as the VTK export's block-local stencil."""
@inline function _lbmio_deriv(vel, c::Int, lin::Int, m::Int, p::Int, inv_dx::Float32, inv_2dx::Float32)
    @inbounds begin
        (m != 0 && p != 0) && return (vel[p + c] - vel[m + c]) * inv_2dx
        p != 0 && return (vel[p + c] - vel[lin + c]) * inv_dx
        m != 0 && return (vel[lin + c] - vel[m + c]) * inv_dx
    end
    return NaN32
end

function _lbmio_grids_key(grids)
    h = hash(objectid(grids))
    for g in grids
        h = hash((length(g.active_block_coords), g.grid_dim_x, g.grid_dim_y, g.grid_dim_z, g.dx), h)
    end
    return h
end

"""CPU copies of what the plane geometry needs (block pointer, obstacle) for every level."""
function _lbmio_level_info(grids)
    return [(grid_dim_x=g.grid_dim_x, grid_dim_y=g.grid_dim_y, grid_dim_z=g.grid_dim_z, dx=Float64(g.dx),
             block_pointer=Array(g.block_pointer), obs=Array(g.obstacle),
             nblocks=length(g.active_block_coords), coords=g.active_block_coords) for g in grids]
end

@inline function _lbmio_cell_lin(info, gx::Int, gy::Int, gz::Int)
    bs = BLOCK_SIZE
    (gx < 1 || gy < 1 || gz < 1 ||
     gx > info.grid_dim_x * bs || gy > info.grid_dim_y * bs || gz > info.grid_dim_z * bs) && return 0
    b = Int(info.block_pointer[(gx - 1) ÷ bs + 1, (gy - 1) ÷ bs + 1, (gz - 1) ÷ bs + 1])
    b == 0 && return 0
    return ((gx - 1) % bs + 1) + bs * ((gy - 1) % bs) + bs * bs * ((gz - 1) % bs) + bs * bs * bs * (b - 1)
end

@inline function _lbmio_global(plane::Symbol, g1::Int, g2::Int, gn::Int)
    plane == :xz && return (g1, gn, g2)
    plane == :xy && return (g1, g2, gn)
    return (gn, g1, g2)                      # :yz
end

function _lbmio_build_geom(p::LBMIOPlane, grids, info)
    nlev = length(grids)
    bs = BLOCK_SIZE
    plane = p.plane == :default ? (GROUND_ANALYSIS ? :xy : :xz) : p.plane
    a1, a2, an = _lbmio_axes(plane)
    # the plane: the same selection (and obstacle-seeking nudge) as the PNG of this spec
    level_data = [(grid_dim_x=i.grid_dim_x, grid_dim_y=i.grid_dim_y, grid_dim_z=i.grid_dim_z,
                   dx=i.dx, block_pointer=i.block_pointer, obs=i.obs) for i in info]
    valid = [Set{Tuple{Int,Int,Int}}(i.coords) for i in info]
    meta = _pick_slice_plane(level_data, valid;
                             plane_override=plane,
                             y_override=plane == :xz ? p.coord : nothing,
                             x_override=plane == :yz ? p.coord : nothing,
                             z_override=plane == :xy ? p.coord : nothing,
                             normal_clamp=p.coord === nothing ? :body : :domain)
    cn = meta.coord
    dims(i) = (i.grid_dim_x, i.grid_dim_y, i.grid_dim_z)
    coarse = info[1]
    ext = (dims(coarse)[a1] * bs * coarse.dx, dims(coarse)[a2] * bs * coarse.dx)
    req_level = LBMIO_EXPORT_LEVEL == 0 ? nlev : min(LBMIO_EXPORT_LEVEL, nlev)
    L = req_level
    local lo1, hi1, lo2, hi2, n1, n2, h, mode
    while true
        h = info[L].dx
        if LBMIO_EXPORT_WINDOW === :finest
            fine = info[nlev]
            gnf = floor(Int, cn / fine.dx) + 1
            bn = (gnf - 1) ÷ bs + 1
            l1 = Inf; u1 = -Inf; l2 = Inf; u2 = -Inf
            for c in fine.coords
                c[an] == bn || continue
                l1 = min(l1, (c[a1] - 1) * bs * fine.dx); u1 = max(u1, c[a1] * bs * fine.dx)
                l2 = min(l2, (c[a2] - 1) * bs * fine.dx); u2 = max(u2, c[a2] * bs * fine.dx)
            end
            if isfinite(l1)
                w = (l1, u1, l2, u2); mode = "finest"
            else
                println("[SliceData] export_window=finest: the finest level does not cross plane $(p.tag); using the domain")
                w = (0.0, ext[1], 0.0, ext[2]); mode = "domain"
            end
        elseif LBMIO_EXPORT_WINDOW isa Tuple
            w = LBMIO_EXPORT_WINDOW; mode = "explicit"
        else
            w = (0.0, ext[1], 0.0, ext[2]); mode = "domain"
        end
        lo1 = max(0.0, floor(w[1] / h + 1e-9) * h); hi1 = min(ext[1], ceil(w[2] / h - 1e-9) * h)
        lo2 = max(0.0, floor(w[3] / h + 1e-9) * h); hi2 = min(ext[2], ceil(w[4] / h - 1e-9) * h)
        n1 = round(Int, (hi1 - lo1) / h); n2 = round(Int, (hi2 - lo2) / h)
        (n1 >= 1 && n2 >= 1) || error("advanced.slice.export_window does not overlap the domain for plane $(p.tag)")
        n1 * n2 <= LBMIO_EXPORT_MAX_CELLS && break
        L == 1 && error("plane $(p.tag): $(n1)x$(n2) nodes exceed export_max_cells=$(LBMIO_EXPORT_MAX_CELLS) even at level 1")
        println("[SliceData] plane $(p.tag): $(n1)x$(n2) nodes at level $L exceed export_max_cells=$(LBMIO_EXPORT_MAX_CELLS); using level $(L - 1)")
        L -= 1
    end
    node_level = zeros(UInt8, n1, n2)
    node_lin = zeros(Int, n1, n2)
    obstacle = zeros(UInt8, n1, n2)
    nbs = ntuple(_ -> zeros(Int, n1, n2), 4)
    used = falses(nlev)
    offs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    # Periodic Y (lbm-app-int, 27 Oct 2026, merge review F1): on a periodic run (periodic_y_amr on every level, or the
    # single-level periodic_y) the owner level's four vorticity-neighbour lookups wrap the Y index across the seam, so
    # the first and last Y row of a level get the same central difference as the interior on xy/yz planes. grid_dim_y
    # spans the full period on every level there (BlockLevel periodic_grid_by; a single level is the whole domain).
    # Non-periodic runs (wrap_y false) and xz planes (Y is the normal, always in range) are unchanged.
    wrap_y = periodic_y_amr_active() || (SPAN_PERIODIC_Y && nlev == 1)
    @inbounds for k in 1:n2, i in 1:n1
        c1 = lo1 + (i - 0.5) * h
        c2 = lo2 + (k - 0.5) * h
        for l in L:-1:1
            inf = info[l]
            g1 = floor(Int, c1 / inf.dx) + 1
            g2 = floor(Int, c2 / inf.dx) + 1
            gn = floor(Int, cn / inf.dx) + 1
            lin = _lbmio_cell_lin(inf, _lbmio_global(plane, g1, g2, gn)...)
            lin == 0 && continue
            node_level[i, k] = UInt8(l)
            node_lin[i, k] = lin
            obstacle[i, k] = inf.obs[lin] ? 0x01 : 0x00
            used[l] = true
            for (j, (d1, d2)) in enumerate(offs)
                gxn, gyn, gzn = _lbmio_global(plane, g1 + d1, g2 + d2, gn)
                wrap_y && (gyn = mod1(gyn, inf.grid_dim_y * bs))
                ln = _lbmio_cell_lin(inf, gxn, gyn, gzn)
                nbs[j][i, k] = (ln != 0 && !inf.obs[ln]) ? ln : 0
            end
            break
        end
    end
    return LBMIOPlaneGeom(_lbmio_grids_key(grids), p.tag, plane, meta, n1, n2, h, L, req_level,
                          (lo1, lo2), mode, node_level, node_lin, obstacle, nbs, findall(used))
end

function _lbmio_geom(p::LBMIOPlane, grids)
    key = _lbmio_grids_key(grids)
    g = get(_LBMIO_GEOM_CACHE, p.tag, nothing)
    if g === nothing || g.key != key
        g = _lbmio_build_geom(p, grids, _lbmio_level_info(grids))
        _LBMIO_GEOM_CACHE[p.tag] = g
    end
    return g
end

"""
    lbmio_export_planes!(label_step, state_step, grids, slices_dir; with_fields=true)

Write one `.lbmslice` file per selected plane to `slices_dir/step_<label>/plane_<tag>_<label>.lbmslice`.
Reads the solver arrays synchronously (call it on the main thread, after the batch whose state
it records). `state_step` is the step the arrays hold (the label can be the slice cadence step
of a batch that ended later; both are written to the header). `with_fields=false` writes only
`level` and `obstacle` (build-only mode).
"""
function lbmio_export_planes!(label_step::Int, state_step::Int, grids, slices_dir::String;
                              with_fields::Bool=true)
    params = get_domain_params()
    nlev = length(grids)
    out_dir = joinpath(slices_dir, @sprintf("step_%06d", label_step))
    isdir(out_dir) || mkpath(out_dir)
    planes = lbmio_export_planes_list()
    geoms = [_lbmio_geom(p, grids) for p in planes]
    # one host copy per level actually sampled by some plane
    needed = sort(unique(vcat([g.levels_used for g in geoms]...)))
    host = Dict{Int,Any}()
    want_p = with_fields && "pressure" in LBMIO_EXPORT_FIELDS
    if with_fields
        for l in needed
            level = grids[l]
            rho = Array(level.rho)
            vel = (l == 1 && iseven(state_step)) ? Array(level.vel_temp) : Array(level.vel)   # as the VTK export
            temp = (want_p && DDF_ENABLED && has_thermal(level)) ? Array(level.temperature) : nothing
            sponge = (want_p && EOS_THERMODYNAMIC && LATTICE_TYPE == :d3q39 &&   # the VTK fallback for non-finite T
                      hasproperty(level, :sponge)) ? Array(level.sponge) : nothing
            host[l] = (rho=rho, vel=vel, temp=temp, sponge=sponge, stride=length(rho), dx=Float32(level.dx))
        end
    end
    vscale = Float32(params.velocity_scale)
    pcoeff = Float32(diagnostic_pressure_coeff())
    u_ref_lat = abs(Float64(U_INLET_CURRENT[])) > 0.0 ? abs(Float64(U_INLET_CURRENT[])) : abs(Float64(U_TARGET))
    t_inf = (DDF_ENABLED || EOS_THERMODYNAMIC) ? Float32(DDF_T_INLET) : 1.0f0
    p_ref = Float64(Float32(IC_UNIFORM_RHO) * pcoeff * t_inf)
    q_ref = Float64(0.5f0 * Float32(IC_UNIFORM_RHO) * Float32(U_TARGET) * Float32(U_TARGET))
    written = String[]
    for (p, g) in zip(planes, geoms)
        a1, a2, an = _lbmio_axes(g.plane)
        n1, n2 = g.n1, g.n2
        arrays = Any[("level", g.node_level, "1", "level whose cell supplied the node (0 = none)"),
                     ("obstacle", g.obstacle, "1", "1 = solid cell on that level")]
        if with_fields
            rho_m = fill(NaN32, n1, n2); p_m = fill(NaN32, n1, n2)
            u_m = ntuple(_ -> fill(NaN32, n1, n2), 3)
            um_m = fill(NaN32, n1, n2); w_m = fill(NaN32, n1, n2)
            @inbounds for k in 1:n2, i in 1:n1
                l = Int(g.node_level[i, k]); l == 0 && continue
                hd = host[l]; lin = g.node_lin[i, k]; s = hd.stride
                r = hd.rho[lin]
                vx = hd.vel[lin]; vy = hd.vel[lin + s]; vz = hd.vel[lin + 2s]
                rho_m[i, k] = r
                u_m[1][i, k] = vx * vscale; u_m[2][i, k] = vy * vscale; u_m[3][i, k] = vz * vscale
                um_m[i, k] = sqrt(vx * vx + vy * vy + vz * vz) * vscale
                if want_p
                    stored_T = hd.temp !== nothing ? hd.temp[lin] : NaN32
                    sp = hd.sponge !== nothing ? hd.sponge[lin] : 0.0f0
                    t_val = isfinite(stored_T) ? stored_T :
                            Float32(diagnostic_effective_temperature(r, vx, vy, vz, NaN, sp, u_ref_lat))
                    p_m[i, k] = r * pcoeff * t_val
                end
                # slice-normal vorticity on the owner level's own grid (same stencil arithmetic as the VTK)
                g.obstacle[i, k] == 0x01 && continue
                inv_dx = 1.0f0 / hd.dx
                inv_2dx = 0.5f0 * inv_dx
                ca1 = (a1 - 1) * s; ca2 = (a2 - 1) * s
                m1 = g.nb_lin[1][i, k]; p1 = g.nb_lin[2][i, k]
                m2 = g.nb_lin[3][i, k]; p2 = g.nb_lin[4][i, k]
                du1_d2 = _lbmio_deriv(hd.vel, ca1, lin, m2, p2, inv_dx, inv_2dx)   # d(u_a1)/d(a2)
                du2_d1 = _lbmio_deriv(hd.vel, ca2, lin, m1, p1, inv_dx, inv_2dx)   # d(u_a2)/d(a1)
                wv = g.plane == :xz ? (du1_d2 - du2_d1) : (du2_d1 - du1_d2)
                w_m[i, k] = wv * vscale
            end
            fieldmap = Dict("density" => (rho_m, "1", "density rho / rho_ref (lattice)"),
                            "pressure" => (p_m, "lattice", "pressure as the VTK export: rho * pressure_coeff * T (lattice units)"),
                            "u" => (u_m[1], "m/s", "x velocity"),
                            "v" => (u_m[2], "m/s", "y velocity"),
                            "w" => (u_m[3], "m/s", "z velocity"),
                            "velocity_magnitude" => (um_m, "m/s", "|u|"),
                            "vorticity" => (w_m, "1/s", "slice-normal vorticity omega_$(_LBMIO_AXIS_NAME[an]), central differences on the owner level (one-sided next to solid cells and missing blocks; NaN in solid cells)"))
            for f in LBMIO_EXPORT_FIELDS
                a, units, desc = fieldmap[f]
                name = f == "vorticity" ? "vorticity_" * _LBMIO_AXIS_NAME[an] : f
                push!(arrays, (name, a, units, desc))
            end
        end
        meta = g.meta
        header = Dict{String,Any}(
            "case" => basename(CASE_DIR),
            "solver" => "LUDWIG LBM (src/main.jl)",
            "lattice" => string(LATTICE_TYPE),
            "collision_operator" => string(COLLISION_OPERATOR),
            "plane" => string(g.plane),
            "plane_tag" => g.tag,
            "axis1" => _LBMIO_AXIS_NAME[a1], "axis2" => _LBMIO_AXIS_NAME[a2],
            "normal_axis" => _LBMIO_AXIS_NAME[an],
            "normal_coord_m" => Float64(meta.coord),
            "normal_target_m" => Float64(meta.target_coord),
            "normal_index_finest" => meta.slice_index,
            "plane_moved_to_cross_obstacle" => meta.used_fallback,
            "plane_crosses_obstacle" => meta.intersects_geometry,
            "frame" => "domain",
            "frame_note" => "domain coordinates (the VTK frame): a point of the alpha/beta-rotated STL is at its STL position + mesh_offset_m",
            "mesh_offset_m" => collect(Float64.(params.mesh_offset)),
            "domain_size_m" => collect(Float64.(params.domain_size)),
            "origin_m" => [g.origin[1], g.origin[2]],
            "spacing_m" => g.h,
            "spacing_level" => g.spacing_level,
            "spacing_requested_level" => g.spacing_requested_level,
            "window_mode" => g.window_mode,
            "window_m" => [g.origin[1], g.origin[1] + n1 * g.h, g.origin[2], g.origin[2] + n2 * g.h],
            "node_center_rule" => "centre of node (i, k) = (origin_m[1] + (i - 0.5) * spacing_m, origin_m[2] + (k - 0.5) * spacing_m), i = 1..n1 along axis1, k = 1..n2 along axis2",
            "layout" => "column-major, element (i, k) at linear index i + (k - 1) * n1; little-endian",
            "combine_rule" => "finest_active_level_injection",
            "combine_note" => "each node takes the cell value of the finest level (with dx >= spacing_m) whose active block contains the node point; no interpolation or averaging; level 0 = no level, values NaN",
            "num_levels" => nlev,
            "level_dx_m" => [Float64(gr.dx) for gr in grids],
            "coarse_grid_cells" => [grids[1].grid_dim_x * BLOCK_SIZE, grids[1].grid_dim_y * BLOCK_SIZE,
                                    grids[1].grid_dim_z * BLOCK_SIZE],
            "levels_used" => g.levels_used,
            "step" => label_step,
            "state_step" => state_step,
            # lbm-taufix VA3 (W7): a solver step is one COARSEST step: time_s = state_step * time_scale_s * 2^(L-1).
            # With one level the header is unchanged (x 1.0), so single-level files stay byte-identical.
            "time_s" => Float64(state_step) * solver_step_time(params),
            "time_scale_s" => params.time_scale,
            "time_note" => nlev == 1 ? "time_s = state_step * time_scale_s, the convention of csv/convergence.csv" :
                "time_s = state_step * time_per_step_s; time_per_step_s = time_scale_s * 2^(num_levels - 1) (one solver step is one coarsest-level step; time_scale_s is the finest level's step), the convention of csv/convergence.csv",
            "u_inlet_lattice" => Float64(U_INLET_CURRENT[]),
            "u_target_lattice" => Float64(U_TARGET),
            "velocity_scale_m_s" => Float64(vscale),
            "velocity_note" => "u, v, w, velocity_magnitude = Float32(lattice velocity) * Float32(velocity_scale_m_s)",
            "u_ref_m_s" => params.u_physical,
            "rho_ref_kg_m3" => params.rho_physical,
            "reference_chord_m" => params.reference_chord,
            "pressure_ref_lattice" => p_ref,
            "dynamic_pressure_ref_lattice" => q_ref,
            "pressure_note" => "Cp = (pressure - pressure_ref_lattice) / dynamic_pressure_ref_lattice (the slice PNG convention)",
            "with_fields" => with_fields,
        )
        nlev > 1 && (header["time_per_step_s"] = solver_step_time(params))   # lbm-taufix VA3 (W7), multi-level only
        fname = joinpath(out_dir, @sprintf("plane_%s_%06d.lbmslice", g.tag, label_step))
        write_lbmslice(fname, header, arrays)
        push!(written, fname)
        @printf("[SliceData] %s: %s (%dx%d nodes, h=%.6g m, levels %s)\n", g.tag, fname, n1, n2, g.h,
                join(string.(g.levels_used), ","))
    end
    return written
end

# ──────────────────────────────────────────────────────────────────────────────
# Build-only mode
# ──────────────────────────────────────────────────────────────────────────────

"""
    lbmio_build_only_case(runtime_selection, output_claim, sim_start_datetime)

`basic.simulation.build_only: true`: load the STL, voxelise, build the blocks, levels and
Bouzidi q-maps (setup_multilevel_domain, on the CPU), write the AMR viewer data, the
mid-plane obstacle/level file(s) and `build_only_summary.toml`, then return a completed
case outcome with steps = 0. No backend is selected and no GPU call is made here.
"""
function lbmio_build_only_case(runtime_selection, output_claim, sim_start_datetime)
    t0 = time()
    output_dir = output_claim.target.path
    println("[BuildOnly] basic.simulation.build_only = true: domain build only; no GPU, no time step")
    flush(stdout)
    write_provenance(output_dir, basename(CASE_DIR), CFG;
                     extra=Dict("lattice_type" => string(LATTICE_TYPE),
                                "collision_operator" => string(COLLISION_OPERATOR),
                                "build_only" => true,
                                "sim_start" => string(sim_start_datetime)))
    t_dom = time()
    cpu_grids, geometry_mesh, _, _, _ =
        setup_multilevel_domain(runtime_selection, output_claim; num_levels=NUM_LEVELS_CONFIG)
    build_s = time() - t_dom
    @printf("[BuildOnly] domain build done in %.2f s\n", build_s)
    flush(stdout)
    params = get_domain_params()
    try
        export_amr_viz(cpu_grids, params.mesh_offset, output_dir; mesh=geometry_mesh, case_name=basename(CASE_DIR))
    catch e
        @printf("[AMR Viz] Warning: export failed: %s\n", e)
    end
    steps = STEPS
    if DOMAIN_LENGTH_COVERED > 0.0          # the same formula as main.jl
        t_needed = DOMAIN_LENGTH_COVERED * params.domain_size[1] / params.u_physical
        steps = Int(ceil(Int(ceil(t_needed / solver_step_time(params))) / DIAG_FREQ)) * DIAG_FREQ   # lbm-taufix VA3 (W7)
    end
    slices_dir = joinpath(output_dir, "SLICES")
    plane_files = lbmio_export_planes!(0, 0, cpu_grids, slices_dir; with_fields=false)
    levels = Dict{String,Any}[]
    total_cells = 0; total_bytes = 0; total_blocks = 0
    for (l, g) in enumerate(cpu_grids)
        nb = length(g.active_block_coords)
        cells = nb * BLOCK_SIZE^3
        nobs = count(Array(g.obstacle))
        bytes = Base.summarysize(g)
        total_cells += cells; total_bytes += bytes; total_blocks += nb
        push!(levels, Dict{String,Any}("level" => l, "dx_m" => Float64(g.dx), "tau" => Float64(g.tau),
                                       "grid_blocks" => [g.grid_dim_x, g.grid_dim_y, g.grid_dim_z],
                                       "active_blocks" => nb, "cells" => cells,
                                       "obstacle_cells" => nobs, "fluid_cells" => cells - nobs,
                                       "bouzidi_boundary_cells" => g.n_boundary_cells,
                                       "bytes_allocated" => bytes))
        @printf("[BuildOnly] L%d: dx=%.6g m, %d blocks, %d cells (%d solid, %d Bouzidi), %.1f MB allocated\n",
                l, g.dx, nb, cells, nobs, g.n_boundary_cells, bytes / 1e6)
    end
    summary = Dict{String,Any}(
        "case" => basename(CASE_DIR),
        "build_only" => true,
        "written_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z",
        "domain_build_seconds" => build_s,
        "lattice" => string(LATTICE_TYPE), "collision_operator" => string(COLLISION_OPERATOR),
        "num_levels" => length(cpu_grids),
        "coarse_grid_cells" => [params.nx_coarse, params.ny_coarse, params.nz_coarse],
        "domain_size_m" => collect(Float64.(params.domain_size)),
        "mesh_offset_m" => collect(Float64.(params.mesh_offset)),
        "reynolds" => params.re_number, "u_physical_m_s" => params.u_physical,
        "u_lattice_target" => Float64(U_TARGET), "time_scale_s" => params.time_scale,
        "time_per_step_s" => solver_step_time(params),   # lbm-taufix VA3 (W7): one solver step = one coarsest step
        "velocity_scale_m_s" => params.velocity_scale,
        "steps_configured" => steps,
        "total_blocks" => total_blocks, "total_cells" => total_cells,
        "bytes_allocated_total" => total_bytes,
        "memory_note" => "bytes_allocated = Base.summarysize of the built CPU grids (all arrays of each BlockLevel); the GPU grid upload copies these arrays; CUDA context, kernels, lattice tables and slice/force scratch come on top",
        "levels" => levels,
        "plane_files" => [replace(relpath(f, output_dir), "\\" => "/") for f in plane_files],
        "geometry_triangles" => length(geometry_mesh.triangles),
    )
    open(joinpath(output_dir, "build_only_summary.toml"), "w") do io
        TOML.print(io, summary; sorted=true)
    end
    @printf("[BuildOnly] %d level(s), %d blocks, %d cells, %.1f MB allocated; summary: %s\n",
            length(cpu_grids), total_blocks, total_cells, total_bytes / 1e6,
            joinpath(output_dir, "build_only_summary.toml"))
    flush(stdout)
    end_dt = Dates.now()
    return BatchOutcomeContract.lbm_case_outcome((
        case_name=basename(CASE_DIR), status=:completed, error_msg="",
        re_number=params.re_number, u_physical=params.u_physical,
        alpha_deg=Float64(ALPHA_DEG), beta_deg=Float64(BETA_DEG),
        total_cells=total_cells, wall_time=time() - t0, mlups=0.0, steps=0,
        start_time=sim_start_datetime, end_time=max(end_dt, sim_start_datetime),
        Cd=0.0, Cl=0.0, Cs=0.0, Cdp=0.0, Cdv=0.0, Cmx=0.0, Cmy=0.0, Cmz=0.0,
        avg_Cd=0.0, avg_Cl=0.0, avg_Cs=0.0, avg_Cdp=0.0, avg_Cdv=0.0, avg_Cmx=0.0, avg_Cmy=0.0, avg_Cmz=0.0,
        n_samples=0, min_Cd=0.0, max_Cd=0.0, min_Cl=0.0, max_Cl=0.0, min_Cs=0.0, max_Cs=0.0,
        force_divergence_flag=false, force_peak_Cd=0.0, force_peak_Cl=0.0, force_peak_Cs=0.0,
        force_ratio_Cd=0.0, force_ratio_Cl=0.0, force_ratio_Cs=0.0,
        collision_operator=COLLISION_OPERATOR, ma_physical=params.u_physical / 343.0,
        compressibility_correction=COMPRESSIBILITY_CORRECTION, output_dir=abspath(OUT_DIR)))
end
