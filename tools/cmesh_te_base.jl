# cmesh_te_base.jl -- the trailing-edge base of a blunt channel C-mesh, shared by the driver and the wall instruments.
#
# Blunt trailing edge plan (memory/project_docs/2026-10-01_BLUNT_TRAILING_EDGE_PLAN.md), section 5.4: every tool that walks the foil
# range takes the base faces from the receipt through ONE helper and reports them separately, so the surface stations stay
# monotone in x and no instrument re-derives the base on its own.
#
# Topology (the midpoint split, plan section 2). The foil range holds NIF = n_foil + 2 n_base wall faces. Foil-relative
# face k (absolute cell i = NIW + k, k = 1..NIF) lies between foil-relative nodes k and k+1 (absolute nodes NIW+k and
# NIW+k+1). The wake cut meets the foil at the base midpoint m, which is foil-relative node 1 and node NIF+1:
#   faces 1 .. n_base                      lower half-base, from m to the lower corner (node n_base+1)
#   faces n_base+1 .. n_base+n_foil        the surface, lower corner -> leading edge -> upper corner (node n_base+n_foil+1)
#   faces n_base+n_foil+1 .. NIF           upper half-base, from the upper corner back to m
# With n_base = 0 the base is empty and the surface is 1:NIF, so every caller's ranges are today's, bit for bit.
#
# Strip topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md sections 6 and 9, `te_topology=strip`): the foil range
# is the open surface alone, NIF = n_foil (lower corner = node 1, upper corner = node NIF+1), and the base is the strip
# block's column 1, 2 n_base faces from the lower corner to the upper. `te_base_partition(NIF, n_base; strip=true)` then
# returns surface = 1:NIF with the strip's faces APPENDED (lower_base = NIF+1 : NIF+n_base beside the lower corner,
# upper_base = NIF+n_base+1 : NIF+2 n_base), so partition_sums and base_mean work on a concatenated per-face vector
# [foil faces; strip column-1 faces]. The receipt's settings carry te_topology (parsed by name) and its readings the strip's
# construction and measures (`te_base_realised(...; strip=block)`).
#
# The receipt line (printed by the driver when a blunt-edge mesh knob is set) has two parts separated by " | ":
#   settings: the generator keywords that differ from their defaults, as key=value, written so that they parse back to the
#             identical value (Int %d, Float64 %.17g, Symbol by name). wall_mesh splats them into generate_cmesh, so the
#             regenerated mesh reproduces the coordinate receipt.
#   realised: measured quantities for the reader (gap, corners, spacings); never passed to the generator.
# Only the keys in TE_BASE_SETTING_KEYS are accepted as settings; anything else is refused.
#
# Gauge pressure. Any per-part force must use p - p_ref: the surface alone is not a closed loop on a blunt mesh (its face
# vectors sum to minus the base's), so an absolute-pressure "base drag" is meaningless (about -0.039 in C_D units at
# sigma 6 on the 0.006 c base, against a physical 0.0005-0.0007).
module CMeshTEBase
using Printf
export TE_BASE_PREFIX, TE_BASE_SETTING_KEYS, te_base_partition, surface_view, surface_kle, te_base_receipt_line,
       parse_te_base_receipt, te_base_receipt_from_lines, partition_sums, base_mean, same_settings,
       te_base_restart_guard, te_base_realised

const TE_BASE_PREFIX = "REALISED TRAILING-EDGE BASE: "

# The generator keywords a receipt may carry (src/cmesh/cmesh_gen.jl, generate_cmesh). Extend together with the
# generator; an unknown key is refused so that nothing unreviewed is ever splatted into generate_cmesh. te_topology (5 Oct
# 2026) is a Symbol, aplus or strip, parsed by name.
const TE_BASE_SETTING_KEYS = (:n_base, :te_ds, :wake_near_ds, :wake_near_len, :wake_near_growth, :te_topology)
const TE_TOPOLOGIES = (:aplus, :strip)

"""
    te_base_partition(NIF, n_base; strip=false)

Foil-relative ranges of the midpoint-split foil: `lower_base`, `surface`, `upper_base` (face indices 1..NIF), the corner
nodes `lower_corner = n_base+1` and `upper_corner = NIF-n_base+1`, and `n_foil = NIF - 2 n_base`. `base` lists both
half-bases in index order. With n_base = 0, `surface == 1:NIF` and both half-bases are empty.

With `strip = true` (the strip topology) the foil range is the open surface, `surface = 1:NIF`, `n_foil = NIF`, the corners
are nodes 1 and NIF+1, and the base faces are the strip's column-1 faces appended after the surface: `lower_base =
NIF+1 : NIF+n_base` (the rows beside the lower corner, j_s = 1..n_base) and `upper_base = NIF+n_base+1 : NIF+2 n_base`, so
that `partition_sums` and `base_mean` work on a concatenated per-face vector [foil faces; strip column-1 faces]. The tuple
carries `strip` for readers.
"""
function te_base_partition(NIF::Integer, n_base::Integer; strip::Bool=false)
    n_base >= 0 || throw(ArgumentError("n_base must be >= 0 (got $n_base)"))
    if strip
        n_base >= 2 || throw(ArgumentError("the strip topology needs n_base >= 2 (got $n_base)"))
        NIF >= 4 || throw(ArgumentError("foil range of $NIF faces is too short for a surface"))
        lower_base = NIF+1:NIF+n_base
        surface = 1:NIF
        upper_base = NIF+n_base+1:NIF+2n_base
        return (; n_base=Int(n_base), n_foil=Int(NIF), lower_base, surface, upper_base,
                  base=vcat(collect(lower_base), collect(upper_base)),
                  lower_corner=1, upper_corner=Int(NIF+1), strip=true)
    end
    n_foil = NIF - 2n_base
    n_foil >= 4 || throw(ArgumentError("foil range of $NIF faces cannot hold two half-bases of $n_base faces"))
    lower_base = 1:n_base
    surface = n_base+1:n_base+n_foil
    upper_base = n_base+n_foil+1:NIF
    (; n_base=Int(n_base), n_foil=Int(n_foil), lower_base, surface, upper_base,
       base=vcat(collect(lower_base), collect(upper_base)),
       lower_corner=Int(n_base+1), upper_corner=Int(n_base+n_foil+1), strip=false)
end

"""
    surface_view(NIW, NIF, n_base; strip=false)

The (NIW, NIF) pair that makes a foil-range walker see the surface only: `(NIW + n_base, NIF - 2 n_base)`. A
foil-relative index ii of the full range becomes ii - n_base in this view. Identity at n_base = 0, and with the strip
topology (whose foil range is the surface already).
"""
surface_view(NIW::Integer, NIF::Integer, n_base::Integer; strip::Bool=false) =
    strip ? (Int(NIW), Int(NIF)) : (Int(NIW + n_base), Int(te_base_partition(NIF, n_base).n_foil))

"""
    surface_kle(X, NIW, NIF, n_base; strip=false)

Foil-relative node index of the leading edge: the minimum-x node among the surface nodes n_base+1 .. n_base+n_foil+1
(row j = 1 of the node array X). Faces kle:n_base+n_foil are the upper surface and n_base+1:kle-1 the lower one. At
n_base = 0 it is the all-node minimum the instruments use today (first minimum, as argmin); with the strip topology the
surface nodes are 1 .. NIF+1.
"""
function surface_kle(X::AbstractMatrix, NIW::Integer, NIF::Integer, n_base::Integer; strip::Bool=false)
    p = te_base_partition(NIF, n_base; strip)
    nodes = NIW + p.lower_corner : NIW + p.upper_corner
    p.lower_corner - 1 + argmin(view(X, nodes, 1))
end

_fmt(v::Integer) = @sprintf("%d", v)
_fmt(v::AbstractFloat) = @sprintf("%.17g", Float64(v))
_fmt(v::Symbol) = String(v)
_fmt(v::Tuple) = join(map(_fmt, v), ",")
_fmt(v::AbstractString) = String(v)
# readings: seven significant digits (te_base_realised rounds to seven; %.7g prints that decimal exactly)
_fmt_r(v::Integer) = @sprintf("%d", v)
_fmt_r(v::AbstractFloat) = @sprintf("%.7g", Float64(v))
_fmt_r(v::Tuple) = join(map(_fmt_r, v), ",")
_fmt_r(v::AbstractString) = String(v)
_fmt_r(v::Symbol) = String(v)

"""
    te_base_receipt_line(settings::NamedTuple, realised::NamedTuple) -> String

The driver's receipt line. `settings` are generator keywords (subset of TE_BASE_SETTING_KEYS, in the order given);
`realised` are measured values (numbers or tuples of numbers), printed for the reader only.
"""
function te_base_receipt_line(settings::NamedTuple, realised::NamedTuple)
    for k in keys(settings)
        k in TE_BASE_SETTING_KEYS || throw(ArgumentError("unknown trailing-edge base setting $k"))
    end
    haskey(settings, :n_base) || throw(ArgumentError("the trailing-edge base receipt must carry n_base"))
    haskey(settings, :te_topology) && !(settings.te_topology isa Symbol && settings.te_topology in TE_TOPOLOGIES) &&
        throw(ArgumentError("te_topology must be :aplus or :strip (got $(settings.te_topology))"))
    s = join((string(k, "=", _fmt(v)) for (k, v) in pairs(settings)), " ")
    r = join((string(k, "=", _fmt_r(v)) for (k, v) in pairs(realised)), " ")
    for (k, v) in pairs(realised)
        occursin(r"[\s|=]", _fmt_r(v)) && throw(ArgumentError("realised value of $k is not a single token"))
        k in TE_BASE_SETTING_KEYS && throw(ArgumentError("realised field $k shadows a setting"))
    end
    TE_BASE_PREFIX * s * " | " * r
end

function _parse_setting(k::Symbol, v::AbstractString)
    k === :n_base && return parse(Int, v)
    if k === :te_topology                               # by name, never a number
        t = Symbol(v)
        t in TE_TOPOLOGIES || throw(ArgumentError("unknown trailing-edge topology $v (te_topology must be aplus or strip)"))
        return t
    end
    x = parse(Float64, v)
    isfinite(x) || throw(ArgumentError("non-finite trailing-edge base setting $k=$v"))
    x
end

function _parse_realised(v::AbstractString)
    parts = split(v, ',')
    vals = map(p -> something(tryparse(Float64, p), String(p)), parts)
    length(vals) == 1 ? vals[1] : Tuple(vals)
end

"""
    parse_te_base_receipt(line) -> (settings=NamedTuple, realised=NamedTuple)

Inverse of `te_base_receipt_line`. Refuses a malformed line, a duplicated key, a setting outside TE_BASE_SETTING_KEYS
and a receipt without n_base.
"""
function parse_te_base_receipt(line::AbstractString)
    line = lstrip(line)                                # the driver indents its receipt lines
    startswith(line, TE_BASE_PREFIX) || throw(ArgumentError("not a trailing-edge base receipt"))
    body = line[length(TE_BASE_PREFIX)+1:end]
    halves = split(body, " | "; limit=2)
    length(halves) == 2 || throw(ArgumentError("malformed trailing-edge base receipt (no ' | ' separator)"))
    function fields(txt)
        pieces = split.(split(strip(txt)), '='; limit=2)
        all(p -> length(p) == 2 && !isempty(p[1]) && !isempty(p[2]), pieces) ||
            throw(ArgumentError("malformed trailing-edge base receipt field"))
        ks = Symbol.(first.(pieces))
        allunique(ks) || throw(ArgumentError("duplicated trailing-edge base receipt field"))
        ks, last.(pieces)
    end
    sk, sv = fields(halves[1]); rk, rv = fields(halves[2])
    for k in sk
        k in TE_BASE_SETTING_KEYS || throw(ArgumentError("unknown trailing-edge base setting $k"))
    end
    :n_base in sk || throw(ArgumentError("the trailing-edge base receipt must carry n_base"))
    isempty(intersect(sk, rk)) || throw(ArgumentError("a trailing-edge base field is both a setting and a reading"))
    settings = NamedTuple{Tuple(sk)}(Tuple(_parse_setting(k, v) for (k, v) in zip(sk, sv)))
    settings.n_base >= 0 || throw(ArgumentError("n_base must be >= 0"))
    realised = NamedTuple{Tuple(rk)}(Tuple(_parse_realised(v) for v in rv))
    (; settings, realised)
end

"""
    te_base_receipt_from_lines(lines) -> nothing | (settings, realised)

The receipt of a run log: `nothing` when the run printed none (a default mesh), the parsed receipt when it printed
exactly one, and an error when it printed more than one.
"""
function te_base_receipt_from_lines(lines)
    found = filter(l -> startswith(lstrip(l), TE_BASE_PREFIX), lines)
    isempty(found) && return nothing
    length(found) == 1 || throw(ArgumentError("trailing-edge base receipt is duplicated"))
    parse_te_base_receipt(only(found))
end

"""
    te_base_realised(X, Y, NIW, NIF, n_base; strip=nothing)

Measured trailing-edge quantities of a mesh (node arrays, row j = 1), for the receipt's reading half: the corner-to-corner
gap, the corner coordinates, the shortest and longest base face, the surface faces at the two corners, the first wake
interval beside the base midpoint (or the sharp trailing edge), the longest wake interval within ten gaps of it (NaN for
a sharp edge), and the wake column count. Seven significant digits: these are readings, not settings.

With `strip` the mesh's strip block (any object with the node arrays `X, Y` and the cell counts `NI, NJ`; the generator's
`StripBlock`), the strip topology's readings (plan section 4.7), every one measured on the two blocks: `strip_ns`,
`strip_niw`, `strip_min_sine` (the strip's smallest corner sine, production's corner order), `strip_skew_max` (degrees:
90 minus the smallest corner angle), `base_ds_min`/`base_ds_max` (the strip's column-1 faces, the base), `corner_ds_lower`/
`corner_ds_upper` (the surface faces at the corners), `wake_ds0` (the upper leg's first interval), `wake_ds_max_10gap`,
`leg_row1_min`/`leg_row1_max` (the first j-edge of the legs' columns, the corners included), `leg_ramp_stations` (the
leg columns whose first-row height is below the strip's edge interval gap/(2 n_base) by more than 1e-6 relative, the
larger of the two legs' counts), `leg_row1_ratio = leg_row1_min / (gap/(2 n_base))` and `base_tilt` (degrees between
the strip's first i-edge at the base's middle node and the base normal: the strip cells' skew at the base).
"""
function te_base_realised(X::AbstractMatrix, Y::AbstractMatrix, NIW::Integer, NIF::Integer, n_base::Integer; strip=nothing)
    strip === nothing || return _te_strip_realised(X, Y, NIW, NIF, n_base, strip)
    p = te_base_partition(NIF, n_base)
    node(k) = (Float64(X[NIW+k, 1]), Float64(Y[NIW+k, 1]))                    # foil-relative node k
    flen(k) = hypot((node(k+1) .- node(k))...)                                 # foil-relative face k
    r7(v) = round(v; sigdigits=7)
    lc = node(p.lower_corner); uc = node(p.upper_corner)
    gap = hypot((uc .- lc)...)
    bl = [flen(k) for k in p.base]
    m = node(NIF + 1)                                                           # the base midpoint (sharp: the edge)
    wnode(k) = (Float64(X[NIW+NIF+1+k, 1]), Float64(Y[NIW+NIF+1+k, 1]))         # upper wake leg, k = 0 is m
    nwake = size(X, 1) - (NIW + NIF + 1)
    wake_ds0 = hypot((wnode(1) .- wnode(0))...)
    wmax = NaN
    if gap > 0
        wmax = 0.0
        for k in 1:nwake
            hypot((wnode(k-1) .- m)...) <= 10gap || break
            wmax = max(wmax, hypot((wnode(k) .- wnode(k-1))...))
        end
    end
    # Base tilt (DECISION.md 4.6): the angle in degrees between the first j-edge at the middle node of each base half and
    # the base normal (perpendicular to the corner-to-corner chord). NaN without a base.
    function tilt(k)
        n_base > 0 || return NaN
        e = (Float64(X[NIW+k, 2]) - Float64(X[NIW+k, 1]), Float64(Y[NIW+k, 2]) - Float64(Y[NIW+k, 1]))
        t = (uc[1] - lc[1], uc[2] - lc[2])
        c = abs(e[1]*t[1] + e[2]*t[2])/(hypot(e...)*hypot(t...))       # |cos| of the angle to the base itself
        90.0 - acosd(clamp(c, 0.0, 1.0))
    end
    (; gap=r7(gap), lower_corner=r7.(lc), upper_corner=r7.(uc),
       base_ds_min=isempty(bl) ? NaN : r7(minimum(bl)), base_ds_max=isempty(bl) ? NaN : r7(maximum(bl)),
       corner_ds_lower=r7(flen(p.lower_corner)), corner_ds_upper=r7(flen(p.upper_corner - 1)),
       wake_ds0=r7(wake_ds0), wake_ds_max_10gap=isnan(wmax) ? NaN : r7(wmax), n_wake=Int(NIW),
       base_tilt_lower=r7(tilt(1 + n_base ÷ 2)), base_tilt_upper=r7(tilt(NIF + 1 - n_base ÷ 2)))
end

"Smallest corner sine over the cells of a node array (CMeshGen.corner_sine_min's arithmetic, production's corner order)."
function _corner_sine_min(X, Y, NI::Integer, NJ::Integer)
    smin = Inf
    for j in 1:NJ, i in 1:NI
        p = ((Float64(X[i,j]), Float64(Y[i,j])), (Float64(X[i+1,j]), Float64(Y[i+1,j])),
             (Float64(X[i+1,j+1]), Float64(Y[i+1,j+1])), (Float64(X[i,j+1]), Float64(Y[i,j+1])))
        for c in 1:4
            a = p[c]; nx = p[mod1(c+1, 4)]; pv = p[mod1(c-1, 4)]
            e1 = (nx[1]-a[1], nx[2]-a[2]); e2 = (pv[1]-a[1], pv[2]-a[2])
            den = hypot(e1...)*hypot(e2...)
            s = den > 0 ? (e1[1]*e2[2] - e1[2]*e2[1])/den : 0.0
            s < smin && (smin = s)
        end
    end
    smin
end

# The strip topology's readings (te_base_realised's docstring lists them).
function _te_strip_realised(X::AbstractMatrix, Y::AbstractMatrix, NIW::Integer, NIF::Integer, n_base::Integer, s)
    NS = 2n_base
    te_base_partition(NIF, n_base; strip=true)                                   # the strip's own refusals (n_base >= 2)
    (s.NI == NIW && s.NJ == NS && size(s.X) == (NIW + 1, NS + 1) && size(s.Y) == size(s.X)) ||
        throw(ArgumentError("strip block $(size(s.X)) with NI=$(s.NI) NJ=$(s.NJ) does not match n_wake=$NIW and 2 n_base=$NS"))
    node(k) = (Float64(X[NIW+k, 1]), Float64(Y[NIW+k, 1]))                    # foil-relative node k (1 = CL, NIF+1 = CU)
    flen(k) = hypot((node(k+1) .- node(k))...)                                 # foil-relative face k
    r7(v) = round(v; sigdigits=7)
    lc = node(1); uc = node(NIF + 1)
    gap = hypot((uc .- lc)...)
    snode(k, j) = (Float64(s.X[k, j]), Float64(s.Y[k, j]))
    bl = [hypot((snode(1, j+1) .- snode(1, j))...) for j in 1:NS]               # the strip's column-1 faces: the base
    wnode(k) = (Float64(X[NIW+NIF+1+k, 1]), Float64(Y[NIW+NIF+1+k, 1]))         # upper leg, k = 0 is the upper corner
    nwake = size(X, 1) - (NIW + NIF + 1)
    wake_ds0 = hypot((wnode(1) .- wnode(0))...)
    wmax = 0.0
    for k in 1:nwake
        hypot((wnode(k-1) .- uc)...) <= 10gap || break
        wmax = max(wmax, hypot((wnode(k) .- wnode(k-1))...))
    end
    jedge(i) = hypot(Float64(X[i, 2]) - Float64(X[i, 1]), Float64(Y[i, 2]) - Float64(Y[i, 1]))   # the column's first interval
    lower_cols = 1:NIW+1; upper_cols = NIW+NIF+1:size(X, 1)
    legs = vcat([jedge(i) for i in lower_cols], [jedge(i) for i in upper_cols])
    edge = gap/NS                                                               # the strip's edge cross interval
    ramp = max(count(i -> jedge(i) < edge*(1 - 1e-6), lower_cols), count(i -> jedge(i) < edge*(1 - 1e-6), upper_cols))
    smin = _corner_sine_min(s.X, s.Y, s.NI, s.NJ)
    mid = n_base + 1                                                            # the base's middle node (row j_s = n_base+1)
    e = (Float64(s.X[2, mid]) - Float64(s.X[1, mid]), Float64(s.Y[2, mid]) - Float64(s.Y[1, mid]))
    t = (uc[1] - lc[1], uc[2] - lc[2])
    c = abs(e[1]*t[1] + e[2]*t[2])/(hypot(e...)*hypot(t...))                   # |cos| of the angle to the base itself
    tilt = 90.0 - acosd(clamp(c, 0.0, 1.0))
    (; gap=r7(gap), lower_corner=r7.(lc), upper_corner=r7.(uc), strip_ns=Int(NS), strip_niw=Int(NIW),
       strip_min_sine=r7(smin), strip_skew_max=r7(acosd(clamp(smin, 0.0, 1.0))),
       base_ds_min=r7(minimum(bl)), base_ds_max=r7(maximum(bl)),
       corner_ds_lower=r7(flen(1)), corner_ds_upper=r7(flen(NIF)),
       wake_ds0=r7(wake_ds0), wake_ds_max_10gap=r7(wmax), n_wake=Int(NIW),
       leg_row1_min=r7(minimum(legs)), leg_row1_max=r7(maximum(legs)), leg_ramp_stations=Int(ramp),
       leg_row1_ratio=r7(minimum(legs)/edge), base_tilt=r7(tilt))
end

"""
    same_settings(a, b)

True when two settings NamedTuples carry the same keys (in any order) with `isequal` values and identical types.
"""
same_settings(a::NamedTuple, b::NamedTuple) =
    Set(keys(a)) == Set(keys(b)) && all(k -> isequal(a[k], b[k]) && typeof(a[k]) === typeof(b[k]), keys(a))
same_settings(a, b) = false

"""
    te_base_restart_guard(metadata, active)

Bind a restart to the trailing-edge mesh settings, as `fixed_outer_restart_guard` binds the fixed-outer recipe. With no
active settings (`nothing`: a default mesh) the checkpoint metadata must not carry a `te_base` entry; with active
settings it must carry exactly them. The solver's own configuration and geometry checks still run; this guard names the
cause. Returns nothing on success.
"""
function te_base_restart_guard(metadata, active)
    marked = hasproperty(metadata, :te_base) ||
        (metadata isa AbstractDict && (haskey(metadata, :te_base) || haskey(metadata, "te_base")))
    if active === nothing
        marked && throw(ArgumentError("trailing-edge base checkpoint requires the matching mesh settings"))
        return nothing
    end
    active isa NamedTuple && haskey(active, :n_base) || throw(ArgumentError("malformed active trailing-edge settings"))
    marked && metadata isa NamedTuple ||
        throw(ArgumentError("checkpoint lacks the trailing-edge base settings of this run"))
    same_settings(metadata.te_base, active) ||
        throw(ArgumentError("checkpoint trailing-edge base settings differ from this run's: " *
                            "$(metadata.te_base) vs $active"))
    nothing
end

"""
    partition_sums(values, part) -> (lower_base, surface, upper_base, base, total)

Sums of a per-face vector over the parts of `te_base_partition` (values indexed by foil-relative face 1..NIF). The base
is summed as lower + upper; the total is the plain sum over all faces, so `surface + base == total` holds to round-off,
and exactly when the summation order agrees (the caller tests it with its own tolerance).
"""
function partition_sums(values::AbstractVector, part::NamedTuple)
    length(values) == length(part.lower_base) + part.n_foil + length(part.upper_base) ||
        throw(ArgumentError("value vector length differs from the foil range"))
    lo = sum(values[part.lower_base]; init=zero(eltype(values)))
    su = sum(values[part.surface]; init=zero(eltype(values)))
    up = sum(values[part.upper_base]; init=zero(eltype(values)))
    (; lower_base=lo, surface=su, upper_base=up, base=lo + up, total=sum(values; init=zero(eltype(values))))
end

"""
    base_mean(values, lengths, part)

Face-length-weighted mean of a per-face quantity over the two half-bases (e.g. the base pressure coefficient from gauge
pressure). NaN when the base is empty.
"""
function base_mean(values::AbstractVector, lengths::AbstractVector, part::NamedTuple)
    isempty(part.base) && return NaN
    w = lengths[part.base]
    sum(values[part.base] .* w) / sum(w)
end

end # module
