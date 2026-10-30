# Gurney flap geometry for the Airfoil Lab (23 Oct 2026, second design). Pure Julia Base, no solver code.
#
# The user's design: "A gurney flap must be like a wall at the trailing edge ... preserve the airfoil until almost the end
# and then have a sharp (or as sharp as possible) end plate". Their answers: zero thickness; exactly sharp corners at the
# root and at the tip; perpendicular to the chord; "symmetric" means h/2 above and h/2 below the trailing edge (a T).
#
# Construction (`gurney_plate`): the airfoil is untouched. A plate of zero thickness stands on the trailing-edge station
# x = x_TE (the app's sections have the chord along x, which is checked, as is that both trailing-edge points share x):
#   lower:     from the lower TE point (x_TE, y_l) down to the tip (x_TE, y_l - h);
#   upper:     from the upper TE point (x_TE, y_u) up to the tip (x_TE, y_u + h);
#   symmetric: from (x_TE, y_u + h/2) down to (x_TE, y_l - h/2), through the native gap.
# 28 Oct 2026: this module is the PLATE'S GEOMETRY RECORD (the drawing, the labels, the LBM's host section). The low-speed CFD
# models the plate as a baffle -- a wall inside the section's own finite-trailing-edge C+H mesh (PBCav's PB_BAFFLE) -- and
# does not use the C-block/H-block wall contour of the first plate design (the plate as part of the outer boundary), which is
# retired. The record's base_height is the extent of the wall at the trailing-edge station: the native gap g = y_u - y_l plus h.
module AirfoilGurney

using Printf

export GURNEY_MODES, GURNEY_REFINE_LEVELS, GURNEY_REFINE_KNOBS, gurney_settings, gurney_refine_level, gurney_plate, gurney_note

const GURNEY_MODES = ("symmetric", "lower", "upper")
const GURNEY_MODE_LABELS = Dict("symmetric"=>"symmetric T (h/2 above and h/2 below the trailing edge)",
                                "lower"=>"one side, lower (pressure) side", "upper"=>"one side, upper (suction) side")
const GURNEY_HEIGHT_RANGE = (0.001, 0.05)      # h/c
const GURNEY_TE_X_TOL = 1e-6                   # both trailing-edge points on one x station (chords)
const GURNEY_CHORD_ANGLE_TOL_DEG = 0.5         # the chord (leading edge -> trailing-edge midpoint) along x
const GURNEY_VERSION = "gurney_plate_v1"
# 30 Oct 2026: the mesh refinement of the plate's free edge (PBCav's PB_BAFFLE_REFINE, branch pbcav-flapref). `gurney_refine` is
# off (the default of the API: an absent key leaves every request as before), standard (tip refinement factor 4) or fine (6);
# the page sends standard unless the user changes it. The factors are the knob's `q`: the cells at the plate's tip are 1/q of
# the section's own C+H mesh cells there, growing back at a constant ratio; the extra rows are shared by every column.
const GURNEY_REFINE_LEVELS = ("off", "standard", "fine")
const GURNEY_REFINE_KNOBS = Dict("standard"=>"tip:q=4", "fine"=>"tip:q=6")

_key(d, k) = haskey(d, k) ? d[k] : (haskey(d, Symbol(k)) ? d[Symbol(k)] : nothing)

"""Validated Gurney settings from a request's flow map, or `nothing` when the flap is off. The flap is on when
`gurney_height` (the plate height h/c) is present and not `nothing`: h in [0.001, 0.05] and `gurney_mode` symmetric,
lower or upper. The ramp of the first design is gone: `gurney_ramp_length` is refused."""
function gurney_settings(flow::AbstractDict)
    h = _key(flow, "gurney_height"); mode = _key(flow, "gurney_mode")
    _key(flow, "gurney_ramp_length") === nothing || throw(ArgumentError(
        "gurney_ramp_length is no longer accepted: the Gurney flap is a zero-thickness plate perpendicular to the chord at the trailing edge, with no ramp"))
    if h === nothing
        mode === nothing || throw(ArgumentError("gurney_mode requires gurney_height (the plate height h/c)"))
        _key(flow, "gurney_refine") === nothing || throw(ArgumentError("gurney_refine requires gurney_height (the plate height h/c)"))
        return nothing
    end
    gurney_refine_level(flow)
    (h isa Real && !(h isa Bool) && isfinite(h)) || throw(ArgumentError("gurney_height must be a finite number (the plate height h/c)"))
    lo, hi = GURNEY_HEIGHT_RANGE
    lo <= h <= hi || throw(ArgumentError(@sprintf("gurney_height must lie between %.3f and %.2f chords (got %g)", lo, hi, h)))
    mode isa AbstractString || throw(ArgumentError("gurney_mode is required with gurney_height: symmetric, lower or upper"))
    m = lowercase(strip(String(mode)))
    m in GURNEY_MODES || throw(ArgumentError("gurney_mode must be symmetric, lower or upper (got $(mode))"))
    (height=Float64(h), mode=m)
end

"""The mesh refinement of the plate's free edge the request asks for: "off" (the default; an absent key), "standard" or "fine".
Refused when it is anything else, or not a string."""
function gurney_refine_level(flow::AbstractDict)
    v = _key(flow, "gurney_refine")
    v === nothing && return "off"
    v isa AbstractString || throw(ArgumentError("gurney_refine must be a string: off, standard or fine"))
    lv = lowercase(strip(String(v)))
    lv in GURNEY_REFINE_LEVELS || throw(ArgumentError("gurney_refine must be off, standard or fine (got $(v))"))
    lv
end

"Plate extents below the lower and above the upper trailing-edge point."
gurney_extents(h, mode) = mode == "lower" ? (Float64(h), 0.0) : mode == "upper" ? (0.0, Float64(h)) : (0.5Float64(h), 0.5Float64(h))

const GURNEY_TRIM_MAX = 2e-3                   # the largest trailing-edge x difference trimmed (chords)

"""    gurney_trim_te(x, y) -> (x, y, metadata)

The plate stands on ONE trailing-edge station. A section whose two trailing-edge points do not share x (a cambered NACA
four-digit section: its thickness is laid normal to the camber line, so NACA 4412's points are 3.3e-4 c apart) is
trimmed: the longer surface is cut at the shorter one's x (linear interpolation along its own contour), so both end on
x_TE = min(x_lower, x_upper). Nothing changes when they already share x within 1e-9 c; above 2e-3 c it is refused."""
function gurney_trim_te(x0::AbstractVector{<:Real}, y0::AbstractVector{<:Real})
    x = Float64.(collect(x0)); y = Float64.(collect(y0)); n = length(x)
    chord = 0.5(x[1] + x[n]) - minimum(x)
    dx = x[n] - x[1]
    meta = Dict{String,Any}("te_x_difference"=>abs(dx), "trimmed"=>false)
    abs(dx) <= 1e-9*chord && return (x, y, meta)
    abs(dx) <= GURNEY_TRIM_MAX*chord || throw(ArgumentError(@sprintf(
        "the Gurney plate stands perpendicular to the chord on one trailing-edge station, but this section's two trailing-edge points are %.2e c apart in x (the app trims at most %.0e c): use a section whose trailing-edge points share x.",
        abs(dx), GURNEY_TRIM_MAX)))
    ile = argmin(x)
    if dx > 0                                  # the upper surface runs past the lower trailing edge: cut it at x[1]
        xc = x[1]; k = n
        while k > ile && x[k-1] > xc; k -= 1; end   # x[k-1] <= xc < x[k]
        t = (xc - x[k-1])/(x[k] - x[k-1]); yc = y[k-1] + t*(y[k] - y[k-1])
        removed = sum(hypot(x[j+1] - x[j], y[j+1] - y[j]) for j in k-1:n-1) - hypot(xc - x[k-1], yc - y[k-1])
        x = [x[1:k-1]; xc]; y = [y[1:k-1]; yc]; side = "upper"
    else                                       # the lower surface runs past the upper trailing edge: cut it at x[n]
        xc = x[n]; k = 1
        while k < ile && x[k+1] > xc; k += 1; end   # x[k+1] <= xc < x[k]
        t = (xc - x[k+1])/(x[k] - x[k+1]); yc = y[k+1] + t*(y[k] - y[k+1])
        removed = sum(hypot(x[j+1] - x[j], y[j+1] - y[j]) for j in 1:k) - hypot(x[k+1] - xc, y[k+1] - yc)
        x = [xc; x[k+1:end]]; y = [yc; y[k+1:end]]; side = "lower"
    end
    x[1] == x[end] || (x[end] = x[1] = min(x[1], x[end]))
    merge!(meta, Dict{String,Any}("trimmed"=>true, "trimmed_surface"=>side, "trimmed_length"=>removed, "x_te"=>x[1],
        "note"=>@sprintf("the %s surface trimmed by %.2e c (its last %.2e c in x) so that both trailing-edge points lie on x/c = %.6f, the plate's station", side, removed, abs(dx), x[1])))
    (x, y, meta)
end

"""    gurney_plate(x, y; height, mode) -> (x, y, metadata)

The C-block wall contour of the section with the plate: the source contour (wrap order, lower trailing edge -> leading
edge -> upper trailing edge, chord units) unchanged, with the tip node(s) appended as exact straight segments. Refused
(ArgumentError, with the reason) when the chord is not along x within 0.5 deg, when the two trailing-edge points do not
share x within 1e-6 c, or when the upper trailing-edge point lies below the lower one."""
function gurney_plate(x0::AbstractVector{<:Real}, y0::AbstractVector{<:Real}; height::Real, mode::AbstractString)
    n = length(x0)
    n == length(y0) >= 21 || throw(ArgumentError("the Gurney flap needs at least 21 contour points with matching x and y"))
    all(isfinite, x0) && all(isfinite, y0) || throw(ArgumentError("airfoil coordinates must be finite"))
    s = gurney_settings(Dict{String,Any}("gurney_height"=>Float64(height), "gurney_mode"=>String(mode)))
    x = Float64.(collect(x0)); y = Float64.(collect(y0))
    ile = argmin(x)
    3 <= ile <= n - 2 || throw(ArgumentError("the leading edge (minimum x) must lie inside the contour"))
    xl, yl, xu, yu = x[1], y[1], x[n], y[n]
    chord = 0.5(xl + xu) - x[ile]
    chord > 0 || throw(ArgumentError("the trailing edge must lie downstream of the leading edge"))
    dx_te = abs(xu - xl)
    dx_te <= GURNEY_TE_X_TOL*chord || throw(ArgumentError(@sprintf(
        "the Gurney plate stands perpendicular to the chord on the trailing-edge station, but this section's two trailing-edge points are %.2e c apart in x (limit %.0e c): its base would not be one straight line through the plate. (NACA four-digit sections with camber have this: their thickness is laid normal to the camber line.) Use a section whose trailing-edge points share x.",
        dx_te, GURNEY_TE_X_TOL)))
    angle = atand(0.5(yl + yu) - y[ile], 0.5(xl + xu) - x[ile])
    abs(angle) <= GURNEY_CHORD_ANGLE_TOL_DEG || throw(ArgumentError(@sprintf(
        "the Gurney plate is drawn perpendicular to the chord, which the app takes along x; this section's chord (leading edge to trailing-edge midpoint) is inclined %.2f deg (limit %.1f deg). Use a section normalized with its chord along x.",
        angle, GURNEY_CHORD_ANGLE_TOL_DEG)))
    g = yu - yl
    g >= -1e-12 || throw(ArgumentError("the upper trailing-edge point lies below the lower one"))
    g = max(g, 0.0)
    xte = 0.5(xl + xu)
    hl, hu = gurney_extents(s.height*chord, s.mode)
    tip_lo = (xte, yl - hl); tip_up = (xte, yu + hu)
    wx = copy(x); wy = copy(y)
    hl > 0 && (pushfirst!(wx, xte); pushfirst!(wy, yl - hl))
    hu > 0 && (push!(wx, xte); push!(wy, yu + hu))
    # the trailing-edge points themselves take the station's x exactly (they already share it to 1e-6 c)
    hl > 0 && (wx[2] = xte); hu > 0 && (wx[end-1] = xte)
    # fluid angle at each root (between the plate and the surface it stands on) and at each tip
    function root_angle(kroot, kprev, dirplate)
        tx, ty = wx[kprev] - wx[kroot], wy[kprev] - wy[kroot]
        acosd(clamp((tx*dirplate[1] + ty*dirplate[2])/hypot(tx, ty), -1.0, 1.0))
    end
    meta = Dict{String,Any}("version"=>GURNEY_VERSION, "gurney_height"=>s.height, "gurney_mode"=>s.mode,
        "gurney_mode_label"=>GURNEY_MODE_LABELS[s.mode], "chord"=>chord, "x_te"=>xte,
        "source_te_gap"=>g, "base_height"=>g + s.height*chord, "plate_below_lower_te"=>hl, "plate_above_upper_te"=>hu,
        "lower_te"=>[xl, yl], "upper_te"=>[xu, yu],
        "lower_tip"=>[tip_lo[1], tip_lo[2]], "upper_tip"=>[tip_up[1], tip_up[2]],
        "base"=>"x = $(xte): from ($(xte), $(tip_lo[2])) to ($(xte), $(tip_up[2]))",
        "plate"=>"zero thickness, perpendicular to the chord on the trailing-edge station, exact sharp root and tip corners",
        "chord_angle_deg"=>angle, "te_x_difference"=>dx_te, "points"=>length(wx), "source_points"=>n,
        "airfoil_unchanged"=>true)
    hl > 0 && (meta["lower_root_fluid_angle_deg"] = root_angle(2, 3, (0.0, -1.0)); meta["lower_tip_angles_deg"] = Dict("C"=>270.0, "H"=>90.0))
    hu > 0 && (meta["upper_root_fluid_angle_deg"] = root_angle(length(wx) - 1, length(wx) - 2, (0.0, 1.0)); meta["upper_tip_angles_deg"] = Dict("C"=>270.0, "H"=>90.0))
    (x=wx, y=wy, metadata=meta)
end

"One line for notes and logs."
gurney_note(meta::AbstractDict) = @sprintf("CFD geometry change: Gurney flap, a zero-thickness plate perpendicular to the chord at the trailing edge (x/c = %.4g), h/c = %.4g, %s; tips at y/c = %s; the wall at the trailing-edge station then spans %.5f c (source trailing-edge gap %.5f c + h); the airfoil itself is unchanged (in the low-speed CFD the plate is a baffle, a wall inside the section's own C+H mesh)",
    meta["x_te"], meta["gurney_height"], meta["gurney_mode_label"],
    join([@sprintf("%.5f", t[2]) for t in (meta["lower_tip"], meta["upper_tip"])], " and "), meta["base_height"], meta["source_te_gap"])

end # module AirfoilGurney
