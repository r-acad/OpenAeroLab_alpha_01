# =====================================================================================================================
# knobs.jl -- the PB_ knob registry (plan 6.2: version 2's knobs take the prefix PB_, so audit_knobs.jl and
# audit_env_knobs.jl see no TOPO_ collision).
#
# THE RULE. A knob exists only if it is in PB_KNOBS below, with its default, its parser and what it does; the driver reads
# knobs only through read_knobs, which REFUSES any PB_* variable in the environment that is not registered (a knob set
# and read by nobody is the defect class of preflight gate 1, 2026-08-05) and any TOPO_*, TEM_* or CAV_MOVIE_* variable
# (version 2 reads none of version 1's knobs; one set in its environment is a run configured for the other solver).
# A knob joins the registry in the commit whose code reads it, never before.
# =====================================================================================================================

"One registered knob: its environment name, default (as written in the environment), parser and description."
struct PBKnob
    name::String
    default::String
    parse::Function
    doc::String
end

function _parse_mesh(s::AbstractString)
    v = split(s, ",")
    length(v) == 6 || throw(ArgumentError("PB_MESH must be n_foil,n_normal,dn1,growth,smooth,n_wake (six fields; got \"$s\")"))
    nf = parse(Int, v[1]); nn = parse(Int, v[2]); dn1 = parse(Float64, v[3]); g = parse(Float64, v[4])
    sm = parse(Int, v[5]); nw = parse(Int, v[6])
    nf >= 8 && nn >= 4 && dn1 > 0 && isfinite(dn1) && g >= 1 && isfinite(g) && sm >= 0 && nw >= 8 ||
        throw(ArgumentError("PB_MESH fields out of range (got \"$s\")"))
    (n_foil=nf, n_normal=nn, dn1=dn1, growth=g, smooth=sm, n_wake=nw)
end
function _parse_wake_near(s::AbstractString)
    isempty(s) && return nothing
    v = parse.(Float64, split(s, ","))
    length(v) in (2, 3) && all(isfinite, v) && v[1] > 0 && v[2] > 0 && (length(v) == 2 || v[3] >= 1) ||
        throw(ArgumentError("PB_MESH_WAKE_NEAR must be ds,len[,growth] with ds > 0, len > 0, growth >= 1 (got \"$s\")"))
    (wake_near_ds=v[1], wake_near_len=v[2], wake_near_growth=length(v) == 3 ? v[3] : 1.05)
end
_parse_float(lo, hi; name) = s -> (x = parse(Float64, s); isfinite(x) && lo <= x <= hi ||
                                    throw(ArgumentError("$name must lie in [$lo, $hi] (got \"$s\")")); x)
_parse_choice(choices; name) = s -> (s in choices || throw(ArgumentError("$name must be one of $(join(choices, ", ")) (got \"$s\")"));
                                     Symbol(s))
_parse_int(lo, hi; name) = s -> (x = parse(Int, s); lo <= x <= hi ||
                                  throw(ArgumentError("$name must lie in [$lo, $hi] (got \"$s\")")); x)
# ---- the free-air mesh remap's fields (W4, 22 Oct 2026; src/pbcav/mesh_remap.jl) ----------------------------------------
function _parse_remap_keep(s::AbstractString)
    v = split(s, ",")
    length(v) == 2 || throw(ArgumentError("PB_MESH_REMAP_KEEP must be n_keep,n_blend (got \"$s\")"))
    k = parse(Int, v[1]); b = parse(Int, v[2])
    2 <= k < b <= 1000 || throw(ArgumentError("PB_MESH_REMAP_KEEP needs 2 <= n_keep < n_blend <= 1000 (got \"$s\")"))
    (n_keep=k, n_blend=b)
end
function _parse_remap_law(s::AbstractString)
    v = split(s, ",")
    length(v) == 3 || throw(ArgumentError("PB_MESH_REMAP_LAW must be rows,ratio,y_law (got \"$s\")"))
    n = parse(Int, v[1]); r = parse(Float64, v[2]); y = parse(Float64, v[3])
    (n == 0 || 8 <= n <= 2000) && isfinite(r) && (r == 0 || 1 < r <= 1.5) && isfinite(y) && 0 <= y <= 1e3 ||
        throw(ArgumentError("PB_MESH_REMAP_LAW: rows 0 or 8-2000, ratio 0 or in (1, 1.5], y_law 0 or in (0, 1000] (got \"$s\")"))
    (rows=n, ratio=r, y_law=y)
end
function _parse_remap_wake(s::AbstractString)
    v = parse.(Float64, split(s, ","))
    length(v) == 2 && all(isfinite, v) && ((v[1] == 0 && v[2] == 0) || (v[1] > 0 && v[2] > 0)) ||
        throw(ArgumentError("PB_MESH_REMAP_WAKE must be ar_far,ar_outlet, both > 0, or 0,0 for no wake growth (got \"$s\")"))
    (ar=v[1], ar_outlet=v[2])
end
function _parse_remap_stall(s::AbstractString)
    isempty(s) && return nothing
    v = parse.(Float64, split(s, ","))
    length(v) in (2, 3) && all(isfinite, v) && 0 < v[1] < v[2] <= 100 && (length(v) == 2 || 0 < v[3] <= 1) ||
        throw(ArgumentError("PB_MESH_REMAP_STALL must be h,y[,f] with 0 < h < y and 0 < f <= 1 (got \"$s\"), or empty for none"))
    (h=v[1], y=v[2], f=length(v) == 3 ? v[3] : 0.4)
end
# ---- W4b (22 Oct 2026, after W4's review; src/pbcav/mesh_remap.jl header item 7) --------------------------------------------
function _parse_remap_strip(s::AbstractString)
    isempty(s) && return nothing
    v = split(s, ",")
    v[1] == "graded" && length(v) in (1, 2) ||
        throw(ArgumentError("PB_MESH_REMAP_STRIP must be empty (the strip untouched), graded, or graded,edge (got \"$s\")"))
    length(v) == 1 && return (edge=0.0,)
    e = parse(Float64, v[2])
    isfinite(e) && 0 < e < 0.01 || throw(ArgumentError("PB_MESH_REMAP_STRIP=graded,edge needs 0 < edge < 0.01 chords (got \"$s\")"))
    (edge=e,)
end
# ---- the merge of W4/W4b (22 Oct 2026; review of W4b, F3): the graded strip's capped base -------------------------------------
function _parse_remap_strip_cap(s::AbstractString)
    isempty(s) && return nothing
    s == "parent" && return (cap=-1.0,)
    c = tryparse(Float64, s)
    c !== nothing && isfinite(c) && 0 < c < 0.01 ||
        throw(ArgumentError("PB_MESH_REMAP_STRIP_CAP must be empty (no cap), parent (the parent's uniform base interval) or a " *
                            "cap in chords, 0 < cap < 0.01 (got \"$s\")"))
    (cap=c,)
end

"PB_START_RAMP: T,r0 (the ramp start of freeair.jl; T > 0 chord-times, 0 < r0 <= 1)."
function _parse_start_ramp(s::AbstractString)
    v = split(s, ",")
    length(v) == 2 || throw(ArgumentError("PB_START_RAMP must be T,r0 (got \"$s\")"))
    T = tryparse(Float64, v[1]); r0 = tryparse(Float64, v[2])
    (T !== nothing && r0 !== nothing && isfinite(T) && 0 < T <= 1e4 && 0 < r0 <= 1) ||
        throw(ArgumentError("PB_START_RAMP needs 0 < T <= 1e4 and 0 < r0 <= 1 (got \"$s\")"))
    (T=T, r0=r0)
end

# ---- interior walls (24 Oct 2026; src/pbcav/baffle.jl): the Gurney plate on a C column --------------------------------------
"""PB_BAFFLE: empty (THE DEFAULT: no baffle, every mesh and case bit for bit the one before), or
`gurney:side=lower|upper|symmetric,h=<h/c>[,snap=exact|nearest][,straighten=<deg>|off][,shear=<s>]` -- a zero-thickness plate
on the C column from the trailing-edge corner (both corners when symmetric, h/2 each), of height h (0 <= h <= 0.2; h = 0 makes
no plate); snap = exact (the default: the line's node nearest h moved to h exactly) or nearest (the node as it is); straighten
= the largest deviation from the chord's normal left as it is (default 0.5 degrees; off: never straightened); shear = the
largest relative shear the node movement may add to a cell (default 0.1; sets its window). Refused when malformed."""
function _parse_baffle(s::AbstractString)
    isempty(s) && return nothing
    kind, rest = occursin(":", s) ? split(s, ":"; limit=2) : (s, "")
    kind == "gurney" || throw(ArgumentError("PB_BAFFLE must be empty or gurney:side=...,h=... (got \"$s\")"))
    d = Dict{String,String}()
    for kv in split(rest, ",")
        isempty(kv) && continue
        occursin("=", kv) || throw(ArgumentError("PB_BAFFLE: field \"$kv\" is not key=value (got \"$s\")"))
        k, v = split(kv, "="; limit=2)
        k in ("side", "h", "snap", "straighten", "shear") || throw(ArgumentError("PB_BAFFLE: unknown field \"$k\" (got \"$s\")"))
        haskey(d, k) && throw(ArgumentError("PB_BAFFLE: field \"$k\" given twice (got \"$s\")"))
        d[k] = v
    end
    haskey(d, "side") && haskey(d, "h") || throw(ArgumentError("PB_BAFFLE=gurney needs side and h (got \"$s\")"))
    side = d["side"]; side in ("lower", "upper", "symmetric") ||
        throw(ArgumentError("PB_BAFFLE: side must be lower, upper or symmetric (got \"$side\")"))
    h = tryparse(Float64, d["h"]); (h !== nothing && isfinite(h) && 0 <= h <= 0.2) ||
        throw(ArgumentError("PB_BAFFLE: h must be a number in [0, 0.2] chords (got \"$(d["h"])\")"))
    snap = get(d, "snap", "exact"); snap in ("exact", "nearest") ||
        throw(ArgumentError("PB_BAFFLE: snap must be exact or nearest (got \"$snap\")"))
    st = get(d, "straighten", "0.5")
    tol = st == "off" ? Inf : tryparse(Float64, st)
    (tol !== nothing && tol >= 0 && !isnan(tol)) ||
        throw(ArgumentError("PB_BAFFLE: straighten must be off or an angle >= 0 degrees (got \"$st\")"))
    sh = tryparse(Float64, get(d, "shear", "0.1")); (sh !== nothing && 0 < sh <= 0.5) ||
        throw(ArgumentError("PB_BAFFLE: shear must lie in (0, 0.5] (got \"$(get(d, "shear", ""))\")"))
    (kind=:gurney, side=Symbol(side), h=h, snap=Symbol(snap), straighten=tol, shear=sh)
end

"""PB_BAFFLE_REFINE: empty (THE DEFAULT: no refinement, every baffle mesh bit for bit the one before), or
`tip[:q=<q>][,ratio=<r>][,wake=<c>|auto][,wake_taper=<c>|auto][,foil=<h>][,foil_taper=<h>][,keep=<rows>][,strict=yes|no]` --
rows added around the Gurney plate's free edge (its tip) and along the shear layer that leaves it (baffle.jl `baffle_refine`):
q the refinement factor at the tip (the tip cells are 1/q of the mesh's own, 1 < q <= 8, default 3), ratio the growth ratio of
the transition back to the mesh's own cells (1.1-1.6, default 1.25), wake / wake_taper the distance downstream along the wake
leg (chords) over which the refinement is at full strength and then fades (auto: 30 plate heights capped to 0.09-0.30 c,
then 0.8-1.0 times that distance), foil / foil_taper the same along the foil's surface upstream of the plate (in plate
heights, default 2 and 4), keep the
rows next to the wall that are never touched (default 3; density is moved from the root and TE wake leg and three rows are
added across the flap-height recirculation region; the finite-TE H block's cross rows are coarsened while its interface curves
and first edge cells are kept), strict = refuse (rather than fall back to the unrefined mesh, with a
recorded reason) when the refined mesh fails its admission checks. Needs PB_BAFFLE. Refused when malformed."""
function _parse_baffle_refine(s::AbstractString)
    isempty(s) && return nothing
    kind, rest = occursin(":", s) ? split(s, ":"; limit=2) : (s, "")
    kind == "tip" || throw(ArgumentError("PB_BAFFLE_REFINE must be empty or tip[:q=...] (got \"$s\")"))
    d = Dict{String,String}()
    for kv in split(rest, ",")
        isempty(kv) && continue
        occursin("=", kv) || throw(ArgumentError("PB_BAFFLE_REFINE: field \"$kv\" is not key=value (got \"$s\")"))
        k, v = split(kv, "="; limit=2)
        k in ("q", "ratio", "wake", "wake_taper", "foil", "foil_taper", "keep", "strict") ||
            throw(ArgumentError("PB_BAFFLE_REFINE: unknown field \"$k\" (got \"$s\")"))
        haskey(d, k) && throw(ArgumentError("PB_BAFFLE_REFINE: field \"$k\" given twice (got \"$s\")"))
        d[k] = v
    end
    num(k, dflt; lo, hi) = (v = get(d, k, nothing); v === nothing ? dflt :
        (x = tryparse(Float64, v); (x !== nothing && isfinite(x) && lo <= x <= hi) ||
            throw(ArgumentError("PB_BAFFLE_REFINE: $k must be a number in [$lo, $hi] (got \"$v\")")); x))
    autoc(k) = (v = get(d, k, "auto"); v == "auto" ? 0.0 :
        (x = tryparse(Float64, v); (x !== nothing && isfinite(x) && 0 < x <= 20) ||
            throw(ArgumentError("PB_BAFFLE_REFINE: $k must be auto or a distance in chords in (0, 20] (got \"$v\")")); x))
    q = num("q", 3.0; lo=1.2, hi=8.0); ratio = num("ratio", 1.25; lo=1.1, hi=1.6)
    foil = num("foil", 2.0; lo=0.0, hi=50.0); foilt = num("foil_taper", 4.0; lo=0.5, hi=100.0)
    kp = get(d, "keep", "3"); keep = tryparse(Int, kp)
    (keep !== nothing && 3 <= keep <= 40) || throw(ArgumentError("PB_BAFFLE_REFINE: keep must be an integer in [3, 40] (got \"$kp\")"))
    st = get(d, "strict", "no"); st in ("yes", "no") || throw(ArgumentError("PB_BAFFLE_REFINE: strict must be yes or no (got \"$st\")"))
    (kind=:tip, q=q, ratio=ratio, wake=autoc("wake"), wake_taper=autoc("wake_taper"), foil=foil, foil_taper=foilt, keep=keep,
     strict=st == "yes")
end

"""The registry. Defaults are N60R's mesh (docs/2026-10-09_N60R_REGISTRATION.md), the version-1 knobs' values for that
mesh written as PB_ knobs, so `PB_STOP=mesh` with no other knob prints N60R's registered MESH COORDINATE RECEIPT."""
const PB_KNOBS = [
    PBKnob("PB_BACKEND", "cpu", _parse_choice(("cpu", "cuda"); name="PB_BACKEND"),
           "the KernelAbstractions backend: cpu, or cuda (stage P3, W3: CUDA.jl's CUDABackend, loaded by the driver only " *
           "when asked; the mesh is built on the CPU and moved, so its metrics are the CPU's bit for bit)"),
    PBKnob("PB_STOP", "run", _parse_choice(("mesh", "run"); name="PB_STOP"),
           "mesh: build the mesh, print its receipts and stop; run: march (stage P0b's second half builds the march)"),
    PBKnob("PB_FOIL", "tools/naca66mod_t09_f020_a08_blunt.dat", s -> String(s),
           "the section file, absolute or relative to the repository root (version 1's TOPO_FOIL)"),
    PBKnob("PB_MESH", "256,108,5e-5,1.20,80,136", _parse_mesh,
           "n_foil,n_normal,dn1,growth,smooth,n_wake (version 1's TOPO_MESH, all six fields required)"),
    PBKnob("PB_MESH_LE_DS", "3e-4", _parse_float(0.0, 0.0999; name="PB_MESH_LE_DS"),
           "leading-edge surface interval, chord units; 0 the default distribution (TOPO_MESH_LE_DS)"),
    PBKnob("PB_MESH_TE_BASE", "8", _parse_int(0, 10_000; name="PB_MESH_TE_BASE"),
           "cells per half of the trailing-edge base; 0 a sharp edge (TOPO_MESH_TE_BASE)"),
    PBKnob("PB_MESH_TE_DS", "5.62e-4", _parse_float(0.0, 0.0999; name="PB_MESH_TE_DS"),
           "trailing-edge surface interval; 0 the default (TOPO_MESH_TE_DS)"),
    PBKnob("PB_MESH_WAKE_NEAR", "7.4925e-4,0.05994", _parse_wake_near,
           "near-wake ds,len[,growth]; empty for none (TOPO_MESH_WAKE_NEAR)"),
    PBKnob("PB_MESH_TE_TOPOLOGY", "strip", _parse_choice(("aplus", "strip"); name="PB_MESH_TE_TOPOLOGY"),
           "trailing-edge topology of a blunt base (TOPO_MESH_TE_TOPOLOGY)"),
    PBKnob("PB_MESH_LEVEL", "0", _parse_int(0, 4; name="PB_MESH_LEVEL"),
           "bilinear subdivisions of the generated mesh (the nested levels of plan 10.1)"),
    PBKnob("PB_CHANNEL_H", "5.0", _parse_float(1e-3, 1e3; name="PB_CHANNEL_H"),
           "channel height, chords (TOPO_CHANNEL_H; version 2 builds channel meshes only)"),
    PBKnob("PB_CHANNEL_XMIN", "-5.0", _parse_float(-1e3, 0.0; name="PB_CHANNEL_XMIN"),
           "inlet station along the channel, chords (TOPO_CHANNEL_XMIN)"),
    PBKnob("PB_CHANNEL_XMAX", "6.83", _parse_float(1.0, 1e3; name="PB_CHANNEL_XMAX"),
           "outlet station along the channel, chords (TOPO_CHANNEL_XMAX)"),
    PBKnob("PB_ALPHA", "4.0", _parse_float(-30.0, 30.0; name="PB_ALPHA"),
           "incidence, degrees: the channel's angle to the foil (TOPO_ALPHA); with PB_DOMAIN=freeair the freestream's angle " *
           "to the foil, the mesh not rotated (its outer domain at PB_FARFIELD_ANGLE)"),
    # the turbulence model (W1, plan Amendment 12, section 32; 21 Oct 2026): read by `turbulence_settings`, which gives the
    # PBConfig keywords; laminar, the default, is the path before W1 bit for bit
    PBKnob("PB_TURBULENCE", "laminar", _parse_choice(("laminar", "sst"); name="PB_TURBULENCE"),
           "the turbulence model: laminar (the default) or sst (SST-2003m, assemble_komega.jl)"),
    PBKnob("PB_SST_TU_INF", "0.001", _parse_float(1e-8, 0.199; name="PB_SST_TU_INF"),
           "SST inflow turbulence intensity (version 1's TOPO_SST_TU_INF and default)"),
    PBKnob("PB_SST_MUT_RATIO", "0.1", _parse_float(1e-8, 1e4; name="PB_SST_MUT_RATIO"),
           "SST inflow eddy-viscosity ratio mu_t/mu (version 1's TOPO_SST_MUT_RATIO and default)"),
    PBKnob("PB_WALL_OMEGA", "v1", _parse_choice(("v1", "auto"); name="PB_WALL_OMEGA"),
           "SST wall omega: v1 (W-a, 60 nu/(beta1 d1^2), version 1's) or auto (W-b, Menter's blend: a stub until stage P2)"),
    # the SST solve's settings (the W1+W2 integration, 22 Oct 2026; re-review of W1b, R1 d): every default is PBConfig's, so a
    # run that sets none of them is the configuration before the knobs existed; read by `turbulence_settings`
    PBKnob("PB_TURB_COUPLING", "step", _parse_choice(("step", "outer"); name="PB_TURB_COUPLING"),
           "SST: the k-omega coupling, step (one update after the accepted flow iterate, the default) or outer (PBConfig.turb_coupling)"),
    PBKnob("PB_TURB_TOL", "1e-8", _parse_float(1e-15, 1.0; name="PB_TURB_TOL"),
           "SST: turb_tol, the k-omega tolerance (:outer's acceptance; the line solve's goal floor is 0.1 turb_tol under :outer, " *
           "0.1 min(turb_tol, turb_step_tol) under :step)"),
    PBKnob("PB_TURB_STEP_TOL", "Inf", s -> (x = parse(Float64, s); (x > 0 && !isnan(x)) ||
                                            throw(ArgumentError("PB_TURB_STEP_TOL must be > 0 (Inf: one update per step; got \"$s\")")); x),
           "SST: turb_step_tol, the per-step k-omega tolerance under :step (Inf: one update per step, the steady pseudo-time form)"),
    PBKnob("PB_TURB_NONORTH", "1.0", _parse_float(0.0, 1.0; name="PB_TURB_NONORTH"),
           "SST: turb_nonorth, the k-omega non-orthogonal diffusion (1: the full correction with W1b's positivity split)"),
    PBKnob("PB_TURB_OUTER_TOL", "0", _parse_float(0.0, 1.0; name="PB_TURB_OUTER_TOL"),
           "SST, turb_coupling = :outer: the per-step k-omega acceptance tolerance on its own (0: turb_tol, the default; W6)"),
    PBKnob("PB_TURB_OUTER_RAMP", "0", _parse_int(0, 1_000_000; name="PB_TURB_OUTER_RAMP"),
           "SST: turb_outer_ramp, the committed steps taken as :step before :outer (needed by :outer from the uniform start)"),
    # W6 (a robust start for steady SST in free air, 22 Oct 2026): the defaults are PBConfig's (W6's; PB_DEFAULTS_BEFORE_W6)
    PBKnob("PB_TURB_MUT", "frozen", _parse_choice(("iterate", "frozen"); name="PB_TURB_MUT"),
           "SST: turb_mut, the momentum's mu_t within a :step-treated step: frozen (formed once at the step's first outer " *
           "iteration, the default since W6; the same steady fixed point) or iterate (re-formed at every outer iterate, W1's)"),
    # the A2F_a15 diagnosis (27 Oct 2026, branch pbcav-mutbase; records air/mutbase/): the default is PBConfig's (W6's), and the
    # knob is printed in PB KNOBS only when its value is not the default (RECEIPT_QUIET_AT_DEFAULT); free air only
    PBKnob("PB_TURB_MUT_BASE", "extrapolated", _parse_choice(("extrapolated", "committed"); name="PB_TURB_MUT_BASE"),
           "SST, PB_TURB_MUT=frozen, free air only: turb_mut_base, the state the step's frozen mu_t is formed at: extrapolated " *
           "(the step's start iterate, u and v extrapolated from n and n-1: the default, W6's) or committed (the committed level n; " *
           "the same steady fixed point; removes the neutral travelling mode the A2F_a15 diagnosis measured under the stress limiter)"),
    PBKnob("PB_TURB_OMEGA_NONORTH", "wall_off", _parse_choice(("full", "wall_off", "limited"); name="PB_TURB_OMEGA_NONORTH"),
           "SST: turb_omega_nonorth, omega's non-orthogonal diffusion: wall_off (none on the faces of wall-adjacent cells, the " *
           "default since W6), full (as k's, W1b's) or limited (clamped to the two-point flux)"),
    PBKnob("PB_TURB_OMEGA_BOUND", "0.2", _parse_float(0.0, 0.999; name="PB_TURB_OMEGA_BOUND"),
           "SST: turb_omega_bound, omega's net negative non-orthogonal correction at most this fraction of the cell's steady " *
           "positive budget (production, positive cross diffusion, boundary inflow, sum_N |a_N| omega_N): 0.2 the default since " *
           "W6b (the review of W6, R1); 0 none (W6's)"),
    # ---- free air (work item W2, plan section 32; 21 Oct 2026). Every default below leaves the tunnel case bitwise:
    # PB_DOMAIN=tunnel (the default) reads none of the other knobs of this block. -------------------------------------
    PBKnob("PB_DOMAIN", "tunnel", _parse_choice(("tunnel", "freeair"); name="PB_DOMAIN"),
           "tunnel: the channel of PB_CHANNEL_* with slip walls (plan section 9, the default); freeair: version 1's generator " *
           "channel of PB_FARFIELD_* with every outer face tagged far field and the downstream end a pressure outlet (freeair.jl)"),
    PBKnob("PB_FARFIELD_H", "60.0", _parse_float(1.0, 1e4; name="PB_FARFIELD_H"),
           "free air: the outer domain's height, chords (the app's free-air meshes: 60)"),
    PBKnob("PB_FARFIELD_XMIN", "-29.5", _parse_float(-1e4, 0.0; name="PB_FARFIELD_XMIN"),
           "free air: the upstream far-field station, chords (the app's: -29.5)"),
    PBKnob("PB_FARFIELD_XMAX", "30.5", _parse_float(1.0, 1e4; name="PB_FARFIELD_XMAX"),
           "free air: the downstream pressure-outlet station, chords (the app's: 30.5)"),
    PBKnob("PB_FARFIELD_ANGLE", "0.0", _parse_float(-45.0, 45.0; name="PB_FARFIELD_ANGLE"),
           "free air: the outer domain's angle to the foil, degrees (0: one mesh serves a whole polar; the incidence is PB_ALPHA)"),
    PBKnob("PB_RE", "1e4", _parse_float(1.0, 1e9; name="PB_RE"),
           "free air run: the chord Reynolds number rho U c/mu (rho = U = c = 1, so mu = 1/PB_RE; the model is PB_TURBULENCE)"),
    PBKnob("PB_DT", "0.05", _parse_float(1e-8, 1e3; name="PB_DT"),
           "free air run: the step (chords/U), backward Euler; with PB_STEADY_TOL > 0 a pseudo-time step"),
    PBKnob("PB_MAX_STEPS", "2000", _parse_int(1, 10_000_000; name="PB_MAX_STEPS"),
           "free air run: the most committed steps of the one march"),
    PBKnob("PB_STEADY_TOL", "1e-6", _parse_float(0.0, 1.0; name="PB_STEADY_TOL"),
           "free air run: stop when the steady residual (rho U^2/c per unit volume) falls to it; 0 never"),
    PBKnob("PB_WALL_CAP", "3600", _parse_float(1.0, 1e7; name="PB_WALL_CAP"),
           "free air run: the run's wall cap, seconds (pbcav_run!: INCOMPLETE at the cap)"),
    PBKnob("PB_N_OUTER_MAX", "12", _parse_int(1, 1000; name="PB_N_OUTER_MAX"),
           "free air run: the outer iterations (coupled solves) allowed per step, N_max of plan 4.7"),
    PBKnob("PB_CPL_AU", "0.8", _parse_float(0.05, 1.0; name="PB_CPL_AU"),
           "free air run: cpl_au, SIMPLEC's alpha_u in the coupled step's preconditioner only (the momentum block's diagonal a/au, " *
           "D = V/(a^t + (1/au - 1) a)); 0.8 PBConfig's default; W6b measured 0.99 on A2's high-aspect-ratio near wake (the review " *
           "of W6, R2)"),
    PBKnob("PB_CPL_VISC_NONORTH", "jacobian", _parse_choice(("jacobian", "deferred"); name="PB_CPL_VISC_NONORTH"),
           "free air run: cpl_visc_nonorth, the coupled step's J with (jacobian, PBConfig's default) or without (deferred: " *
           "converged by the outer iteration) the momentum's viscous non-orthogonal part (W6b; the review of W6, R2)"),
    PBKnob("PB_VORTEX", "none", _parse_choice(("none", "point"); name="PB_VORTEX"),
           "free air run: none, the far field at the uniform freestream; point, the incompressible point-vortex correction " *
           "(Thomas and Salas 1986) with the circulation from the current lift, lagged and relaxed (freeair.jl)"),
    PBKnob("PB_VORTEX_RELAX", "0.5", _parse_float(1e-3, 1.0; name="PB_VORTEX_RELAX"),
           "free air run, PB_VORTEX=point: the circulation's relaxation factor per update"),
    PBKnob("PB_VORTEX_EVERY", "10", _parse_int(1, 1_000_000; name="PB_VORTEX_EVERY"),
           "free air run, PB_VORTEX=point: committed steps between circulation updates"),
    PBKnob("PB_OUT_DIR", "", s -> String(s),
           "free air run: the directory for the load history and the surface C_p CSVs; empty, none written"),
    PBKnob("PB_START", "projection", _parse_choice(("projection", "kutta", "ramp"); name="PB_START"),
           "free air run: the start, projection (the potential flow without circulation from the uniform freestream, the default), " *
           "kutta (the lifting potential flow whose circulation zeroes the velocity jump behind the sharp trailing edge; W6) " *
           "or ramp (24 Oct 2026: the freestream accelerated smoothly from r0 U to U over t = T, PB_START_RAMP; freeair.jl)"),
    PBKnob("PB_START_RAMP", "2,0.05", _parse_start_ramp,
           "PB_START=ramp only: T,r0 -- the ramp's duration in chord-times (c/U) and the initial fraction of the freestream, " *
           "r(t) = r0 + (1 - r0)(1 - cos(pi t/T))/2 for t < T, then 1 (T > 0, 0 < r0 <= 1)"),
    PBKnob("PB_TURB_STEADY_TOL", "1e-7", _parse_float(1e-15, 1.0; name="PB_TURB_STEADY_TOL"),
           "free air run, PB_TURBULENCE=sst: the steady stop also needs the k and omega relative steady residuals at or below it " *
           "(W1's rule 1e-7; it must lie above the line solve's goal floor: 0.1 PB_TURB_TOL under PB_TURB_COUPLING=outer, " *
           "0.1 min(PB_TURB_TOL, PB_TURB_STEP_TOL) under step)"),
    # ---- speed (work item W7, plan Amendment 13, section 33; 22 Oct 2026): read by `speed_settings`; every default is the
    # path before W7, bit for bit (PBConfig's docstring and pimple.jl's "THE SPEED SETTINGS") --------------------------------
    PBKnob("PB_SYNC", "all", _parse_choice(("all", "reads"); name="PB_SYNC"),
           "W7: the step's device synchronisations: all (the default), or reads (on a GPU only the host reads drain the stream)"),
    PBKnob("PB_LINE_SOLVER", "thomas", _parse_choice(("thomas", "pcr"); name="PB_LINE_SOLVER"),
           "W7: every line solve (multigrid, coupled preconditioner, k-omega): thomas (the default) or pcr (workgroup PCR)"),
    PBKnob("PB_PREC_PRECISION", "f64", _parse_choice(("f64", "f32"); name="PB_PREC_PRECISION"),
           "W7: the coupled preconditioner (momentum sweeps and pressure V-cycle): f64 (the default) or f32"),
    PBKnob("PB_KW_PRECISION", "f64", _parse_choice(("f64", "f32"); name="PB_KW_PRECISION"),
           "W7: the k-omega line solves' corrections: f64 (the default) or f32 (residual and test stay Float64)"),
    PBKnob("PB_CPL_GS", "mgs", _parse_choice(("mgs", "mgs_dev", "cgs2"); name="PB_CPL_GS"),
           "W7: FGMRES's Gram-Schmidt: mgs (the default), mgs_dev (the same operations, coefficients on the device) or cgs2 (classical, twice)"),
    PBKnob("PB_GRAPHS", "off", _parse_choice(("off", "on"); name="PB_GRAPHS"),
           "W7: on a GPU, the coupled preconditioner and J replayed as CUDA graphs: off (the default) or on (needs PB_SYNC=reads)"),
    PBKnob("PB_DOT", "chunk", _parse_choice(("chunk", "tree"); name="PB_DOT"),
           "W7: the deterministic dot's order: chunk (the default) or tree (coalesced workgroup tree; rounding differs)"),
    PBKnob("PB_CPU_LAUNCH", "fixed", _parse_choice(("fixed", "balanced"); name="PB_CPU_LAUNCH"),
           "W7: the CPU's workgroup shape: fixed (256, the default) or balanced (small kernels split over every thread; bitwise)"),
    PBKnob("PB_SPEED", "off", _parse_choice(("off", "auto"); name="PB_SPEED"),
           "W7: off (the default: every speed knob as given); auto: every speed knob NOT set in the environment takes the value " *
           "adopted for the run's backend and case class by W7's registered measurements (PB_SPEED_TABLE, apply_speed_auto!)"),
    PBKnob("PB_SPEED_DEV", "0", s -> (s in ("0", "1") || throw(ArgumentError("PB_SPEED_DEV must be 0 or 1 (got \"$s\")")); s == "1"),
           "W7 (second review, R1): 1 acknowledges a DEVELOPMENT run on CUDA with PB_SPEED=auto or a non-default numerical speed " *
           "knob while the CUDA row is PROVISIONAL (refused without it; printed in the receipts; not for a registered reading)"),
    # ---- the free-air mesh remap (work item W4, 22 Oct 2026; src/pbcav/mesh_remap.jl). With PB_MESH_REMAP=none (the
    # default) the other knobs of this block are parsed (a malformed value is refused) but nothing uses them, so every
    # existing mesh and case is bitwise unchanged. -----------------------------------------------------------------------
    PBKnob("PB_MESH_REMAP", "none", _parse_choice(("none", "common"); name="PB_MESH_REMAP"),
           "none: the generator's mesh; common: free air only, the C columns' nodes re-placed along the parent's columns " *
           "(the wall rows kept, one shared spacing law, the wake's first spacing grown along the cut, optional stall layer)"),
    PBKnob("PB_MESH_REMAP_KEEP", "6,24", _parse_remap_keep,
           "remap: n_keep,n_blend -- rows 0..n_keep of every foil column are the parent's nodes bit for bit; the parent's rows " *
           "hand over to the shared law by row n_blend (smoothstep)"),
    PBKnob("PB_MESH_REMAP_LAW", "0,0,0", _parse_remap_law,
           "remap: rows,ratio,y_law -- the remapped NJ (0 the parent's; the multigrid needs 2 NJ <= 700), the shared law's " *
           "ratio (0 the foil's median realised ratio over the kept rows), its extent in chords (0: 5 c), beyond which each " *
           "column has its own outer section to the boundary; the wake columns merge into the shared rows below it"),
    PBKnob("PB_MESH_REMAP_WAKE", "500,750", _parse_remap_wake,
           "remap: ar_far,ar_outlet -- the wake columns' first spacing grows smoothly with the distance along the cut so the " *
           "first row's aspect ratio is about ar_far, capped at the outlet interval over ar_outlet; 0,0 keeps the parent's"),
    PBKnob("PB_MESH_REMAP_STALL", "", _parse_remap_stall,
           "remap: h,y[,f] -- the stall layer, uniform cells of height h (chords) from the inner growth to y chords from the wall " *
           "in every column, then growth again (the app's 2.5e-3,0.2; y below the law's extent); the wake's first spacing " *
           "is then capped at f h (default 0.4) so that every wake column merges inside the plateau; empty for none"),
    # W4b (22 Oct 2026, after W4's review): empty (the default) leaves W4's construction bit for bit.
    PBKnob("PB_MESH_REMAP_STRIP", "", _parse_remap_strip,
           "remap, strip topology: empty, the strip block untouched (W4); graded[,edge] -- the legs' first spacing grown by the " *
           "wake law from edge (default the corner columns' first spacing), the strip's cross distribution following the legs " *
           "(the base graded toward the corners) and the strip widened downstream along the legs' C columns"),
    # The merge of W4/W4b (22 Oct 2026; review of W4b, F3): empty (the default) leaves W4b's graded strip bit for bit.
    PBKnob("PB_MESH_REMAP_STRIP_CAP", "", _parse_remap_strip_cap,
           "remap, PB_MESH_REMAP_STRIP=graded only: empty, no cap (W4b); parent or a cap in chords -- the base's cross cells capped " *
           "(from the corner cells at the uncapped base's ratio, levelling off below the cap by a smooth order-2 switch; parent: " *
           "the parent's uniform base interval), the strip's cell count NS raised to the fewest that fill the base, and every " *
           "station of the strip given that NS: the same ratios from its edges, one switch parameter solved to fill it (tag V3)"),
    # Interior walls (24 Oct 2026; src/pbcav/baffle.jl): empty (the default) builds every mesh bit for bit as before.
    PBKnob("PB_BAFFLE", "", _parse_baffle,
           "interior wall: empty, none; gurney:side=lower|upper|symmetric,h=<h/c>[,snap=exact|nearest][,straighten=<deg>|off]" *
           "[,shear=<s>] -- a zero-thickness Gurney plate on the C column from the trailing-edge corner (C-meshes; free air), its " *
           "faces cut into walls on both sides (baffle.jl); prints PB BAFFLE"),
    # Free-edge refinement of the Gurney plate (30 Oct 2026; baffle.jl `baffle_refine`): empty (the default) changes nothing.
    PBKnob("PB_BAFFLE_REFINE", "", _parse_baffle_refine,
           "refinement of the Gurney plate's free edge and its shear layer: empty, none; tip[:q=<q>][,ratio=<r>][,wake=<c>|auto]" *
           "[,wake_taper=<c>|auto][,foil=<h>][,foil_taper=<h>][,keep=<rows>][,strict=yes|no] -- rows added around the plate's tip " *
           "(the tip cells 1/q of the mesh's own, growing back at <r> per cell) and along the grid row through the tip downstream " *
           "(needs PB_BAFFLE; baffle.jl); prints PB BAFFLE REFINE"),
]

"""
    speed_settings(vals) -> NamedTuple

W7's PBConfig keywords (plan Amendment 13): sync, line_solver, prec_precision, kw_precision, cpl_gs, graphs and dot (PB_CPU_LAUNCH is
not a PBConfig field: the driver applies it with `pb_cpu_launch!`). Their defaults are PBConfig's.
"""
speed_settings(vals::AbstractDict) = (sync=get(vals, "PB_SYNC", :all), line_solver=get(vals, "PB_LINE_SOLVER", :thomas),
                                      prec_precision=get(vals, "PB_PREC_PRECISION", :f64),
                                      kw_precision=get(vals, "PB_KW_PRECISION", :f64), cpl_gs=get(vals, "PB_CPL_GS", :mgs),
                                      graphs=get(vals, "PB_GRAPHS", :off), dot=get(vals, "PB_DOT", :chunk))

"""THE ADOPTED SPEED SETTINGS PER BACKEND AND CASE CLASS (W7, plan Amendment 13; registration air/w7_speed/W7_SPEED_REGISTRATION.md
section 4.5; the readings air/w7_speed/README.md). Keys: (backend, class), backend :cpu or :cuda, class :air (PB_DOMAIN=freeair)
or :water (the tunnel: N60R-type cases; the pressure cycle stays Float64 there by plan 5.2's precision rule, whose contrast test
the Float32 cycle failed). Values: the knob values `apply_speed_auto!` gives every speed knob the environment does not set.
Since the W7 review (22 Oct 2026, finding 4) `apply_speed_auto!` reads only the :air rows (the tunnel refuses the speed knobs
until stage P3 wires its run); the :water rows are the measured setting kept for that stage, and the Float32 preconditioner is
refused by the configuration and state checks wherever the density can be non-uniform, whatever the row.
W7 AMENDMENT 1 (22 Oct 2026, the review's finding 2; registration section 7): the GPU air row's Float32 preconditioner is
WITHDRAWN (it failed rule 4.3 and the 10% rule on CH in its old form, and its form has changed: finding 1); it returns only if
the corrected form passes the CUDA runs that amendment registers. The CUDA rows were PROVISIONAL (PB_SPEED_PROVISIONAL); the
(:cuda, :air) row's gate was lifted on 23 Oct 2026 (W7 README 7.16.10-7.16.11), the (:cuda, :water) row stays PROVISIONAL."""
const PB_SPEED_TABLE = Dict(
    (:cuda, :air)   => Dict("PB_SYNC" => :reads, "PB_GRAPHS" => :on, "PB_CPL_GS" => :cgs2, "PB_LINE_SOLVER" => :pcr,
                            "PB_PREC_PRECISION" => :f64, "PB_KW_PRECISION" => :f64, "PB_DOT" => :tree, "PB_CPU_LAUNCH" => :fixed),
    (:cuda, :water) => Dict("PB_SYNC" => :reads, "PB_GRAPHS" => :on, "PB_CPL_GS" => :cgs2, "PB_LINE_SOLVER" => :pcr,
                            "PB_PREC_PRECISION" => :f64, "PB_KW_PRECISION" => :f64, "PB_DOT" => :tree, "PB_CPU_LAUNCH" => :fixed),
    (:cpu, :air)    => Dict("PB_SYNC" => :all, "PB_GRAPHS" => :off, "PB_CPL_GS" => :mgs, "PB_LINE_SOLVER" => :thomas,
                            "PB_PREC_PRECISION" => :f64, "PB_KW_PRECISION" => :f64, "PB_DOT" => :chunk, "PB_CPU_LAUNCH" => :balanced),
    (:cpu, :water)  => Dict("PB_SYNC" => :all, "PB_GRAPHS" => :off, "PB_CPL_GS" => :mgs, "PB_LINE_SOLVER" => :thomas,
                            "PB_PREC_PRECISION" => :f64, "PB_KW_PRECISION" => :f64, "PB_DOT" => :chunk, "PB_CPU_LAUNCH" => :balanced))

"""The rows of PB_SPEED_TABLE that are PROVISIONAL (W7 Amendment 1, the review's finding 2), with the reason the driver prints:
a registered campaign may not use them before the CUDA checks owed by that amendment (and the minimum GPU check of the W7
README) have passed. ENFORCED since the second review (R1): while a row is here, `speed_gate_refusals` refuses that
backend's PB_SPEED=auto and non-default numerical speed knobs without PB_SPEED_DEV=1; the row is removed only in the commit
that records README 7.7 passing on the merged head.
THE (:cuda, :air) ROW IS REMOVED (23 Oct 2026, W7 README 7.16.10-7.16.11): README 7.7's minimum GPU check passed on the merged
head (pbcav d62a577f inside pbcav-speed 1ba14de7) under parity rule v3, with Amendment 1's CUDA runs, so CUDA free air admits
PB_SPEED=auto and the adopted row's knobs without PB_SPEED_DEV.
THE TWO ROWS THAT STAY (the merge review of cf838408, M1-M2; README 7.17):
  (:cuda, :f32)    the Float32 forms on CUDA free air: PB_PREC_PRECISION=f32 (withdrawn from the GPU air row by Amendment 1,
                   7.2 (a), until its CUDA runs pass) and PB_KW_PRECISION=f32 (never adopted, 7.1). What 7.7 passed is the
                   adopted row (Float64 preconditioner and k-omega), not these two knobs: they stay gated (SPEED_GATED_F32).
  (:cuda, :water)  every other CUDA configuration (the tunnel, water, any configuration not classified constant-density free
                   air: `speed_gate_class`, `config_gate_class`): never converged; every SPEED_GATED knob stays gated, at knob
                   time and on the configuration that runs, until stage P3."""
const PB_SPEED_PROVISIONAL = Dict(
    (:cuda, :f32)   => "PROVISIONAL (W7 Amendment 1: the Float32 preconditioner is withdrawn from the GPU air row until its CUDA runs pass, and kw_precision f32 was never adopted; the merge review's M1)",
    (:cuda, :water) => "PROVISIONAL (W7 Amendment 1: never converged; the tunnel refuses the speed knobs until stage P3)")

"The W7 speed knobs (PB_SPEED aside)."
const SPEED_KNOB_NAMES = ("PB_SYNC", "PB_LINE_SOLVER", "PB_PREC_PRECISION", "PB_KW_PRECISION", "PB_CPL_GS", "PB_GRAPHS",
                          "PB_DOT", "PB_CPU_LAUNCH")

"""THE PROVISIONAL GATE (the second W7 review's R1, 22 Oct 2026): the class-N speed knobs and their values that change no
arithmetic class N measures (the default; for PB_CPL_GS also the class-B :mgs_dev). While PB_SPEED_PROVISIONAL holds the
row of the run's class, a CUDA run with PB_SPEED=auto or with any of these knobs at another value is REFUSED unless
PB_SPEED_DEV=1 acknowledges a development run (printed in the receipts). The class-B knobs (PB_SYNC, PB_GRAPHS,
PB_CPL_GS=mgs_dev, PB_CPU_LAUNCH) are outside the gate: each is bit for bit the default (registration 3; CUDA with FMA off).
The gate is lifted only by emptying PB_SPEED_PROVISIONAL's row in the commit that records README 7.7's GPU check passing on
the merged head (for CUDA free air: done on 23 Oct 2026, W7 README 7.16.10-7.16.11, except the Float32 forms, which stay
gated through the (:cuda, :f32) row: SPEED_GATED_F32)."""
const SPEED_GATED = ("PB_LINE_SOLVER" => (:thomas,), "PB_CPL_GS" => (:mgs, :mgs_dev), "PB_DOT" => (:chunk,),
                     "PB_PREC_PRECISION" => (:f64,), "PB_KW_PRECISION" => (:f64,))
"The knobs the (:cuda, :f32) row keeps gated on CUDA free air after the lift (the merge review's M1)."
const SPEED_GATED_F32 = ("PB_PREC_PRECISION" => (:f64,), "PB_KW_PRECISION" => (:f64,))

"""
    speed_gate_row(be, cls) -> (row key, gated knobs) or nothing

The PROVISIONAL row that governs a run of class `cls` (:air, constant-density free air; :water, anything else) on backend
`be`, and the knobs it gates: the class's own row with every SPEED_GATED knob (and PB_SPEED=auto) while it is PROVISIONAL;
else, for free air, the backend's (be, :f32) row with SPEED_GATED_F32 while that is PROVISIONAL; else nothing (not gated).
"""
function speed_gate_row(be::Symbol, cls::Symbol)
    haskey(PB_SPEED_PROVISIONAL, (be, cls)) && return ((be, cls), SPEED_GATED)
    (cls === :air && haskey(PB_SPEED_PROVISIONAL, (be, :f32))) && return ((be, :f32), SPEED_GATED_F32)
    nothing
end
"The class of a knob set for the gate: :air for PB_DOMAIN=freeair (always constant density), :water for anything else."
speed_gate_class(vals::AbstractDict) = get(vals, "PB_DOMAIN", :tunnel) === :freeair ? :air : :water

"""
    speed_gate_refusals(vals) -> Vector{String}

The knobs the PROVISIONAL gate refuses for this run (empty when it may run): without PB_SPEED_DEV=1, on a backend and class
whose row is PROVISIONAL (`speed_gate_row`), every knob that row gates whose value is not one of its ungated values (and
PB_SPEED=auto when the class's own row is PROVISIONAL), however it was set. Applied by `apply_speed_auto!` (before the table
is applied) and again by `freeair_config` (after it), so a run that skips the first still meets the second.
"""
function speed_gate_refusals(vals::AbstractDict)
    get(vals, "PB_SPEED_DEV", false) && return String[]
    gr = speed_gate_row(get(vals, "PB_BACKEND", :cpu), speed_gate_class(vals))
    gr === nothing && return String[]
    key, gated = gr
    bad = String[]
    (gated === SPEED_GATED && get(vals, "PB_SPEED", :off) === :auto) && push!(bad, "PB_SPEED=auto")
    for (k, ok) in gated
        haskey(vals, k) && !(vals[k] in ok) && push!(bad, "$k=$(vals[k])")
    end
    bad
end
"Refuse (ArgumentError) what `speed_gate_refusals` lists."
function check_speed_gate(vals::AbstractDict)
    bad = speed_gate_refusals(vals)
    isempty(bad) && return nothing
    be = vals["PB_BACKEND"]
    key, _ = speed_gate_row(be, speed_gate_class(vals))
    throw(ArgumentError("$(join(bad, ", ")) on PB_BACKEND=$be is refused: " * PB_SPEED_PROVISIONAL[key] *
        ". Set PB_SPEED_DEV=1 to acknowledge a development run (printed in the receipts; such a run is not a registered " *
        "reading). The class-B knobs (PB_SYNC, PB_GRAPHS, PB_CPL_GS=mgs_dev, PB_CPU_LAUNCH) are not gated (the second W7 " *
        "review's R1)"))
end


"""
    apply_speed_auto!(vals, src) -> Vector{String}

With PB_SPEED=auto, give every speed knob that `src` (read_knobs's sources) marks :default the value of PB_SPEED_TABLE for
the run's backend (PB_BACKEND) and case class, and mark its source :auto (so `knob_receipt` prints the value that runs: the
W7 review's finding 3); a knob set in the environment is kept. Returns the knobs changed ("name=value"). With PB_SPEED=off,
nothing changes. The W7 review's corrections (22 Oct 2026):
  * THE CLASS (finding 4): :air is the free-air domain, today the only driver path and single-phase at uniform density; the
    Float32 preconditioner is ALSO refused, whatever the class or how it was set, by `check_speed_config` (a configuration
    not classified density-neutral) and at every step by `check_f32_state` (a state whose density is not uniform), so a later
    free-air cavitating case cannot inherit it silently.
  * THE TUNNEL (finding 4): the tunnel domain's run does not consume the speed knobs (its PB_STOP=run is stage P3's), so any
    speed knob set there, and PB_SPEED=auto, is REFUSED here rather than announced and ignored.
  * GRAPHS (finding 9): the table's PB_GRAPHS=on is applied only when the resolved PB_SYNC is reads (a PB_SYNC=all from the
    environment keeps graphs off, reported in the returned list), so auto cannot produce the refused combination.
  * THE PROVISIONAL GATE (the second review's R1): `speed_gate_refusals` -- on CUDA, while its row is PROVISIONAL, auto and
    every non-default numerical speed knob are REFUSED without PB_SPEED_DEV=1. Checked here, before anything is applied, so
    the driver refuses at knob time.
"""
function apply_speed_auto!(vals::AbstractDict, src::AbstractDict)
    auto = get(vals, "PB_SPEED", :off) === :auto
    if get(vals, "PB_DOMAIN", :tunnel) !== :freeair
        set = sort([k for k in (SPEED_KNOB_NAMES..., "PB_SPEED_DEV") if get(src, k, :default) === :env])
        (auto || !isempty(set)) && throw(ArgumentError("the tunnel domain applies no W7 speed knob (its run is stage P3's and " *
            "does not read them): unset " * join(vcat(auto ? ["PB_SPEED=auto"] : String[], set), ", ") *
            " (the W7 review's finding 4: refused rather than announced and ignored)"))
    end
    check_speed_gate(vals)
    auto || return String[]
    tab = PB_SPEED_TABLE[(vals["PB_BACKEND"], :air)]; out = String[]
    sync = get(src, "PB_SYNC", :default) === :default ? tab["PB_SYNC"] : vals["PB_SYNC"]
    for (k, v) in sort(collect(tab); by=first)
        get(src, k, :default) === :default || continue
        if k == "PB_GRAPHS" && v === :on && sync !== :reads
            push!(out, "PB_GRAPHS=off(kept: PB_SYNC=$sync, graphs need reads)")
            continue
        end
        vals[k] === v || push!(out, "$k=$v")
        vals[k] = v; src[k] = :auto
    end
    out
end

"""
    turbulence_settings(vals) -> NamedTuple

The PBConfig keywords of the turbulence knobs (`PBConfig(; turbulence_settings(vals)..., ...)`): turbulence, sst_tu,
sst_mut_ratio and wall_omega (:auto is accepted here and refused loudly when the work space is built, stage P2), and since the
W1+W2 integration (re-review of W1b, R1 d) turb_coupling, turb_tol, turb_step_tol, turb_nonorth and turb_outer_ramp, and since
W6 (22 Oct 2026) turb_mut, turb_omega_nonorth and turb_outer_tol; since 27 Oct 2026 turb_mut_base (PBConfig's default
:extrapolated when the dictionary has no PB_TURB_MUT_BASE).
"""
turbulence_settings(vals::AbstractDict) = (turbulence=vals["PB_TURBULENCE"], sst_tu=vals["PB_SST_TU_INF"],
                                           sst_mut_ratio=vals["PB_SST_MUT_RATIO"], wall_omega=vals["PB_WALL_OMEGA"],
                                           turb_coupling=vals["PB_TURB_COUPLING"], turb_tol=vals["PB_TURB_TOL"],
                                           turb_step_tol=vals["PB_TURB_STEP_TOL"], turb_nonorth=vals["PB_TURB_NONORTH"],
                                           turb_outer_ramp=vals["PB_TURB_OUTER_RAMP"], turb_mut=vals["PB_TURB_MUT"],
                                           turb_mut_base=get(vals, "PB_TURB_MUT_BASE", :extrapolated),
                                           turb_omega_nonorth=vals["PB_TURB_OMEGA_NONORTH"], turb_outer_tol=vals["PB_TURB_OUTER_TOL"],
                                           turb_omega_bound=vals["PB_TURB_OMEGA_BOUND"])
const _PB_KNOB_NAMES = Set(k.name for k in PB_KNOBS)
const _V1_PREFIXES = ("TOPO_", "TEM_", "CAV_MOVIE_")

"""
    read_knobs(env=ENV) -> (values::Dict{String,Any}, source::Dict{String,Symbol})

Every registered knob's parsed value, from `env` or its default (`source` says which). Refuses an unregistered PB_*
variable and any version-1 knob (TOPO_*, TEM_*, CAV_MOVIE_*) in `env`.
"""
function read_knobs(env::AbstractDict=ENV)
    unknown = sort([k for k in keys(env) if startswith(k, "PB_") && !(k in _PB_KNOB_NAMES)])
    isempty(unknown) || throw(ArgumentError("unregistered PB_ knob(s) $(join(unknown, ", ")): version 2 reads only the " *
        "knobs of src/pbcav/knobs.jl (PB_KNOBS); a knob that nothing reads is refused"))
    v1 = sort([k for k in keys(env) if any(p -> startswith(k, p), _V1_PREFIXES)])
    isempty(v1) || throw(ArgumentError("version-1 knob(s) $(join(v1, ", ")) in the environment: version 2 reads none of " *
        "them (its knobs are PB_*); unset them before running tools/pbcav_driver.jl"))
    vals = Dict{String,Any}(); src = Dict{String,Symbol}()
    for k in PB_KNOBS
        raw = get(env, k.name, nothing)
        vals[k.name] = k.parse(raw === nothing ? k.default : String(raw))
        src[k.name] = raw === nothing ? :default : :env
    end
    # W6b (the review of W6b, F12; W7's convention, its review's finding 4): the free-air run's coupled-step and bound knobs are
    # REFUSED in the tunnel domain, whose run (stage P3) reads none of them, rather than printed in PB KNOBS and ignored
    if vals["PB_DOMAIN"] !== :freeair
        set = [n for n in FREEAIR_ONLY_KNOBS if src[n] === :env]
        isempty(set) || throw(ArgumentError("the tunnel domain applies no free-air-only knob (FREEAIR_ONLY_KNOBS: W6b's, " *
            "PB_TURB_MUT_BASE, the baffle's and the ramp start's; its run is stage P3's and does not read them): unset " *
            "$(join(set, ", ")), or set PB_DOMAIN=freeair (the review of W6b, F12: refused rather than announced and ignored)"))
    end
    # the ramp's parameters are read only by the ramp start (the baffle review's F4, 27 Oct 2026): refused rather than ignored
    src["PB_START_RAMP"] === :env && vals["PB_START"] !== :ramp &&
        throw(ArgumentError("PB_START_RAMP is read only by PB_START=ramp (got PB_START=$(vals["PB_START"])): unset it, or set " *
                            "PB_START=ramp (refused rather than announced and ignored)"))
    # the plate's refinement acts on the plate (30 Oct 2026): without PB_BAFFLE there is nothing to refine
    vals["PB_BAFFLE_REFINE"] !== nothing && vals["PB_BAFFLE"] === nothing &&
        throw(ArgumentError("PB_BAFFLE_REFINE refines the Gurney plate's tip and shear layer and needs PB_BAFFLE (got PB_BAFFLE empty): " *
                            "unset it, or set PB_BAFFLE=gurney:side=...,h=... (refused rather than announced and ignored)"))
    vals, src
end
"The W6b knobs only the free-air run reads (`freeair_config`), refused in the tunnel domain by `read_knobs`."
const W6B_FREEAIR_ONLY_KNOBS = ("PB_TURB_OMEGA_BOUND", "PB_CPL_AU", "PB_CPL_VISC_NONORTH")
"""Every knob only the free-air run reads, refused in the tunnel domain by `read_knobs`: W6b's, PB_TURB_MUT_BASE (27 Oct 2026,
records air/mutbase/), and PB_BAFFLE and PB_START_RAMP (27 Oct 2026, the baffle review's F4: the plate's loads and the ramp
start belong to the free-air run). (The merge of pbcav-baffle into pbcav, 27 Oct 2026: the union of both branches' lists.)"""
const FREEAIR_ONLY_KNOBS = (W6B_FREEAIR_ONLY_KNOBS..., "PB_TURB_MUT_BASE", "PB_BAFFLE", "PB_START_RAMP", "PB_BAFFLE_REFINE")

"""
    mesh_call(vals, src=nothing) -> CMeshCall

The version-1 generator call the mesh knobs describe (the version-1 driver's construction: the trailing-edge keywords
only when a base is set; the strip topology needs a base of at least 2 cells per half).
FREE AIR (PB_DOMAIN=freeair; W2, 21 Oct 2026): the channel is PB_FARFIELD_H/XMIN/XMAX at PB_FARFIELD_ANGLE (the foil is not
rotated; PB_ALPHA is the freestream's angle and is not a mesh input), and a sharp edge (PB_MESH_TE_BASE=0) is the plain C-mesh
whatever PB_MESH_TE_TOPOLOGY says (the topology is a property of a base; without one there is nothing to build a strip from).
In free air the tunnel's trailing-edge clustering defaults (PB_MESH_TE_DS 5.62e-4, PB_MESH_WAKE_NEAR) reach the generator
ONLY WHEN SET in the environment (`src`, read_knobs's sources; the W1+W2 integration, review of W2 R8 d: before, with
PB_MESH_TE_BASE=0 they silently clustered the trailing edge, or made the generator refuse the call when dn1 > te_ds), so a free
-air call needs `src` (ArgumentError without it).
The tunnel domain is unchanged: there PB_MESH_TE_TOPOLOGY=strip with a base below 2 cells is refused, as before.
"""
function mesh_call(vals::AbstractDict, src=nothing)
    m = vals["PB_MESH"]; nb = vals["PB_MESH_TE_BASE"]; tds = vals["PB_MESH_TE_DS"]; wn = vals["PB_MESH_WAKE_NEAR"]
    topo = vals["PB_MESH_TE_TOPOLOGY"]
    fa = get(vals, "PB_DOMAIN", :tunnel) === :freeair
    if fa
        src === nothing && throw(ArgumentError("mesh_call: a free-air call needs the knobs' sources (mesh_call(vals, src), " *
            "src from read_knobs): the tunnel's trailing-edge defaults PB_MESH_TE_DS and PB_MESH_WAKE_NEAR enter only when set"))
        get(src, "PB_MESH_TE_DS", :default) === :env || (tds = 0.0)
        get(src, "PB_MESH_WAKE_NEAR", :default) === :env || (wn = nothing)
    end
    fa && nb == 0 && (topo = :aplus)
    topo === :strip && nb < 2 && throw(ArgumentError("PB_MESH_TE_TOPOLOGY=strip needs PB_MESH_TE_BASE >= 2 (got $nb)"))
    te = (; n_base=nb)
    tds > 0 && (te = merge(te, (te_ds=tds,)))
    wn === nothing || (te = merge(te, wn))
    nb > 0 && topo === :strip && (te = merge(te, (te_topology=:strip,)))
    te = nb == 0 && length(te) == 1 ? nothing : te
    H, xin, xout, al = fa ? (vals["PB_FARFIELD_H"], vals["PB_FARFIELD_XMIN"], vals["PB_FARFIELD_XMAX"], vals["PB_FARFIELD_ANGLE"]) :
                            (vals["PB_CHANNEL_H"], vals["PB_CHANNEL_XMIN"], vals["PB_CHANNEL_XMAX"], vals["PB_ALPHA"])
    CMeshCall(; foil=vals["PB_FOIL"], n_foil=m.n_foil, n_normal=m.n_normal, dn1=m.dn1, growth=m.growth, smooth=m.smooth,
              n_wake=m.n_wake, le_ds=vals["PB_MESH_LE_DS"], te=te, H=H, xin=xin, xout=xout, alpha=al)
end

"""The knobs the receipt line (PB KNOBS) prints ONLY WHEN THEIR VALUE IS NOT THE DEFAULT, so the receipt of every run that
leaves them at the default is, byte for byte, the receipt from before they existed: PB_TURB_MUT_BASE (27 Oct 2026, records
air/mutbase/; the A2 campaign's receipts are compared as written), and PB_BAFFLE and PB_START_RAMP (27 Oct 2026, the baffle
review's F4; records air/baffle/). (The merge of pbcav-baffle into pbcav, 27 Oct 2026: the union of both branches' lists.)"""
const RECEIPT_QUIET_AT_DEFAULT = ("PB_TURB_MUT_BASE", "PB_BAFFLE", "PB_START_RAMP", "PB_BAFFLE_REFINE")

"""The knob receipt line: every registered knob, its value as read, and whether it came from the environment; a knob that
PB_SPEED=auto set (source :auto, `apply_speed_auto!`) is printed with the value it set and "(auto)" (the W7 review's
finding 3: before, the receipt printed the default for it). A knob of RECEIPT_QUIET_AT_DEFAULT is left out while its value is
its default's (wherever that value came from)."""
function knob_receipt(vals::AbstractDict, src::AbstractDict, env::AbstractDict=ENV)
    parts = String[]
    for k in PB_KNOBS
        (k.name in RECEIPT_QUIET_AT_DEFAULT && isequal(vals[k.name], k.parse(k.default))) && continue
        if src[k.name] === :auto
            push!(parts, string(k.name, "=", vals[k.name], "(auto)"))
        else
            raw = get(env, k.name, k.default)
            push!(parts, string(k.name, "=", raw, src[k.name] === :default ? "(default)" : ""))
        end
    end
    "PB KNOBS: " * join(parts, " ")
end
