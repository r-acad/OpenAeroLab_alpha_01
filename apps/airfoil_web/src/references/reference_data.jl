module AirfoilReferenceData

export reference_for_case, benchmark_catalog

const APP = normpath(joinpath(@__DIR__, "..", ".."))
const ROOT = normpath(joinpath(APP, "..", ".."))
const PRIVATE_ROOT = normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
    joinpath(ROOT, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
const DATA = joinpath(APP, "data", "references")
const MACH_TOL = 5.0e-4
const RE_REL_TOL = 1.0e-4
const ALPHA_TOL = 5.0e-3
const TRIP_TOL = 1.0e-4

_point(x, cp) = Dict{String,Float64}("x"=>Float64(x), "cp"=>Float64(cp))

function _ordered_cp(upper, lower)
    sort!(upper; by=p->p["x"])
    sort!(lower; by=p->p["x"])
    Dict{String,Any}("upper"=>upper, "lower"=>lower)
end

function _rae_case9_cp()
    path = joinpath(DATA, "rae2822_case9_exp_full.csv")
    upper = Dict{String,Float64}[]
    lower = Dict{String,Float64}[]
    for line in eachline(path)
        startswith(line, "#") && continue
        fields = split(strip(line), ',')
        length(fields) == 3 || continue
        target = fields[1] == "upper" ? upper : fields[1] == "lower" ? lower : nothing
        target === nothing && continue
        push!(target, _point(parse(Float64,fields[2]), parse(Float64,fields[3])))
    end
    length(upper) == 52 && length(lower) == 50 ||
        error("RAE 2822 AGARD Case 9 Cp tap count changed")
    _ordered_cp(upper,lower)
end

function _rae_case6_cp()
    path = joinpath(DATA, "rae2822_case6_cp.gen")
    rows = Tuple{Float64,Float64}[]
    for line in eachline(path)
        fields = split(strip(line))
        length(fields) == 2 || continue
        x = tryparse(Float64,fields[1])
        cp_plot = tryparse(Float64,fields[2])
        (x === nothing || cp_plot === nothing) && continue
        push!(rows,(x,cp_plot))
    end
    length(rows) == 103 || error("RAE 2822 NASA Case 6 Cp tap count changed")
    # NASA's GENPLOT file contains -Cp for plotting suction upward (see Study #4).
    ile = findfirst(row->row[1] == 0.0, rows)
    ile === nothing && error("RAE 2822 NASA Cp leading-edge split missing")
    upper = [_point(x,-cp_plot) for (x,cp_plot) in rows[1:ile]]
    lower = [_point(x,-cp_plot) for (x,cp_plot) in rows[ile+1:end]]
    _ordered_cp(upper,lower)
end

function _oat_cp()
    path = joinpath(DATA, "onera_oat15a_cp.dat")
    zones = Dict{Float64,Dict{String,Any}}()
    angle = nothing
    rows = Tuple{Float64,Float64}[]
    function finish_zone!()
        angle === nothing && return
        ile = findlast(row -> row[1] >= 0.9999, rows)
        ile === nothing && error("ONERA OAT15A Cp surface split missing")
        upper = [_point(x,cp) for (x,cp) in rows[1:ile]]
        lower = [_point(x,cp) for (x,cp) in rows[ile+1:end]]
        length(upper) >= 40 && length(lower) >= 20 ||
            error("ONERA OAT15A Cp taps incomplete at alpha=$angle")
        zones[angle] = _ordered_cp(upper,lower)
    end
    for line in eachline(path)
        s = strip(line)
        if startswith(s,"ZONE")
            finish_zone!()
            m = match(r"M = 0\.73 - a = ([0-9.]+)",s)
            m === nothing && error("unrecognized ONERA OAT15A Cp zone")
            angle = parse(Float64,m.captures[1])
            empty!(rows)
        elseif angle !== nothing
            fields = split(s)
            length(fields) == 2 || continue
            push!(rows,(parse(Float64,fields[1]),parse(Float64,fields[2])))
        end
    end
    finish_zone!()
    sort!(collect(keys(zones))) == [2.5,3.0,3.1,3.25,3.5,3.9] ||
        error("ONERA OAT15A experimental Cp zones changed")
    zones
end

const RAE_CASE9_CP = isfile(joinpath(DATA,"rae2822_case9_exp_full.csv")) ? _rae_case9_cp() : nothing
const RAE_CASE6_CP = isfile(joinpath(DATA,"rae2822_case6_cp.gen")) ? _rae_case6_cp() : nothing
const OAT_CP = isfile(joinpath(DATA,"onera_oat15a_cp.dat")) ? _oat_cp() : nothing

function _flow_matches(flow, mach, reynolds; trip=nothing)
    m = get(flow,"mach",NaN)
    re = get(flow,"reynolds",NaN)
    m isa Real && re isa Real || return false
    abs(Float64(m)-mach) <= MACH_TOL || return false
    abs(Float64(re)-reynolds) <= RE_REL_TOL*reynolds || return false
    if trip !== nothing
        for side in ("upper","lower")
            t = get(flow,"trip_"*side,side == "upper" ? 0.03 : 0.05)
            t isa Real && abs(Float64(t)-trip) <= TRIP_TOL || return false
        end
    end
    true
end

function _selected_points(alphas, available)
    [Dict{String,Any}("alpha_deg"=>a, "cp"=>cp, "cl"=>nothing,
                      "cd"=>nothing, "cm"=>nothing)
        for (a,cp) in available
        if any(requested -> abs(Float64(requested)-a) <= ALPHA_TOL, alphas)]
end

"""
    reference_for_case(case)

Return experimental surface-pressure data only when the published reference
Mach, Reynolds number, angle, and (when reported) trip locations match the
submitted case. RAE2822 Case 9 also has published experimental CL/CD; other
force values remain `nothing`. Cp is dimensionless with the usual aerodynamic
sign (suction negative).
"""
function reference_for_case(case::AbstractDict)
    foil = get(case,"airfoil",Dict())
    flow = get(case,"flow",Dict())
    alphas = get(case,"alphas",Float64[])
    kind = get(foil,"kind","")
    if kind == "naca4" && get(foil,"id","")=="0012" && _flow_matches(flow,0.15,6e6) &&
       !haskey(flow,"gurney_height") && get(flow,"cfd_te_thickness",0)==0
        points=Dict{String,Any}[];active=false
        for line in eachline(joinpath(DATA,"naca0012_ladson.dat"))
            startswith(line,"zone") && active && break
            startswith(line,"zone") && (active=occursin("80 grit",line);continue)
            active||continue
            a,cl,cd=parse.(Float64,split(strip(line)))
            push!(points,Dict("alpha_deg"=>a,"cl"=>cl,"cd"=>cd,"cm"=>nothing,"cp"=>Dict()))
        end
        return Dict{String,Any}("name"=>"NACA 0012 · Ladson 80-grit trip",
            "source_url"=>"https://tmbwg.github.io/turbmodels/naca0012_val.html",
            "data_url"=>"https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat",
            "note"=>"NASA TM 4074: M=0.15, Re=6 million, 80-grit transition trip. Experimental CL/CD polar; no Cp or CM is supplied by this dataset. The pressure solver is incompressible and fully turbulent: Mach and trip matching are approximations, not an exact model of the tunnel experiment. Steady RANS near stall can be branch dependent.",
            "conditions"=>Dict("mach"=>.15,"reynolds"=>6e6,"alpha_tolerance_deg"=>ALPHA_TOL),
            "points"=>points)
    end
    (haskey(flow,"gurney_height") || get(flow,"cfd_te_thickness",0)>0 ||
     get(flow,"cfd_fully_turbulent",false) === true || get(flow,"cfd_transition_model","") == "fully_turbulent") && return nothing
    if kind == "rae2822"
        cases = Dict{String,Any}[]
        if RAE_CASE9_CP !== nothing && _flow_matches(flow,0.734,6.5e6; trip=0.03)
            pts = _selected_points(alphas, [(2.79,RAE_CASE9_CP)])
            if !isempty(pts)
                # Experimental forces from Veldman, AIAA 2005-4801, Table 2;
                # the tunnel angle 3.19 deg maps to corrected free-air 2.79 deg.
                pts[1]["cl"] = 0.803
                pts[1]["cd"] = 0.0168
            end
            isempty(pts) || push!(cases, Dict{String,Any}(
                "name"=>"RAE 2822 · AGARD Case 9 (corrected free air)",
                "source_url"=>"https://cfd.ku.edu/hiocfd/case_c2.2.html",
                "data_url"=>"https://apps.dtic.mil/sti/tr/pdf/ADA073982.pdf",
                "force_source_url"=>"https://pure.rug.nl/ws/portalfiles/portal/2914855/2005PaperAIAAVeldman.pdf",
                "note"=>"Experimental Cp from AGARD AR-138 Table 6.7; wind-tunnel readings at M=0.730, alpha=3.19°. DLR corrected free-air CFD conditions are M=0.734, alpha=2.79°, Re=6.5e6, trips at 3% on both sides.",
                "conditions"=>Dict("mach"=>0.734,"reynolds"=>6.5e6,
                    "trip_upper"=>0.03,"trip_lower"=>0.03,
                    "mach_tolerance"=>MACH_TOL,"reynolds_relative_tolerance"=>RE_REL_TOL,
                    "alpha_tolerance_deg"=>ALPHA_TOL,"trip_tolerance"=>TRIP_TOL),
                "points"=>pts))
        end
        if RAE_CASE6_CP !== nothing && _flow_matches(flow,0.729,6.5e6)
            pts = _selected_points(alphas, [(2.31,RAE_CASE6_CP)])
            isempty(pts) || push!(cases, Dict{String,Any}(
                "name"=>"RAE 2822 · AGARD Case 6 (NASA corrected conditions)",
                "source_url"=>"https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf04/raetaf04.html",
                "data_url"=>"https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf04/cp.exp.gen",
                "note"=>"Experimental Cp from NASA NPARC, shown at the archive's CFD conditions M=0.729, alpha=2.31°, Re=6.5e6. Original wind-tunnel settings were M=0.725, alpha=2.92°. Transition matching is not asserted by this reference.",
                "conditions"=>Dict("mach"=>0.729,"reynolds"=>6.5e6,
                    "mach_tolerance"=>MACH_TOL,"reynolds_relative_tolerance"=>RE_REL_TOL,
                    "alpha_tolerance_deg"=>ALPHA_TOL),
                "points"=>pts))
        end
        isempty(cases) && return nothing
        return first(cases)
    elseif kind == "oat15a" && OAT_CP !== nothing && _flow_matches(flow,0.73,3.0e6; trip=0.07)
        available = [(a,OAT_CP[a]) for a in sort!(collect(keys(OAT_CP)))]
        pts = _selected_points(alphas,available)
        isempty(pts) && return nothing
        return Dict{String,Any}(
            "name"=>"ONERA OAT15A · Jacquin et al. wind tunnel Cp",
            "source_url"=>"https://www.aiaa-dpw.org/experiment.html#oat",
            "data_url"=>"https://dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/Experimental_Data/onera-oat15a.zip",
            "note"=>"ONERA/AIAA DPW mean experimental Cp at M=0.73, Re=3.0e6, trips at 7% on both sides. The DPW extended polar from alpha=3.25 degrees requires unsteady analysis; compare time-averaged CFD with the measured mean. ONERA retains intellectual property; acknowledge ONERA and pass on the sharing conditions in apps/airfoil_web/data/references/README.md.",
            "conditions"=>Dict("mach"=>0.73,"reynolds"=>3.0e6,
                "trip_upper"=>0.07,"trip_lower"=>0.07,
                "mach_tolerance"=>MACH_TOL,"reynolds_relative_tolerance"=>RE_REL_TOL,
                "alpha_tolerance_deg"=>ALPHA_TOL,"trip_tolerance"=>TRIP_TOL),
            "points"=>pts)
    end
    nothing
end

include("benchmark_presets.jl")

end # module
