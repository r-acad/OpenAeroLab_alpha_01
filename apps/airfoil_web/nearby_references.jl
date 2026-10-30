module AirfoilNearbyReferences

# Wind-tunnel lift at NEARBY conditions, drawn on the CL polar of jobs with the low-speed CFD (23 Oct 2026). Never a matched
# reference: the conditions differ (Re, Mach, transition), so each series carries its own conditions and "digitized,
# approximate". The matched-reference mechanism (reference_data.jl, reference_for_case) is not used or changed.

using SHA

const DATA_DIR = joinpath(normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
    joinpath(@__DIR__, "..", "..", "..", "02_PRIVATE_PROJECT_DEVELOPMENT"))),
    "apps", "airfoil_web", "validation", "data", "naca0012_wind_tunnel_cl")
const SETS = [
    (file="CL_Gregory_expdata.dat", sha256="0b7381a45412de50fcf0e0dddf67f1e8e35ba1a278bd4734fb7f16ad08df1f74",
     name="Gregory & O'Reilly wind tunnel (ARC R&M 3726)", reynolds=2.88e6, mach=0.16,
     conditions="Re 2.88M, M 0.16, free transition", key="gregory_oreilly_1970"),
    (file="0012.abbottdata.cl.dat", sha256="3ee6a0c198f6402c9fc5fd8affc1115c1e87ff943b8b02775df86a7cfa3c55f3",
     name="Abbott & von Doenhoff wind tunnel", reynolds=6e6, mach=nothing,
     conditions="Re 6M", key="abbott_von_doenhoff"),
]

function _points(path)
    pts = Dict{String,Any}[]
    for line in eachline(path)
        v = split(strip(line))
        length(v) == 2 || continue
        a = tryparse(Float64, v[1]); c = tryparse(Float64, v[2])
        (a === nothing || c === nothing) && continue
        push!(pts, Dict{String,Any}("alpha_deg"=>a, "cl"=>c))
    end
    pts
end

"Nearby-condition wind-tunnel CL series for a job's request (airfoil, backends), or an empty list."
function nearby_experiments(request)
    request isa AbstractDict || return Any[]
    foil = get(request, "airfoil", Dict()); backends = get(request, "backends", String[])
    (get(foil, "kind", "") == "naca4" && get(foil, "id", "") == "0012" && "lowspeed" in backends) || return Any[]
    # A Gurney flap changes the section: the clean-section tunnel lift is not drawn (23 Oct 2026).
    flow = get(request, "flow", Dict())
    flow isa AbstractDict && get(flow, "gurney_height", nothing) !== nothing && return Any[]
    out = Any[]
    for s in SETS
        path = joinpath(DATA_DIR, s.file)
        isfile(path) && bytes2hex(open(sha256, path)) == s.sha256 || continue
        push!(out, Dict{String,Any}("key"=>s.key, "name"=>s.name, "reynolds"=>s.reynolds, "mach"=>s.mach,
            "conditions"=>s.conditions, "quantity"=>"cl", "role"=>"nearby-condition experiment (not matched)",
            "note"=>"$(s.name), $(s.conditions); digitized by NASA TMR, approximate. Nearby conditions, not a matched reference.",
            "source_file"=>s.file, "sha256"=>s.sha256, "points"=>_points(path)))
    end
    out
end

end
