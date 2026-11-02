# Included in AirfoilReferenceData. These are physical comparison conditions;
# iteration counts and meshes are application choices, not experimental data.
function benchmark_catalog()
    items = Dict{String,Any}[]
    definitions = [("rae2822-case9", "rae2822", "RAE 2822 · AGARD Case 9", 0.734, 6.5e6, 2.79, 0.03, "constant_dynamic", 288.15),
                   ("rae2822-case6", "rae2822", "RAE 2822 · NASA Case 6", 0.729, 6.5e6, 2.31, 0.03, "sutherland", 460*5/9)]
    append!(definitions, [("oat15a-$(a)", "oat15a", "ONERA OAT15A · α=$(a)°", 0.73, 3.0e6, a, 0.07, "sutherland", 271.0)
                          for a in (2.5,3.0,3.1,3.25,3.5,3.9)])
    for (id,kind,label,mach,re,alpha,trip,viscosity,temp) in definitions
        unsteady = kind == "oat15a" && alpha >= 3.25
        flow = Dict{String,Any}("mach"=>mach,"reynolds"=>re,"alpha_deg"=>alpha,"ncrit"=>9.0,
            "trip_upper"=>trip,"trip_lower"=>trip,"cfd_viscosity_model"=>viscosity,
            "cfd_freestream_temperature_k"=>temp,"cfd_te_topology"=>"aplus",
            "cfd_mesh"=>"aplus_nested_refined","cfd_resolution"=>"fine",
            "cfd_gpu"=>true,"cfd_time_mode"=>unsteady ? "physical" : "steady")
        case = Dict{String,Any}("airfoil"=>Dict("kind"=>kind,"id"=>kind),"flow"=>flow,
            "backends"=>["transonic"],"sweep"=>Dict("enabled"=>false),"alphas"=>[alpha])
        reference = reference_for_case(case)
        reference === nothing && error("Bundled reference is missing for $id")
        note = kind == "rae2822" ? (id == "rae2822-case9" ?
            "Corrected free-air conditions; tunnel readings used M=0.730 and α=3.19°. Constant dynamic viscosity; the reference temperature is immaterial for this nondimensional case." :
            "NASA corrected free-air conditions; tunnel readings used M=0.725 and α=2.92°. T∞=460°R. NASA does not prescribe matching trip locations; the 3% trips are an application choice.") :
            "M=0.73, Re=3 million, T∞=271 K, experimental trips at 7% chord on both sides. " *
            (unsteady ? "DPW V1b unsteady case: compare time-averaged Cp after burn-in; one instantaneous field is not the measured mean." : "DPW V1a steady comparison point.")
        push!(items, Dict("id"=>id,"kind"=>kind,"label"=>label,"case"=>case,"reference"=>reference,
            "default"=>(id in ("rae2822-case9","oat15a-2.5")),"unsteady"=>unsteady,"note"=>note,
            "numerics_note"=>"Fine A-plus mesh retains the native profile and prescribed trips/viscosity. SST, mesh resolution, time step, run duration and stopping criteria are numerical choices; convergence and time/mesh independence must be checked.",
            "conditions_url"=>kind=="oat15a" ? "https://www.aiaa-dpw.org/TestCases/testcases.html" : reference["source_url"]))
    end
    flow=Dict{String,Any}("mach"=>.15,"reynolds"=>6e6,"alpha_deg"=>10.0,"ncrit"=>9.0,
        "trip_upper"=>.05,"trip_lower"=>.05,"cfd_viscosity_model"=>"constant_kinematic",
        "cfd_freestream_temperature_k"=>288.15,"cfd_time_mode"=>"steady",
        "lowspeed_resolution"=>"fine","lowspeed_gpu"=>true,"lowspeed_max_steps"=>12000,
        "lowspeed_wall_cap_s"=>7200.0,"lowspeed_tu"=>0.001,"lowspeed_mut_ratio"=>0.1)
    case=Dict{String,Any}("airfoil"=>Dict("kind"=>"naca4","id"=>"0012"),"flow"=>flow,
        "backends"=>["lowspeed"],"sweep"=>Dict("enabled"=>true,"start"=>10.0,"stop"=>18.0,"step"=>2.0),
        "alphas"=>collect(10.0:2.0:18.0))
    push!(items,Dict("id"=>"naca0012-stall","kind"=>"naca4","label"=>"NACA 0012 · NASA stall polar",
        "case"=>case,"reference"=>reference_for_case(case),"default"=>true,"unsteady"=>false,
        "note"=>"Ladson 80-grit trip, M=0.15, Re=6 million; angles 10, 12, 14, 16 and 18 degrees bracket stall.",
        "numerics_note"=>"GPU pressure-based SST solver, Fine mesh, fully turbulent and incompressible approximation. Finite-Mach, trip, mesh and unsteady effects remain comparison limitations; a force plateau does not prove a steady flow exists.",
        "conditions_url"=>"https://tmbwg.github.io/turbmodels/naca0012_val.html"))
    items
end
