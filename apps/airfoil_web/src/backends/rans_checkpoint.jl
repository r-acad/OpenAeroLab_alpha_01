# Checked, atomic steady-RANS state transfer between adjacent polar points.
# The mesh artifact hash covers every node and the topology. A different mesh
# starts cold; interpolation across RANS grids needs a separate validation.
using Serialization
Base.include(@__MODULE__,joinpath(@__DIR__,"rans_source_snapshot.jl"))

const RANS_RESTART_FIELDS = (:rho, :ux, :uy, :T, :tk, :tom, :nut, :tnl)

function _rans_restart_identity(cfg, mesh_path, topology)
    keys = ("mach", "reynolds", "time_mode", "viscosity_model", "mesh_mode",
            "trip_upper", "trip_lower", "fully_turbulent", "irsj",
            "sst_wall_metric_correction", "sst_nonorth_correction",
            "refresh_flow_bc_before_sst", "characteristic_i_outlet",
            "rk_weight_closure", "tu_inf", "mu_t_ratio", "freestream_temperature_k")
    # Solver hashes belong to the worker's validated file-load cohort. Editing
    # disk after startup cannot relabel state evolved by the already-loaded code.
    merge(_rans_restart_loaded_source_identity(),Dict{String,Any}(
        "format"=>2, "topology"=>topology,
        "source_provenance"=>_RANS_RESTART_SOURCE_PROVENANCE,
        "mesh_sha256"=>bytes2hex(open(sha256, mesh_path)),
        "settings"=>Dict(key=>(key=="sst_wall_metric_correction" ? Bool(get(cfg,key,true)) : get(cfg,key,nothing)) for key in keys)))
end

function _rans_restart_physical(src, field, solver)
    interior = @view src[solver.g+1:solver.g+solver.NI, solver.g+1:solver.g+solver.NJ]
    field in (:rho, :T) && return all(>(0), interior)
    field === :tom && return all(solver.sst ? >(0) : >=(0), interior)
    field in (:tk, :nut, :tnl) && return all(>=(0), interior)
    true
end

function _rans_restart_arrays(blocks)
    arrays = Dict{String,Any}(string(block.name, "/", field)=>Array(getfield(block.solver, field))
        for block in blocks for field in RANS_RESTART_FIELDS)
    for block in blocks
        hasproperty(block, :extras) || continue
        for (key, value) in pairs(block.extras)
            arrays[string(block.name, "/", key)] = Array(value)
        end
    end
    arrays
end

function _rans_load_restart!(cfg, mesh_path, topology, blocks)
    path = get(cfg, "warm_start", nothing)
    path === nothing && return ""
    try
        saved = open(deserialize, path)
        saved["identity"] == _rans_restart_identity(cfg, mesh_path, topology) ||
            error("mesh, solver source, or physical settings differ")
        abs(Float64(saved["alpha_deg"]) - Float64(cfg["alpha_deg"])) <= 2.5 ||
            error("incidence jump exceeds 2.5 degrees")
        for block in blocks
            isfinite(convert(typeof(block.solver.gam),saved["circulation"][String(block.name)])) ||
                error("invalid $(block.name) circulation")
        end
        for block in blocks, field in RANS_RESTART_FIELDS
            src = saved["arrays"][string(block.name, "/", field)]
            destination=getfield(block.solver,field)
            size(src) == size(destination) && eltype(src) == eltype(destination) && all(isfinite, src) ||
                error("invalid $(block.name) $field state")
            _rans_restart_physical(src, field, block.solver) ||
                error("nonphysical $(block.name) $field state")
        end
        for block in blocks
            hasproperty(block, :extras) || continue
            for (key, value) in pairs(block.extras)
                src = saved["arrays"][string(block.name, "/", key)]
                size(src) == size(value) && eltype(src) == eltype(value) && all(isfinite, src) ||
                    error("invalid $(block.name) $key state")
                if key in (:rk,:rw)
                    all(>(0),src) || error("nonphysical $(block.name) $key state")
                    solver=block.solver
                    interior=(solver.g+1:solver.g+solver.NI,solver.g+1:solver.g+solver.NJ)
                    rho=@view saved["arrays"][string(block.name,"/rho")][interior...]
                    primitive=@view saved["arrays"][string(block.name,key===:rk ? "/tk" : "/tom")][interior...]
                    expected=rho.*primitive
                    all(abs.(src.-expected) .<= 32eps(eltype(src)).*max.(abs.(src),abs.(expected))) ||
                        error("inconsistent $(block.name) $key state")
                end
            end
        end
        for block in blocks, field in RANS_RESTART_FIELDS
            copyto!(getfield(block.solver, field), saved["arrays"][string(block.name, "/", field)])
        end
        for block in blocks
            hasproperty(block, :extras) || continue
            for (key, value) in pairs(block.extras)
                copyto!(value, saved["arrays"][string(block.name, "/", key)])
            end
        end
        for block in blocks
            block.solver.gam = convert(typeof(block.solver.gam), saved["circulation"][String(block.name)])
        end
        println("RANS WARM START: checked state from alpha ", saved["alpha_deg"], " deg: ", path)
        flush(stdout)
        String(path)
    catch err
        println("RANS WARM START: refused (", sprint(showerror, err), "); starting from freestream")
        flush(stdout)
        ""
    end
end

function _rans_save_restart(cfg, mesh_path, topology, blocks)
    saved = Dict{String,Any}(
        "identity"=>_rans_restart_identity(cfg, mesh_path, topology),
        "alpha_deg"=>Float64(cfg["alpha_deg"]),
        "circulation"=>Dict(String(block.name)=>Float64(block.solver.gam) for block in blocks),
        "arrays"=>_rans_restart_arrays(blocks))
    path = joinpath(cfg["outdir"], "state_checkpoint.jls")
    tmp = path * ".tmp"
    open(io -> serialize(io, saved), tmp, "w")
    mv(tmp, path; force=true)
    path
end
