module RANSAcceleration

using SHA

export policy, configure_compressible!, implicit_method, implicit_memory_guard

const METHODS=("standard","irsj","line_implicit","jfnk_native","jfnk_krylov")
const PRECONDITIONERS=("line","adi","cut_line","block_adi","cut_block_adi",
    "swept_line","cut_swept_line","kinetic_block_adi","kinetic_fullvisc_block_adi",
    "kinetic_weno_fullvisc_block_adi","kinetic_block_unfloored_adi",
    "kinetic_weno_fullvisc_unfloored_adi")
implicit_method(method)=method in ("line_implicit","jfnk_native","jfnk_krylov")

function numerical_source_receipt()
    root=normpath(joinpath(@__DIR__,"..","..","..",".."))
    paths=vcat([joinpath("src","compressible",name) for name in readdir(joinpath(root,"src","compressible"))
        if startswith(name,"cmesh_steady_")&&endswith(name,".jl")],
        [joinpath("src","compressible","cmesh_solver_ka.jl"),
         joinpath("src","compressible","cmesh_geometric_reconstruction_ka.jl"),joinpath("src","cmesh","cmesh_flux_core.jl"),
         joinpath("src","cmesh","cmesh_wallfun.jl"),joinpath("src","cmesh","cmesh_metrics.jl"),joinpath("src","collision","lattice.jl")])
    Dict(replace(path,'\\'=>'/')=>bytes2hex(open(sha256,joinpath(root,path))) for path in paths)
end

function option(flow,key,default,lo,hi;integer=false)
    value=get(flow,key,default)
    value isa Real && !(value isa Bool) && isfinite(value) && lo<=value<=hi ||
        throw(ArgumentError("$key must be a finite number in [$lo, $hi]"))
    integer && !isinteger(value) && throw(ArgumentError("$key must be an integer"))
    integer ? Int(value) : Float64(value)
end

"Conservative device/host workspace bound, independent of the total iteration budget."
function implicit_memory_guard(ni::Integer,nj::Integer,g::Integer,m::Integer,pc::AbstractString,
                               method::AbstractString;available_bytes=nothing,
                               state_scalar_type::Type{<:AbstractFloat}=Float32,
                               block_scalar_type::Type{<:AbstractFloat}=Float32,
                               lagged_component_limiter::Bool=false,
                               muscl_ls_factors::Bool=false)
    ni>0&&nj>0&&g>=0&&2<=m<=96 || throw(ArgumentError("Invalid implicit workspace dimensions"))
    pc in PRECONDITIONERS || throw(ArgumentError("Unknown implicit preconditioner"))
    method in ("jfnk_native","jfnk_krylov") || throw(ArgumentError("Unknown Newton–Krylov method"))
    state_scalar_type in (Float32,Float64)&&block_scalar_type in (Float32,Float64)||
        throw(ArgumentError("Implicit workspace precision must be Float32 or Float64"))
    !lagged_component_limiter||state_scalar_type===Float64||
        throw(ArgumentError("Lagged component-limiter Jv currently requires Float64 state precision"))
    # Count the native V/Z bases, all seven four-field scratch vectors, twelve
    # line arrays and thirteen coupled rollback fields. The Krylov adapter
    # keeps the native allocation and adds its own packed basis and scratch.
    n=BigInt(ni)*nj;ng=(BigInt(ni)+2g)*(BigInt(nj)+2g)
    sb=sizeof(state_scalar_type);bb=sizeof(block_scalar_type)
    bytes=4sb*n*(2m+8)+8sb*n+4sb*ng+13sb*ng
    pc in ("adi","block_adi","cut_block_adi","swept_line","cut_swept_line") && (bytes+=5sb*n)
    pc in ("swept_line","cut_swept_line") && (bytes+=4sb*n)
    startswith(pc,"kinetic_") && (bytes+=160bb*n+8sb*n+32bb*(BigInt(ni)+nj))
    method=="jfnk_krylov" && (bytes+=4sb*n*(2m+16))
    # Solver-API-only experiments, not new web/default selections. Lagged R1
    # owns four factors, four baseline residuals and one error-rollback clock
    # field. The separately selected R3 PC has its own four-factor storage.
    lagged_component_limiter && (bytes+=9sb*n)
    muscl_ls_factors && (bytes+=4sb*n)
    # Leave room for implementation scratch/allocator rounding. On the 10 GiB
    # reference GPU the measured native m60 block workspace is 696 MiB; this
    # guard also reserves existing solver arrays, CUDA and desktop allocations.
    bound=cld(5bytes,4)
    hard_limit=BigInt(2)*1024^3
    limit=available_bytes===nothing ? hard_limit : min(hard_limit,BigInt(available_bytes)÷2)
    limit>0&&bound<=limit || throw(ArgumentError(
        "Implicit workspace needs at most $(round(Float64(bound)/1024^2;digits=1)) MiB including reserve, above the $(round(Float64(max(limit,0))/1024^2;digits=1)) MiB safety budget. Reduce restart length or mesh size."))
    receipt=(;estimated_workspace_bytes=Int(bound),workspace_budget_bytes=Int(limit),
      state_scalar_type=string(state_scalar_type),block_scalar_type=string(block_scalar_type),
      reserve_factor=1.25,scope="Conservative workspace bound; excludes the already allocated flow solver. Total iterations do not increase restart storage.")
    lagged_component_limiter||muscl_ls_factors ?
        merge(receipt,(;lagged_component_limiter,muscl_ls_factors)) : receipt
end

"Select an actual iteration method; never silently substitute a different solver."
function configure_compressible!(cfg,flow;coupled=false)
    method=String(get(flow,"rans_method",get(cfg,"irsj",false) ? "irsj" : "standard"))
    method in METHODS || throw(ArgumentError("Unknown steady RANS method: $method"))
    method!="standard" && get(cfg,"time_mode","steady")!="steady" &&
        throw(ArgumentError("The selected RANS iteration method requires steady mode"))
    coupled && implicit_method(method) && throw(ArgumentError(
        "Line-implicit and Newton–Krylov currently require the single C / A+ mesh. Select the A+ trailing-edge topology, or use Standard / residual smoothing for C+H."))
    cfg["rans_method"]=method
    cfg["rans_acceleration"]=method!="standard"
    labels=Dict("standard"=>"standard explicit", "irsj"=>"j-line implicit residual smoothing",
        "line_implicit"=>"experimental line-implicit", "jfnk_native"=>"experimental Newton–Krylov (native FGMRES)",
        "jfnk_krylov"=>"experimental Newton–Krylov (Krylov.jl GMRES)")
    cfg["rans_acceleration_method"]=labels[method]
    cfg["irsj"]=method=="irsj"
    if implicit_method(method)
        globalization=String(get(flow,"cfd_implicit_globalization","steady"))
        globalization in ("steady","pseudo_transient") || throw(ArgumentError("Unknown implicit globalization"))
        method=="line_implicit"&&globalization!="steady" && throw(ArgumentError("Backward-Euler globalization requires a Newton–Krylov method"))
        cfg["implicit_globalization"]=globalization
        cfl_limit=globalization=="pseudo_transient" ? 1e5 : 10.
        cfg["implicit_cfl"]=option(flow,"cfd_implicit_cfl",1.,.01,cfl_limit)
        cfg["implicit_max_cfl"]=option(flow,"cfd_implicit_max_cfl",max(2.,cfg["implicit_cfl"]),.01,cfl_limit)
        cfg["implicit_max_cfl"]>=cfg["implicit_cfl"] || throw(ArgumentError("Maximum implicit CFL must be at least the initial CFL"))
        cfg["implicit_warmup_steps"]=option(flow,"cfd_implicit_warmup_steps",1000,0,100000;integer=true)
        cfg["implicit_warmup_steps"]<cfg["steps"] || throw(ArgumentError("Implicit warm-up steps must be smaller than the total iteration limit"))
        cfg["implicit_basis"]=option(flow,"cfd_implicit_basis",16,2,96;integer=true)
        cfg["implicit_linear_maxiter"]=option(flow,"cfd_implicit_linear_maxiter",cfg["implicit_basis"],1,10000;integer=true)
        cfg["implicit_linear_tolerance"]=option(flow,"cfd_implicit_linear_tolerance",.01,1e-6,.9)
        cfg["implicit_physical_fraction"]=option(flow,"cfd_implicit_physical_fraction",.2,.01,.5)
        cfg["implicit_sst_substeps"]=option(flow,"cfd_implicit_sst_substeps",10,1,100;integer=true)
        cfg["implicit_nonmonotone_window"]=option(flow,"cfd_implicit_nonmonotone_window",8,0,20;integer=true)
        cfg["implicit_recovery_steps"]=option(flow,"cfd_implicit_recovery_steps",0,0,100;integer=true)
        preconditioner=String(get(flow,"cfd_implicit_preconditioner","line"))
        preconditioner in PRECONDITIONERS || throw(ArgumentError("Unknown implicit preconditioner"))
        cfg["implicit_preconditioner"]=preconditioner
        cfg["implicit_source_sha256"]=numerical_source_receipt()
        cfg["sfd_start"]=cfg["steps"]+1
        cfg["sfd_release"]=false
        cfg["rk_weight_closure"]=false
        cfg["mass_probe_stride"]=0
        cfg["equation_probe"]=true
        cfg["implicit_maturity"]="Research: CPU/GPU tests pass, but reliable full airfoil convergence and a speed advantage are not established. Inspect errors, mass balance and loads; diagnostic results remain viewable after a stop."
    end
    cfg
end

"""Steady RANS acceleration selected by the airfoil app. The default preserves legacy numerics.
The low-speed pressure solver is already implicit; its accelerated setting raises only
the requested pseudo-time ceiling, leaving its nonlinear rejection/retry in charge.
These settings are experimental and do not imply a measured wall-clock speedup."""
function policy(enabled::Bool, family::Symbol; physical::Bool=false)
    family in (:compressible, :compressible_ch, :lowspeed, :lowspeed_ch, :species_air) ||
        throw(ArgumentError("unsupported steady RANS family: $family"))
    enabled && physical && throw(ArgumentError("RANS convergence acceleration applies to steady pseudo-time only"))
    method = !enabled ? "standard" : family in (:compressible, :compressible_ch) ?
        "j-line implicit residual smoothing" : family in (:lowspeed, :lowspeed_ch) ?
        "adaptive implicit backward Euler, requested pseudo-time ceiling 0.5 c/U" :
        "wall-normal line implicit relaxation"
    (; enabled, method,
       irsj=enabled && family in (:compressible, :compressible_ch),
       pseudo_dt=enabled && family in (:lowspeed, :lowspeed_ch) ? 0.5 : 0.25,
       species_line=enabled && family === :species_air)
end

end
