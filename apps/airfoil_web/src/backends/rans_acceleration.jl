module RANSAcceleration

export policy, configure_compressible!, implicit_method

const METHODS=("standard","irsj","line_implicit","jfnk_native","jfnk_krylov")
implicit_method(method)=method in ("line_implicit","jfnk_native","jfnk_krylov")

function option(flow,key,default,lo,hi;integer=false)
    value=get(flow,key,default)
    value isa Real && !(value isa Bool) && isfinite(value) && lo<=value<=hi ||
        throw(ArgumentError("$key must be a finite number in [$lo, $hi]"))
    integer && !isinteger(value) && throw(ArgumentError("$key must be an integer"))
    integer ? Int(value) : Float64(value)
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
        cfg["implicit_cfl"]=option(flow,"cfd_implicit_cfl",1.,.01,10.)
        cfg["implicit_warmup_steps"]=option(flow,"cfd_implicit_warmup_steps",1000,0,100000;integer=true)
        cfg["implicit_warmup_steps"]<cfg["steps"] || throw(ArgumentError("Implicit warm-up steps must be smaller than the total iteration limit"))
        cfg["implicit_basis"]=option(flow,"cfd_implicit_basis",16,2,24;integer=true)
        cfg["implicit_sst_substeps"]=option(flow,"cfd_implicit_sst_substeps",10,1,100;integer=true)
        cfg["implicit_nonmonotone_window"]=option(flow,"cfd_implicit_nonmonotone_window",8,0,20;integer=true)
        cfg["implicit_recovery_steps"]=option(flow,"cfd_implicit_recovery_steps",0,0,100;integer=true)
        preconditioner=String(get(flow,"cfd_implicit_preconditioner","line"))
        preconditioner in ("line","adi","cut_line") || throw(ArgumentError("Unknown implicit preconditioner"))
        cfg["implicit_preconditioner"]=preconditioner
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
