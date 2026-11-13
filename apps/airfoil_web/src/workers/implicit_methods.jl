# App adapter for the opt-in single C / A+ steady flow methods.
# The native solver remains responsible for state rollback and physical admission.
function _implicit_runtime(M,s,cfg)
    method=String(get(cfg,"rans_method","standard"))
    RANSAcceleration.implicit_method(method) || return nothing
    cfl=Float32(get(cfg,"implicit_cfl",1.))
    globalization=Symbol(get(cfg,"implicit_globalization","steady"))
    globalization in (:steady,:pseudo_transient) || throw(ArgumentError("Unknown implicit globalization"))
    cfl_limit=globalization==:pseudo_transient ? 1f5 : 10f0
    0<cfl<=cfl_limit || throw(ArgumentError("Implicit pseudo CFL exceeds the selected globalization bound"))
    max_cfl=Float32(get(cfg,"implicit_max_cfl",max(2.f0,cfl)))
    cfl<=max_cfl<=cfl_limit || throw(ArgumentError("Invalid maximum implicit pseudo CFL"))
    if method=="line_implicit"
        globalization==:steady || throw(ArgumentError("Backward-Euler globalization requires Newton–Krylov"))
        return (method,workspace=M.build_implicit_line_workspace(s),cfl=Ref(cfl),controller=nothing,
            max_cfl,memory_receipt=nothing)
    end
    basis=Int(get(cfg,"implicit_basis",16))
    preconditioner=String(get(cfg,"implicit_preconditioner","line"))
    available=M.KA.get_backend(s.rho)==M.KA.CPU() ? nothing : CUDA.free_memory()
    memory_receipt=RANSAcceleration.implicit_memory_guard(s.NI,s.NJ,s.g,basis,preconditioner,method;
        available_bytes=available,state_scalar_type=eltype(s.rho))
    cfg["implicit_workspace_memory"]=Dict(string(k)=>v for (k,v) in pairs(memory_receipt))
    cfg["implicit_source_sha256"]=RANSAcceleration.numerical_source_receipt()
    controller=M.build_jfnk_airfoil_controller(s;
        m=basis,initial_cfl=cfl,min_cfl=min(.005f0,cfl),max_cfl,max_attempts=8,
        linear_solver=method=="jfnk_krylov" ? :krylov : :native,
        nonmonotone_window=Int(get(cfg,"implicit_nonmonotone_window",8)),
        preconditioner=Symbol(preconditioner),globalization,
        physical_fraction=Float32(get(cfg,"implicit_physical_fraction",.2)),
        recovery_steps=Int(get(cfg,"implicit_recovery_steps",0)),recovery_limit=200)
    (;method,workspace=nothing,cfl=Ref(cfl),controller,max_cfl,memory_receipt)
end

function _implicit_step!(M,s,runtime,cfg)
    substeps=Int(get(cfg,"implicit_sst_substeps",10))
    1<=substeps<=100 || throw(ArgumentError("SST substeps must be in [1,100]"))
    if runtime.controller!==nothing
        return M.step_jfnk_rans_ka!(s,runtime.controller;max_backtracks=16,sst_substeps=substeps,
            linear_tol=Float64(get(cfg,"implicit_linear_tolerance",.01)),
            linear_maxiter=Int(get(cfg,"implicit_linear_maxiter",get(cfg,"implicit_basis",16))),
            corrected_wall_metric=Bool(get(cfg,"sst_wall_metric_correction",false)),
            nonorth_diffusion=Bool(get(cfg,"sst_nonorth_correction",false)))
    end
    result=nothing;attempts=0;used=runtime.cfl[]
    for attempt in 1:8
        attempts=attempt
        used=runtime.cfl[]
        result=M.step_implicit_line_ka!(s,runtime.workspace,used;max_backtracks=16)
        if result.accepted
            for _ in 1:substeps
                M.prepare_dt_ka!(s,.45f0)
                M.sst_step_ka!(s;corrected_wall_metric=Bool(get(cfg,"sst_wall_metric_correction",false)),
                    nonorth_diffusion=Bool(get(cfg,"sst_nonorth_correction",false)))
            end
            runtime.cfl[]=min(runtime.max_cfl,used*1.05f0)
            M.refresh_raw_residual_ka!(s)
            return merge(result,(attempts=attempt,pseudo_cfl_used=used,pseudo_cfl_next=runtime.cfl[],
                flow_trial_merit_after=result.merit_after,merit_after=M._implicit_merit(s),
                step_kind=:implicit,linear_iterations=0,linear_relative=NaN,recovery_steps=0,recovery_total=0))
        end
        runtime.cfl[]=used*.5f0
        runtime.cfl[]<.005f0 && break
    end
    merge(result,(attempts,pseudo_cfl_used=used,pseudo_cfl_next=runtime.cfl[],
        step_kind=:rejected,linear_iterations=0,linear_relative=NaN,recovery_steps=0,recovery_total=0))
end

function _implicit_log!(io,n,result,elapsed)
    value(name,default)=hasproperty(result,name) ? getproperty(result,name) : default
    println(io,join((n,elapsed,result.accepted,value(:step_kind,:implicit),
        result.merit_before,result.merit_after,value(:pseudo_cfl_used,NaN),
        value(:linear_iterations,0),value(:linear_relative,NaN),value(:attempts,1),
        value(:recovery_steps,0),value(:recovery_total,0)),','));flush(io)
end

# Keep the legacy eleven-column prefix stable. Raw SST equations and the
# prescribed-omega constraint are additional diagnostics, not new stop limits.
const _IMPLICIT_RAW_SST_METRICS=("turb_k_equation_l2","turb_omega_equation_l2",
    "turb_k_equation_linf","turb_omega_equation_linf",
    "turb_omega_wall_constraint_relative_linf","turb_omega_all_transport_l2")
_implicit_metric_names()=("iteration","mass_l2","momentum_x_l2","momentum_y_l2","energy_l2",
    "turb_k_update_defect","turb_omega_update_defect","mass_net_frac","cl","cd","cm",
    _IMPLICIT_RAW_SST_METRICS...)
function _implicit_metric_values(n,metrics,mass,cl,cd,cm)
    (n,metrics.flow_l2_convective...,metrics.turb_k_update_defect,
        metrics.turb_omega_update_defect,mass,cl,cd,cm,
        (getproperty(metrics,Symbol(name)) for name in _IMPLICIT_RAW_SST_METRICS)...)
end
_implicit_raw_sst_available(rows)=!isempty(rows) && all(row->
    all(name->get(row,name,nothing) isa Real && isfinite(row[name]),_IMPLICIT_RAW_SST_METRICS),rows)
