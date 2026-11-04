# App adapter for the opt-in single C / A+ steady flow methods.
# The native solver remains responsible for state rollback and physical admission.
function _implicit_runtime(M,s,cfg)
    method=String(get(cfg,"rans_method","standard"))
    RANSAcceleration.implicit_method(method) || return nothing
    cfl=Float32(get(cfg,"implicit_cfl",1.))
    0<cfl<=10 || throw(ArgumentError("Implicit pseudo CFL must be in (0,10]"))
    if method=="line_implicit"
        return (method,workspace=M.build_implicit_line_workspace(s),cfl=Ref(cfl),controller=nothing)
    end
    controller=M.build_jfnk_airfoil_controller(s;
        m=Int(get(cfg,"implicit_basis",16)),initial_cfl=cfl,min_cfl=min(.005f0,cfl),
        max_cfl=max(2.f0,cfl),max_attempts=8,
        linear_solver=method=="jfnk_krylov" ? :krylov : :native,
        nonmonotone_window=Int(get(cfg,"implicit_nonmonotone_window",8)),
        preconditioner=Symbol(get(cfg,"implicit_preconditioner","line")),
        recovery_steps=Int(get(cfg,"implicit_recovery_steps",0)),recovery_limit=200)
    (;method,workspace=nothing,cfl=Ref(cfl),controller)
end

function _implicit_step!(M,s,runtime,cfg)
    substeps=Int(get(cfg,"implicit_sst_substeps",10))
    1<=substeps<=100 || throw(ArgumentError("SST substeps must be in [1,100]"))
    if runtime.controller!==nothing
        return M.step_jfnk_rans_ka!(s,runtime.controller;max_backtracks=16,sst_substeps=substeps,
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
            runtime.cfl[]=min(2.f0,used*1.05f0)
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
