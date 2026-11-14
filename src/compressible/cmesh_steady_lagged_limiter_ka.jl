# Opt-in R1 model for a thermal MUSCL Newton linearization. The physical R0,
# its nonlinear/BE merit, positivity guards and stopping diagnostics stay
# current. Only four component limiter factors are lagged during Jv samples.

mutable struct AirfoilLaggedLimiterWorkspace{S,A,V,D}
    owner::S
    component_factors::A
    reference_residual::V
    saved_dt::D
    ready::Bool
    refreshes::Int
end

function _airfoil_lagged_limiter_admit(s::KSol)
    eltype(s.rho)===Float64||throw(ArgumentError("Lagged component-limiter Jv currently requires a Float64 thermal state"))
    s.cav==0&&s.outer_mode==0&&s.outlet_mode==0&&s.ff_mix==0&&s.emix==0&&s.wall_flux||
        throw(ArgumentError("Lagged component-limiter Jv requires a single free-air, impermeable thermal airfoil"))
    s.recon==_AIRFOIL_MUSCL_RECON&&haskey(_AIRFOIL_MUSCL,s)||
        throw(ArgumentError("Lagged component-limiter Jv requires an explicitly configured original-node MUSCL workspace"))
    mw=_AIRFOIL_MUSCL[s]
    eltype(mw.geometry.area)===Float64&&eltype(mw.states)===Float64||
        throw(ArgumentError("Lagged component-limiter Jv requires Float64 MUSCL geometry and traces"))
    mw
end

function _build_airfoil_lagged_limiter_workspace(s::KSol)
    _airfoil_lagged_limiter_admit(s)
    AirfoilLaggedLimiterWorkspace(s,similar(s.dtf,Float64,s.NI,s.NJ,4),
        ntuple(_->similar(s.dtf),4),similar(s.dtf),false,0)
end

function _airfoil_lagged_owner(s::KSol,r::AirfoilLaggedLimiterWorkspace)
    r.owner===s||throw(ArgumentError("Lagged component-limiter workspace belongs to another solver"))
    mw=_airfoil_lagged_limiter_admit(s)
    _airfoil_check_component_factors(s,r.component_factors)
    all(a->eltype(a)===Float64&&size(a)==size(s.dtf)&&KA.get_backend(a)==s.backend,r.reference_residual)&&
        eltype(r.saved_dt)===Float64&&size(r.saved_dt)==size(s.dtf)&&KA.get_backend(r.saved_dt)==s.backend||
        throw(ArgumentError("Lagged component-limiter baseline/rollback storage does not match its solver"))
    mw
end

function _airfoil_lagged_residual_ka!(s::KSol,r::AirfoilLaggedLimiterWorkspace)
    _airfoil_lagged_owner(s,r)
    applyBC_ka!(s,Int32(2))
    _raw_residual_body_ka!(s,Int32(2),Int32(1),false,nothing;
        airfoil_component_factors=r.component_factors)
end

function _airfoil_refresh_lagged_limiter!(s::KSol,r::AirfoilLaggedLimiterWorkspace,line)
    mw=_airfoil_lagged_owner(s,r);a=mw.geometry;r.ready=false
    try
        _k_airfoil_muscl_block_theta!(s.backend)(r.component_factors,s.rho,s.ux,s.uy,s.T,
            a.offsets,a.weights,a.neighbors,a.area,mw.limiter,mw.smooth_coefficient,
            Int32(s.NI),Int32(s.NJ),Int32(s.g),eltype(s.rho)(hypot(s.uxi,s.uyi));ndrange=(s.NI,s.NJ))
        _airfoil_check_component_factors(s,r.component_factors;check_values=true)
        _airfoil_lagged_residual_ka!(s,r)
        for (dst,src) in zip(r.reference_residual,(s.dc_r,s.dc_mx,s.dc_my,s.dc_e));copyto!(dst,src);end
        r.refreshes+=1;r.ready=true
    catch
        r.ready=false
        rethrow()
    finally
        # Even BC/flux/validation failures cannot change the linearization's
        # primitive state. Coupled SST rollback remains the controller's job.
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T),(line.rho0,line.ux0,line.uy0,line.T0));copyto!(dst,src);end
    end
    nothing
end

function _airfoil_jacobian_residual_ka!(s::KSol,w)
    w.lagged_limiter===nothing ? _raw_residual_ka!(s,Int32(2),Int32(1)) :
        _airfoil_lagged_residual_ka!(s,w.lagged_limiter)
end

function _airfoil_jacobian_reference(w)
    r=w.lagged_limiter
    r===nothing&&return w.base_residual
    r.ready||throw(ArgumentError("Lagged component-limiter Jv requires a fresh nonlinear linearization"))
    r.reference_residual
end

"Declared linearization model; this is not a physical-convergence certificate."
function airfoil_jacobian_contract(w)
    r=w.lagged_limiter
    (;jacobian_model=r===nothing ? :current : :lagged_component_limiter,
      jacobian_refreshes=r===nothing ? 0 : r.refreshes,
      physical_residual_model=:current,physical_merit_model=:current,
      frozen_base_ready=r!==nothing&&r.ready)
end
