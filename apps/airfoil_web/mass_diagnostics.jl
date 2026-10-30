module CFDMassDiagnostics

export assess_mass_balance, assess_peak_dynamics, nonfatal_diagnostic

"""Keep optional reporting errors from discarding a completed CFD field."""
function nonfatal_diagnostic(f::Function, label::AbstractString; logger::Function=println)
    try
        return f()
    catch err
        message = sprint(showerror,err)
        logger("CMESH2D optional $(label) diagnostic failed: $(message)\n" *
               sprint(showerror,err,catch_backtrace()))
        return Dict{String,Any}("diagnostic_error"=>message)
    end
end

"""Summarize the final finite-volume continuity residual and boundary fluxes.

`dc` must use the solver's convention
`Fi[i+1,j]-Fi[i,j] + Fj[i,j+1]-Fj[i,j]`. `vol` is positive cell area.
The flux scale is freestream density × speed × chord (2-D unit span); the
volume-scaled residual uses density × speed / chord. All inputs are read-only.
"""
function assess_mass_balance(dc::AbstractMatrix, vol::AbstractMatrix,
                             fi::AbstractMatrix, fj::AbstractMatrix,
                             xc::AbstractMatrix, yc::AbstractMatrix;
                             niw::Integer, nif::Integer, nbase::Integer,
                             rho_inf::Real, speed_inf::Real, chord::Real)
    ni,nj = size(dc)
    size(vol) == size(xc) == size(yc) == (ni,nj) ||
        throw(ArgumentError("mass diagnostic cell arrays have different shapes"))
    size(fi) == (ni+1,nj) || throw(ArgumentError("i-face flux shape mismatch"))
    size(fj) == (ni,nj+1) || throw(ArgumentError("j-face flux shape mismatch"))
    niw >= 0 && nif > 0 && 2niw+nif == ni ||
        throw(ArgumentError("mass diagnostic wake/foil columns do not cover grid"))
    0 <= nbase <= nif÷2 || throw(ArgumentError("invalid base-face count"))
    all(v -> isfinite(v) && v > 0, vol) ||
        throw(ArgumentError("mass diagnostic requires positive finite cell volumes"))
    all(isfinite,dc) && all(isfinite,fi) && all(isfinite,fj) ||
        throw(ArgumentError("mass diagnostic requires finite fluxes and residuals"))
    rho = Float64(rho_inf); speed = Float64(speed_inf); c = Float64(chord)
    rho > 0 && speed > 0 && c > 0 ||
        throw(ArgumentError("mass diagnostic scales must be positive"))
    flux_scale = rho*speed*c
    rate_scale = rho*speed/c
    near_rows = min(8,max(1,nj÷4))
    outer_start = nj-near_rows+1
    first_body = niw+1
    last_body = niw+nif
    first_surface = first_body+nbase
    last_surface = last_body-nbase
    function region(i,j)
        j >= outer_start && return "farfield_ring"
        if j <= near_rows
            (i < first_body || i > last_body) && return "wake_cut_nearwall"
            (i < first_surface || i > last_surface) && return "base_nearwall"
            return "foil_nearwall"
        end
        "interior"
    end
    acc = Dict{String,Dict{String,Float64}}()
    peak = (-Inf,0,0,"")
    for j in 1:nj, i in 1:ni
        label = region(i,j)
        a = get!(acc,label) do
            Dict("cells"=>0.0,"sum_squared"=>0.0,"max"=>0.0,
                 "signed_divergence"=>0.0,"absolute_divergence"=>0.0)
        end
        signed = Float64(dc[i,j])
        r = abs(signed)/Float64(vol[i,j])
        a["cells"] += 1
        a["sum_squared"] += r*r
        a["max"] = max(a["max"],r)
        a["signed_divergence"] += signed
        a["absolute_divergence"] += abs(signed)
        if r > peak[1]
            peak = (r,i,j,label)
        end
    end
    groups = Dict{String,Any}()
    for (label,a) in acc
        groups[label] = Dict{String,Any}(
            "cells"=>Int(a["cells"]),
            "linf_volume_scaled"=>a["max"],
            "rms_volume_scaled"=>sqrt(a["sum_squared"]/a["cells"]),
            "linf_relative_to_freestream_rate"=>a["max"]/rate_scale,
            "signed_divergence_over_freestream_flux"=>a["signed_divergence"]/flux_scale,
            "absolute_divergence_over_freestream_flux"=>a["absolute_divergence"]/flux_scale)
    end
    boundary = Dict{String,Float64}(
        "left"=>-sum(Float64,view(fi,1,:)),
        "right"=>sum(Float64,view(fi,ni+1,:)),
        "outer"=>sum(Float64,view(fj,:,nj+1)),
        "inner_wake_lower"=>sum((i->-Float64(fj[i,1])),1:niw; init=0.0),
        "inner_wake_upper"=>sum((i->-Float64(fj[i,1])),last_body+1:ni; init=0.0),
        "inner_base"=>sum((i->-Float64(fj[i,1])),first_body:first_surface-1; init=0.0) +
                      sum((i->-Float64(fj[i,1])),last_surface+1:last_body; init=0.0),
        "inner_foil"=>sum((i->-Float64(fj[i,1])),first_surface:last_surface; init=0.0))
    boundary_normalized = Dict(label=>value/flux_scale for (label,value) in boundary)
    total_boundary = sum(values(boundary))
    total_cell = sum(Float64,dc)
    Dict{String,Any}(
        "scope"=>"final continuity residual; mutually exclusive near-wall/outer/core cells",
        "nearwall_rows"=>near_rows,"outer_ring_rows"=>near_rows,
        "flux_scale_rho_u_chord"=>flux_scale,
        "rate_scale_rho_u_over_chord"=>rate_scale,
        "regions"=>groups,
        "peak"=>Dict("i"=>peak[2],"j"=>peak[3],"region"=>peak[4],
                     "x"=>Float64(xc[peak[2],peak[3]]),
                     "y"=>Float64(yc[peak[2],peak[3]]),
                     "linf_volume_scaled"=>peak[1]),
        "boundary_signed_flux_over_freestream_flux"=>boundary_normalized,
        "boundary_total_over_freestream_flux"=>total_boundary/flux_scale,
        "cell_divergence_total_over_freestream_flux"=>total_cell/flux_scale,
        "telescoping_error_over_freestream_flux"=>(total_cell-total_boundary)/flux_scale)
end

"""Read-only timestep and storage-aware SFD audit at the mass-residual peak cell.

The two dt estimates reproduce `_k_dtfield!` for the current saved primitives;
`dtf` was computed immediately before the last RK step, so a small difference
from the estimates is expected. SFD increment tests predict the next filter
update using the saved filter array's Float32 or Float64 storage precision;
the flow state remains Float32 in both cases.
"""
function assess_peak_dynamics(i::Integer,j::Integer,a::NamedTuple;
                              g::Integer,cfl::Real,nu_lam::Real,sst::Bool,
                              irsj::Bool,sfd::Bool,sfd_alpha::Real,sfd_chi::Real,
                              viscosity_model::AbstractString="constant_kinematic",
                              rho_inf::Real=1.0,sutherland_ratio::Real=0.0)
    ia=i+g; ja=j+g
    u=Float64(a.ux[ia,ja]); v=Float64(a.uy[ia,ja]); temp=Float64(a.temp[ia,ja])
    c=3sqrt(max(temp,1e-3))
    siw=hypot(Float64(a.six[i,j]),Float64(a.siy[i,j]))
    sie=hypot(Float64(a.six[i+1,j]),Float64(a.siy[i+1,j]))
    sjs=hypot(Float64(a.sjx[i,j]),Float64(a.sjy[i,j]))
    sjn=hypot(Float64(a.sjx[i,j+1]),Float64(a.sjy[i,j+1]))
    lami=(abs(u*a.six[i,j]+v*a.siy[i,j])+
          abs(u*a.six[i+1,j]+v*a.siy[i+1,j]))/2 + c*(siw+sie)/2
    lamj=(abs(u*a.sjx[i,j]+v*a.sjy[i,j])+
          abs(u*a.sjx[i,j+1]+v*a.sjy[i,j+1]))/2 + c*(sjs+sjn)/2
    volume=Float64(a.vol[i,j])
    viscosity_model in ("constant_kinematic","sutherland","constant_dynamic") ||
        throw(ArgumentError("unknown viscosity model in peak diagnostic"))
    nu_molecular=if viscosity_model == "sutherland"
        theta=Float32(a.temp[ia,ja]); sr=Float32(sutherland_ratio)
        factor=theta*sqrt(theta)*(1.0f0+sr)/(theta+sr)
        Float64(Float32(rho_inf)*Float32(nu_lam)*factor/Float32(a.rho[ia,ja]))
    elseif viscosity_model == "constant_dynamic"
        Float64(Float32(rho_inf)*Float32(nu_lam)/Float32(a.rho[ia,ja]))
    else
        Float64(nu_lam)
    end
    nueff=nu_molecular+(sst ? Float64(a.nut[ia,ja]) : 0.0)
    acoustic=Float64(cfl)*volume/max(irsj ? lami : lami+lamj,1e-30)
    dn=volume/max(irsj ? max(siw,sie) : max(siw,sie,sjs,sjn),1e-30)
    viscous=nueff > 0 ? 0.2dn^2/nueff : Inf
    actual=Float64(a.dtf[i,j])
    filter_type=eltype(a.sb_r)
    filter_type in (Float32,Float64) ||
        throw(ArgumentError("SFD diagnostic requires Float32 or Float64 filter storage"))
    all(eltype(field) === filter_type for field in
        (a.sb_u,a.sb_v,a.sb_T)) ||
        throw(ArgumentError("SFD filter arrays have different storage types"))
    result=Dict{String,Any}(
        "dtf_actual"=>actual,
        "dt_acoustic_estimate"=>acoustic,
        "dt_viscous_estimate"=>viscous,
        "dt_limiter_estimate"=>acoustic < viscous ? "acoustic" : "viscous",
        "dtf_over_min_estimate"=>actual/min(acoustic,viscous),
        "viscous_to_acoustic_dt_ratio"=>viscous/acoustic,
        "effective_kinematic_viscosity"=>nueff,
        "molecular_kinematic_viscosity"=>nu_molecular,
        "viscosity_model"=>String(viscosity_model),
        "local_cfl"=>Float64(cfl),
        "sfd"=>Dict{String,Any}("engaged"=>sfd,
                               "filter_storage_type"=>string(filter_type),
                               "alpha"=>Float64(sfd_alpha),
                               "chi"=>Float64(sfd_chi)))
    if sfd
        fields=Dict{String,Any}()
        for (name,state,filter) in
                (("rho",a.rho[ia,ja],a.sb_r[ia,ja]),
                 ("ux",a.ux[ia,ja],a.sb_u[ia,ja]),
                 ("uy",a.uy[ia,ja],a.sb_v[ia,ja]),
                 ("temperature",a.temp[ia,ja],a.sb_T[ia,ja]))
            state32=Float32(state)
            # Reproduce the device expression with its actual filter array
            # element type. The Float32 branch is deliberately unchanged.
            filter_value=filter_type(filter)
            gap=state32-filter_value
            proposed=Float32(sfd_alpha)*gap
            next_filter=filter_value+proposed
            actual_filter_step=next_filter-filter_value
            chi_proposed=Float32(sfd_chi)*(state32-next_filter)
            actual_state_step=Float32(state32-chi_proposed)-state32
            filter_ulp=Float64(nextfloat(filter_value)-filter_value)
            state_ulp=Float64(nextfloat(state32)-state32)
            field_report=Dict{String,Any}(
                "state"=>Float64(state32),"filter"=>Float64(filter_value),
                "state_minus_filter"=>Float64(gap),
                "proposed_filter_step"=>Float64(proposed),
                "actual_filter_step"=>Float64(actual_filter_step),
                "filter_step_to_ulp"=>abs(Float64(proposed))/filter_ulp,
                "filter_step_rounds_to_zero"=>actual_filter_step == 0,
                "proposed_chi_step"=>Float64(chi_proposed),
                "actual_float32_state_step"=>Float64(actual_state_step),
                "chi_step_to_ulp"=>abs(Float64(chi_proposed))/state_ulp,
                "chi_step_rounds_to_zero"=>actual_state_step == 0)
            if filter_type === Float32
                field_report["actual_float32_filter_step"] = Float64(actual_filter_step)
            else
                field_report["actual_float64_filter_step"] = Float64(actual_filter_step)
            end
            fields[name]=field_report
        end
        result["sfd"]["fields"] = fields
    end
    result
end

end # module
