"""Report the steady airfoil equations independently of iteration acceptance.

Flow entries are RMS/maximum flux divergences per volume, normalized by the
freestream conservative scales. `flow_l2_convective` additionally uses the
unit-chord convection time. Mass imbalance uses rho_inf * U_inf * chord.
SST entries are relative fixed-point update defects at the fixed safe 0.45
pseudo-CFL, not raw turbulence-equation residuals. The probe restores its
evolved fields and local clock, so recording diagnostics does not march SST.
"""
function steady_airfoil_diagnostics_ka(s::KSol;probe_sst::Bool=true,
    corrected_wall_metric::Bool=false,nonorth_diffusion::Bool=false)
    refresh_raw_residual_ka!(s)
    vol=s.vol;r=max(Float64(s.rho_inf),1e-12)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u)
    scales=(r,r*u,r*u,r*e);raw=_airfoil_residual(s);n=length(vol)
    l2=ntuple(k->sqrt(Float64(sum(abs2,raw[k]./vol))/n)/scales[k],4)
    linf=ntuple(k->Float64(maximum(abs,raw[k]./vol))/scales[k],4)
    mass=abs(Float64(sum(s.dc_r)))/(r*u)
    cl,cd=loads_ka(s)
    kdef=NaN;wdef=NaN;kmax=NaN;wmax=NaN
    if probe_sst && s.sst
        names=(:tk,:tom,:nut,:tnl,:pvort,:exx,:exy,:eyy,:dtf)
        saved=map(name->copy(getproperty(s,name)),names)
        ir=(s.g+1):(s.g+s.NI);jr=(s.g+1):(s.g+s.NJ)
        try
            prepare_dt_ka!(s,0.45f0)
            sst_step_ka!(s;corrected_wall_metric,nonorth_diffusion)
            k0=@view saved[1][ir,jr];w0=@view saved[2][ir,jr]
            k1=@view s.tk[ir,jr];w1=@view s.tom[ir,jr]
            dk=@. abs(k1-k0)/max(abs(k0),s.k_inf,1.0f-12)
            dw=@. abs(w1-w0)/max(abs(w0),s.w_inf,1.0f-6)
            kdef=sqrt(Float64(sum(abs2,dk))/n)
            wdef=sqrt(Float64(sum(abs2,dw))/n)
            kmax=Float64(maximum(dk));wmax=Float64(maximum(dw))
        finally
            for (name,array) in zip(names,saved);copyto!(getproperty(s,name),array);end
        end
    end
    (;flow_l2=l2,flow_linf=linf,flow_l2_convective=l2./u,
      normalized_mass_imbalance=mass,cl,cd,
      turb_k_update_defect=kdef,turb_omega_update_defect=wdef,
      turb_k_update_linf=kmax,turb_omega_update_linf=wmax,
      turbulence_metric="relative fixed-safe-clock update defect; not a raw equation residual",
      all_six_equation_residuals_available=false)
end

export steady_airfoil_diagnostics_ka
