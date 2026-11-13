"Return signed numerical outer flux and the total absolute and incoming fluxes."
function _airfoil_outer_mass_fluxes(fi,fj)
    left=@view fi[1,:]; right=@view fi[end,:]; outer=@view fj[:,end]
    # i increases left-to-right, j increases away from the airfoil. These
    # flux arrays already include face length and the numerical face flux.
    # Accumulate face values in Float64: the net is the small difference
    # between much larger inlet and outlet fluxes on fine Float32 meshes.
    net=sum(Float64,right)-sum(Float64,left)+sum(Float64,outer)
    throughput=sum(x->abs(Float64(x)),right)+sum(x->abs(Float64(x)),left)+
               sum(x->abs(Float64(x)),outer)
    incoming=sum(x->max(Float64(x),0.0),left)+
             sum(x->max(-Float64(x),0.0),right)+
             sum(x->max(-Float64(x),0.0),outer)
    (;net,throughput,incoming)
end

"""Report the steady airfoil equations independently of iteration acceptance.

Flow entries are RMS/maximum flux divergences per volume, normalized by the
freestream conservative scales. `flow_l2_convective` additionally uses the
unit-chord convection time. `normalized_mass_imbalance` uses rho_inf * U_inf *
unit chord; it is not the fraction of inlet flow. Separate outer-boundary
ratios report the numerical net outflow relative to total absolute face flux
and incoming flux. These ratios are NaN if their denominator is zero.
SST equation entries are unscaled primitive transport residuals before
point-implicit division and clipping. Separate update-defect entries retain
the fixed safe 0.45 pseudo-CFL probe. Equation residuals require their own
reference scaling when used as stopping criteria. The probe restores its
evolved fields and local clock, so recording diagnostics does not march SST.
"""
function steady_airfoil_diagnostics_ka(s::KSol;probe_sst::Bool=true,
    corrected_wall_metric::Bool=_sst_default_wall_metric(s),nonorth_diffusion::Bool=false)
    refresh_raw_residual_ka!(s)
    vol=s.vol;r=max(Float64(s.rho_inf),1e-12)
    u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u)
    scales=(r,r*u,r*u,r*e);raw=_airfoil_residual(s);n=length(vol)
    l2=ntuple(k->sqrt(Float64(sum(abs2,raw[k]./vol))/n)/scales[k],4)
    linf=ntuple(k->Float64(maximum(abs,raw[k]./vol))/scales[k],4)
    mass=abs(Float64(sum(s.dc_r)))/(r*u)
    outer_mass=_airfoil_outer_mass_fluxes(s.Fi_r,s.Fj_r)
    outer_fraction=outer_mass.throughput>0 ? abs(outer_mass.net)/outer_mass.throughput : NaN
    incoming_fraction=outer_mass.incoming>0 ? abs(outer_mass.net)/outer_mass.incoming : NaN
    cl,cd=loads_ka(s)
    kdef=NaN;wdef=NaN;kmax=NaN;wmax=NaN
    keq=NaN;weq=NaN;keqmax=NaN;weqmax=NaN
    wallomega=NaN;allomega=NaN
    omega_wall_cells=s.keps==0.0f0 ? min(2,s.NJ)*s.NIF : 0
    omega_transport_cells=n-omega_wall_cells
    if probe_sst && s.sst
        names=(:tk,:tom,:nut,:tnl,:pvort,:exx,:exy,:eyy,:dtf)
        saved=map(name->copy(getproperty(s,name)),names)
        ir=(s.g+1):(s.g+s.NI);jr=(s.g+1):(s.g+s.NJ)
        saved_clock=s.last_global_dt
        try
            prepare_dt_ka!(s,0.45f0)
            if s.keps==0.0f0
                # Read the same transport/source expressions used by SST,
                # before the local time step, implicit sink and floors can
                # conceal an equation defect. The finally block restores
                # ghost values and all evolved turbulence auxiliaries.
                eq=sst_step_ka!(s;corrected_wall_metric,nonorth_diffusion,raw_equations=true)
                kr=@view eq.k[ir,jr];wr=@view eq.omega[ir,jr]
                keq=sqrt(sum(x->Float64(x)^2,kr)/n)
                weq=sqrt(sum(x->Float64(x)^2,wr)/max(1,omega_transport_cells))
                keqmax=Float64(maximum(abs,kr));weqmax=Float64(maximum(abs,wr))
                wallomega=Float64(maximum(abs,view(eq.omega_wall_constraint_relative,ir,jr)))
                allomega=sqrt(sum(x->Float64(x)^2,view(eq.omega_all_transport,ir,jr))/n)
            end
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
            s.last_global_dt=saved_clock
        end
    end
    (;flow_l2=l2,flow_linf=linf,flow_l2_convective=l2./u,
      normalized_mass_imbalance=mass,
      mass_outer_net=outer_mass.net,
      mass_outer_abs_flux=outer_mass.throughput,
      mass_outer_incoming_flux=outer_mass.incoming,
      mass_outer_net_fraction=outer_fraction,
      mass_outer_net_incoming_fraction=incoming_fraction,cl,cd,
      turb_k_update_defect=kdef,turb_omega_update_defect=wdef,
      turb_k_update_linf=kmax,turb_omega_update_linf=wmax,
      turb_k_equation_l2=keq,turb_omega_equation_l2=weq,
      turb_k_equation_linf=keqmax,turb_omega_equation_linf=weqmax,
      turb_omega_wall_constraint_relative_linf=wallomega,
      turb_omega_all_transport_l2=allomega,
      turb_omega_wall_constraint_cells=omega_wall_cells,
      turb_omega_transport_cells=omega_transport_cells,
      turbulence_metric="raw primitive SST equation residuals and relative fixed-safe-clock update defects",
      all_six_equation_residuals_available=probe_sst && s.sst && s.keps==0.0f0)
end

export steady_airfoil_diagnostics_ka
