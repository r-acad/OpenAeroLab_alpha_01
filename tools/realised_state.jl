# Shared diagnostic banner. Read the built state; distinguish reference scales from
# measured propagation, convergence and phase-change behavior. Plain include, no module.
using Printf

function _realised_flux_mode(s)
    hasproperty(s, :cav) || return "flux mode unavailable (state has no cav field)"
    cav = Int(s.cav)
    cav >= 4 && return "mixture AUSM+-up flux on rho,u,p_mix,a_m"
    cav == 3 && return "KFVS carrier + pexc + Layer C + negative-compressibility correction"
    cav == 2 && return "KFVS carrier + pexc + Layer C"
    cav == 1 && return "KFVS carrier + pexc"
    cav == 0 && return "KFVS carrier (no cavitation pressure correction)"
    return "unrecognised cav mode $cav"
end

_realised_fields(state, names) = join((string(name, "=",
    hasproperty(state, name) ? getproperty(state, name) : "NOT_STORED") for name in names), " ")

"""
    realised_config_pairs(tem, s)

Compact scalar receipt for `print_realised_state(...; extra=...)`. Values describe the
state at the call, so call again after changing warm-up/march settings. Missing fields
are explicit; no requested constants or device arrays are read. `Re_unit_chord` is
`hypot(s.uxi,s.uyi)/s.nu_lam`, using the unit-chord convention. The solver CFL is a step
argument, not a stored KSol field; the SST Reboud exponent is also a step keyword.
Both need a separate producer/step receipt. Likewise
this is not a mesh-coordinate, physical-clock or toolchain admission check.
"""
function realised_config_pairs(tem, s)
    re = if all(name -> hasproperty(s, name), (:uxi, :uyi, :nu_lam))
        hypot(Float64(s.uxi), Float64(s.uyi))/Float64(s.nu_lam)
    else
        "NOT_STORED"
    end
    Pair{String,Any}[
        "config.flow" => _realised_fields(s, (:uxi, :uyi, :nu_lam, :cav, :recon, :wall_flux, :fullvisc)) * " Re_unit_chord=$re",
        "config.step" => "solver_CFL=NOT_STORED(step_argument) " * _realised_fields(s, (:cav_cfl, :irsj, :irs_ac, :acou, :sfd, :last_global_dt)),
        "config.turbulence" => _realised_fields(s, (:sst, :keps, :k_inf, :w_inf, :wallfun, :wf_larsson, :suth, :rhok, :earsm, :lag, :fbm_cdelta, :pans_fk)) * " reboud_n=NOT_STORED(step_keyword)",
        "config.energy" => _realised_fields(s, (:drot, :eint, :emix, :eint_v, :latent)),
        "config.boundary" => _realised_fields(s, (:ff_mix, :ff_p_ref, :ff_a2m, :ff_Tmin, :ff_Tmax, :outer_mode, :outlet_mode, :outlet_p, :sponge_sigma, :sponge_j0f)),
        "config.tem_eos" => _realised_fields(tem, (:Tmin, :Tmax, :eos, :wallis, :vaploc, :vapbound, :psm, :am_floor_frac, :ncg, :dk_cmin2, :dk_a2v, :lq)),
        "config.tem_rates" => _realised_fields(tem, (:kprod, :kdest, :omega, :src_jac, :src_smooth)),
        "config.tem_transport" => _realised_fields(tem, (:cons, :arecon, :enf, :beta2, :p3, :cflp, :dtboost, :dtb_dt, :advimp, :gs)),
        "config.tem_bubbles" => _realised_fields(tem, (:rp, :slip, :slip_tau, :cloudint, :poly, :hk, :wallnuc, :btke, :pvturb, :pvvort)),
        "config.tem_thermal" => _realised_fields(tem, (:tvp, :tfree, :tmix, :tmix_lat)),
    ]
end

"""
    print_realised_state(tem, s, U; label="", extra=nothing)

Print built saturation densities, branch stiffness reference scales, ambient cavitation
number, interior carrier-temperature statistics and reference source timescales. These
diagnostics do not measure the effective acoustic speed or establish a physical regime.

`U` is the freestream speed in lattice units (`hypot(s.uxi, s.uyi)`). CTU = 1/U for
the unit-chord mesh. Calculations use the built state and this caller-supplied speed.
`extra` may be a vector of key/value pairs, e.g. `realised_config_pairs(tem, s)`.
"""
function print_realised_state(tem, s, U::Float64; label::AbstractString="", extra=nothing)
    q  = 0.5*Float64(s.rho_inf)*U^2
    pv = Float64(tem.p_v)
    p_ref = tem_ambient(tem, s)[1]
    rls = Float64(tem.rho_l_sat); rvs = Float64(tem.rho_v_sat)
    a2l = Float64(tem.a2l);       a2v = Float64(tem.a2v)

    isempty(label) || @printf("  REALISED STATE [%s]\n", label)
    @printf("  REALISED: rho_l_sat=%.6f  rho_v_sat=%.4e  ratio_eff=%.1f  a2l=%.4f  a2v=%.4g\n",
            rls, rvs, rls/max(rvs, 1e-30), a2l, a2v)
    @printf("            p_v=%.4e  p_ref=%.4e  SIGMA_eff=(p_ref-p_v)/q=%.3f   c_liq/U=%.2f  a_v/U=%.2f\n",
            pv, p_ref, (p_ref - pv)/q, sqrt(a2l)/U, sqrt(a2v)/U)
    println("            c_liq/U and a_v/U above are sqrt(a2l)/U and sqrt(a2v)/U: EOS branch reference scales.")
    println("            active flux: ", _realised_flux_mode(s))

    # Preserve the historical scalar calculations and labels for log comparison.
    # _k_dtfield! uses 3sqrt(max(T,1e-3)) locally, together with velocity, geometry,
    # viscosity and optional modifiers. Its bound does not measure a propagated wave.
    # For cav>=4 the face flux uses p_mix and a_m; the legacy KFVS inference does not apply.
    let Th = Array(s.T), g0 = Int(s.g), sub = @view(Th[(g0+1):(g0+Int(s.NI)), (g0+1):(g0+Int(s.NJ))]),
        Tmean = Float64(sum(sub))/length(sub), Tmin_f = Float64(minimum(sub)), Tmax_f = Float64(maximum(sub)),
        c_car = 3.0*sqrt(Tmean), c_hyd = sqrt(1.4*Float64(TEM_CS2)*Tmean)
        @printf("            c_carrier/U=%.2f  (=3*sqrt(T)/U at FIELD mean T=%.4f [min %.4f max %.4f]; timestep reference scale)\n",
                c_car/U, Tmean, Tmin_f, Tmax_f)
        @printf("            c_hydro/U  =%.2f  (=sqrt(1.4*cs2*T)/U; ideal-gas reference, not a measured mixture acoustic speed)\n", c_hyd/U)
        @printf("            c_thermal/U=%.2f  (=sqrt(cs2*T)/U; carrier thermal reference scale)\n",
                sqrt(Float64(TEM_CS2)*Tmean)/U)
        println("            The local timestep also depends on flow, geometry, viscosity and enabled modifiers.")
        println("            These mean-field scales do not establish wave propagation or numerical contact diffusion.")
    end

    # Merkle reference rate at |dp|=p_v, converted to CTU from built coefficients.
    # This remains the same diagnostic under other source modes, where it need not
    # equal the active source relaxation time. No actual driving-pressure bound is inferred.
    let kp = Float64(tem.kprod), kd = Float64(tem.kdest)
        tau_e = kp > 0 ? U/(kp*pv) : Inf
        tau_c = kd > 0 ? U/(kd*pv) : Inf
        @printf("            tau_evap=%.4g CTU   tau_cond=%.4g CTU   (Merkle coefficient reference at |dp| = p_v)\n",
                tau_e, tau_c)
        println("            Actual relaxation depends on local pressure, active source model and time integration.")
        println("            A short reference time alone does not establish equilibrium or erase transported history.")
    end

    # Retain the isothermal gas-loaded Blake reference formula and scalar values.
    # Its comparison with p_v cannot decide inception for a polytropic, transported,
    # polydisperse or thermally modified model, nor diagnose a zero-alpha field.
    if hasproperty(tem, :rp) && Float64(tem.rp) > 0
        S = Float64(tem.rp_S); pg0 = Float64(tem.rp_pg0); R0 = Float64(tem.rp_R0)
        @printf("            R-P: mode=%g  R0=%.4e  n0=%.4e  S=%.4g  pg0=%.4g  kappa=%.3g\n",
                Float64(tem.rp), R0, Float64(tem.rp_n0), S, pg0, Float64(tem.rp_kappa))
        if S <= 0.0
            println("            R-P: S <= 0; no positive surface-tension Blake reference is evaluated.")
        else
            Rc  = sqrt(3.0*pg0*R0^3/(2.0*S))
            dpc = 4.0*S/(3.0*Rc)
            ratio = dpc/max(pv, 1e-30)
            @printf("            R-P: isothermal Blake reference R_c=%.4e  dp_crit=%.4e  vs reference p_v=%.4e  (ratio %.2f)\n",
                    Rc, dpc, pv, ratio)
            if ratio > 1.0
                println("            R-P: isothermal reference dp_crit exceeds p_v; this alone does not decide nucleus growth.")
            else
                @printf("            R-P: reference p_v/dp_crit=%.1f; this alone does not decide nucleus growth.\n", 1.0/max(ratio, 1e-30))
            end
            println("            The formula assumes isothermal gas (kappa=1); check the active model and local pressure history.")
        end
    end

    if extra !== nothing
        for (k, v) in extra
            @printf("            %-16s = %s\n", k, string(v))
        end
    end
    flush(stdout)
    return nothing
end
