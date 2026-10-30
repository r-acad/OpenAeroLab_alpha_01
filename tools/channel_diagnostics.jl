# Shared realised-state diagnostics for the rotated rectangular-channel controls.
# Include after CMeshCavTEM so TEM_CS2 is available. This file deliberately consumes the built
# metrics/state rather than the requested ChannelSpec: the campaign's provenance rule is to report
# what reached the solver, not what an environment variable claimed.

using Printf

const _CHANNEL_INLET = UInt8(1)
const _CHANNEL_LOWER = UInt8(2)
const _CHANNEL_UPPER = UInt8(3)

"""Hard geometry/GCL admission gate and realised banner for either legacy or channel meshes. A strip mesh (the strip
trailing-edge topology, 5 Oct 2026) must come with its strip block's metrics (`strip_met`, from `build_strip_metrics`):
the second block's GCL, folds and corner sines are then gated and printed too, and `wake_pair_err` is the strip edges'
coincidence with the legs (P1'), 0 required."""
function admit_outer_geometry(mesh, met; label::AbstractString="", growth_limit::Real=1.20, strip_met=nothing)
    mq = mesh_quality(mesh)
    gworst, gmean = gcl_check(met)
    mq.nfold == 0 || error("$label mesh has $(mq.nfold) non-positive cells")
    gworst < 1e-6 || error("$label mesh GCL $gworst exceeds 1e-6")
    if mesh isa ChannelMesh
        q = channel_quality(mesh)
        q.lower_corner_count == 1 && q.upper_corner_count == 1 ||
            error("$label channel corners are not unique")
        q.lower_wall_eta_err < 1e-10 && q.upper_wall_eta_err < 1e-10 &&
            q.inlet_xi_err < 1e-10 && q.outlet_xi_err < 1e-10 ||
            error("$label channel boundary nodes miss their exact rotated lines")
        q.wall_parallel_err < 1e-6 || error("$label channel wall alignment gate failed")
        q.spacing_ratio_symmetric_max <= Float64(growth_limit) + 512eps(Float64) ||
            error("$label channel realised adjacent spacing ratio " *
                  "$(q.spacing_ratio_symmetric_max) exceeds growth limit $growth_limit")
        q.wake_pair_err == 0.0 || error(mesh.strip === nothing ? "$label channel wake partners are not bitwise coincident" :
                                        "$label strip edges are not the wake legs bitwise (P1')")
        q.wake_alignment_err < 1e-12 || error("$label channel wake is not streamwise aligned")
        q.wake_outlet_xi_err < 1e-10 || error("$label channel wake misses the outlet plane")
        q.outer_monotonicity_violations == 0 || error("$label channel outer path is non-monotone")
        @printf("REALISED CHANNEL QUALITY%s: H=%.8g folds=%d area_min=%.8e growth[nominal/realised]=%.8g/%.8g GCL[max/mean]=%.3e/%.3e wall_parallel=%.3e wake_pair=%.1f corners=%d/%d\n",
                isempty(label) ? "" : " [$label]", q.height_target, q.nfold, q.area_min,
                q.growth_max_used, q.spacing_ratio_symmetric_max, gworst, gmean,
                q.wall_parallel_err, q.wake_pair_err,
                q.lower_corner_count, q.upper_corner_count)
        if mesh.strip === nothing
            strip_met === nothing || error("$label mesh has no strip block but strip metrics were passed")
            return merge(q, (gcl_worst=gworst, gcl_mean=gmean))
        end
        # strip topology: the second block, from its own metrics (fail closed without them)
        strip_met === nothing && error("$label mesh carries a strip block: pass its metrics (build_strip_metrics) as strip_met")
        sb = mesh.strip
        (strip_met.NI, strip_met.NJ, strip_met.NIW, strip_met.NIF) == (sb.NI, sb.NJ, sb.NI, 0) ||
            error("$label strip metrics $((strip_met.NI, strip_met.NJ, strip_met.NIW, strip_met.NIF)) do not describe the strip block $((sb.NI, sb.NJ))")
        all(==(OUTER_STRIP_INTERFACE), strip_met.outer_face_kind) || error("$label strip metrics carry a channel tag")
        sworst, smean = gcl_check(strip_met)
        sworst < 1e-6 || error("$label strip GCL $sworst exceeds 1e-6")
        ssine, sat = corner_sine_min(sb.X, sb.Y, 1:sb.NI, 1:sb.NJ)
        ssine >= 0.2 || error("$label strip minimum corner sine $ssine at $sat is below 0.2")
        @printf("REALISED STRIP QUALITY%s: cells=%dx%d (n_base %d) edge_coincidence=%.1f min_sine=%.6f at (%d,%d) GCL[max/mean]=%.3e/%.3e wdist[min/max]=%.3e/%.3e\n",
                isempty(label) ? "" : " [$label]", sb.NI, sb.NJ, sb.n_base, q.wake_pair_err, ssine, sat[1], sat[2],
                sworst, smean, minimum(strip_met.wdist), maximum(strip_met.wdist))
        return merge(q, (gcl_worst=gworst, gcl_mean=gmean, strip_gcl_worst=sworst, strip_gcl_mean=smean, strip_sine_min=ssine))
    end
    # ★ FREE-AIR WAKE-CUT GATE (2026-08-07). The channel branch above has asserted wake-partner
    # coincidence since it was written; THIS branch -- the one every cavitation battery runs -- never
    # did. The C-mesh branch cut requires node (i,1) and node (NI+2-i,1) to be the same physical point
    # for i in 1..NIW+1; ~8 solver sites exchange ghosts across it on that assumption.
    # Measured while scoping the blunt trailing edge: a finite-TE foil silently drives this to
    # 4.000e-3 against 6.78e-21 for the sharp production foil -- no error, no warning, wrong answers.
    # Threshold 1e-12 is ~1e9x the realised sharp value, so it cannot fire on a legitimate mesh.
    mq.wake_pair_err <= 1e-12 || error(
        "$label wake-cut partners are not coincident (max |node(i,1) - node(NI+2-i,1)| = " *
        "$(mq.wake_pair_err)). The C-mesh branch cut is broken and the solver will exchange ghosts " *
        "between cells that are not the same point. A finite-thickness trailing edge does this: the " *
        "free-air path does not mesh the base. See docs/2026-08-07_BLUNT_TE_SCOPED.md.")
    @printf("REALISED LEGACY MESH QUALITY%s: folds=%d area_min=%.8e growth_max=%.8g GCL[max/mean]=%.3e/%.3e wake_pair=%.3e\n",
            isempty(label) ? "" : " [$label]", mq.nfold,
            mq.area_min, mq.growth_max_used, gworst, gmean, mq.wake_pair_err)
    merge(mq, (enabled=false, gcl_worst=gworst, gcl_mean=gmean))
end

"Return outer-face indices used for the measured upstream state."
function _upstream_outer_faces(s, met)
    kinds = met.outer_face_kind
    inlet = findall(==(_CHANNEL_INLET), kinds)
    !isempty(inlet) && return inlet

    # Legacy C far field: take the eight outer cells furthest upstream in the realised freestream
    # frame. This is an explicit measurement convention for arm F, whose semicircle has no inlet
    # mask. The W/C arms instead use every face in their exact inlet segment above.
    xi = [Float64(met.xc[i, met.NJ])*Float64(s.ca) +
          Float64(met.yc[i, met.NJ])*Float64(s.sa) for i in 1:met.NI]
    sortperm(xi)[1:min(8, met.NI)]
end

"Return the eight channel-inlet cells nearest the realised freestream centreline."
function _matched_upstream_outer_faces(s, met)
    inlet = findall(==(_CHANNEL_INLET), met.outer_face_kind)
    isempty(inlet) && return _upstream_outer_faces(s, met)
    ca, sa = Float64(s.ca), Float64(s.sa)
    ordered = sort(inlet; by=i ->
        (abs(-(Float64(met.xc[i, met.NJ])-0.5)*sa + Float64(met.yc[i, met.NJ])*ca), i))
    ordered[1:min(8, length(ordered))]
end

"Measure upstream state/wall fluxes; optional mixture_pressure supplies the actual full-ghost flux pressure."
function channel_diagnostics(s, tem, met, Uref::Real; label::AbstractString="", mixture_pressure=nothing)
    kinds = met.outer_face_kind
    length(kinds) == met.NI || error("outer-face mask length $(length(kinds)) != NI=$(met.NI)")
    inlet = _upstream_outer_faces(s, met)

    g, NJ = s.g, s.NJ
    rho = Array(s.rho); ux = Array(s.ux); uy = Array(s.uy); T = Array(s.T)
    pexc = Array(s.pexc)
    pressure = mixture_pressure === nothing ? nothing : Array(mixture_pressure)
    if pressure !== nothing
        size(pressure) == size(rho) && all(p -> isfinite(p) && p > 0, pressure) ||
            throw(ArgumentError("channel diagnostic mixture pressure must be finite, positive and full-ghost"))
    end
    Sjx = Array(s.Sjx); Sjy = Array(s.Sjy)
    weights = [hypot(Float64(Sjx[i, NJ+1]), Float64(Sjy[i, NJ+1])) for i in inlet]
    wsum = sum(weights)
    wsum > 0 || error("upstream diagnostic selected zero-length outer faces")
    wmean(f) = sum(weights[k]*f(inlet[k]) for k in eachindex(inlet))/wsum

    rhou = wmean(i -> Float64(rho[i+g, NJ+g]))
    uxu = wmean(i -> Float64(ux[i+g, NJ+g]))
    uyu = wmean(i -> Float64(uy[i+g, NJ+g]))
    Tu = wmean(i -> Float64(T[i+g, NJ+g]))
    pu = pressure === nothing ? wmean(i -> Float64(rho[i+g, NJ+g])*Float64(TEM_CS2)*
                     Float64(T[i+g, NJ+g]) +
                     (s.cav >= Int32(1) ? Float64(pexc[i+g, NJ+g]) : 0.0)) :
         wmean(i -> Float64(pressure[i+g, NJ+g]))
    Umag = hypot(uxu, uyu)
    q = wmean(i -> 0.5*Float64(rho[i+g, NJ+g]) *
                     (Float64(ux[i+g, NJ+g])^2 + Float64(uy[i+g, NJ+g])^2))
    sigma_eff = (pu - Float64(tem.p_v))/max(q, eps(Float64))
    Re_eff = Umag/max(Float64(s.nu_lam), eps(Float64))

    walls = findall(k -> k == _CHANNEL_LOWER || k == _CHANNEL_UPPER, kinds)
    mass_abs = 0.0; energy_abs = 0.0; mass_net = 0.0; energy_net = 0.0
    mass_norm = 0.0; energy_norm = 0.0
    if !isempty(walls)
        Fr = Array(s.Fj_r); Fe = Array(s.Fj_e)
        for i in walls
            mass_abs += abs(Float64(Fr[i, NJ+1])); energy_abs += abs(Float64(Fe[i, NJ+1]))
            mass_net += Float64(Fr[i, NJ+1]); energy_net += Float64(Fe[i, NJ+1])
        end
        wall_length = sum(hypot(Float64(Sjx[i, NJ+1]), Float64(Sjy[i, NJ+1])) for i in walls)
        mass_scale = max(rhou*max(Umag, Float64(Uref))*wall_length, eps(Float64))
        hscale = Float64(s.eint)*max(Tu, 0.0) + 0.5*max(Umag, Float64(Uref))^2
        energy_scale = max(mass_scale*hscale, eps(Float64))
        mass_norm = mass_abs/mass_scale; energy_norm = energy_abs/energy_scale
    end

    tag = isempty(label) ? "" : " [$label]"
    pressure === nothing || println("UPSTREAM PRESSURE SOURCE", tag, ": supplied physical mixture pressure")
    @printf("REALISED UPSTREAM%s: faces=%d rho=%.8g p=%.8e U=%.8g Re_eff=%.8g sigma_eff=%.8g\n",
            tag, length(inlet), rhou, pu, Umag, Re_eff, sigma_eff)
    if !isempty(walls)
        # The all-inlet mean above is the historical W/C diagnostic.  This additional eight-cell
        # measurement is approximately location-matched to legacy F's upstream-most eight-cell
        # convention when the rectangular inlet is placed at the same far extent.  Its printed
        # physical support makes the remaining quadrature mismatch explicit.  It is read-only and
        # leaves the historical line and return tuple unchanged.
        matched = _matched_upstream_outer_faces(s, met)
        mweights = [hypot(Float64(Sjx[i, NJ+1]), Float64(Sjy[i, NJ+1])) for i in matched]
        msum = sum(mweights)
        msum > 0 || error("matched upstream diagnostic selected zero-length outer faces")
        mmean(f) = sum(mweights[k]*f(matched[k]) for k in eachindex(matched))/msum
        mrho = mmean(i -> Float64(rho[i+g, NJ+g]))
        mux = mmean(i -> Float64(ux[i+g, NJ+g]))
        muy = mmean(i -> Float64(uy[i+g, NJ+g]))
        mp = pressure === nothing ? mmean(i -> Float64(rho[i+g, NJ+g])*Float64(TEM_CS2)*
                        Float64(T[i+g, NJ+g]) +
                        (s.cav >= Int32(1) ? Float64(pexc[i+g, NJ+g]) : 0.0)) :
             mmean(i -> Float64(pressure[i+g, NJ+g]))
        mU = hypot(mux, muy)
        mq = mmean(i -> 0.5*Float64(rho[i+g, NJ+g]) *
                          (Float64(ux[i+g, NJ+g])^2 + Float64(uy[i+g, NJ+g])^2))
        msigma = (mp-Float64(tem.p_v))/max(mq, eps(Float64))
        mRe = mU/max(Float64(s.nu_lam), eps(Float64))
        ca, sa = Float64(s.ca), Float64(s.sa)
        mxi = [0.5 + (Float64(met.xc[i, NJ])-0.5)*ca + Float64(met.yc[i, NJ])*sa
               for i in matched]
        meta = [-(Float64(met.xc[i, NJ])-0.5)*sa + Float64(met.yc[i, NJ])*ca
                for i in matched]
        @printf("REALISED UPSTREAM MATCHED8%s: faces=%d face_length=%.8g xi=[%.8g,%.8g] eta=[%.8g,%.8g] rho=%.8g p=%.8e U=%.8g Re_eff=%.8g sigma_eff=%.8g\n",
                tag, length(matched), msum, minimum(mxi), maximum(mxi), minimum(meta), maximum(meta),
                mrho, mp, mU, mRe, msigma)
    end
    if isempty(walls)
        @printf("REALISED OUTER WALL BUDGET%s: no masked tunnel-wall faces (legacy far field)\n", tag)
    else
        @printf("REALISED OUTER WALL BUDGET%s: faces=%d mass_net=%+.8e mass_abs=%.8e norm=%.3e energy_net=%+.8e energy_abs=%.8e norm=%.3e\n",
                tag, length(walls), mass_net, mass_abs, mass_norm, energy_net, energy_abs, energy_norm)
    end

    (rho=rhou, p=pu, ux=uxu, uy=uyu, U=Umag, Re_eff=Re_eff, sigma_eff=sigma_eff,
     inlet_faces=length(inlet), wall_faces=length(walls), wall_mass_net=mass_net,
     wall_mass_abs=mass_abs, wall_mass_norm=mass_norm, wall_energy_net=energy_net,
     wall_energy_abs=energy_abs, wall_energy_norm=energy_norm)
end
