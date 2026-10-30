# =====================================================================================================================
# assemble_pressure.jl -- the pressure equation of plan 4.5-4.6 in its psi = 0 limit (design A's volumetric pressure
# equation, interPhaseChangeFoam's pEqn without the source), and the conservative corrections. Stage P0b, 14 Oct 2026.
#
# THE EQUATION, for the correction dp = p^(k) - p^(k-1) of one PISO corrector (cell P, outward faces f):
#   sum_f D_f |S|^2/(S.d) (dp_P - dp_N) + sum_outlets D_b |S_b|^2/(S_b.d_b) dp_P
#        = V_P (Theta^n_P/dt + m_P) + s_P - sum_f F*_f(out) - sum_b F*_b
# with F* and D from the momentum interpolation (rhie_chow_flux!), Theta^n = rho^n/rho_l - 1 the closure defect of level n
# (the "provisional update's closure-defect rate" of 4.5: it drives the transported mixture mass back onto the volume
# constraint, so Theta^{n+1} is the linear solve's residual), m the MMS mass source, and the source hook (the phase-change
# term of 4.5, -V (1/rho_v - 1/rho_l) dGamma/dp on the diagonal and its value s_P on the right) PRESENT AND ZERO in P0b
# (`source_diag`, `source_rhs` arguments, all zeros; P5a fills them).
# The matrix is the two-point Laplacian of D: symmetric, positive diagonal, non-positive couplings, so SPD with any
# pressure-Dirichlet face. Without one (a closed box) the pressure is fixed by a reference cell: its row gains its own
# diagonal kappa and the right side kappa (p_ref - p_r), which leaves the solution of a consistent system unchanged and
# makes the matrix SPD (FCG + the Schaffer cycle solve it; the coarsest inverse needs a nonsingular matrix).
# THE CORRECTIONS (correct_fluxes!): F = F* - D |S|^2/(S.d) (dp_N - dp_P) on internal faces, F_b = F*_b + D_b (...) dp_P on
# outlets (the flux through every other boundary face is fixed); p += dp; u from the cell equation with the new pressure,
# u = HbyA - (V/A) (grad p^(k-1))^GG - D^c (grad dp)^GG, D^c = V/(a^t + (1/lambda - 1) a) (plan 4.3's D at the cell).
# The mixture mass is then transported in flux form with the final flux (transport_mass!): exact to round-off.
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

@kernel function _k_p_face!(up, lo, Df, fa)
    f = @index(Global)
    @inbounds begin
        c = -Df[f]*fa[f]; up[f] = c; lo[f] = c
    end
end
@kernel function _k_p_cell!(dg, rhs, up, lo, Fs, Fbs, Db, ba, bkind, cface, nf::Int32, vol, rhon, rho_l::Float64,
                            dt::Float64, xc, yc, mms, tn::Float64, sdiag, srhs, ktb::Int32, tbar::Float64)
    c = @index(Global)
    @inbounds begin
        d = sdiag[c]; div = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf; div += Fbs[b]
                is_pressure_bc(bkind[b]) && (d += Db[b]*ba[b])
            elseif code > 0
                d -= up[code]; div += Fs[code]
            else
                d -= lo[-code]; div -= Fs[-code]
            end
        end
        V = vol[c]
        dg[c] = d
        th = rhon[c]/rho_l - 1
        ktb == Int32(1) && (th -= tbar)                  # theta_drive = :zero_mean (opt-in; pimple.jl, PBConfig)
        rhs[c] = V*(th/dt + mms_source(mms, EQ_MASS, xc[c], yc[c], tn)) + srhs[c] - div
    end
end
@kernel function _k_p_ref!(dg, rhs, p, r::Int32, pref::Float64)
    t = @index(Global)
    @inbounds begin
        kap = dg[r]; dg[r] = 2kap; rhs[r] += kap*(pref - p[r])
    end
end

"""
    assemble_pressure!(w, mesh, st, bc, cfg, t_new; tbar=0.0)

The corrector's pressure equation for dp into w.psys, from w.Fs, w.Fbs, w.Df, w.Db (rhie_chow_flux!). The source hook
(w.sdiag, w.srhs) is zero in P0b. `tbar`, when not 0 (only under the opt-in theta_drive = :zero_mean, from
`theta_drive_mean`), is taken off Theta^n in the closure-defect drive; at 0 the kernel's arithmetic is the old one.
"""
function assemble_pressure!(w, m::PBMesh, st::PBState, bc, cfg, t_new::Real; tbar::Real=0.0)
    be = cfg.backend; ps = w.psys
    _launch!(_k_p_face!, be, m.nf, ps.upper, ps.lower, w.Df, m.fa)
    _launch!(_k_p_cell!, be, m.nc, ps.diag, ps.rhs, ps.upper, ps.lower, w.Fs, w.Fbs, w.Db, m.ba, bc.kind, m.cface,
             Int32(m.nf), m.vol, lev(st.rho, 2), cfg.rho_l, st.dt, m.xc, m.yc, cfg.mms, Float64(t_new), w.sdiag, w.srhs,
             Int32(tbar == 0 ? 0 : 1), Float64(tbar))
    if bc.npdir == 0
        r = cfg.p_ref_cell > 0 ? cfg.p_ref_cell : 1
        _launch!(_k_p_ref!, be, 1, ps.diag, ps.rhs, lev(st.p, 1), Int32(r), Float64(cfg.p_ref))
    end
    pb_sync(be)
    ps
end

@kernel function _k_corr_face!(F, Fs, Df, fa, dp, own, nei)
    f = @index(Global)
    @inbounds F[f] = Fs[f] - Df[f]*fa[f]*(dp[nei[f]] - dp[own[f]])
end
@kernel function _k_corr_bface!(Fb, Fbs, Db, ba, dp, bkind, bown)
    b = @index(Global)
    @inbounds Fb[b] = is_pressure_bc(bkind[b]) ? Fbs[b] + Db[b]*ba[b]*dp[bown[b]] : Fbs[b]
end
@kernel function _k_corr_cell!(u, v, p, dp, Hx, Hy, A, a, at, un, vn, upv, vpv, ggx, ggy, gdx, gdy, vol, lam::Float64)
    c = @index(Global)
    @inbounds begin
        rel = (1 - lam)/lam*a[c]; Ac = A[c]; V = vol[c]; Dc = V/(at[c] + (1/lam - 1)*a[c])
        u[c] = (Hx[c] + at[c]*un[c] + rel*upv[c])/Ac - (V/Ac)*ggx[c] - Dc*gdx[c]
        v[c] = (Hy[c] + at[c]*vn[c] + rel*vpv[c])/Ac - (V/Ac)*ggy[c] - Dc*gdy[c]
        p[c] += dp[c]
    end
end

"""
    correct_fluxes!(w, mesh, st, bc, cfg, lam)

The conservative corrections of one corrector from w.dp (the solved correction): face fluxes, pressure, velocity.
The velocity uses w.ggx, w.ggy (the Green-Gauss gradient of p^(k-1), from the corrector's start).
"""
function correct_fluxes!(w, m::PBMesh, st::PBState, bc, cfg, lam::Real)
    be = cfg.backend
    pressure_gradients!(w.gdlx, w.gdly, w.gdx, w.gdy, w.pbf2, w.dp, w, m, bc, cfg; correction=true)
    _launch!(_k_corr_face!, be, m.nf, lev(st.F, 1), w.Fs, w.Df, m.fa, w.dp, m.own, m.nei)
    _launch!(_k_corr_bface!, be, m.nb, lev(st.Fb, 1), w.Fbs, w.Db, m.ba, w.dp, bc.kind, m.bown)
    _launch!(_k_corr_cell!, be, m.nc, lev(st.ux, 1), lev(st.uy, 1), lev(st.p, 1), w.dp, w.Hx, w.Hy, w.msx.diag, _arel(w, cfg), w.at,
             lev(st.ux, 2), lev(st.uy, 2), w.uprev, w.vprev, w.ggx, w.ggy, w.gdx, w.gdy, m.vol, Float64(lam))
    pb_sync(be)
    nothing
end

@kernel function _k_transport!(rho, theta, rhon, F, Fb, cface, nf::Int32, vol, rho_l::Float64, dt::Float64, xc, yc, mms,
                               tn::Float64)
    c = @index(Global)
    @inbounds begin
        s = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                s += Fb[code - nf]
            elseif code > 0
                s += F[code]
            else
                s -= F[-code]
            end
        end
        r = rhon[c] - dt/vol[c]*rho_l*s + dt*rho_l*mms_source(mms, EQ_MASS, xc[c], yc[c], tn)
        rho[c] = r; theta[c] = r/rho_l - 1
    end
end

"""
    transport_mass!(w, mesh, st, cfg, t_new)

rho^{n+1} = rho^n - (dt/V) sum_f rho_l F_f(out) (+ the MMS mass source), with the final fluxes; Theta = rho/rho_l - 1 into
w.theta. Flux form, each face once with opposite signs on its two cells: the mass inventory changes by the boundary
fluxes alone, to round-off (ledger.jl measures it).
"""
function transport_mass!(w, m::PBMesh, st::PBState, cfg, t_new::Real)
    _launch!(_k_transport!, cfg.backend, m.nc, lev(st.rho, 1), w.theta, lev(st.rho, 2), lev(st.F, 1), lev(st.Fb, 1),
             m.cface, Int32(m.nf), m.vol, cfg.rho_l, st.dt, m.xc, m.yc, cfg.mms, Float64(t_new))
    pb_sync(cfg.backend)
    nothing
end
