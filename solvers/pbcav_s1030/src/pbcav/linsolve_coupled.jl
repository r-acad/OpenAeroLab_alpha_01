# =====================================================================================================================
# linsolve_coupled.jl -- the coupled fallback of plan 5.3 (graft from design C): the (u, v, p) system of one outer
# iteration solved together by FGMRES(30), right-preconditioned by C's SIMPLEC-type block preconditioner, in place of the
# segregated predictor-corrector loop of pimple.jl. Stage P0b (psi = 0, laminar, single phase), 15 Oct 2026; built opt-in
# (plan 27 registered its comparison) and ADOPTED by plan Amendment 8 (28.3): PBConfig.coupling = :coupled is the default
# since then; :segregated stays selectable, bitwise the old step (PB_DEFAULTS_BEFORE_A8 restores every old default).
#
# THE OUTER ITERATION (Picard; plan 4.7's tolerances unchanged). At the iterate x^m = (u, v, p)^m with the convecting flux
# F^m (the face flux of the iterate):
#   1. the momentum system is assembled exactly as the segregated loop assembles it (assemble_momentum!: implicit
#      first-order upwind in F^m, two-point viscous, backward Euler; the deferred terms -- the convection increment, the
#      non-orthogonal viscous part, the Green-Gauss pressure force on skewness-corrected faces -- evaluated at x^m); its
#      unrelaxed residual in increment form is the step's momentum criterion, evaluated on the iterate itself;
#   2. the face flux of x^m by the momentum interpolation at lambda = 1 (rhie_chow_flux!, Choi's term on F^n, no Majumdar
#      term), F_RC(x^m), and the continuity residual R_c = V (Theta^n/dt + m) - sum_f F_RC(out): the pressure equation's
#      right side (assemble_pressure!), so Theta^{m+1} = dt R_c(x^{m+1}) / V;
#   3. J dx = R is solved by FGMRES, J the derivative of (R_u, R_v, R_c) with the coefficients frozen at x^m (the face
#      coefficients, a, a^t, the convecting flux): EVERY term linear in (u, v, p) is in J -- the implicit stencil, the
#      deferred convection increment and non-orthogonal viscous term (so only the limiter-free forms :upwind and
#      :linear_upwind_unlimited are admitted), the Green-Gauss pressure force with its skewness correction, and in the
#      continuity rows the momentum interpolation's dependence on u (through H^s, with its skewness correction) and on p
#      (the compact two-point part and the non-orthogonal part). J is applied matrix-free by the kernels below, which are
#      the assemblers' own face and cell formulas run on the increment with homogeneous boundary data; the suite checks that
#      R(x + dx) - R(x) = -J dx to round-off. With cpl_newton (the default) J's momentum rows also carry the convection's
#      derivative in the face flux (rho F_lin(dx) (u_f - u_P) on the owner, minus (u_f - u_N) on the neighbour, u_f the
#      face value the assembly convects), and the right side the same term on the frozen flux's own change
#      F_RC(x^m) - F^m; without it the convecting flux is lagged (Picard). What stays lagged either way is the momentum
#      interpolation's dependence on the convecting flux (through H^s and a): development probes D and E (plan 27.1)
#      measured that it, not the linear solve, sets the outer contraction (about tenfold per solve on LE256).
#   4. x^{m+1} = x^m + dx, F^{m+1} = F_RC(x^m) + F_lin(dx) (the same face formula, so the flux is the momentum
#      interpolation of x^{m+1} with the frozen coefficients), the mixture mass transported in flux form with F^{m+1}
#      (transport_mass!: exact to round-off), Theta^{m+1}.
#   The step is accepted at the start of an outer iteration (after at least one solve) when the iterate meets every
#   criterion of 4.7 (momentum residual <= mom_tol, max |Theta| <= theta_tol, the last Krylov solve converged, bounds); it is
#   rejected with the segregated loop's typed reasons and rules (prediction at iteration 7, :outer_stalled, N_max solves).
#   There is no under-relaxation and no separate final iteration: the iterate checked is the iterate committed.
#
# THE ROW SCALING (block scaling, as C, written in the criteria's units): momentum rows times L/(rho U^2 V mom_tol),
#   continuity rows times dt/(V theta_tol), so a scaled residual of max-norm 1 is the step tolerance. FGMRES minimises the
#   scaled 2-norm and stops at max(cpl_eta ||r0||, cpl_abs) (cpl_eta the forcing term, 1e-1 by default since Amendment 8, 1e-2 before; cpl_abs 0.1:
#   below it every row is under a tenth of its tolerance).
#
# THE PRECONDITIONER (C's SIMPLEC block factorisation, BlockPC kind :simplec, au = cpl_au = 0.8):
#   u* = A~^-1 r_u      cpl_sweeps alternating coloured line Gauss-Seidel sweeps (j then i) from zero on A~ = the momentum
#                       matrix with diagonal a/au (line_gs_sweep!, the segregated loop's line kernels); v* likewise;
#   dp = S~^-1 (r_c - B u*)   B the first-order part of the continuity rows' u-block (the face flux of H = sum a_N u*_N,
#                       no deferred or skewness part); S~ the two-point Laplacian with SIMPLEC's D_f = Vbar/(abar^t +
#                       (1/au - 1) abar) (the segregated pressure equation's operator at lambda = au), one V(1,1) cycle
#                       of the P0a Schaffer multigrid, the hierarchy kept by the F5 policy (mg_update!);
#   du = u* - D^c (grad dp)^GG,   D^c = V/(a^t + (1/au - 1) a).
#   It is a fixed linear map per outer iteration, so FGMRES's flexibility is not needed for it (FGMRES is C's choice and
#   tolerates a lagged hierarchy).
# THE ALPHA SLOT. The block holds CPL_NF = 3 fields (u, v, p); the phase alpha (C's fourth field, with its local (p, alpha)
#   elimination in the Schur complement) is slot 4 and joins at stage P5a; nothing here reads it.
# KERNELS. Every loop over cells or faces is a KernelAbstractions kernel; the Krylov scalars are read on the host (one read
#   per dot product here; W7 (plan 33): cpl_gs = :mgs_dev keeps them on the device, :cgs2 fuses them, graphs = :on replays J
#   and M as CUDA graphs). No @Const (KernelAbstractions
#   issue 652, PBCav.jl).
# =====================================================================================================================

"Fields of the coupled block, in order. The phase alpha is slot CPL_ALPHA_SLOT (stage P5a); it is not in the block yet."
const CPL_FIELDS = (:u, :v, :p)
const CPL_NF = length(CPL_FIELDS)
const CPL_ALPHA_SLOT = 4

"""
    CoupledWork

The coupled path's work space (allocated once per PBWork on the first coupled step): the block vectors (length
CPL_NF nc, field k at rows (k-1) nc + 1 : k nc), the FGMRES bases, the row scaling, the preconditioner's momentum systems
(shared off-diagonals, diagonal a/au, one right side per component), D^c, the frozen convecting flux, and the scratch of
the matrix-free operator.
"""
mutable struct CoupledWork{V,M,FS}
    nc::Int; N::Int; restart::Int
    Wt::V; R::V; Y::V; T1::V; T2::V; dx::V
    Vb::M; Zb::M
    H::Matrix{Float64}; g::Vector{Float64}; cs::Vector{Float64}; sn::Vector{Float64}
    Pu::FS; Pv::FS; pdiag::V; Dc::V
    Fconv::V; Fbconv::V
    zb::V; zc::V; ubl::V; vbl::V; pbl::V
    gux::V; guy::V; gvx::V; gvy::V; cux::V; cuy::V; cvx::V; cvy::V
    ups::V; los::V; fcx::V; fcy::V; bdgs::V; bfx::V; bfy::V
    gpx::V; gpy::V; ggx::V; ggy::V
    Hx::V; Hy::V; hbx::V; hby::V; hxx::V; hxy::V; hyx::V; hyy::V
    Fl::V; Fbl::V; dvc::V; rc2::V; rw2::V; zw2::V
    eox::V; eoy::V; enx::V; eny::V; dF0::V
    part::V; sc::V
    kref::Float64; rref::Int
    matvecs::Int; precs::Int
    w7::Any                  # W7 (speed): the Float32 preconditioner copies and the device Gram-Schmidt scalars, or nothing
end

function CoupledWork(m::PBMesh, w, cfg)
    be = cfg.backend; nc, nf, nb = m.nc, m.nf, m.nb; N = CPL_NF*nc; rs = cfg.cpl_restart
    z(n) = KernelAbstractions.zeros(be, Float64, n)
    pdiag = z(nc)
    Pu = FaceSystem(pdiag, w.msx.upper, w.msx.lower, z(nc)); Pv = FaceSystem(pdiag, w.msx.upper, w.msx.lower, z(nc))
    CoupledWork(nc, N, rs, z(N), z(N), z(N), z(N), z(N), z(N),
                KernelAbstractions.zeros(be, Float64, N, rs + 1), KernelAbstractions.zeros(be, Float64, N, rs),
                zeros(rs + 1, rs), zeros(rs + 1), zeros(rs), zeros(rs),
                Pu, Pv, pdiag, z(nc), z(nf), z(nb),
                z(max(nb, 1)), z(nc), z(nb), z(nb), z(nb),
                z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc),
                z(nf), z(nf), z(nf), z(nf), z(nb), z(nb), z(nb),
                z(nc), z(nc), z(nc), z(nc),
                z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc),
                z(nf), z(nb), z(nc), z(nc), z(nc), z(nc),
                z(nf), z(nf), z(nf), z(nf), z(nf),
                z(max(cld(N, PB_DOT_CHUNK), 1)), z(8),
                0.0, 0, 0, 0, nothing)
end

"The field k (1 u, 2 v, 3 p) of a block vector: a contiguous view."
@inline cfield(x, k::Integer, nc::Integer) = view(x, (k-1)*nc+1:k*nc)

"Check that a configuration can take the coupled path (the Jacobian is exact only for the limiter-free forms)."
function check_coupled_config(cfg)
    cfg.coupling in (:segregated, :coupled) ||
        throw(ArgumentError("PBConfig.coupling must be :segregated or :coupled (got :$(cfg.coupling))"))
    cfg.coupling === :coupled || return nothing
    cfg.convection in (:upwind, :linear_upwind_unlimited) ||
        throw(ArgumentError("the coupled path takes the limiter-free convection forms only (:upwind, :linear_upwind_unlimited; " *
                            "got :$(cfg.convection)): a limited form is not linear, so its Jacobian is not the one applied"))
    (cfg.nonorth == 0 || cfg.nonorth == 1) ||
        throw(ArgumentError("the coupled path takes nonorth 0 or 1 (the limited non-orthogonal correction is not linear; got $(cfg.nonorth))"))
    cfg.relax_form === :patankar || throw(ArgumentError("the coupled path has no relaxation form (relax_form must be :patankar)"))
    (0 < cfg.cpl_au <= 1) || throw(ArgumentError("PBConfig.cpl_au must lie in (0, 1] (got $(cfg.cpl_au))"))
    (cfg.cpl_restart >= 1 && cfg.cpl_maxit >= 1 && cfg.cpl_sweeps >= 1 && cfg.cpl_vcycles >= 1) ||
        throw(ArgumentError("PBConfig: cpl_restart, cpl_maxit, cpl_sweeps and cpl_vcycles must be >= 1"))
    (0 < cfg.cpl_eta < 1 && cfg.cpl_abs >= 0) || throw(ArgumentError("PBConfig: 0 < cpl_eta < 1 and cpl_abs >= 0 required"))
    cfg.cpl_visc_nonorth in (:jacobian, :deferred) ||
        throw(ArgumentError("PBConfig.cpl_visc_nonorth must be :jacobian or :deferred (got :$(cfg.cpl_visc_nonorth))"))
    nothing
end

# ---- kernels ----------------------------------------------------------------------------------------------------------
"The momentum residual vectors in increment form (_k_mom_resid!'s formula, signed, unscaled)."
@kernel function _k_cpl_mom_res!(ru, rv, bsx, bsy, ggx, ggy, u, v, un, vn, as_, at, up, lo, cface, own, nei, nf::Int32, vol)
    c = @index(Global)
    @inbounds begin
        V = vol[c]; ax = as_[c]*u[c]; ay = as_[c]*v[c]
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                ax += up[code]*u[nei[code]]; ay += up[code]*v[nei[code]]
            elseif code < 0
                ax += lo[-code]*u[own[-code]]; ay += lo[-code]*v[own[-code]]
            end
        end
        ru[c] = bsx[c] - V*ggx[c] - ax - at[c]*(u[c] - un[c])
        rv[c] = bsy[c] - V*ggy[c] - ay - at[c]*(v[c] - vn[c])
    end
end

"""J's momentum rows on (du, dv, dp): yu = a du + sum_N up du_N - S(du) + V (grad dp)^GG, S the deferred terms' gather
(_k_mom_cell!'s), and the increment's H = S(du) - sum_N up du_N (H^s's linear part)."""
@kernel function _k_cpl_mom_apply!(yu, yv, Hx, Hy, du, dv, a, up, lo, fcx, fcy, bfx, bfy, ggx, ggy, cface, own, nei,
                                   nf::Int32, vol)
    c = @index(Global)
    @inbounds begin
        sx = 0.0; sy = 0.0; ox = 0.0; oy = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf; sx += bfx[b]; sy += bfy[b]
            elseif code > 0
                sx += fcx[code]; sy += fcy[code]; ox += up[code]*du[nei[code]]; oy += up[code]*dv[nei[code]]
            else
                sx -= fcx[-code]; sy -= fcy[-code]; ox += lo[-code]*du[own[-code]]; oy += lo[-code]*dv[own[-code]]
            end
        end
        V = vol[c]
        Hx[c] = sx - ox; Hy[c] = sy - oy
        yu[c] = a[c]*du[c] + ox - sx + V*ggx[c]
        yv[c] = a[c]*dv[c] + oy - sy + V*ggy[c]
    end
end

"H = sum_N a_N u_N = -sum_N up u_N (the preconditioner's first-order B: no deferred part)."
@kernel function _k_cpl_hoff!(Hx, Hy, u, v, up, lo, cface, own, nei, nf::Int32)
    c = @index(Global)
    @inbounds begin
        hx = 0.0; hy = 0.0
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                hx -= up[code]*u[nei[code]]; hy -= up[code]*v[nei[code]]
            elseif code < 0
                hx -= lo[-code]*u[own[-code]]; hy -= lo[-code]*v[own[-code]]
            end
        end
        Hx[c] = hx; Hy[c] = hy
    end
end

"""The momentum interpolation's linear part at lambda = 1 (_k_rc_face! without Choi's term): internal faces,
Fl = (Hbar . S + skew term - Vbar (|S|^2/(S.d) (dp_N - dp_P) + k . (grad dp)bar)) / abar."""
@kernel function _k_cpl_rc_face!(Fl, Hx, Hy, hxx, hxy, hyx, hyy, as_, a, vol, dp, gpx, gpy, own, nei, fw, fa, kx, ky, skx,
                                 sky, Sx, Sy, lamn::Float64, skew::Float64, du, dv, klin::Float64, krc::Float64)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; w = fw[f]; w1 = 1 - w
        ab = w*a[P] + w1*a[N]; Vb = w*vol[P] + w1*vol[N]
        HS = (w*Hx[P] + w1*Hx[N])*Sx[f] + (w*Hy[P] + w1*Hy[N])*Sy[f]
        klin != 0 && (HS += _rc_lin_term(as_, du, dv, P, N, w, w1, Sx[f], Sy[f]))     # plan 28.6 (F3)
        if skew != 0
            asb = w*as_[P] + w1*as_[N]; rx = skx[f]; ry = sky[f]
            HS += skew*asb*0.5*(((hxx[P] + hxx[N])*rx + (hxy[P] + hxy[N])*ry)*Sx[f] +
                                ((hyx[P] + hyx[N])*rx + (hyy[P] + hyy[N])*ry)*Sy[f])
        end
        tp = fa[f]*(dp[N] - dp[P]); np_ = 0.0
        lamn > 0 && (np_ = kx[f]*(w*gpx[P] + w1*gpx[N]) + ky[f]*(w*gpy[P] + w1*gpy[N]))
        # rhie_chow = :off (krc 0; plan 28.11): the linear face velocity of the increment, no pressure dependence
        Fl[f] = krc != 0 ? (HS - Vb*(tp + np_))/ab : (w*du[P] + w1*du[N])*Sx[f] + (w*dv[P] + w1*dv[N])*Sy[f]
    end
end

"The linear part at lambda = 1 on boundary faces: a pressure outlet's (_k_rc_bface!, p_b fixed); every other face 0."
@kernel function _k_cpl_rc_bface!(Fbl, Hx, Hy, a, vol, dp, gpx, gpy, bkind, bown, bSx, bSy, ba, bkx, bky, lamn::Float64,
                                  krc::Float64, du, dv)
    b = @index(Global)
    @inbounds begin
        P = bown[b]
        if is_pressure_bc(bkind[b])
            tp = -ba[b]*dp[P]; np_ = lamn > 0 ? bkx[b]*gpx[P] + bky[b]*gpy[P] : 0.0
            Fbl[b] = krc != 0 ? (Hx[P]*bSx[b] + Hy[P]*bSy[b] - vol[P]*(tp + np_))/a[P] :
                                du[P]*bSx[b] + dv[P]*bSy[b]           # rhie_chow = :off (plan 28.11)
        else
            Fbl[b] = 0.0
        end
    end
end

"The signed outward flux sum of every cell."
@kernel function _k_cpl_div!(y, F, Fb, cface, nf::Int32)
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
        y[c] = s
    end
end
@kernel function _k_cpl_ref!(y, dp, r::Int32, kap::Float64)
    t = @index(Global)
    @inbounds y[r] += kap*dp[r]
end
"SIMPLEC's D on faces at au (the Schur operator's), and on pressure outlets; 0 elsewhere."
@kernel function _k_cpl_dpc_face!(Df, a, at, vol, own, nei, fw, au::Float64)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; w = fw[f]; w1 = 1 - w
        Df[f] = (w*vol[P] + w1*vol[N])/((w*at[P] + w1*at[N]) + (1/au - 1)*(w*a[P] + w1*a[N]))
    end
end
@kernel function _k_cpl_dpc_bface!(Db, a, at, vol, bkind, bown, au::Float64)
    b = @index(Global)
    @inbounds begin
        P = bown[b]
        Db[b] = is_pressure_bc(bkind[b]) ? vol[P]/(at[P] + (1/au - 1)*a[P]) : 0.0
    end
end
@kernel function _k_cpl_cell_pc!(pdiag, Dc, a, at, vol, au::Float64)
    c = @index(Global)
    @inbounds begin
        pdiag[c] = a[c]/au; Dc[c] = vol[c]/(at[c] + (1/au - 1)*a[c])
    end
end
@kernel function _k_cpl_weights!(Wt, vol, nc::Int32, sm::Float64, sp::Float64)
    i = @index(Global)
    @inbounds begin
        c = (i - 1) % nc + 1
        Wt[i] = (i <= 2nc ? sm : sp)/vol[c]
    end
end
@kernel function _k_cpl_vcorr!(du, dv, Dc, ggx, ggy)
    c = @index(Global)
    @inbounds begin
        du[c] -= Dc[c]*ggx[c]; dv[c] -= Dc[c]*ggy[c]
    end
end
@kernel function _k_cpl_mul!(y, a, x)
    i = @index(Global)
    @inbounds y[i] = a[i]*x[i]
end
@kernel function _k_cpl_div_by!(y, x, a)
    i = @index(Global)
    @inbounds y[i] = x[i]/a[i]
end
@kernel function _k_cpl_axpy!(y, al::Float64, x)
    i = @index(Global)
    @inbounds y[i] += al*x[i]
end
@kernel function _k_cpl_scale!(y, al::Float64)
    i = @index(Global)
    @inbounds y[i] *= al
end
"y = W (R - Y)"
@kernel function _k_cpl_wres!(y, W, R, Yv)
    i = @index(Global)
    @inbounds y[i] = W[i]*(R[i] - Yv[i])
end
@kernel function _k_cpl_sub!(y, x)
    i = @index(Global)
    @inbounds y[i] -= x[i]
end
@kernel function _k_cpl_add2!(y, a, b)
    i = @index(Global)
    @inbounds y[i] = a[i] + b[i]
end

"""The convective face differences of the iterate for the Newton term (cpl_newton): with u_f the face value the momentum
assembly convects (the upwind cell's value plus, for :linear_upwind_unlimited, its increment r . g, _k_mom_face!'s),
eo = u_f - u_own and en = u_f - u_nei. The convective part of the owner's residual is -rho F (u_f - u_own) and the
neighbour's +rho F (u_f - u_nei) (the continuity-error-subtracted form, assemble_momentum.jl), so their derivatives in F
are -rho eo and +rho en (the upwind switch and the increment's side are held)."""
@kernel function _k_cpl_conv_faces!(eox, eoy, enx, eny, F, u, v, cux, cuy, cvx, cvy, own, nei, xf, yf, ftx, fty, xc, yc,
                                    lu::Int32)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]
        if F[f] >= 0
            uf = u[P]; vf = v[P]
            if lu >= Int32(1)
                rx = xf[f] - xc[P]; ry = yf[f] - yc[P]; uf += cux[P]*rx + cuy[P]*ry; vf += cvx[P]*rx + cvy[P]*ry
            end
        else
            uf = u[N]; vf = v[N]
            if lu >= Int32(1)
                rx = xf[f] - (xc[N] + ftx[f]); ry = yf[f] - (yc[N] + fty[f]); uf += cux[N]*rx + cuy[N]*ry; vf += cvx[N]*rx + cvy[N]*ry
            end
        end
        eox[f] = uf - u[P]; eoy[f] = vf - v[P]; enx[f] = uf - u[N]; eny[f] = vf - v[N]
    end
end
"""y += s rho sum_f dF_f e_f (owner faces +eo, neighbour faces -en): s = +1 J's Newton term on dF = F_lin(dx), s = -1 the
residual's correction for the frozen flux's own change dF0 = F_RC(x^m) - F^m."""
@kernel function _k_cpl_newton!(yu, yv, dF, eox, eoy, enx, eny, cface, nf::Int32, rho_l::Float64, s::Float64)
    c = @index(Global)
    @inbounds begin
        ax = 0.0; ay = 0.0
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                ax += dF[code]*eox[code]; ay += dF[code]*eoy[code]
            elseif code < 0
                ax -= dF[-code]*enx[-code]; ay -= dF[-code]*eny[-code]
            end
        end
        yu[c] += s*rho_l*ax; yv[c] += s*rho_l*ay
    end
end
@kernel function _k_cpl_sub2!(y, a, b)
    i = @index(Global)
    @inbounds y[i] = a[i] - b[i]
end

"x.y into the host (the deterministic two-pass reduction on the coupled work's preallocated partials)."
function cpl_dot(cw::CoupledWork, x, y; backend=CPU())
    _dot_dev!(cw.sc, 1, x, y, cw.part; backend)
    pb_sync(backend)
    _host(view(cw.sc, 1:1))[1]
end

# ---- the operator J -----------------------------------------------------------------------------------------------------
"""
    cpl_apply!(y, x, cw, w, mesh, bc, cfg) -> y

y = J x (matrix-free, the module header's J) with the coefficients last frozen by `cpl_freeze!`; also leaves the
increment's face fluxes F_lin(x) in cw.Fl, cw.Fbl.
"""
function _cpl_apply_body!(y, x, cw::CoupledWork, w, m::PBMesh, bc, cfg)
    be = cfg.backend; nc = m.nc; G = w.lsq
    du = cfield(x, 1, nc); dv = cfield(x, 2, nc); dp = cfield(x, 3, nc)
    _launch!(_k_bvel!, be, m.nb, cw.ubl, cw.vbl, du, dv, bc.kind, cw.zb, cw.zb, m.bown, m.bSx, m.bSy, m.bSmag)
    for (g1, g2, phi, phib) in ((cw.gux, cw.guy, du, cw.ubl), (cw.gvx, cw.gvy, dv, cw.vbl))
        _launch!(_k_lsq_grad!, be, nc, g1, g2, phi, phib, G.u11, G.u12, G.u22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(0))
    end
    lu = Int32(cfg.convection === :linear_upwind_unlimited ? 2 : 0)
    unitg = lu >= 1 && cfg.conv_gradient === :unit
    if unitg
        for (g1, g2, phi, phib) in ((cw.cux, cw.cuy, du, cw.ubl), (cw.cvx, cw.cvy, dv, cw.vbl))
            _launch!(_k_lsq_grad!, be, nc, g1, g2, phi, phib, G.c11, G.c12, G.c22, m.cface, m.own, m.nei, m.dx, m.dy,
                     m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(1))
        end
    end
    cg = unitg ? (cw.cux, cw.cuy, cw.cvx, cw.cvy) : (cw.gux, cw.guy, cw.gvx, cw.gvy)
    # the momentum's face kernels on the increment (laminar: the scalar mu; SST: the frozen mu_eff and the transpose term)
    # cpl_visc_nonorth = :deferred (W6b): J without the viscous non-orthogonal part (the residual keeps it; the outer loop
    # converges it as a deferred correction); :jacobian, the default, J carries it (bit for bit the path before W6b)
    _mom_face_kernels!(be, m, cfg, w, cw.ups, cw.los, cw.fcx, cw.fcy, cw.Fconv, cw.gux, cw.guy, cw.gvx, cw.gvy, cg, du, dv,
                       cw.bdgs, cw.bfx, cw.bfy, cw.Fbconv, bc.kind, cw.ubl, cw.vbl, lu;
                       lamn=cfg.cpl_visc_nonorth === :deferred ? 0.0 : Float64(cfg.nonorth))
    pressure_gradients!(cw.gpx, cw.gpy, cw.ggx, cw.ggy, cw.pbl, dp, w, m, bc, cfg; correction=true)
    _launch!(_k_cpl_mom_apply!, be, nc, cfield(y, 1, nc), cfield(y, 2, nc), cw.Hx, cw.Hy, du, dv, w.a, w.msx.upper,
             w.msx.lower, cw.fcx, cw.fcy, cw.bfx, cw.bfy, cw.ggx, cw.ggy, m.cface, m.own, m.nei, Int32(m.nf), m.vol)
    if cfg.skew
        _launch!(_k_hhat!, be, nc, cw.hbx, cw.hby, cw.Hx, cw.Hy, w.as_)
        for (g1, g2, phi) in ((cw.hxx, cw.hxy, cw.hbx), (cw.hyx, cw.hyy, cw.hby))
            _launch!(_k_lsq_grad!, be, nc, g1, g2, phi, cw.ubl, G.i11, G.i12, G.i22, m.cface, m.own, m.nei, m.dx, m.dy,
                     m.bdx, m.bdy, bc.kind, Int32(m.nf), LSQ_INT, Int32(0), Int32(0))
        end
    end
    _launch!(_k_cpl_rc_face!, be, m.nf, cw.Fl, cw.Hx, cw.Hy, cw.hxx, cw.hxy, cw.hyx, cw.hyy, w.as_, w.a, m.vol, dp, cw.gpx,
             cw.gpy, m.own, m.nei, m.fw, m.fa, m.kx, m.ky, m.skx, m.sky, m.Sx, m.Sy, Float64(cfg.nonorth), cfg.skew ? 1.0 : 0.0,
             du, dv, _rc_klin(cfg), _rc_on(cfg))
    _launch!(_k_cpl_rc_bface!, be, m.nb, cw.Fbl, cw.Hx, cw.Hy, w.a, m.vol, dp, cw.gpx, cw.gpy, bc.kind, m.bown, m.bSx, m.bSy,
             m.ba, m.bkx, m.bky, Float64(cfg.nonorth), _rc_on(cfg), du, dv)
    _launch!(_k_cpl_div!, be, nc, cfield(y, 3, nc), cw.Fl, cw.Fbl, m.cface, Int32(m.nf))
    cfg.cpl_newton && _launch!(_k_cpl_newton!, be, nc, cfield(y, 1, nc), cfield(y, 2, nc), cw.Fl, cw.eox, cw.eoy, cw.enx, cw.eny,
                                 m.cface, Int32(m.nf), cfg.rho_l, 1.0)
    cw.rref > 0 && _launch!(_k_cpl_ref!, be, 1, cfield(y, 3, nc), dp, Int32(cw.rref), cw.kref)
    pb_sync(be)
    y
end

# ---- the preconditioner ---------------------------------------------------------------------------------------------------
"""
    cpl_precond!(z, r, cw, w, mesh, bc, cfg) -> z

z = M r, C's SIMPLEC block preconditioner (the module header): momentum line sweeps, the first-order B, cpl_vcycles V(1,1)
cycles of the hierarchy w.mg on the Schur operator w.psys, the velocity correction.
"""
function _cpl_precond_body!(z, r, cw::CoupledWork, w, m::PBMesh, bc, cfg)
    be = cfg.backend; nc = m.nc
    zu = cfield(z, 1, nc); zv = cfield(z, 2, nc); zp = cfield(z, 3, nc)
    if cfg.prec_precision === :f32         # W7 (opt-in): the momentum sweeps in Float32 on the copies cpl_freeze! loaded
        F = _w7(cw)[:f32]
        pb_convert!(F.Pu.rhs, cfield(r, 1, nc); backend=be); pb_convert!(F.Pv.rhs, cfield(r, 2, nc); backend=be)
        fill!(F.zu, 0f0); fill!(F.zv, 0f0)
        for _ in 1:cfg.cpl_sweeps
            for (x, sys) in ((F.zu, F.Pu), (F.zv, F.Pv))
                # (22 Oct correction: the excess residual; the PCR on the Float64 families w.fj, w.fi)
                line_gs_sweep!(x, sys, m, F.fj, F.rw, F.zw, cfg.line_solver; backend=be, fam64=w.fj, ex=F.ex)
                line_gs_sweep!(x, sys, m, F.fi, F.rw, F.zw, cfg.line_solver; backend=be, fam64=w.fi, ex=F.ex)
            end
        end
        pb_convert!(zu, F.zu; backend=be); pb_convert!(zv, F.zv; backend=be)
    else
        copyto!(cw.Pu.rhs, cfield(r, 1, nc)); copyto!(cw.Pv.rhs, cfield(r, 2, nc))
        fill!(zu, 0.0); fill!(zv, 0.0)
        for _ in 1:cfg.cpl_sweeps
            for (x, sys) in ((zu, cw.Pu), (zv, cw.Pv))
                if cfg.line_solver === :thomas
                    line_gs_sweep!(x, sys, m, w.fj, w.rw, w.zw; backend=be)
                    line_gs_sweep!(x, sys, m, w.fi, w.rw, w.zw; backend=be)
                else                       # W7 (opt-in): the workgroup PCR in Float64
                    line_gs_sweep!(x, sys, m, w.fj, w.rw, w.zw, cfg.line_solver; backend=be)
                    line_gs_sweep!(x, sys, m, w.fi, w.rw, w.zw, cfg.line_solver; backend=be)
                end
            end
        end
    end
    # r_c - B u*: the face flux of H = sum a_N u*_N, first order (no deferred part, no skewness term)
    _launch!(_k_cpl_hoff!, be, nc, cw.Hx, cw.Hy, zu, zv, w.msx.upper, w.msx.lower, m.cface, m.own, m.nei, Int32(m.nf))
    _launch!(_k_cpl_rc_face!, be, m.nf, cw.Fl, cw.Hx, cw.Hy, cw.hxx, cw.hxy, cw.hyx, cw.hyy, w.as_, w.a, m.vol, cw.zc,
             cw.gpx, cw.gpy, m.own, m.nei, m.fw, m.fa, m.kx, m.ky, m.skx, m.sky, m.Sx, m.Sy, 0.0, 0.0, zu, zv, 0.0, 1.0)
    _launch!(_k_cpl_rc_bface!, be, m.nb, cw.Fbl, cw.Hx, cw.Hy, w.a, m.vol, cw.zc, cw.gpx, cw.gpy, bc.kind, m.bown, m.bSx,
             m.bSy, m.ba, m.bkx, m.bky, 0.0, 1.0, zu, zv)
    _launch!(_k_cpl_div!, be, nc, cw.dvc, cw.Fl, cw.Fbl, m.cface, Int32(m.nf))
    copyto!(cw.rc2, cfield(r, 3, nc)); _launch!(_k_cpl_sub!, be, nc, cw.rc2, cw.dvc)
    pb_sync(be)
    h = w.mg
    mg_vcycle!(zp, h, cw.rc2)
    for _ in 2:cfg.cpl_vcycles          # further cycles by residual correction on the Schur operator
        _launch!(_k_fs_resid!, be, nc, cw.rw2, cw.rc2, zp, w.psys.diag, w.psys.upper, w.psys.lower, m.cface, m.own, m.nei,
                 Int32(m.nf))
        mg_vcycle!(cw.zw2, h, cw.rw2); _launch!(_k_cpl_axpy!, be, nc, zp, 1.0, cw.zw2)
    end
    pressure_gradients!(cw.gpx, cw.gpy, cw.ggx, cw.ggy, cw.pbl, zp, w, m, bc, cfg; correction=true)
    _launch!(_k_cpl_vcorr!, be, nc, zu, zv, cw.Dc, cw.ggx, cw.ggy)
    pb_sync(be)
    z
end

# ---- W7: the coupled step's W7 store, and the CUDA graphs of J and M (opt-in, PBConfig.graphs) ----------------------------
"The coupled work's W7 store (made on first use): :f32 (the Float32 preconditioner copies), :graphs, :gx, :gz."
_w7(cw::CoupledWork) = cw.w7 === nothing ? (cw.w7 = Dict{Symbol,Any}()) : cw.w7
_graphs_on(cfg) = cfg.graphs === :on && !(cfg.backend isa CPU) && PB_GRAPH[] !== nothing

"""Run `body!(out, inp)` as a captured graph (W7, graphs = :on): the graph of `key` is captured on first use, after one
eager run (which also compiles every kernel, and gives this call's result), then replayed. `out`, `inp` must be the same
arrays at every call of a key (the caller copies into fixed buffers). A key names everything a capture depends on: the
hierarchy and its companion objects (a rebuild makes new arrays), and any scalar a kernel takes by value."""
function _cpl_graph!(body!, cw::CoupledWork, key, out, inp)
    G = get!(() -> Dict{Any,Any}(), _w7(cw), :graphs)
    ex = get(G, key, nothing)
    if ex === nothing
        body!(out, inp)
        gb = PB_GRAPH[]
        gc0 = GC.enable(false)       # no finalizer may free (and so enqueue) on the capturing stream (W7 review, finding 11)
        g = try
            gb.capture(() -> body!(out, inp))
        finally
            GC.enable(gc0)
        end
        ex = gb.instantiate(g)
        filter!(kv -> kv[1][1] !== key[1], G)            # one live graph per kind
        G[key] = ex
        return out
    end
    PB_GRAPH[].launch(ex)
    out
end

"""
    cpl_apply!(y, x, cw, w, mesh, bc, cfg) -> y

y = J x (matrix-free, the module header's J) with the coefficients last frozen by `cpl_freeze!`; also leaves the
increment's face fluxes F_lin(x) in cw.Fl, cw.Fbl. With graphs = :on (W7, a GPU) x is copied into a fixed buffer and the
operator's kernels are replayed from their captured graph (the same kernels on the same arrays).
"""
function cpl_apply!(y, x, cw::CoupledWork, w, m::PBMesh, bc, cfg)
    if _graphs_on(cfg) && y === cw.Y
        gx = get!(() -> similar(cw.dx), _w7(cw), :gx)
        copyto!(gx, x)
        _cpl_graph!((o, i) -> _cpl_apply_body!(o, i, cw, w, m, bc, cfg), cw, (:J, cw.rref, cw.kref, objectid(w.lsq), cfg.cpl_visc_nonorth), cw.Y, gx)
    else
        _cpl_apply_body!(y, x, cw, w, m, bc, cfg)
    end
    cw.matvecs += 1
    y
end

"""
    cpl_precond!(z, r, cw, w, mesh, bc, cfg) -> z

z = M r, C's SIMPLEC block preconditioner (the module header): momentum line sweeps, the first-order B, cpl_vcycles V(1,1)
cycles of the hierarchy w.mg on the Schur operator w.psys, the velocity correction. With graphs = :on (W7, a GPU) its kernels
are replayed from their captured graph into a fixed buffer and copied to z.
"""
function cpl_precond!(z, r, cw::CoupledWork, w, m::PBMesh, bc, cfg)
    if _graphs_on(cfg) && r === cw.T2
        gz = get!(() -> similar(cw.dx), _w7(cw), :gz)
        h = w.mg
        key = (:M, objectid(h), h.fast === nothing ? UInt(0) : objectid(h.fast), haskey(_w7(cw), :f32) ? objectid(_w7(cw)[:f32]) : UInt(0))
        _cpl_graph!((o, i) -> _cpl_precond_body!(o, i, cw, w, m, bc, cfg), cw, key, gz, cw.T2)
        copyto!(z, gz)
    else
        _cpl_precond_body!(z, r, cw, w, m, bc, cfg)
    end
    cw.precs += 1
    z
end

# ---- FGMRES(restart), right preconditioned, on the scaled system --------------------------------------------------------
"""
    cpl_fgmres!(cw, w, mesh, bc, cfg) -> NamedTuple

Solve J dx = R (cw.R) into cw.dx: FGMRES(cfg.cpl_restart), right-preconditioned by `cpl_precond!`, on the row-scaled
system W J (W = cw.Wt), from dx = 0; modified Gram-Schmidt, Givens rotations on the host. Stops when the explicitly
recomputed scaled residual falls to max(cpl_eta ||W R||, cpl_abs) or after cpl_maxit iterations. Returns the iterations,
the convergence flag, and the initial and final true scaled residual norms. The Arnoldi estimate triggers a check, but
cannot itself declare convergence: rounding or an inexact operator can leave a residual gap.
"""
cpl_fgmres!(cw::CoupledWork, w, m::PBMesh, bc, cfg) =
    _cpl_fgmres!(cw, w, m, bc, cfg, cpl_apply!, cpl_precond!, cpl_dot)

# Operator hooks allow the same iteration to be checked against small known systems.
function _cpl_fgmres!(cw, w, m, bc, cfg, apply!, precond!, dotfn)
    be = cfg.backend; N = cw.N; rs = cw.restart
    fill!(cw.dx, 0.0)
    _launch!(_k_cpl_mul!, be, N, cw.T1, cw.Wt, cw.R); pb_sync(be)
    beta0 = sqrt(dotfn(cw, cw.T1, cw.T1; backend=be))
    isfinite(beta0) || return (iters=0, converged=false, r0=beta0, rest=beta0)
    goal = max(cfg.cpl_eta*beta0, cfg.cpl_abs)
    beta0 <= goal && return (iters=0, converged=true, r0=beta0, rest=beta0)
    its = 0; beta = beta0; conv = false; est = beta0
    while true
        V1 = view(cw.Vb, :, 1); copyto!(V1, cw.T1); _launch!(_k_cpl_scale!, be, N, V1, 1/beta)
        fill!(cw.H, 0.0); fill!(cw.g, 0.0); cw.g[1] = beta; jl = 0
        for j in 1:rs
            jl = j; its += 1
            Vj = view(cw.Vb, :, j); Zj = view(cw.Zb, :, j); Vn = view(cw.Vb, :, j + 1)
            _launch!(_k_cpl_div_by!, be, N, cw.T2, Vj, cw.Wt)
            precond!(Zj, cw.T2, cw, w, m, bc, cfg)
            apply!(cw.Y, Zj, cw, w, m, bc, cfg)
            _launch!(_k_cpl_mul!, be, N, Vn, cw.Wt, cw.Y)
            if cfg.cpl_gs === :cgs2            # W7 (opt-in): classical Gram-Schmidt twice, 9 launches and one read per iteration
                hs = _cpl_cgs2!(cw, Vn, j, be)
                for i in 1:j; cw.H[i, j] = hs[i] + hs[j + i]; end
                hn = sqrt(hs[2j+1]); cw.H[j+1, j] = hn
            elseif cfg.cpl_gs === :mgs_dev     # W7 (opt-in): the same MGS, its coefficients on the device, one host read
                hsc = _cpl_hsc!(cw, be)
                for i in 1:j
                    _dot_dev!(hsc, i, Vn, view(cw.Vb, :, i), cw.part; backend=be)
                    _launch!(_k_cpl_axpy_dev!, be, N, Vn, hsc, Int32(i), view(cw.Vb, :, i))
                end
                _dot_dev!(hsc, j + 1, Vn, Vn, cw.part; backend=be)
                _launch!(_k_cpl_scale_dev!, be, N, Vn, hsc, Int32(j + 1))
                hs = _host(view(hsc, 1:j+1))
                for i in 1:j; cw.H[i, j] = hs[i]; end
                hn = sqrt(hs[j+1]); cw.H[j+1, j] = hn
            else
                for i in 1:j
                    hij = dotfn(cw, Vn, view(cw.Vb, :, i); backend=be); cw.H[i, j] = hij
                    _launch!(_k_cpl_axpy!, be, N, Vn, -hij, view(cw.Vb, :, i))
                end
                hn = sqrt(dotfn(cw, Vn, Vn; backend=be)); cw.H[j+1, j] = hn
                hn > 0 && _launch!(_k_cpl_scale!, be, N, Vn, 1/hn)
            end
            for i in 1:j-1
                t = cw.cs[i]*cw.H[i, j] + cw.sn[i]*cw.H[i+1, j]
                cw.H[i+1, j] = -cw.sn[i]*cw.H[i, j] + cw.cs[i]*cw.H[i+1, j]; cw.H[i, j] = t
            end
            den = hypot(cw.H[j, j], cw.H[j+1, j])
            # A null or invalid Arnoldi column cannot form a valid rotation or solution.
            # The last verified residual is retained instead of reporting NaN or success.
            (isfinite(den) && den > 0) || return (iters=its, converged=false, r0=beta0, rest=beta)
            cw.cs[j] = cw.H[j, j]/den; cw.sn[j] = cw.H[j+1, j]/den
            cw.H[j, j] = den; cw.H[j+1, j] = 0.0; cw.g[j+1] = -cw.sn[j]*cw.g[j]; cw.g[j] = cw.cs[j]*cw.g[j]
            est = abs(cw.g[j+1])
            isfinite(est) || break
            est <= goal && break
            its >= cfg.cpl_maxit && break
        end
        isfinite(est) || return (iters=its, converged=false, r0=beta0, rest=beta)
        y = UpperTriangular(cw.H[1:jl, 1:jl]) \ cw.g[1:jl]
        for i in 1:jl; _launch!(_k_cpl_axpy!, be, N, cw.dx, y[i], view(cw.Zb, :, i)); end
        pb_sync(be)
        apply!(cw.Y, cw.dx, cw, w, m, bc, cfg)                         # verify and restart from the true residual
        _launch!(_k_cpl_wres!, be, N, cw.T1, cw.Wt, cw.R, cw.Y); pb_sync(be)
        beta = sqrt(dotfn(cw, cw.T1, cw.T1; backend=be)); est = beta
        isfinite(beta) || return (iters=its, converged=false, r0=beta0, rest=beta)
        beta <= goal && (conv = true; break)
        its >= cfg.cpl_maxit && break
    end
    (iters=its, converged=conv, r0=beta0, rest=est)
end

# ---- the outer iteration's set-up: the residual, the frozen coefficients, the preconditioner ------------------------------
"""
    cpl_freeze!(cw, w, mesh, st, bc, cfg, t_new, first_outer)

After assemble_momentum! at the iterate: the momentum residual vectors into cw.R (rows 1-2), the flux F_RC of the iterate
(w.Fs, w.Fbs; rhie_chow_flux! at lambda 1), SIMPLEC's D at cpl_au (w.Df, w.Db), the Schur operator and the continuity
residual (assemble_pressure!: w.psys, its right side into cw.R rows 3), the hierarchy (mg_update!), the preconditioner's
momentum matrix (diagonal a/au) and its line tridiagonals, D^c, the row weights and the frozen convecting flux.
"""
function cpl_freeze!(cw::CoupledWork, w, m::PBMesh, st::PBState, bc, cfg, t_new::Real, first_outer::Bool)
    be = cfg.backend; nc = m.nc
    _launch!(_k_cpl_mom_res!, be, nc, cfield(cw.R, 1, nc), cfield(cw.R, 2, nc), w.bsx, w.bsy, w.ggx, w.ggy, lev(st.ux, 1),
             lev(st.uy, 1), lev(st.ux, 2), lev(st.uy, 2), w.as_, w.at, w.msx.upper, w.msx.lower, m.cface, m.own, m.nei,
             Int32(m.nf), m.vol)
    copyto!(cw.Fconv, lev(st.F, 1)); copyto!(cw.Fbconv, lev(st.Fb, 1))
    copyto!(w.Fprev, lev(st.F, 1)); copyto!(w.Fbprev, lev(st.Fb, 1))
    copyto!(w.msx.diag, w.a)                       # A = a at lambda = 1 (rhie_chow_flux! reads msx.diag)
    momentum_H!(w, m, st, cfg)
    rhie_chow_flux!(w, m, st, bc, cfg, 1.0)
    if cfg.cpl_newton
        lu = Int32(cfg.convection === :linear_upwind_unlimited ? 2 : 0)
        cg = lu >= 1 && cfg.conv_gradient === :unit ? (w.cux, w.cuy, w.cvx, w.cvy) : (w.gux, w.guy, w.gvx, w.gvy)
        _launch!(_k_cpl_conv_faces!, be, m.nf, cw.eox, cw.eoy, cw.enx, cw.eny, cw.Fconv, lev(st.ux, 1), lev(st.uy, 1), cg...,
                 m.own, m.nei, m.xf, m.yf, m.ftx, m.fty, m.xc, m.yc, lu)
        _launch!(_k_cpl_sub2!, be, m.nf, cw.dF0, w.Fs, cw.Fconv)
        _launch!(_k_cpl_newton!, be, nc, cfield(cw.R, 1, nc), cfield(cw.R, 2, nc), cw.dF0, cw.eox, cw.eoy, cw.enx, cw.eny,
                 m.cface, Int32(m.nf), cfg.rho_l, -1.0)
    end
    _launch!(_k_cpl_dpc_face!, be, m.nf, w.Df, w.a, w.at, m.vol, m.own, m.nei, m.fw, cfg.cpl_au)
    _launch!(_k_cpl_dpc_bface!, be, m.nb, w.Db, w.a, w.at, m.vol, bc.kind, m.bown, cfg.cpl_au)
    pb_sync(be)
    w.nonpos_step += _count_nonpos_D(w, m, bc, cfg)
    fill!(w.sdiag, 0.0); fill!(w.srhs, 0.0)
    assemble_pressure!(w, m, st, bc, cfg, t_new; tbar=w.tbar)
    copyto!(cfield(cw.R, 3, nc), w.psys.rhs)
    if bc.npdir == 0
        cw.rref = cfg.p_ref_cell > 0 ? cfg.p_ref_cell : 1
        cw.kref = _host(view(w.psys.diag, cw.rref:cw.rref))[1]/2
    else
        cw.rref = 0; cw.kref = 0.0
    end
    mg_update!(w, m, cfg, first_outer)
    _launch!(_k_cpl_cell_pc!, be, nc, cw.pdiag, cw.Dc, w.a, w.at, m.vol, cfg.cpl_au)
    pb_sync(be)
    line_tridiag!(w.fj, cw.Pu, m; backend=be); line_tridiag!(w.fi, cw.Pu, m; backend=be)
    cfg.prec_precision === :f32 && _cpl_f32_load!(cw, w, m, be)          # W7 (opt-in)
    _launch!(_k_cpl_weights!, be, cw.N, cw.Wt, m.vol, Int32(nc), cfg.L_ref/(cfg.rho_l*cfg.U_ref^2*cfg.mom_tol),
             st.dt/cfg.theta_tol)
    pb_sync(be)
    nothing
end

"""W7 (prec_precision = :f32): the Float32 copies of the preconditioner's momentum system (diagonal a/au, the face
couplings; one right side per component), its row sums (formed in Float64, rounded: the sweeps' excess residual, the
correction of 22 Oct 2026) and its two line families (the factors formed in Float64 by line_tridiag! and rounded), with the
sweeps' Float32 work vectors; made on the first freeze, loaded at every freeze."""
function _cpl_f32_load!(cw::CoupledWork, w, m::PBMesh, be)
    if !haskey(_w7(cw), :f32)
        z(n) = KernelAbstractions.zeros(be, Float32, n)
        dg = z(m.nc); up = z(m.nf); lo = z(m.nf)
        _w7(cw)[:f32] = (Pu=FaceSystem(dg, up, lo, z(m.nc)), Pv=FaceSystem(dg, up, lo, z(m.nc)),
                 fj=line_family_like(w.fj, Float32; backend=be), fi=line_family_like(w.fi, Float32; backend=be),
                 zu=z(m.nc), zv=z(m.nc), rw=z(m.nc), zw=z(m.nc), ex=z(m.nc))
    end
    F = _w7(cw)[:f32]
    pb_convert!(F.Pu.diag, cw.Pu.diag; backend=be); pb_convert!(F.Pu.upper, cw.Pu.upper; backend=be)
    pb_convert!(F.Pu.lower, cw.Pu.lower; backend=be)
    fs_rowsum!(F.ex, cw.Pu, m; backend=be)
    line_copy!(F.fj, w.fj; backend=be); line_copy!(F.fi, w.fi; backend=be)
    pb_sync(be)
    F
end

"""CGS2 (W7, cpl_gs = :cgs2): the new Arnoldi vector w orthogonalised against V[:, 1:j] by classical Gram-Schmidt twice
(h1 = V' w, w -= V h1; h2 = V' w, w -= V h2; Giraud, Langou and Rozloznik 2005: as stable as MGS in floating point), then
normalised. Each pass is one multi-dot kernel (a coalesced workgroup tree per 2,048-entry block and column), one kernel
summing the blocks per column, and one update kernel; then the norm (tree dot) and the scaling: 9 launches and one host read
of (h1, h2, |w|^2) per iteration, whatever j is. Deterministic; not MGS's order (class N)."""
function _cpl_cgs2!(cw::CoupledWork, w, j::Int, be)
    N = cw.N; nb = cld(N, PB_TREE_BLOCK); rs = cw.restart
    S = get!(() -> (part=KernelAbstractions.zeros(be, Float64, nb, rs), sc=KernelAbstractions.zeros(be, Float64, 2rs + 2),
                    p1=KernelAbstractions.zeros(be, Float64, max(nb, 1))), _w7(cw), :cgs2)
    for (pass, off) in ((1, 0), (2, j))
        _launch_wg!(_k_mdot_tree!, be, nb*j*PB_TREE_WG, PB_TREE_WG, S.part, cw.Vb, w, Int32(N), Int32(nb))
        _launch_wg!(_k_msum_tree!, be, j*PB_TREE_WG, PB_TREE_WG, S.sc, S.part, Int32(nb), Int32(off))
        _launch!(_k_cgs_update!, be, N, w, cw.Vb, S.sc, Int32(j), Int32(off))
    end
    _launch_wg!(_k_dot_tree!, be, nb*PB_TREE_WG, PB_TREE_WG, S.p1, w, w, Int32(N))
    _launch_wg!(_k_sum_tree!, be, PB_TREE_WG, PB_TREE_WG, S.sc, S.p1, Int32(nb), Int32(2j + 1))
    _launch!(_k_cpl_scale_dev!, be, N, w, S.sc, Int32(2j + 1))
    _host(view(S.sc, 1:2j+1))
end
@kernel function _k_mdot_tree!(part, V, w, n::Int32, nb::Int32)
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    sh = @localmem Float64 (256,)
    @inbounds begin
        b = (g - 1) % nb + 1; i = (g - 1) ÷ nb + 1
        base = (b - 1)*2048; s = 0.0
        for k in 0:7
            idx = base + k*256 + t
            idx <= n && (s += V[idx, i]*w[idx])
        end
        sh[t] = s
    end
    @synchronize
    for st in 1:8
        @inbounds begin
            wd = 256 >> st
            t <= wd && (sh[t] += sh[t + wd])
        end
        @synchronize
    end
    @inbounds if t == 1
        b = (g - 1) % nb + 1; i = (g - 1) ÷ nb + 1
        part[b, i] = sh[1]
    end
end
@kernel function _k_msum_tree!(sc, part, nb::Int32, off::Int32)
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    sh = @localmem Float64 (256,)
    @inbounds begin
        s = 0.0; k = t
        while k <= nb
            s += part[k, g]; k += 256
        end
        sh[t] = s
    end
    @synchronize
    for st in 1:8
        @inbounds begin
            wd = 256 >> st
            t <= wd && (sh[t] += sh[t + wd])
        end
        @synchronize
    end
    @inbounds t == 1 && (sc[off + g] = sh[1])
end
"w -= sum_i V[:, i] sc[off + i], i = 1..j in order (CGS's update)."
@kernel function _k_cgs_update!(w, V, sc, j::Int32, off::Int32)
    idx = @index(Global)
    @inbounds begin
        s = 0.0
        for i in Int32(1):j; s += V[idx, i]*sc[off + i]; end
        w[idx] -= s
    end
end

"The FGMRES device scalars of cpl_gs = :mgs_dev (W7): one slot per Gram-Schmidt coefficient and the norm, made once."
_cpl_hsc!(cw::CoupledWork, be) = (h = cw.sc; length(h) >= cw.restart + 2 ? h :
                                  (cw.sc = KernelAbstractions.zeros(be, Float64, max(cw.restart + 2, 8))))
"y -= sc[k] x (W7: MGS's update with its coefficient on the device; the host form passes -h as al: the same arithmetic)."
@kernel function _k_cpl_axpy_dev!(y, sc, k::Int32, x)
    i = @index(Global)
    @inbounds y[i] += (-sc[k])*x[i]
end
"y *= 1/sqrt(sc[k]) when sqrt(sc[k]) > 0 (W7: the Arnoldi vector's normalisation with its norm on the device)."
@kernel function _k_cpl_scale_dev!(y, sc, k::Int32)
    i = @index(Global)
    @inbounds begin
        hn = sqrt(sc[k])
        hn > 0 && (y[i] *= 1/hn)
    end
end

"""
    coupled_outer!(st, mesh, bc, w, cfg, t_new, theta0) -> (accepted, reason, info)

The coupled path's outer loop of one step (called by pimple_step! after its step-0 prologue; the module header). `info`
as pimple_step!'s, with `outer` the number of coupled solves, `fcg` the FGMRES iterations of each solve (the field name
kept for the march's records), and `linres` each solve's estimated final scaled residual.
INTERNAL: callers go through `pimple_step!` (which sets W7's modes and applies its checks). Called directly, it still applies
W7's Float32 rule itself (the second W7 review's R2): under prec_precision = :f32 the configuration check and the state's
uniform-density check run here too. `cpl_precond!`, `cpl_apply!` and `cpl_fgmres!` are internal as well and check nothing.
"""
function coupled_outer!(st::PBState, m::PBMesh, bc, w, cfg, t_new::Real, theta0::Real)
    check_coupled_config(cfg)
    cfg.prec_precision === :f32 && (check_speed_config(cfg); check_f32_state(st, m, w, cfg))   # W7 (second review, R2)
    be = cfg.backend; nc = m.nc
    cw = w.cpl === nothing ? (w.cpl = CoupledWork(m, w, cfg)) : w.cpl
    u = lev(st.ux, 1); v = lev(st.uy, 1); p = lev(st.p, 1)
    theta = theta0; Rhist = Float64[]; Rm = Float64[]; Rt = Float64[]; kits = Int[]; lres = Float64[]; linf = Float64[]; lin0 = Float64[]; resmom = NaN; nsolve = 0
    # SST (W1, opt-in; nothing below runs on the laminar path): R_turb per outer iteration (the k-omega relative residual at
    # the iterate over turb_tol, joined into R under turb_coupling = :outer), the k-omega solves' sweeps and floors; W1b: the
    # step's k-omega updates and final relative residual under :step (turb_iters, turb_res), the solves continued past their
    # first budget (turb_retries); :outer takes the :step treatment while st.nstep < turb_outer_ramp (review F2)
    sst = cfg.turbulence === :sst
    outer_turb = sst && cfg.turb_coupling === :outer && st.nstep >= cfg.turb_outer_ramp; Ru = Float64[]
    titer = Ref(0); tres = Ref(NaN)
    tinfo() = sst ? (Rturb=Ru, turb_sweeps=copy(w.turb.sweeps), turb_floors=(w.turb.nfloor_k, w.turb.nfloor_w),
                     turb_iters=titer[], turb_res=tres[], turb_retries=w.turb.retries) : (Rturb=Ru,)
    reject(reason) = (restore_level!(st); return (false, reason, merge((outer=nsolve, R=Rhist, fcg=kits, theta=theta, mom=resmom,
                                                                   mom_pre=resmom, linres=lres, Rm=Rm, Rt=Rt, linf=linf, lin0=lin0), tinfo())))
    for it in 1:cfg.n_outer_max + 1
        if cfg.cpl_reproject && nsolve >= 1        # the iterate's flux re-taken as its momentum interpolation (one fixed point)
            assemble_momentum!(w, m, st, bc, cfg, t_new)
            copyto!(w.msx.diag, w.a); momentum_H!(w, m, st, cfg); rhie_chow_flux!(w, m, st, bc, cfg, 1.0)
            copyto!(lev(st.F, 1), w.Fs); copyto!(lev(st.Fb, 1), w.Fbs)
            transport_mass!(w, m, st, cfg, t_new); theta = pb_maxabs(w.theta; backend=be)
        end
        rturb = 0.0
        if sst                                     # mu_t (and mu_eff) of the iterate; its k-omega residual
            # turb_mut = :frozen (W6; the default since W6): formed at the step's first outer iteration only and held through its
            # solves, in a step that takes the :step treatment; under :outer's treatment re-formed at every outer iteration.
            # turb_mut_base (27 Oct 2026): the frozen mu_t from the step's start iterate (level 1, :extrapolated, the default) or
            # from the committed level n (:committed; frozen_mut_level)
            (cfg.turb_mut === :iterate || outer_turb || it == 1) &&
                (turb_update!(w, m, st, bc, cfg; level=it == 1 ? frozen_mut_level(cfg, outer_turb) : 1) || return reject(:nonfinite))
            if outer_turb
                rk_, rw_ = komega_residual(w, m, st, bc, cfg)
                rturb = max(rk_, rw_)/_turb_outer_tol(cfg)        # turb_outer_tol (W6; 0: turb_tol, bit for bit)
                isfinite(rturb) || return reject(:nonfinite)
            end
            push!(Ru, rturb)
        end
        resmom = assemble_momentum!(w, m, st, bc, cfg, t_new)
        isfinite(resmom) || return reject(:nonfinite)
        R = max(resmom/cfg.mom_tol, theta/cfg.theta_tol)
        outer_turb && (R = max(R, rturb))
        push!(Rhist, R); push!(Rm, resmom/cfg.mom_tol); push!(Rt, theta/cfg.theta_tol)
        if nsolve >= 1 && R <= 1
            _launch!(_k_bounds!, be, nc, w.scr, lev(st.rho, 1), lev(st.q, 1), cfg.rho_l)
            pb_maxabs(w.scr; backend=be) > cfg.bound_tol && return reject(:bounds)
            _launch!(_k_conterr!, be, nc, w.scr, lev(st.rho, 1), lev(st.rho, 2), lev(st.F, 1), lev(st.Fb, 1), m.cface,
                     Int32(m.nf), m.vol, cfg.rho_l, st.dt)
            w.cont_step = pb_sum(w.scr; backend=be)*st.dt
            if sst && !outer_turb                  # turb_coupling = :step: the k-omega update on the accepted flow iterate
                # turb_mut = :frozen (W6): the SST terms re-formed at the accepted iterate first, so the k-omega update is
                # :iterate's (its production from the accepted strain), whatever mu_t the step's solves held
                cfg.turb_mut === :frozen && (turb_update!(w, m, st, bc, cfg) || return reject(:nonfinite))
                why = _komega_step_update!(w, m, st, bc, cfg, titer, tres)
                why === :accepted || return reject(why)
            end
            return (true, :accepted, merge((outer=nsolve, R=Rhist, fcg=kits, theta=theta, mom=resmom, mom_pre=resmom, linres=lres, Rm=Rm, Rt=Rt, linf=linf, lin0=lin0), tinfo()))
        end
        it > cfg.n_outer_max && break
        if cfg.predict_reject && it == 7 && length(Rhist) >= 7
            rate = (Rhist[7]/Rhist[4])^(1/3)
            more = rate < 1 ? log(Rhist[7])/(-log(rate)) : Inf
            (it - 1 + more + 1 > cfg.n_outer_max) && return reject(_nonconv(:outer_predicted, Rhist))
        end
        cpl_freeze!(cw, w, m, st, bc, cfg, t_new, it == 1)
        r = cpl_fgmres!(cw, w, m, bc, cfg); nsolve += 1
        push!(kits, r.iters); push!(lres, r.rest)
        r.converged || return reject(:linear_failed)
        # the update: x += dx; F = F_RC(x^m) + F_lin(dx) (the same face formula, frozen coefficients)
        _launch!(_k_cpl_axpy!, be, nc, u, 1.0, cfield(cw.dx, 1, nc)); _launch!(_k_cpl_axpy!, be, nc, v, 1.0, cfield(cw.dx, 2, nc))
        _launch!(_k_cpl_axpy!, be, nc, p, 1.0, cfield(cw.dx, 3, nc))
        cpl_apply!(cw.Y, cw.dx, cw, w, m, bc, cfg)
        _launch!(_k_cpl_wres!, be, cw.N, cw.T1, cw.Wt, cw.R, cw.Y); pb_sync(be)
        push!(linf, pb_maxabs(cw.T1; backend=be)); push!(lin0, pb_maxabs(_wr0!(cw, be); backend=be))
        _launch!(_k_cpl_add2!, be, m.nf, lev(st.F, 1), w.Fs, cw.Fl)
        _launch!(_k_cpl_add2!, be, m.nb, lev(st.Fb, 1), w.Fbs, cw.Fbl)
        pb_sync(be)
        transport_mass!(w, m, st, cfg, t_new)
        theta = pb_maxabs(w.theta; backend=be)
        fin = max(pb_maxabs(u; backend=be), pb_maxabs(v; backend=be), pb_maxabs(p; backend=be), theta)
        isfinite(fin) || return reject(:nonfinite)
        if outer_turb                              # W1: the k-omega solve after the flow update (its flux, rho^{n+1})
            turb_update!(w, m, st, bc, cfg) || return reject(:nonfinite)
            r = komega_solve!(w, m, st, bc, cfg); titer[] += 1
            r.finite || return reject(:nonfinite)
            r.converged || return reject(:turb_linear_failed)
        end
    end
    reject(_nonconv(:outer_not_converged, Rhist))
end

"""
    _komega_step_update!(w, mesh, st, bc, cfg, iters, res) -> Symbol

The k-omega update of turb_coupling = :step (W1b, review F2 and N3) on the accepted flow iterate: komega_solve! (its line
solves continued once when they miss their goal), turb_update! (the committed mu_t of the new k, omega) and the k-omega
relative residual of the system re-assembled there, with its time term (`res[]`, reported on every step); with a finite
turb_step_tol the update is repeated from the new iterate until that residual is within it, at most turb_step_maxit updates
(`iters[]`). :accepted, or the typed reason: :nonfinite, :turb_linear_failed (a line solve missed its goal after its
continuation), :turb_not_converged (turb_step_tol not met). With turb_step_tol = Inf (the default) it is W1's single update
followed by one residual evaluation, which writes no state (the systems are re-assembled before any later use).
"""
function _komega_step_update!(w, m::PBMesh, st::PBState, bc, cfg, iters::Ref{Int}, res::Ref{Float64})
    assembled = false
    while true
        r = komega_solve!(w, m, st, bc, cfg; assemble=!assembled); iters[] += 1
        r.finite || return :nonfinite
        r.converged || return :turb_linear_failed
        turb_update!(w, m, st, bc, cfg) || return :nonfinite             # the committed mu_t of the new k, omega
        rk_, rw_ = komega_residual(w, m, st, bc, cfg); assembled = true  # the systems now hold this iterate's
        res[] = max(rk_, rw_)
        isfinite(res[]) || return :nonfinite
        isfinite(cfg.turb_step_tol) || return :accepted                  # one update (the default)
        res[] <= cfg.turb_step_tol && return :accepted
        iters[] >= cfg.turb_step_maxit && return :turb_not_converged
    end
end

"`:outer`'s per-step k-omega acceptance tolerance: PBConfig.turb_outer_tol, or turb_tol when it is 0 (the default; W6)."
_turb_outer_tol(cfg) = cfg.turb_outer_tol > 0 ? cfg.turb_outer_tol : cfg.turb_tol

"The scaled right side W R into cw.T2 (its max-norm: the linear solve's starting residual in the criteria's units)."
function _wr0!(cw::CoupledWork, be)
    _launch!(_k_cpl_mul!, be, cw.N, cw.T2, cw.Wt, cw.R); pb_sync(be)
    cw.T2
end
