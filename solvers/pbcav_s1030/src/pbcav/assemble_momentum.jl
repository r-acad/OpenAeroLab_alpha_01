# =====================================================================================================================
# assemble_momentum.jl -- the momentum predictor (plan 4.1, 4.2, 4.7 step 1.3) and the momentum interpolation of the face
# flux (plan 4.3). Stage P0b: laminar, constant mu, psi = 0, backward Euler. 14 Oct 2026.
#
# THE CELL EQUATION (per component; P the cell, N its neighbours across internal faces):
#   a_P u_P - sum_N a_N u_N = b^s_P + a^t_P u^n_P - V_P (grad p)^GG_P,          a_P = a^t_P + a^s_P
#   * time: backward Euler with the continuity-error term kept, ddt(rho u) + div(phi u) - [ddt(rho) + div(phi)] u, which
#     leaves a^t_P = rho^n_P V_P / dt and the source a^t u^n (the term's integral is ledgered; with the masses transported
#     by the same flux it is round-off, ledger.jl);
#   * convection: implicit first-order upwind in phi = rho_l F (F the volumetric face flux), a_N = max(-phi_out, 0); the
#     deferred linear-upwind correction phi (u_f^LU - u_up), u_f^LU = u_up + (grad u)^LSQ_up . (x_f - x_up), on the right;
#     LIMITED as plan 4.1 states (review F2, 14 Oct; the core had it unlimited): u_f^LU - u_up is clamped to the interval
#     between 0 and u_down - u_up, so the face value lies between the two cells' values (convection = :linear_upwind). The
#     unlimited form (:linear_upwind_unlimited) grew a mode in N60R's far-wake outlet-row cells (aspect ratio ~4000, face
#     skew 1.3-3.7 |d|) by about 1.23x per step (plan 23.5) on the 1/|d|^2 gradient. ON THE UNIT-WEIGHT GRADIENT (below) it
#     is form LUW, plan 4.1's form and THE DEFAULT since plan Amendment 8 (28.4 item 1; convection =
#     :linear_upwind_unlimited, conv_gradient = :unit); the limited form above stays selectable.
#     THE INCREMENT'S GRADIENT (R2, 15 Oct 2026, plan 26): by default (conv_gradient = :inverse_distance) the velocity's
#     least-squares gradient with weights 1/|d|^2, shared with the non-orthogonal term; with conv_gradient = :unit its own
#     gradient on the same stencil with unit weights (assemble_system.jl, _lsq_w). Diagnosis (records p0b/r2_convection):
#     the 1/|d|^2 weights let a thin cell's short, tangentially offset j-neighbours set the long direction's component, so
#     the increment r . g over the long cell amplifies a neighbour perturbation up to 458 times on N60R's outlet rows
#     (convection_gain), and the semi-discrete operator there has a real eigenvalue of +852 (probe D): the growth of 23.5 is
#     that instability, which backward Euler amplifies at any dt. With unit weights the gains stay <= 0.87 and the
#     eigenvalue is negative; both gradients are exact for a linear field, so the form stays second order.
#   * viscous: the two-point part mu |S|^2/(S.d) implicit; the non-orthogonal part mu k.(grad u)_f deferred, (grad u)_f the
#     linear interpolation of the cells' least-squares gradients, limited at lambda_n (cfg.nonorth: 1 = full correction,
#     0 = none, 0 < lambda_n < 1: |corr| <= lambda_n/(1 - lambda_n) |two-point part|, OpenFOAM's `limited corrected`);
#     the transpose term div(mu grad u^T) is zero for constant mu and divergence-free u and enters with mu_t (stage P2);
#   * the pressure force: Green-Gauss on the skewness-corrected face pressures (assemble_system.jl, plan 4.1);
#   * boundary faces: bc.jl; the MMS source V_P s(x_P, t^{n+1}) in b^s. A boundary face's deferred non-orthogonal term is
#     limited exactly as an internal face's (review F6, 14 Oct): |corr| <= lambda_n/(1 - lambda_n) |two-point part|.
#   The spatial operator has zero row sum (a^s_P = sum a_N + implicit boundary terms), which the momentum interpolation's
#   D of plan 4.3 relies on.
# RELAXATION (lambda_u; 1 on the final outer iteration): (a_P/lambda) u_P = ... + (1 - lambda)/lambda a_P u_P^(m-1).
#   A DIAGNOSTIC general form (review F2, plan 23.1; not a production option): relax_form :inviscid / :temporal /
#   :patankar_gen give diag = a + r, r = (1 - lambda)/lambda arel with arel = a - a^v (a^v the viscous two-point
#   diagonal), a^t, or a; the face flux, both D's and the cell correction use the same r (rhie_chow_flux!, correct_fluxes!).
# THE RESIDUAL (review F4, 14 Oct) is evaluated in increment form, r = b^s - a^s u + sum a_N u_N - V (grad p) - a^t (u - u^n),
# with the spatial diagonal a^s kept apart from a^t: written as (a^t u^n + ...) - (a^t + a^s) u it carried a Float64 floor of
# order eps a^t |u|/V = eps rho |u|/dt even at u = u^n. CORRECTED 14 Oct (re-review R3): the increment form removes that
# floor only at u = u^n (a steady state, where the time term is zero bit for bit). In a transient u - u^n is resolved to one
# ulp of u at best, so the time term a^t (u - u^n) still leaves a floor of about eps rho |u|/dt (measured 2.2e-14 ... 9.8e-9
# over dt 1e-2 ... 1e-8 on the 16 x 16 lid box), which a halving of dt RAISES; a tolerance must therefore be registered
# above the floor at the smallest dt the march may reach. The STEADY RESIDUAL (steady_residual) is r with the time term
# removed, at a committed state.
#
# THE FACE FLUX (rhie_chow_flux!, plan 4.3) is the cell equation written at the face and dotted with S (Bartholomew et
# al. 2018's form): with A = a/lambda, H^s_P = b^s_P + sum_N a_N u_N (the current velocity), and overbars the linear
# interpolation (owner weight fw):
#   Abar F*_f = Hbar^s_f . S + abar^t_f F^n_f [Choi] + (1 - lambda)/lambda abar_f F^(m-1)_f [Majumdar]
#               - Vbar_f ( |S|^2/(S.d) (p_N - p_P) + k . (grad p)bar_f )
# At a converged steady state F^n = F^(m-1) = F and the terms in a^t and lambda cancel exactly, leaving
#   abar^s F = Hbar^s . S - Vbar (grad p)^c . S:  the face flux does not depend on dt or lambda_u (plan 10.2's gate).
# SKEWNESS (plan 4.1 as amended 14 Oct): Hbar^s carries the skewness correction of the face values (assemble_system.jl) in
# the form that keeps H and the coefficients consistent: Hbar^s = (plain Hbar^s) + abar^s 1/2 (grad Hhat_P + grad Hhat_N) .
# (x_f - x_f'), Hhat = H^s/a^s the momentum's pseudo-velocity, its gradient the internal least-squares one. A uniform flow
# has a uniform Hhat, so it stays an exact discrete solution (correcting H itself would not: H^s = a^s u varies with a^s);
# a linear field with a uniform a^s gets its exact face value; on a mesh without skew the form is the plain one bitwise.
# The pressure correction uses D_f = Vbar/(abar^t + (1/lambda - 1) abar), positive by construction (plan 4.3).
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

"Boundary velocity values of every face for the least-squares stencil: the condition's value, or (slip) the owner's
tangential velocity, or (outlet) the owner's velocity."
@kernel function _k_bvel!(ubf, vbf, u, v, bkind, ub, vb, bown, bSx, bSy, bSmag)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]; P = bown[b]
        if is_velocity_bc(k)
            ubf[b] = ub[b]; vbf[b] = vb[b]
        elseif k == UInt8(BC_SLIP)
            nx = bSx[b]/bSmag[b]; ny = bSy[b]/bSmag[b]; un = u[P]*nx + v[P]*ny
            ubf[b] = u[P] - un*nx; vbf[b] = v[P] - un*ny
        else
            ubf[b] = u[P]; vbf[b] = v[P]
        end
    end
end

"Internal faces: the implicit coefficients and the owner's deferred right-side contribution (the neighbour's is minus it)."
@kernel function _k_mom_face!(up, lo, fcx, fcy, F, gux, guy, gvx, gvy, own, nei, fa, fw, kx, ky, xf, yf, ftx, fty, xc, yc,
                              rho_l::Float64, mu::Float64, lu::Int32, lamn::Float64, u, v, cux, cuy, cvx, cvy)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; phi = rho_l*F[f]; G = mu*fa[f]
        up[f] = -(G + max(-phi, 0.0)); lo[f] = -(G + max(phi, 0.0))
        cu = 0.0; cv = 0.0
        if lu >= Int32(1)
            if phi >= 0
                rx = xf[f] - xc[P]; ry = yf[f] - yc[P]
                du = cux[P]*rx + cuy[P]*ry; dv = cvx[P]*rx + cvy[P]*ry; eu = u[N] - u[P]; ev = v[N] - v[P]
            else
                rx = xf[f] - (xc[N] + ftx[f]); ry = yf[f] - (yc[N] + fty[f])
                du = cux[N]*rx + cuy[N]*ry; dv = cvx[N]*rx + cvy[N]*ry; eu = u[P] - u[N]; ev = v[P] - v[N]
            end
            if lu == Int32(1)       # limited (plan 4.1): the face value kept between the two cells' values
                du = clamp(du, min(0.0, eu), max(0.0, eu)); dv = clamp(dv, min(0.0, ev), max(0.0, ev))
            elseif lu == Int32(3)   # van Albada's smooth limiter on r = 2 du/eu: du <- eu (2 du^2 + du eu)/(4 du^2 + eu^2)
                du = eu*(2du*du + du*eu)/(4du*du + eu*eu + 1e-300); dv = ev*(2dv*dv + dv*ev)/(4dv*dv + ev*ev + 1e-300)
            end
            cu = phi*du; cv = phi*dv
        end
        nu_ = 0.0; nv_ = 0.0
        if lamn > 0
            w = fw[f]
            nu_ = mu*(kx[f]*(w*gux[P] + (1 - w)*gux[N]) + ky[f]*(w*guy[P] + (1 - w)*guy[N]))
            nv_ = mu*(kx[f]*(w*gvx[P] + (1 - w)*gvx[N]) + ky[f]*(w*gvy[P] + (1 - w)*gvy[N]))
            if lamn < 1
                r = lamn/(1 - lamn)
                lu_ = r*abs(G*(u[N] - u[P])); lv_ = r*abs(G*(v[N] - v[P]))
                nu_ = clamp(nu_, -lu_, lu_); nv_ = clamp(nv_, -lv_, lv_)
            end
        end
        fcx[f] = -cu + nu_; fcy[f] = -cv + nv_
    end
end

"Boundary faces: the implicit diagonal term and the right side of the owner, by condition (bc.jl)."
@kernel function _k_mom_bface!(bdg, bfx, bfy, Fb, bkind, ubf, vbf, u, v, gux, guy, gvx, gvy, bown, ba, bkx, bky,
                               rho_l::Float64, mu::Float64, lu::Int32, lamn::Float64)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]; P = bown[b]; phi = rho_l*Fb[b]; G = mu*ba[b]
        dg = 0.0; rx = 0.0; ry = 0.0
        if is_velocity_bc(k)
            ain = max(-phi, 0.0)
            dg = G + ain; rx = (G + ain)*ubf[b]; ry = (G + ain)*vbf[b]
            if lamn > 0
                nu_ = mu*(bkx[b]*gux[P] + bky[b]*guy[P]); nv_ = mu*(bkx[b]*gvx[P] + bky[b]*gvy[P])
                if lamn < 1
                    r = lamn/(1 - lamn)
                    lu_ = r*abs(G*(ubf[b] - u[P])); lv_ = r*abs(G*(vbf[b] - v[P]))
                    nu_ = clamp(nu_, -lu_, lu_); nv_ = clamp(nv_, -lv_, lv_)
                end
                rx += nu_; ry += nv_
            end
            if lu >= Int32(1) && phi > 0            # outflow through a velocity face: deferred correction to the face value
                rx -= phi*(ubf[b] - u[P]); ry -= phi*(vbf[b] - v[P])
            end
        elseif k == UInt8(BC_SLIP)
            dg = G; rx = G*ubf[b]; ry = G*vbf[b]
        else                                         # pressure outlet: no viscous flux; backflow at the cell's velocity
            ain = max(-phi, 0.0)
            dg = ain; rx = ain*u[P]; ry = ain*v[P]
        end
        bdg[b] = dg; bfx[b] = rx; bfy[b] = ry
    end
end

# ---- the turbulent viscous terms (W1, plan 32; 21 Oct 2026; PBConfig.turbulence = :sst only) -------------------------
# THE EFFECTIVE VISCOSITY AND THE FULL STRESS. With SST the stress is tau = mu_eff (grad u + grad u^T), mu_eff = mu + mu_t
# per face (turb_update!: mu + the linear interpolant of the cells' mu_t; mu at a no-slip wall, where mu_t = 0; the inflow
# value at an inflow face), without the isotropic rho k term (SST-2003m, as version 1). The two kernels below are the laminar
# kernels above with mu replaced by the face's mu_eff and the TRANSPOSE TERM added to the deferred right side:
#   internal face: mu_eff,f ((grad u)_f^T . S) (x: du/dx S_x + dv/dx S_y; y: du/dy S_x + dv/dy S_y), (grad u)_f the linear
#   interpolant of the cells' least-squares gradients, not limited; velocity faces: mu_eff,b ((grad u)_P^T . S_b); slip and
#   pressure faces: none (no shear stress there).
# For constant mu and a divergence-free u it sums to mu V grad(div u), about zero, so it matters only where mu_t varies. It is
# linear in u with mu_eff frozen, so the coupled step's J (cpl_apply!, which runs these kernels on the increment) carries it
# exactly. The laminar kernels are untouched: the default path is bit for bit the one before W1.
"Internal faces with SST: _k_mom_face! with the face's mu_eff (muf) and the transpose term (header above)."
@kernel function _k_mom_face_t!(up, lo, fcx, fcy, F, gux, guy, gvx, gvy, own, nei, fa, fw, kx, ky, xf, yf, ftx, fty, xc, yc,
                                rho_l::Float64, muf, lu::Int32, lamn::Float64, u, v, cux, cuy, cvx, cvy, Sx, Sy)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; phi = rho_l*F[f]; mu = muf[f]; G = mu*fa[f]
        up[f] = -(G + max(-phi, 0.0)); lo[f] = -(G + max(phi, 0.0))
        cu = 0.0; cv = 0.0
        if lu >= Int32(1)
            if phi >= 0
                rx = xf[f] - xc[P]; ry = yf[f] - yc[P]
                du = cux[P]*rx + cuy[P]*ry; dv = cvx[P]*rx + cvy[P]*ry; eu = u[N] - u[P]; ev = v[N] - v[P]
            else
                rx = xf[f] - (xc[N] + ftx[f]); ry = yf[f] - (yc[N] + fty[f])
                du = cux[N]*rx + cuy[N]*ry; dv = cvx[N]*rx + cvy[N]*ry; eu = u[P] - u[N]; ev = v[P] - v[N]
            end
            if lu == Int32(1)
                du = clamp(du, min(0.0, eu), max(0.0, eu)); dv = clamp(dv, min(0.0, ev), max(0.0, ev))
            elseif lu == Int32(3)
                du = eu*(2du*du + du*eu)/(4du*du + eu*eu + 1e-300); dv = ev*(2dv*dv + dv*ev)/(4dv*dv + ev*ev + 1e-300)
            end
            cu = phi*du; cv = phi*dv
        end
        w = fw[f]; w1 = 1 - w
        gxu = w*gux[P] + w1*gux[N]; gyu = w*guy[P] + w1*guy[N]; gxv = w*gvx[P] + w1*gvx[N]; gyv = w*gvy[P] + w1*gvy[N]
        nu_ = 0.0; nv_ = 0.0
        if lamn > 0
            nu_ = mu*(kx[f]*gxu + ky[f]*gyu)
            nv_ = mu*(kx[f]*gxv + ky[f]*gyv)
            if lamn < 1
                r = lamn/(1 - lamn)
                lu_ = r*abs(G*(u[N] - u[P])); lv_ = r*abs(G*(v[N] - v[P]))
                nu_ = clamp(nu_, -lu_, lu_); nv_ = clamp(nv_, -lv_, lv_)
            end
        end
        tx = mu*(gxu*Sx[f] + gxv*Sy[f]); ty = mu*(gyu*Sx[f] + gyv*Sy[f])
        fcx[f] = -cu + nu_ + tx; fcy[f] = -cv + nv_ + ty
    end
end
"Boundary faces with SST: _k_mom_bface! with the face's mu_eff (mub) and, on velocity faces, the transpose term."
@kernel function _k_mom_bface_t!(bdg, bfx, bfy, Fb, bkind, ubf, vbf, u, v, gux, guy, gvx, gvy, bown, ba, bkx, bky,
                                 rho_l::Float64, mub, lu::Int32, lamn::Float64, bSx, bSy)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]; P = bown[b]; phi = rho_l*Fb[b]; mu = mub[b]; G = mu*ba[b]
        dg = 0.0; rx = 0.0; ry = 0.0
        if is_velocity_bc(k)
            ain = max(-phi, 0.0)
            dg = G + ain; rx = (G + ain)*ubf[b]; ry = (G + ain)*vbf[b]
            if lamn > 0
                nu_ = mu*(bkx[b]*gux[P] + bky[b]*guy[P]); nv_ = mu*(bkx[b]*gvx[P] + bky[b]*gvy[P])
                if lamn < 1
                    r = lamn/(1 - lamn)
                    lu_ = r*abs(G*(ubf[b] - u[P])); lv_ = r*abs(G*(vbf[b] - v[P]))
                    nu_ = clamp(nu_, -lu_, lu_); nv_ = clamp(nv_, -lv_, lv_)
                end
                rx += nu_; ry += nv_
            end
            rx += mu*(gux[P]*bSx[b] + gvx[P]*bSy[b]); ry += mu*(guy[P]*bSx[b] + gvy[P]*bSy[b])
            if lu >= Int32(1) && phi > 0
                rx -= phi*(ubf[b] - u[P]); ry -= phi*(vbf[b] - v[P])
            end
        elseif k == UInt8(BC_SLIP)
            dg = G; rx = G*ubf[b]; ry = G*vbf[b]
        else
            ain = max(-phi, 0.0)
            dg = ain; rx = ain*u[P]; ry = ain*v[P]
        end
        bdg[b] = dg; bfx[b] = rx; bfy[b] = ry
    end
end

"""The momentum's face and boundary kernels (laminar: _k_mom_face!/_k_mom_bface! with the scalar mu, bit for bit the path
before W1; SST: the _t kernels with w.muf, w.mub), launched on the given velocity, gradients and fluxes into the given
outputs: the assembly (assemble_momentum!) and the coupled step's matrix-free J (cpl_apply!) call the same function."""
function _mom_face_kernels!(be, m::PBMesh, cfg, w, up, lo, fcx, fcy, F, gux, guy, gvx, gvy, cg, u, v, bdg, bfx, bfy, Fb, bkind,
                            ubf, vbf, lu::Int32; lamn::Float64=Float64(cfg.nonorth))
    if cfg.turbulence === :sst
        _launch!(_k_mom_face_t!, be, m.nf, up, lo, fcx, fcy, F, gux, guy, gvx, gvy, m.own, m.nei, m.fa, m.fw, m.kx, m.ky, m.xf,
                 m.yf, m.ftx, m.fty, m.xc, m.yc, cfg.rho_l, w.muf, lu, lamn, u, v, cg..., m.Sx, m.Sy)
        _launch!(_k_mom_bface_t!, be, m.nb, bdg, bfx, bfy, Fb, bkind, ubf, vbf, u, v, gux, guy, gvx, gvy, m.bown, m.ba, m.bkx,
                 m.bky, cfg.rho_l, w.mub, lu, lamn, m.bSx, m.bSy)
    else
        _launch!(_k_mom_face!, be, m.nf, up, lo, fcx, fcy, F, gux, guy, gvx, gvy, m.own, m.nei, m.fa, m.fw, m.kx, m.ky, m.xf,
                 m.yf, m.ftx, m.fty, m.xc, m.yc, cfg.rho_l, cfg.mu, lu, lamn, u, v, cg...)
        _launch!(_k_mom_bface!, be, m.nb, bdg, bfx, bfy, Fb, bkind, ubf, vbf, u, v, gux, guy, gvx, gvy, m.bown, m.ba, m.bkx,
                 m.bky, cfg.rho_l, cfg.mu, lu, lamn)
    end
    nothing
end

"""
    momentum_bface_terms!(bdg, bfx, bfy, mesh, bc, cfg, u, v, Fb, ubf, vbf, gux, guy, gvx, gvy; mub=nothing)

THE MOMENTUM'S OWN BOUNDARY-FACE TERMS of a velocity field (the W1+W2 integration, 22 Oct 2026; review of W2, R1): the boundary
kernel `_mom_face_kernels!` launches in every assembly, on the given velocity `u, v`, boundary flux `Fb`, boundary values `ubf,
vbf` (`_k_bvel!`) and least-squares velocity gradients (`_k_lsq_grad!` with the velocity weights), at the configuration's
convection code and nonorth: laminar `_k_mom_bface!` with cfg.mu; SST `_k_mom_bface_t!` with the face viscosity `mub` (the
momentum's w.mub, or any array with the wall's value: mu_eff = mu at a no-slip wall, where mu_t = 0) and the transpose term.
Per face b of owner P the cell equation carries bdg[b] u_P on its left and bfx[b], bfy[b] on its right, so the force the
boundary applies to the fluid through b is (bfx[b] - bdg[b] u_P, bfy[b] - bdg[b] v_P) (viscous, plus the convective term on
an open face); on the body it is minus that. Used by the free-air loads (`airfoil_loads!`) and by `wall_shear`'s :assembled
form, so the loads, the wall-shear instrument and the momentum are one set of terms. Writes only its outputs."""
function momentum_bface_terms!(bdg, bfx, bfy, m::PBMesh, bc, cfg, u, v, Fb, ubf, vbf, gux, guy, gvx, gvy; mub=nothing)
    be = cfg.backend
    lu = Int32(cfg.convection === :linear_upwind ? 1 : cfg.convection === :linear_upwind_unlimited ? 2 :
               cfg.convection === :linear_upwind_vanalbada ? 3 : 0)
    if cfg.turbulence === :sst
        (mub !== nothing && length(mub) >= m.nb) || throw(ArgumentError("momentum_bface_terms!: PBConfig.turbulence = :sst " *
            "needs the boundary faces' mu_eff (mub: the momentum's w.mub, of length nb)"))
        _launch!(_k_mom_bface_t!, be, m.nb, bdg, bfx, bfy, Fb, bc.kind, ubf, vbf, u, v, gux, guy, gvx, gvy, m.bown, m.ba, m.bkx,
                 m.bky, cfg.rho_l, mub, lu, Float64(cfg.nonorth), m.bSx, m.bSy)
    else
        _launch!(_k_mom_bface!, be, m.nb, bdg, bfx, bfy, Fb, bc.kind, ubf, vbf, u, v, gux, guy, gvx, gvy, m.bown, m.ba, m.bkx,
                 m.bky, cfg.rho_l, cfg.mu, lu, Float64(cfg.nonorth))
    end
    pb_sync(be)
    nothing
end

"Cells: the unrelaxed diagonal a = a^t + a^s, a^t, a^s, b^s and the right side b^s + a^t u^n - V (grad p)^GG."
@kernel function _k_mom_cell!(dg, at, as_, bsx, bsy, rx, ry, up, lo, fcx, fcy, bdg, bfx, bfy, cface, nf::Int32, vol, rhon,
                              un, vn, ggx, ggy, xc, yc, dt::Float64, mms, tn::Float64)
    c = @index(Global)
    @inbounds begin
        V = vol[c]; a_t = rhon[c]*V/dt; d = 0.0; sx = 0.0; sy = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf; d += bdg[b]; sx += bfx[b]; sy += bfy[b]
            elseif code > 0
                d -= up[code]; sx += fcx[code]; sy += fcy[code]
            else
                d -= lo[-code]; sx -= fcx[-code]; sy -= fcy[-code]
            end
        end
        sx += V*mms_source(mms, EQ_XMOM, xc[c], yc[c], tn); sy += V*mms_source(mms, EQ_YMOM, xc[c], yc[c], tn)
        dg[c] = a_t + d; at[c] = a_t; as_[c] = d; bsx[c] = sx; bsy[c] = sy
        rx[c] = sx + a_t*un[c] - V*ggx[c]; ry[c] = sy + a_t*vn[c] - V*ggy[c]
    end
end

"""The unrelaxed momentum residual at the current velocity in increment form (header), max over components, in units
rho U^2 / L per unit volume; tw = 0 drops the time term (the steady residual)."""
@kernel function _k_mom_resid!(res, bsx, bsy, ggx, ggy, u, v, un, vn, as_, at, up, lo, cface, own, nei, nf::Int32, vol,
                               scale::Float64, tw::Float64)
    c = @index(Global)
    @inbounds begin
        V = vol[c]
        ax = as_[c]*u[c]; ay = as_[c]*v[c]
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                ax += up[code]*u[nei[code]]; ay += up[code]*v[nei[code]]
            elseif code < 0
                ax += lo[-code]*u[own[-code]]; ay += lo[-code]*v[own[-code]]
            end
        end
        rx = bsx[c] - V*ggx[c] - ax - tw*at[c]*(u[c] - un[c])
        ry = bsy[c] - V*ggy[c] - ay - tw*at[c]*(v[c] - vn[c])
        res[c] = max(abs(rx), abs(ry))*scale/V
    end
end

"Relaxation: diag = a/lambda, rhs += (1 - lambda)/lambda a u^(m-1)."
@kernel function _k_mom_relax!(diag, rx, ry, a, up_, vp_, lam::Float64)
    c = @index(Global)
    @inbounds begin
        diag[c] = a[c]/lam; r = (1 - lam)/lam*a[c]
        rx[c] += r*up_[c]; ry[c] += r*vp_[c]
    end
end

"Relaxation in a general form: diag = a + r, rhs += r u^(m-1), r = (1 - lambda)/lambda arel."
@kernel function _k_mom_relax_gen!(diag, rx, ry, a, arel, up_, vp_, lam::Float64)
    c = @index(Global)
    @inbounds begin
        r = (1 - lam)/lam*arel[c]; diag[c] = a[c] + r
        rx[c] += r*up_[c]; ry[c] += r*vp_[c]
    end
end

"H^s = b^s + sum_N a_N u_N at the current velocity (a_N = -off-diagonal)."
@kernel function _k_mom_H!(Hx, Hy, bsx, bsy, u, v, up, lo, cface, own, nei, nf::Int32)
    c = @index(Global)
    @inbounds begin
        hx = bsx[c]; hy = bsy[c]
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

"""
    velocity_gradients!(w, mesh, st, bc, cfg; level=1)

The boundary face velocities and the cells' least-squares velocity gradients at the current iterate (level 1, the default);
`level` = 2 reads the committed level n instead (turb_update!'s turb_mut_base = :committed; 27 Oct 2026).
"""
function velocity_gradients!(w, m::PBMesh, st::PBState, bc, cfg; level::Integer=1)
    be = cfg.backend; u = lev(st.ux, level); v = lev(st.uy, level); G = w.lsq
    _launch!(_k_bvel!, be, m.nb, w.ubf, w.vbf, u, v, bc.kind, bc.ub, bc.vb, m.bown, m.bSx, m.bSy, m.bSmag)
    for (g1, g2, phi, phib) in ((w.gux, w.guy, u, w.ubf), (w.gvx, w.gvy, v, w.vbf))
        _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, phib, G.u11, G.u12, G.u22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(0))
    end
    if cfg.conv_gradient === :unit        # the convection increment's own gradient (plan 26): unit weights, same stencil
        for (g1, g2, phi, phib) in ((w.cux, w.cuy, u, w.ubf), (w.cvx, w.cvy, v, w.vbf))
            _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, phib, G.c11, G.c12, G.c22, m.cface, m.own, m.nei, m.dx, m.dy,
                     m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(1))
        end
    end
    pb_sync(be)
    nothing
end

"""
    pressure_gradients!(gx, gy, ggx, ggy, pbf, p, w, mesh, bc, cfg; correction=false)

Least-squares (gx, gy) and Green-Gauss (ggx, ggy) gradients of a pressure field p, with the boundary face pressures pbf
(correction = true: a pressure correction, zero on pressure-Dirichlet faces). The Green-Gauss face values are skewness
corrected with (gx, gy) unless cfg.skew is false.
"""
function pressure_gradients!(gx, gy, ggx, ggy, pbf, p, w, m::PBMesh, bc, cfg; correction::Bool=false)
    be = cfg.backend; G = w.lsq; vz = Int32(correction ? 1 : 0)
    _launch!(_k_lsq_grad!, be, m.nc, gx, gy, p, bc.pb, G.p11, G.p12, G.p22, m.cface, m.own, m.nei, m.dx, m.dy,
             m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(1), vz, Int32(0))
    _launch!(_k_bface_p!, be, m.nb, pbf, p, gx, gy, bc.pb, bc.kind, m.bown, m.bdx, m.bdy, vz)
    _launch!(_k_gg_grad!, be, m.nc, ggx, ggy, p, gx, gy, pbf, m.vol, m.cface, m.own, m.nei, m.fw, m.skx, m.sky, m.Sx, m.Sy,
             m.bSx, m.bSy, Int32(m.nf), cfg.skew ? 1.0 : 0.0)
    pb_sync(be)
    nothing
end

"""
    assemble_momentum!(w, mesh, st, bc, cfg, t_new) -> momentum residual (host Float64)

Assemble the unrelaxed momentum system into w.msx / w.msy (shared diagonal and off-diagonals; the right sides differ) at
the current iterate (level 1: u^(m-1), p^(m-1), F^(m-1)), with level 2 as time level n; return the unrelaxed residual
max_P |r_P| L/(V_P rho U^2) at u^(m-1). `relax_momentum!` then applies lambda.
"""
function assemble_momentum!(w, m::PBMesh, st::PBState, bc, cfg, t_new::Real)
    be = cfg.backend; u = lev(st.ux, 1); v = lev(st.uy, 1)
    cfg.conv_gradient in (:inverse_distance, :unit) ||
        throw(ArgumentError("conv_gradient must be :inverse_distance or :unit (got :$(cfg.conv_gradient))"))
    velocity_gradients!(w, m, st, bc, cfg)
    pressure_gradients!(w.gpx, w.gpy, w.ggx, w.ggy, w.pbf, lev(st.p, 1), w, m, bc, cfg)
    lu = Int32(cfg.convection === :linear_upwind ? 1 : cfg.convection === :linear_upwind_unlimited ? 2 :
               cfg.convection === :linear_upwind_vanalbada ? 3 :
               cfg.convection === :upwind ? 0 :
               throw(ArgumentError("convection must be :linear_upwind, :upwind or :linear_upwind_unlimited (got :$(cfg.convection))")))
    cg = cfg.conv_gradient === :unit ? (w.cux, w.cuy, w.cvx, w.cvy) : (w.gux, w.guy, w.gvx, w.gvy)
    ms = w.msx
    _mom_face_kernels!(be, m, cfg, w, ms.upper, ms.lower, w.fcx, w.fcy, lev(st.F, 1), w.gux, w.guy, w.gvx, w.gvy, cg, u, v,
                       w.bdg, w.bfx, w.bfy, lev(st.Fb, 1), bc.kind, w.ubf, w.vbf, lu)
    _launch!(_k_mom_cell!, be, m.nc, w.a, w.at, w.as_, w.bsx, w.bsy, w.msx.rhs, w.msy.rhs, ms.upper, ms.lower, w.fcx, w.fcy,
             w.bdg, w.bfx, w.bfy, m.cface, Int32(m.nf), m.vol, lev(st.rho, 2), lev(st.ux, 2), lev(st.uy, 2), w.ggx, w.ggy,
             m.xc, m.yc, st.dt, cfg.mms, Float64(t_new))
    if cfg.relax_form !== :patankar
        cfg.turbulence === :sst && throw(ArgumentError("the diagnostic relaxation forms (relax_form :$(cfg.relax_form)) form " *
            "the laminar viscous diagonal; PBConfig.turbulence = :sst takes relax_form = :patankar only"))
        cfg.relax_form in (:inviscid, :temporal, :patankar_gen) ||
            throw(ArgumentError("relax_form must be :patankar, :inviscid, :temporal or :patankar_gen (got :$(cfg.relax_form))"))
        _launch!(_k_visc_diag!, be, m.nc, w.av, w.arel, w.a, w.at, m.cface, Int32(m.nf), m.fa, m.ba, bc.kind, cfg.mu,
                 cfg.relax_form === :temporal ? Int32(1) : cfg.relax_form === :patankar_gen ? Int32(2) : Int32(0))
    end
    momentum_residual!(w, m, st, cfg; time_term=true)
end

"The viscous two-point diagonal a^v (internal faces mu |S|^2/(S.d); boundary faces likewise except pressure outlets) and
the array the relaxation is proportional to, arel = a - a^v (relax_form :inviscid) or a^t (:temporal)."
@kernel function _k_visc_diag!(av, arel, a, at, cface, nf::Int32, fa, ba, bkind, mu::Float64, mode::Int32)
    c = @index(Global)
    @inbounds begin
        s = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf; is_pressure_bc(bkind[b]) || (s += mu*ba[b])
            elseif code > 0
                s += mu*fa[code]
            else
                s += mu*fa[-code]
            end
        end
        av[c] = s; arel[c] = mode == Int32(1) ? at[c] : mode == Int32(2) ? a[c] : a[c] - s
    end
end
"The array the relaxation is proportional to: a itself (Patankar) or w.arel."
_arel(w, cfg) = cfg.relax_form === :patankar ? w.a : w.arel

"""
    momentum_residual!(w, mesh, st, cfg; time_term=true) -> max residual (host Float64)

The unrelaxed residual of the system last assembled (assemble_momentum!) at the current velocity (level 1), in increment
form; `time_term = false` drops a^t (u - u^n): the steady residual.
"""
function momentum_residual!(w, m::PBMesh, st::PBState, cfg; time_term::Bool=true)
    be = cfg.backend
    _launch!(_k_mom_resid!, be, m.nc, w.res, w.bsx, w.bsy, w.ggx, w.ggy, lev(st.ux, 1), lev(st.uy, 1), lev(st.ux, 2),
             lev(st.uy, 2), w.as_, w.at, w.msx.upper, w.msx.lower, m.cface, m.own, m.nei, Int32(m.nf), m.vol,
             cfg.L_ref/(cfg.rho_l*cfg.U_ref^2), time_term ? 1.0 : 0.0)
    pb_sync(be)
    pb_maxabs(w.res; backend=be)
end

"""
    steady_residual(st, mesh, bc, w, cfg) -> Float64

The momentum residual of the state's level 1 with the time term removed (units rho U^2 / L per unit volume): the spatial
operator assembled at level 1 and evaluated there. A steady state is judged by it (review F4), never by a velocity change
divided by dt. Overwrites the momentum work arrays.
"""
function steady_residual(st::PBState, m::PBMesh, bc, w, cfg)
    # SST (W1): that state's mu_eff first; the laminar path is unchanged
    cfg.turbulence === :sst && !turb_update!(w, m, st, bc, cfg) && return NaN
    assemble_momentum!(w, m, st, bc, cfg, st.t)
    momentum_residual!(w, m, st, cfg; time_term=false)
end

"Apply the relaxation lambda to the assembled momentum system (u^(m-1) = w.uprev, w.vprev)."
function relax_momentum!(w, m::PBMesh, cfg, lam::Real)
    if cfg.relax_form === :patankar
        _launch!(_k_mom_relax!, cfg.backend, m.nc, w.msx.diag, w.msx.rhs, w.msy.rhs, w.a, w.uprev, w.vprev, Float64(lam))
    else
        _launch!(_k_mom_relax_gen!, cfg.backend, m.nc, w.msx.diag, w.msx.rhs, w.msy.rhs, w.a, w.arel, w.uprev, w.vprev,
                 Float64(lam))
    end
    pb_sync(cfg.backend)
    nothing
end

"H^s at the current velocity (before each corrector)."
function momentum_H!(w, m::PBMesh, st::PBState, cfg)
    _launch!(_k_mom_H!, cfg.backend, m.nc, w.Hx, w.Hy, w.bsx, w.bsy, lev(st.ux, 1), lev(st.uy, 1), w.msx.upper,
             w.msx.lower, m.cface, m.own, m.nei, Int32(m.nf))
    pb_sync(cfg.backend)
    nothing
end

# ---- the face flux by momentum interpolation (plan 4.3) --------------------------------------------------------------
@kernel function _k_rc_face!(Fs, Df, Hx, Hy, hxx, hxy, hyx, hyy, as_, A, a, at, vol, p, gpx, gpy, Fn, Fp, own, nei, fw, fa,
                             kx, ky, skx, sky, Sx, Sy, lam::Float64, lamn::Float64, skew::Float64, u, v, klin::Float64,
                             krc::Float64)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; w = fw[f]; w1 = 1 - w
        Ab = w*A[P] + w1*A[N]; ab = w*a[P] + w1*a[N]; atb = w*at[P] + w1*at[N]; Vb = w*vol[P] + w1*vol[N]
        HS = (w*Hx[P] + w1*Hx[N])*Sx[f] + (w*Hy[P] + w1*Hy[N])*Sy[f]
        klin != 0 && (HS += _rc_lin_term(as_, u, v, P, N, w, w1, Sx[f], Sy[f]))     # plan 28.6 (F3), rc_weight = :linear
        if skew != 0
            asb = w*as_[P] + w1*as_[N]; rx = skx[f]; ry = sky[f]
            HS += skew*asb*0.5*(((hxx[P] + hxx[N])*rx + (hxy[P] + hxy[N])*ry)*Sx[f] +
                                ((hyx[P] + hyx[N])*rx + (hyy[P] + hyy[N])*ry)*Sy[f])
        end
        tp = fa[f]*(p[N] - p[P]); np_ = 0.0
        if lamn > 0
            np_ = kx[f]*(w*gpx[P] + w1*gpx[N]) + ky[f]*(w*gpy[P] + w1*gpy[N])
            if lamn < 1
                lim = lamn/(1 - lamn)*abs(tp); np_ = clamp(np_, -lim, lim)
            end
        end
        rel = (1 - lam)/lam
        # rhie_chow = :off (krc 0; plan 28.11's negative example): the plain linear face velocity, no pressure in the flux
        Fs[f] = krc != 0 ? (HS + atb*Fn[f] + rel*ab*Fp[f] - Vb*(tp + np_))/Ab :
                           (w*u[P] + w1*u[N])*Sx[f] + (w*v[P] + w1*v[N])*Sy[f]
        Df[f] = Vb/(atb + (1/lam - 1)*ab)
    end
end

"""THE COEFFICIENT-JUMP TERM of the face velocity (plan 28.6, review F3; 15 Oct 2026; `PBConfig.rc_weight = :linear`).
At a converged steady state the face flux above is abar^s F = Hbar^s . S - Vbar (grad p)^c . S with H^s_P = a^s_P u_P +
V_P (grad p)^GG_P, so its velocity part is the a^s-WEIGHTED mean (w a^s_P u_P + w1 a^s_N u_N)/abar^s, not the linear
interpolant w u_P + w1 u_N. The two differ by w w1 (a^s_P - a^s_N)(u_P - u_N)/abar^s: O(h) wherever a^s jumps by O(1)
between neighbours, which it does at every velocity wall (the wall face's two-point viscous coefficient mu |S|^2/(S.d_b),
d_b about h/2, enters a^s of the wall cells only, twice in a corner cell). The face flux is then wrong by O(h^2), the wall
cells' continuity by O(1) per unit volume, and the pressure there by O(mu) at every resolution (probe F3 A/B, records
p0b/fixes_a8/). This term, added to Hbar^s . S, is w w1 (a^s_P - a^s_N)((u_N - u_P) S_x + (v_N - v_P) S_y), which makes the
steady velocity part exactly the linear interpolant; it holds no dt or lambda (a^s only), so plan 10.2's dt/lambda
independence is kept, and it vanishes where a^s is uniform."""
@inline _rc_lin_term(as_, u, v, P, N, w, w1, sx, sy) = w*w1*(as_[P] - as_[N])*((u[N] - u[P])*sx + (v[N] - v[P])*sy)
"1.0 when the face velocity is the linear interpolant (rc_weight = :linear), 0.0 for Bartholomew's coefficient weighting."
function _rc_klin(cfg)
    cfg.rc_weight in (:coefficient, :linear) ||
        throw(ArgumentError("PBConfig.rc_weight must be :coefficient or :linear (got :$(cfg.rc_weight))"))
    cfg.rc_weight === :linear ? 1.0 : 0.0
end
"1.0 for the momentum interpolation (rhie_chow = :on, the default), 0.0 for the plain linear face velocity (:off, plan 28.11's negative example)."
function _rc_on(cfg)
    cfg.rhie_chow in (:on, :off) || throw(ArgumentError("PBConfig.rhie_chow must be :on or :off (got :$(cfg.rhie_chow))"))
    cfg.rhie_chow === :on ? 1.0 : 0.0
end

"The momentum's pseudo-velocity Hhat = H^s/a^s (a^s > 0 whenever mu > 0: every face adds mu |S|^2/(S.d))."
@kernel function _k_hhat!(hbx, hby, Hx, Hy, as_)
    c = @index(Global)
    @inbounds begin
        d = as_[c]; hbx[c] = d > 0 ? Hx[c]/d : 0.0; hby[c] = d > 0 ? Hy[c]/d : 0.0
    end
end

"""Boundary faces: a velocity face's flux is fixed (u_b . S; slip 0); a pressure outlet's comes from the owner's momentum
equation at the face, F* = (H^s_P . S + a^t F^n + (1-lambda)/lambda a F^(m-1) - V_P (|S|^2/(S.d_b)(p_b - p_P) + k_b.grad p_P))/A_P."""
@kernel function _k_rc_bface!(Fbs, Db, Hx, Hy, A, a, at, vol, p, gpx, gpy, Fbn, Fbp, bkind, ub, vb, pb, bown, bSx, bSy,
                              ba, bkx, bky, lam::Float64, lamn::Float64, krc::Float64, u, v)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]; P = bown[b]
        if is_pressure_bc(k)
            tp = ba[b]*(pb[b] - p[P]); np_ = lamn > 0 ? bkx[b]*gpx[P] + bky[b]*gpy[P] : 0.0
            if lamn > 0 && lamn < 1                 # limited as an internal face's (review F6, 14 Oct)
                lim = lamn/(1 - lamn)*abs(tp); np_ = clamp(np_, -lim, lim)
            end
            rel = (1 - lam)/lam
            # rhie_chow = :off (krc 0; plan 28.11): the owner's velocity
            Fbs[b] = krc != 0 ? (Hx[P]*bSx[b] + Hy[P]*bSy[b] + at[P]*Fbn[b] + rel*a[P]*Fbp[b] - vol[P]*(tp + np_))/A[P] :
                                u[P]*bSx[b] + v[P]*bSy[b]
            Db[b] = vol[P]/(at[P] + (1/lam - 1)*a[P])
        elseif is_velocity_bc(k)
            Fbs[b] = ub[b]*bSx[b] + vb[b]*bSy[b]; Db[b] = 0.0
        else
            Fbs[b] = 0.0; Db[b] = 0.0
        end
    end
end

"""
    rhie_chow_flux!(w, mesh, st, bc, cfg, lam)

The face fluxes F* (w.Fs, w.Fbs) and the pressure-correction coefficients D (w.Df, w.Db) of plan 4.3 at the current
velocity (through H^s, `momentum_H!`) and pressure (level 1), with Choi's term on F^n (level 2) and Majumdar's on
F^(m-1) (w.Fprev, w.Fbprev). H^s at the face carries the skewness correction of the header, with the internal
least-squares gradient of Hhat = H^s/a^s computed here (w.hbx, w.hby; w.hxx ... w.hyy).
"""
function rhie_chow_flux!(w, m::PBMesh, st::PBState, bc, cfg, lam::Real)
    be = cfg.backend; p = lev(st.p, 1); G = w.lsq
    if cfg.skew
        _launch!(_k_hhat!, be, m.nc, w.hbx, w.hby, w.Hx, w.Hy, w.as_)
        for (g1, g2, phi) in ((w.hxx, w.hxy, w.hbx), (w.hyx, w.hyy, w.hby))
            _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, w.ubf, G.i11, G.i12, G.i22, m.cface, m.own, m.nei, m.dx, m.dy,
                     m.bdx, m.bdy, bc.kind, Int32(m.nf), LSQ_INT, Int32(0), Int32(0))
        end
    end
    _launch!(_k_rc_face!, be, m.nf, w.Fs, w.Df, w.Hx, w.Hy, w.hxx, w.hxy, w.hyx, w.hyy, w.as_, w.msx.diag, _arel(w, cfg), w.at, m.vol, p,
             w.gpx, w.gpy, lev(st.F, 2), w.Fprev, m.own, m.nei, m.fw, m.fa, m.kx, m.ky, m.skx, m.sky, m.Sx, m.Sy,
             Float64(lam), Float64(cfg.nonorth), cfg.skew ? 1.0 : 0.0, lev(st.ux, 1), lev(st.uy, 1), _rc_klin(cfg), _rc_on(cfg))
    _launch!(_k_rc_bface!, be, m.nb, w.Fbs, w.Db, w.Hx, w.Hy, w.msx.diag, _arel(w, cfg), w.at, m.vol, p, w.gpx, w.gpy,
             lev(st.Fb, 2), w.Fbprev, bc.kind, bc.ub, bc.vb, bc.pb, m.bown, m.bSx, m.bSy, m.ba, m.bkx, m.bky,
             Float64(lam), Float64(cfg.nonorth), _rc_on(cfg), lev(st.ux, 1), lev(st.uy, 1))
    pb_sync(be)
    nothing
end
