# =====================================================================================================================
# assemble_komega.jl -- SST-2003m transport of rho k and rho omega (plan 3.5, 4.7 step 1.1; work item W1 of plan Amendment
# 12, section 32; 21 Oct 2026). OPT-IN: PBConfig.turbulence = :sst (the default :laminar never reaches this file).
#
# THE EQUATIONS (per cell P, outward faces f; the mass flux phi = rho_l F, F the volumetric face flux of the flow iterate):
#   (rho^{n+1} k_P - rho^n k^n_P) V/dt + sum_f phi_f k_f^up - sum_f Gk_f (k_N - k_P) = V rho_l (P_k - beta* omega k_P)
#   (rho^{n+1} w_P - rho^n w^n_P) V/dt + sum_f phi_f w_f^up - sum_f Gw_f (w_N - w_P) = V rho_l (P_w - beta omega^2 + CD)
# THE DENSITIES IN THE INCOMPRESSIBLE LIMIT OF P0b (psi = 0, single phase). The time term carries the mixture densities of the
# state, rho^{n+1} (level 1: transported by the flow's flux, transport_mass!) and rho^n (level 2), so the discrete identity
# below holds; the sources, mu_t = rho_l nu_t and the storage use the liquid's rho_l: the state holds rk = rho_l k and
# rw = rho_l omega, so k = rk/rho_l is defined at every iterate (an intermediate flow iterate's rho^{n+1} can be far from
# rho_l, |Theta| = O(1) in the first outer iterations of a hard step, driven to |Theta| <= theta_tol only at acceptance;
# W1's development probes met a negative rho^{n+1} there). The two differ by (1 + Theta) at the committed state, about
# 1e-9. Phase change (stage P5a) revisits this.
# CONSERVATIVE FORM: each face's convective and diffusive fluxes enter its two cells with opposite signs (the column sums of
# the face terms vanish), and rho^{n+1} is the mixture density the flow's flux-form transport has just produced with the same
# flux (transport_mass!), so the k inventory changes only by the boundary fluxes, the sources and the floors; and, because
# rho^{n+1} V/dt + sum_f phi_f(out) = rho^n V/dt holds to round-off, the matrix keeps a diagonal margin rho^n V/dt over the
# off-diagonal sum (it is an M-matrix).
#   * convection: implicit first-order upwind in phi (the owner's diagonal carries max(phi, 0), its coefficient on the
#     neighbour -max(-phi, 0); the neighbour the reverse);
#   * diffusion: Gk = (mu + sigma_k mu_t)_f |S|^2/(S.d), sigma_k mu_t interpolated linearly (owner weight fw) from the cells;
#     the non-orthogonal part Gamma k.(grad k)_f deferred (PBConfig.turb_nonorth = lambda):
#       lambda = 1, THE DEFAULT since W1b (review of W1, F1; 22 Oct 2026): the FULL correction, unlimited, so the diffusion is
#         second order on a non-orthogonal mesh (the review's distorted periodic boxes: the error falls as h^2); positivity is
#         kept by a sign split of each cell's NET correction C_P = sum_f (+-) Gamma k.(grad x)_f (turb_nonorth_split = true,
#         the default): C_P >= 0 goes to the right side, C_P < 0 to the diagonal as -C_P / x_P^m (x^m the iterate, > 0), the
#         treatment the cross diffusion already has. The right side is then never negative and the matrix stays an M-matrix
#         (its diagonal only grows), so the exact solution is positive; at the iterate the split system's residual NUMERATOR is
#         the unsplit one's (b - A x^m equals it), so the converged state is the full correction's. The relative residuals
#         (R_turb, turb_res, the steady k and omega residuals) divide that numerator by the split system's diagonal a_P, which in
#         a split cell is larger by -C_P/x_P^m, so they read lower there than the unsplit normalisation would (re-review of W1b,
#         R3, corrected wording of the integration, 22 Oct 2026: diagonal ratio at most 1.01 on N60R after 20 steps; up to 1.74
#         in 16 cells of the review's steady distorted cavity, where the reported residual was 6% below the unsplit one). Each
#         face's correction still enters its two cells with opposite signs (conservative); a cell's
#         diagonal share reproduces its C_P exactly at convergence (k = k^m), and between iterates differs from it by
#         C_P (x_P - x_P^m)/x_P^m. turb_nonorth_split = false puts C_P on the right side whatever its sign (a negative example of
#         the suite: the right side can then go negative);
#       0 < lambda < 1: W1's LIMITED form, |corr| <= lambda/(1 - lambda) |two-point flux| per face (the momentum's limited
#         form). INCONSISTENT (review F1, probe R5): where the two-point difference is small against the tangential gradient the
#         clamp removes a fixed fraction of an O(h) flux on an O(h^2) cell, so the error grows like 1/h (0.29 -> 0.78 from n 16
#         to 128 at distortion 0.05). Kept only to reproduce W1's runs; not a production setting;
#       lambda = 0: none (two-point only, as version 1; O(1) inconsistent on a non-orthogonal mesh).
#     W6 (22 Oct 2026; records air/w6_start/): OMEGA's correction is, by default, not formed on the internal faces of
#     wall-adjacent cells (PBConfig.turb_omega_nonorth = :wall_off; :full is W1b's, :limited W1's limiter on omega alone):
#     there the least-squares omega gradient carries the 6 nu/(beta1 y^2) wall singularity, and behind the sharp trailing
#     edge of the NACA 0012 its correction made the first wake cells' omega equation have no positive solution, so the split
#     below held omega at the floor (the review of the integration, F1). Each cell's split share is recorded (kshr, wshr), and
#     the steady residual (turbulence_steady_residual) is normalised by the UNSPLIT diagonal, so a cell the split holds down
#     can no longer read as converged.
#     Boundary faces are two-point (no correction): exact on wall-orthogonal cells (W1b measures the wall faces' |bk|/|S|;
#     the re-review of W1b, R2: N60R's 16 faces above 0.05 are its whole blunt base, every one at 0.1305, where the omission
#     reaches 8.6e-2 (k) and 0.86 (omega) of the two-point flux; on the foil 2e-4 and 2.6e-5; W2's sharp NACA 0012 family has
#     its skewed wall faces at the nose only, up to 0.19-0.38 at n_foil 256 and none above 0.05 at 384 or 512).
#   * sources (production lagged; destruction implicit): the k production P_k = min(nu_t S^2, 10 beta* k w) on the right,
#     the k destruction rho_l beta* omega^m on the diagonal; the omega production on the right, the omega destruction
#     Newton-linearised, rho_l beta (2 omega^m omega - omega^m^2) (2 rho_l beta omega^m V on the diagonal, rho_l beta
#     omega^m^2 V on the right, both >= 0); the cross diffusion CD on the right where it is positive, -CD/omega^m on the diagonal where it
#     is negative (so the right side is never negative from a source). The terms are sst_terms (turbulence.jl) at the
#     iterate (turb_update!): velocity, k and omega least-squares gradients (with boundary values), the wall distance to the
#     no-slip faces of the condition set.
#   * boundary faces (plan 9): no-slip walls k = 0 and omega = omega_wall (option W-a, sst_wall_omega_v1 at the wall-normal
#     distance d1 = S_b.d_b/|S_b| of the owner's centroid), diffusive with Gamma = mu (mu_t = 0 at the wall); velocity
#     inflow faces (inlet, far field) where the flow ENTERS (phi < 0) and backflow through a pressure outlet carry the inflow
#     values (k_in, w_in) convectively, outflow the cell's value, no diffusive flux (version 1's open boundaries); slip faces
#     no flux. The least-squares gradients and the momentum's boundary mu_eff follow the same rule (W1b, review N5): a velocity
#     inflow or far-field face takes k_in, w_in and mu + rho_l k_in/w_in only where the flow enters, the owner's values (zero
#     gradient) and mu + mu_t,P where it leaves.
# THE SOLVE: alternating coloured line Gauss-Seidel sweeps (j-lines then i-lines; linsolve_lines.jl) from the iterate, on k
# then omega, until the relative residual (relres: max_P |r_P| / (a_P max(x_P, x_ref)), x_ref the inflow value) falls by
# turb_lin_rtol or below a tenth of the step's tolerance (turb_tol, or turb_step_tol when smaller; the GOAL FLOOR, below which a
# solve takes no sweep: with W1b's default turb_tol 1e-3 it was 1e-4 and froze k and omega in a steady march, re-review of
# W1b R1; the default is 1e-8 since the W1+W2 integration, and an SST steady stop must register a k-omega rule above the
# floor, pimple.jl), at most turb_sweeps_max
# sweeps; a solve that misses its goal is CONTINUED once from where it stopped with three times that budget more (W1b, review
# N3: counted, `retries`, and ledgered), and only then reported not converged (the caller rejects :turb_linear_failed).
# POSITIVITY: the exact solution of the M-matrix system with its non-negative right side is positive (with turb_nonorth = 1 and
# the split, the default, nothing makes the right side negative); round-off (and the limited forms' deferred term) can still
# take a value below k_min / omega_min, and the floor then restores it, COUNTED and ledgered (cells and the added rho_l x V),
# never silently (PBLedger turb_floor_*).
# THE OUTER CRITERION (pimple.jl, coupled_outer!): R_turb = max(relres_k, relres_w) / turb_tol of the system assembled at
# the iterate (k, omega and the flow iterate), so a step is accepted only when the k-omega equations hold at the committed
# iterate as the flow's do. Under turb_coupling = :step (THE DEFAULT since W1b) the k-omega update follows the accepted flow
# iterate: one update (turb_step_tol = Inf, the default: the steady march's pseudo-time form) or repeated updates until the
# k-omega relative residual re-assembled at the new iterate is within turb_step_tol (pimple.jl; linsolve_coupled.jl).
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

"""
    TurbWork

The SST work of a PBWork (PBConfig.turbulence = :sst): the inflow values and floors, the wall distance to the no-slip faces
(`wdist`), the iterate's primitives and boundary values, their gradients, the cell terms of sst_terms (sigma_k, sigma_w,
P_k, P_w, beta, CD, F1, F2), the wall omega per boundary face, the k and omega face systems with their deferred and boundary
parts, four line families (j and i for each system), this step's floor, sweep and retry counters (W1b: `retries`, the
solves continued past their first budget, review N3), and the number of no-slip faces (`nwall`).
"""
mutable struct TurbWork{V,FS,LF}
    nu::Float64; kin::Float64; win::Float64; kmin::Float64; wmin::Float64
    wdist::V
    k::V; w::V; kb::V; wb::V; wwall::V; d1b::V
    gkx::V; gky::V; gwx::V; gwy::V
    sigk::V; sigw::V; Pk::V; Pw::V; bet::V; cross::V; F1::V; F2::V; stat::V
    ksys::FS; wsys::FS
    kfc::V; wfc::V; kbd::V; kbr::V; wbd::V; wbr::V
    fjk::LF; fik::LF; fjw::LF; fiw::LF
    rw::V; zw::V; res::V; fl::V
    nfloor_k::Int; nfloor_w::Int; added_k::Float64; added_w::Float64
    sweeps::Vector{Int}
    retries::Int
    nwall::Int
    wfmask::V            # W6: per internal face 0.0 where either cell has a no-slip face (turb_omega_nonorth = :wall_off), else 1.0
    nwfmask::Int         # the number of such faces
    kshr::V; wshr::V     # W6 (the review of the integration, F1): each cell's split share -C/x^m on the diagonal (0 where unsplit)
    wcut::V              # W6b: per cell, the part of omega's net negative correction the bound removed (0 where it did not act)
    wbq::V               # W6b (the review of W6b, F1): per cell, the bound's budget Q of the last assembly (0 where the bound is off)
    w7::Any                   # W7: the Float32 copies of the four line families and their work (kw_precision = :f32), else nothing
end

"The condition kinds that are no-slip walls for the turbulence model (k = 0, omega = omega_wall, mu_t = 0)."
@inline is_wall_bc(k::UInt8) = k == UInt8(BC_NOSLIP) || k == UInt8(BC_MOVING_WALL)
"The velocity kinds whose inflow carries the inflow turbulence (inlet, far field)."
@inline is_inflow_bc(k::UInt8) = k == UInt8(BC_INLET) || k == UInt8(BC_FARFIELD) || k == UInt8(BC_FREESTREAM_OUTLET)

@kernel function _k_turb_wallomega!(wwall, d1b, bkind, bSx, bSy, bSmag, bdx, bdy, nu::Float64, mode::Int32)
    b = @index(Global)
    @inbounds begin
        d1 = (bSx[b]*bdx[b] + bSy[b]*bdy[b])/bSmag[b]; d1b[b] = d1
        wwall[b] = is_wall_bc(bkind[b]) ? sst_wall_omega_v1(nu, d1) : 0.0
    end
end

"""
    turbwork(mesh, bc, cfg) -> TurbWork

Allocate the SST work (pbwork calls it when cfg.turbulence = :sst): the inflow values from cfg.U_ref, cfg.sst_tu and
cfg.sst_mut_ratio (sst_freestream, nu = mu/rho_l), the wall distance from every cell's true centroid to the no-slip faces of
`bc` (Inf without one; mesh_faces.jl's kernel), and the wall omega of option W-a on each no-slip face. W-b
(cfg.wall_omega = :auto) is a loud stub (stage P2).
"""
function turbwork(m::PBMesh, bc, cfg)
    be = cfg.backend; nc, nf, nb = m.nc, m.nf, m.nb
    cfg.wall_omega === :v1 || (cfg.wall_omega === :auto ? sst_wall_omega_auto(1.0, 1.0, 1.0) :
        throw(ArgumentError("PBConfig.wall_omega must be :v1 (W-a) or :auto (W-b, stage P2; got :$(cfg.wall_omega))")))
    nu = cfg.mu/cfg.rho_l
    kin, win = sst_freestream(cfg.U_ref, cfg.sst_tu, cfg.sst_mut_ratio, nu)
    z(n) = KernelAbstractions.zeros(be, Float64, n)
    wd = z(nc)
    bk = _host(bc.kind); wq = findall(is_wall_bc, bk)
    if isempty(wq)
        fill!(wd, Inf)
    else
        bna = _host(m.bna)[wq]; bnb = _host(m.bnb)[wq]; xn = _host(m.xn); yn = _host(m.yn)
        _launch!(_k_wdist!, be, nc, wd, m.xc, m.yc, _dev(be, xn[bna]), _dev(be, yn[bna]), _dev(be, xn[bnb]), _dev(be, yn[bnb]),
                 Int32(length(wq)))
        pb_sync(be)
    end
    wwall = z(max(nb, 1)); d1b = z(max(nb, 1))
    _launch!(_k_turb_wallomega!, be, nb, wwall, d1b, bc.kind, m.bSx, m.bSy, m.bSmag, m.bdx, m.bdy, nu, Int32(0))
    pb_sync(be)
    fs() = FaceSystem(m; backend=be)
    # W6: the faces of the wall-adjacent cells (turb_omega_nonorth = :wall_off; host index arithmetic, once per work space)
    bownh = _host(m.bown); wc = falses(nc); for b in wq; wc[bownh[b]] = true; end
    ownh = _host(m.own); neih = _host(m.nei)
    wm = [(wc[ownh[f]] || wc[neih[f]]) ? 0.0 : 1.0 for f in 1:nf]
    wfm = z(max(nf, 1)); nf > 0 && copyto!(wfm, wm)
    TurbWork(nu, kin, win, cfg.k_min, cfg.w_min, wd,
             z(nc), z(nc), z(max(nb, 1)), z(max(nb, 1)), wwall, d1b,
             z(nc), z(nc), z(nc), z(nc),
             z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc),
             fs(), fs(),
             z(max(nf, 1)), z(max(nf, 1)), z(max(nb, 1)), z(max(nb, 1)), z(max(nb, 1)), z(max(nb, 1)),
             line_family(m, :j; backend=be), line_family(m, :i; backend=be),
             line_family(m, :j; backend=be), line_family(m, :i; backend=be),
             z(nc), z(nc), z(nc), z(nc), 0, 0, 0.0, 0.0, Int[], 0, length(wq), wfm, count(==(0.0), wm), z(nc), z(nc), z(nc), z(nc), nothing)
end

"""
    init_turbulence!(st, mesh, bc, cfg; omega_start=:freestream, work=nothing) -> st

Every time level of rk = rho_l k and rw = rho_l omega set to the inflow values, and mu_t to rho_l k_in/w_in; omega_start =
:wall raises omega to max(w_in, 6 nu/(beta1 d^2)) from the wall distance (version 1's TOPO_SST_OMEGA_INIT=wall), with
mu_t = rho_l k/omega. The start of a verification case (set the flow first, init_state!).
"""
function init_turbulence!(st::PBState, m::PBMesh, bc, cfg; omega_start::Symbol=:freestream, work=nothing)
    omega_start in (:freestream, :wall) || throw(ArgumentError("omega_start must be :freestream or :wall"))
    T = work === nothing ? turbwork(m, bc, cfg) : work.turb
    be = cfg.backend
    for L in 1:NTIME
        _launch!(_k_turb_init!, be, m.nc, lev(st.rk, L), lev(st.rw, L), lev(st.mut, L), cfg.rho_l, T.wdist, T.kin, T.win,
                 T.nu, Int32(omega_start === :wall ? 1 : 0))
    end
    pb_sync(be)
    st
end
@kernel function _k_turb_init!(rk, rw, mut, r::Float64, wdist, kin::Float64, win::Float64, nu::Float64, wall::Int32)
    c = @index(Global)
    @inbounds begin
        w = win
        wall == Int32(1) && (d = wdist[c]; w = max(win, 6*nu/(SST_BETA1*d*d)))
        rk[c] = r*kin; rw[c] = r*w; mut[c] = r*kin/w
    end
end

# ---- the cell terms at the iterate -------------------------------------------------------------------------------------
@kernel function _k_turb_prim!(k, w, rk, rw, rho_l::Float64)
    c = @index(Global)
    @inbounds begin
        k[c] = rk[c]/rho_l; w[c] = rw[c]/rho_l
    end
end
"Boundary values of k and omega for the least-squares gradients: the wall's (0, omega_wall), an inflow or far-field face's
(k_in, w_in) where the flow enters (Fb < 0), else the owner's (zero gradient; W1b, review N5: before, every inflow-kind face
took the inflow values, outflow parts of a far field included)."
@kernel function _k_turb_bvals!(kb, wb, k, w, bkind, bown, wwall, kin::Float64, win::Float64, Fb)
    b = @index(Global)
    @inbounds begin
        kd = bkind[b]; P = bown[b]
        if is_wall_bc(kd)
            kb[b] = 0.0; wb[b] = wwall[b]
        elseif is_inflow_bc(kd) && Fb[b] < 0
            kb[b] = kin; wb[b] = win
        else
            kb[b] = k[P]; wb[b] = w[P]
        end
    end
end
@kernel function _k_sst_cell!(mut, sigk, sigw, Pk, Pw, bet, cross, F1, F2, stat, k, w, wdist, r::Float64, gux, guy, gvx, gvy,
                              gkx, gky, gwx, gwy, nu::Float64)
    c = @index(Global)
    @inbounds begin
        kk = k[c]; ww = w[c]
        if kk > 0 && ww > 0 && isfinite(kk) && isfinite(ww)       # never evaluated otherwise (sqrt(k)): flagged instead
            t = sst_terms(kk, ww, wdist[c], nu, r, gux[c], guy[c], gvx[c], gvy[c], gkx[c]*gwx[c] + gky[c]*gwy[c])
            mt = r*t.nut
            mut[c] = mt; sigk[c] = t.sigk; sigw[c] = t.sigw; Pk[c] = t.P; Pw[c] = t.Pw; bet[c] = t.beta; cross[c] = t.cross
            F1[c] = t.F1; F2[c] = t.F2
            good = isfinite(mt) && mt >= 0 && isfinite(t.P) && isfinite(t.Pw) && isfinite(t.cross)
            stat[c] = good ? 0.0 : 1.0
        else
            mut[c] = 0.0; sigk[c] = 0.0; sigw[c] = 0.0; Pk[c] = 0.0; Pw[c] = 0.0; bet[c] = 0.0; cross[c] = 0.0
            F1[c] = 0.0; F2[c] = 0.0; stat[c] = 1.0
        end
    end
end
"The momentum's effective viscosity: internal faces mu + (fw mu_t,P + (1 - fw) mu_t,N)."
@kernel function _k_mueff_face!(muf, mut, fw, own, nei, mu::Float64)
    f = @index(Global)
    @inbounds begin
        w = fw[f]; muf[f] = mu + (w*mut[own[f]] + (1 - w)*mut[nei[f]])
    end
end
"Boundary faces: mu at a no-slip wall (mu_t = 0), mu + rho_l k_in/w_in at an inflow or far-field face where the flow enters
(Fb < 0), mu + mu_t,P elsewhere (W1b, review N5)."
@kernel function _k_mueff_bface!(mub, mut, bkind, bown, mu::Float64, mut_in::Float64, Fb)
    b = @index(Global)
    @inbounds begin
        kd = bkind[b]
        mub[b] = is_wall_bc(kd) ? mu : (is_inflow_bc(kd) && Fb[b] < 0) ? mu + mut_in : mu + mut[bown[b]]
    end
end

"""
    turb_update!(w, mesh, st, bc, cfg; level=1) -> Bool

The SST terms at the iterate (level 1: u, v, rk = rho_l k, rw = rho_l omega): the primitives, the boundary values, the velocity
(velocity_gradients!), k and omega least-squares gradients, sst_terms in every cell, mu_t into the state (level 1) and the
momentum's face viscosities (w.muf, w.mub). False when a cell's terms are not finite or its k or omega is not positive (the
caller rejects the attempt :nonfinite).
`level` (27 Oct 2026, turb_mut_base; records air/mutbase/): the time level the terms are formed FROM -- 1, the iterate (the
default, every caller before it, bit for bit), or 2, the committed level n (u, v, Fb, rk and rw of level 2: the frozen mu_t of
turb_mut_base = :committed, `frozen_mut_level`). The result is written where it always is (mu_t into level 1, the face
viscosities into the work), and no state field of any level is otherwise written. Whatever the level, the work's velocity
gradients and boundary velocities then hold that level's; the momentum assembly re-forms both at level 1 before it reads them.
"""
function turb_update!(w, m::PBMesh, st::PBState, bc, cfg; level::Integer=1)
    (level == 1 || level == 2) ||
        throw(ArgumentError("turb_update!: level must be 1 (the iterate) or 2 (level n: turb_mut_base = :committed), got $level"))
    be = cfg.backend; T = w.turb; G = w.lsq
    _launch!(_k_turb_prim!, be, m.nc, T.k, T.w, lev(st.rk, level), lev(st.rw, level), cfg.rho_l)
    _launch!(_k_turb_bvals!, be, m.nb, T.kb, T.wb, T.k, T.w, bc.kind, m.bown, T.wwall, T.kin, T.win, lev(st.Fb, level))
    pb_sync(be)
    velocity_gradients!(w, m, st, bc, cfg; level)
    for (g1, g2, phi, phib) in ((T.gkx, T.gky, T.k, T.kb), (T.gwx, T.gwy, T.w, T.wb))
        _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, phib, G.u11, G.u12, G.u22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), LSQ_VEL, Int32(0), Int32(0))
    end
    mut = lev(st.mut, 1)
    _launch!(_k_sst_cell!, be, m.nc, mut, T.sigk, T.sigw, T.Pk, T.Pw, T.bet, T.cross, T.F1, T.F2, T.stat, T.k, T.w, T.wdist,
             cfg.rho_l, w.gux, w.guy, w.gvx, w.gvy, T.gkx, T.gky, T.gwx, T.gwy, T.nu)
    _launch!(_k_mueff_face!, be, m.nf, w.muf, mut, m.fw, m.own, m.nei, cfg.mu)
    _launch!(_k_mueff_bface!, be, m.nb, w.mub, mut, bc.kind, m.bown, cfg.mu, cfg.rho_l*T.kin/T.win, lev(st.Fb, level))
    pb_sync(be)
    pb_maxabs(T.stat; backend=be) == 0.0
end

# ---- the k and omega systems -------------------------------------------------------------------------------------------
"""Internal faces of both systems: the coefficient of the owner's row on the neighbour (up) and of the neighbour's row on
the owner (lo) -- the owner's diagonal gets -lo, the neighbour's -up (conservative upwind: column sums of the face terms
vanish) -- and the owner's deferred, limited non-orthogonal diffusion (the neighbour's is minus it)."""
@kernel function _k_kw_face!(kup, klo, wup, wlo, kfc, wfc, F, mut, sigk, sigw, k, w, gkx, gky, gwx, gwy, own, nei, fw, fa,
                             kx, ky, rho_l::Float64, mu::Float64, lamn::Float64, wfm, wmode::Int32)
    f = @index(Global)
    @inbounds begin
        P = own[f]; N = nei[f]; a = fw[f]; a1 = 1 - a
        phi = rho_l*F[f]; pp = max(phi, 0.0); pm = max(-phi, 0.0)
        Gk = (mu + (a*sigk[P]*mut[P] + a1*sigk[N]*mut[N]))*fa[f]
        Gw = (mu + (a*sigw[P]*mut[P] + a1*sigw[N]*mut[N]))*fa[f]
        kup[f] = -(Gk + pm); klo[f] = -(Gk + pp); wup[f] = -(Gw + pm); wlo[f] = -(Gw + pp)
        ck = 0.0; cw = 0.0
        if lamn > 0
            ck = Gk/fa[f]*(kx[f]*(a*gkx[P] + a1*gkx[N]) + ky[f]*(a*gky[P] + a1*gky[N]))
            cw = Gw/fa[f]*(kx[f]*(a*gwx[P] + a1*gwx[N]) + ky[f]*(a*gwy[P] + a1*gwy[N]))
            if lamn < 1
                r = lamn/(1 - lamn)
                lk = r*abs(Gk*(k[N] - k[P])); lw = r*abs(Gw*(w[N] - w[P]))
                ck = clamp(ck, -lk, lk); cw = clamp(cw, -lw, lw)
            end
            # turb_omega_nonorth (W6): 1 = :wall_off (none on a face of a wall-adjacent cell), 2 = :limited (|cw| <= |Gw (w_N - w_P)|)
            if wmode == Int32(1)
                wfm[f] == 0.0 && (cw = 0.0)
            elseif wmode == Int32(2)
                lw2 = abs(Gw*(w[N] - w[P])); cw = clamp(cw, -lw2, lw2)
            end
        end
        kfc[f] = ck; wfc[f] = cw
    end
end
"Boundary faces of both systems: the owner's diagonal (kbd, wbd) and right side (kbr, wbr) by condition (the header)."
@kernel function _k_kw_bface!(kbd, kbr, wbd, wbr, Fb, bkind, ba, wwall, rho_l::Float64, mu::Float64, kin::Float64, win::Float64)
    b = @index(Global)
    @inbounds begin
        kd = bkind[b]; phi = rho_l*Fb[b]
        dk = 0.0; rk = 0.0; dw = 0.0; rw = 0.0
        if is_wall_bc(kd)
            # the wall's Dirichlet values k = 0, omega = omega_wall through the two-point diffusion (mu_t = 0 at the wall);
            # a wall face's flux u_b . S is zero (bc.jl), so it carries no convection
            G = mu*ba[b]
            dk = G; dw = G; rw = G*wwall[b]
        elseif kd == UInt8(BC_SLIP)
            nothing
        else                                            # velocity inflow / far field, pressure outlet: open, convective
            if phi >= 0
                dk = phi; dw = phi
            else
                rk = -phi*kin; rw = -phi*win
            end
        end
        kbd[b] = dk; kbr[b] = rk; wbd[b] = dw; wbr[b] = rw
    end
end
"""Cells: the k and omega diagonals and right sides (time term times tw, sources, the face and boundary parts). wk, ww the
iterate's k and omega (the lagged coefficients); rho1, rhon the mixture densities of levels 1 and 2 (the time term); rkn,
rwn level n's rk = rho_l k, rw = rho_l omega; r = rho_l (the sources). split = 1 (turb_nonorth = 1 with turb_nonorth_split,
W1b's default): the cell's net deferred non-orthogonal correction C goes to the right side when C >= 0 and to the diagonal as
-C/x^m when C < 0 (header); split = 0: each face's correction to the right side, in W1's order (bit for bit W1's kernel)."""
@kernel function _k_kw_cell!(kdg, krhs, wdg, wrhs, kup, klo, wup, wlo, kfc, wfc, kbd, kbr, wbd, wbr, cface, nf::Int32, vol,
                             rho1, rhon, rkn, rwn, ww, Pk, Pw, bet, cross, dt::Float64, tw::Float64, r::Float64, wk,
                             split::Int32, kshr, wshr, own, nei, wbnd::Float64, wcut, wbq)
    c = @index(Global)
    @inbounds begin
        V = vol[c]; wm = ww[c]
        at = tw*rho1[c]*V/dt; an = tw*rhon[c]/r*V/dt
        dk = at + r*SST_BETA_STAR*wm*V; bk = an*rkn[c] + r*Pk[c]*V
        cr = cross[c]
        dw = at + 2*r*bet[c]*wm*V; bw = an*rwn[c] + r*Pw[c]*V + r*bet[c]*wm*wm*V
        if cr >= 0
            bw += r*cr*V
        else
            dw += r*(-cr)/wm*V
        end
        ck = 0.0; cw = 0.0                                  # the net corrections (split = 1 only)
        # W6b: the bound's budget (wbnd > 0 only): the STEADY positive right side (production, positive cross diffusion,
        # boundary inflow; not the time term, not the destruction's linearisation) and sum_N |a_N| omega_N^m
        qs = r*Pw[c]*V + (cr >= 0 ? r*cr*V : 0.0); qn = 0.0
        for q in 1:4
            code = cface[q, c]
            if code > nf
                b = code - nf; dk += kbd[b]; bk += kbr[b]; dw += wbd[b]; bw += wbr[b]; qs += wbr[b]
            elseif code > 0
                dk -= klo[code]; dw -= wlo[code]
                wbnd > 0 && (qn -= wup[code]*ww[nei[code]])
                if split == Int32(1)
                    ck += kfc[code]; cw += wfc[code]
                else
                    bk += kfc[code]; bw += wfc[code]
                end
            else
                dk -= kup[-code]; dw -= wup[-code]
                wbnd > 0 && (qn -= wlo[-code]*ww[own[-code]])
                if split == Int32(1)
                    ck -= kfc[-code]; cw -= wfc[-code]
                else
                    bk -= kfc[-code]; bw -= wfc[-code]
                end
            end
        end
        # turb_omega_bound (W6b; the review of W6, R1): omega's net NEGATIVE correction at most wbnd times the cell's steady
        # positive budget Q = qs + qn (above), so, AS FAR AS THE NON-ORTHOGONAL CORRECTION GOES, the cell's steady equation
        # keeps a positive solution, its transport and destruction balancing at least (1 - wbnd) Q (the negative cross
        # diffusion, on the diagonal above as -cr/omega^m in the same Patankar form, is neither in Q nor bounded: the review of
        # W6b, F5); the time term is not in Q, so the bound cannot let omega decay by a fixed factor per step at small dt, and
        # the bounded scheme's steady fixed point does not depend on dt (it is the unbounded scheme's only where the bound is
        # inactive at convergence: the review of W6b, F1); inactive where -C <= wbnd Q. wbq records Q (the census's cut/Q).
        cut = 0.0; qb = 0.0
        if wbnd > 0 && split == Int32(1)
            qb = qs + qn
            lim = -wbnd*qb
            cw < lim && (cut = lim - cw; cw = lim)
        end
        wcut[c] = cut; wbq[c] = qb
        sk = 0.0; sw = 0.0                                  # the split shares (W6: recorded for the unsplit normaliser)
        if split == Int32(1)
            km = wk[c]
            if ck >= 0 || !(km > 0)
                bk += ck
            else
                sk = -ck/km; dk += sk
            end
            if cw >= 0 || !(wm > 0)
                bw += cw
            else
                sw = -cw/wm; dw += sw
            end
        end
        kdg[c] = dk; krhs[c] = bk; wdg[c] = dw; wrhs[c] = bw; kshr[c] = sk; wshr[c] = sw
    end
end

"""
    assemble_komega!(w, mesh, st, bc, cfg; time_term=true)

The k and omega systems (w.turb.ksys, w.turb.wsys) at the iterate, from the terms of the last `turb_update!`, the flux of
level 1, the mixture densities of levels 1 (rho^{n+1}) and 2 (rho^n), and level 2's rk, rw (k^n, omega^n). `time_term = false` drops the
time term (the steady residual).
"""
function assemble_komega!(w, m::PBMesh, st::PBState, bc, cfg; time_term::Bool=true)
    be = cfg.backend; T = w.turb; ks = T.ksys; ws = T.wsys; mut = lev(st.mut, 1)
    _launch!(_k_kw_face!, be, m.nf, ks.upper, ks.lower, ws.upper, ws.lower, T.kfc, T.wfc, lev(st.F, 1), mut, T.sigk, T.sigw,
             T.k, T.w, T.gkx, T.gky, T.gwx, T.gwy, m.own, m.nei, m.fw, m.fa, m.kx, m.ky, cfg.rho_l, cfg.mu,
             Float64(cfg.turb_nonorth), T.wfmask, Int32(cfg.turb_omega_nonorth === :wall_off ? 1 :
                                                          cfg.turb_omega_nonorth === :limited ? 2 : 0))
    _launch!(_k_kw_bface!, be, m.nb, T.kbd, T.kbr, T.wbd, T.wbr, lev(st.Fb, 1), bc.kind, m.ba, T.wwall, cfg.rho_l, cfg.mu,
             T.kin, T.win)
    _launch!(_k_kw_cell!, be, m.nc, ks.diag, ks.rhs, ws.diag, ws.rhs, ks.upper, ks.lower, ws.upper, ws.lower, T.kfc, T.wfc,
             T.kbd, T.kbr, T.wbd, T.wbr, m.cface, Int32(m.nf), m.vol, lev(st.rho, 1), lev(st.rho, 2), lev(st.rk, 2),
             lev(st.rw, 2), T.w, T.Pk, T.Pw, T.bet, T.cross, st.dt, time_term ? 1.0 : 0.0, cfg.rho_l, T.k,
             Int32(cfg.turb_nonorth == 1 && cfg.turb_nonorth_split ? 1 : 0), T.kshr, T.wshr, m.own, m.nei,
             Float64(cfg.turb_omega_bound), T.wcut, T.wbq)
    pb_sync(be)
    nothing
end
assemble_komega!(sys_k, sys_w, mesh, state, config) =
    throw(ArgumentError("assemble_komega!(w, mesh, st, bc, cfg): the k and omega systems live in w.turb (PBConfig.turbulence = :sst)"))

"max_P |b - A x|_P / (a_P max(x_P, xref)): the relative (Jacobi) residual of a face system at x."
@kernel function _k_relres!(out, b, x, dg, up, lo, cface, own, nei, nf::Int32, xref::Float64)
    c = @index(Global)
    @inbounds begin
        acc = dg[c]*x[c]
        for q in 1:4
            code = cface[q, c]
            if 0 < code <= nf
                acc += up[code]*x[nei[code]]
            elseif code < 0
                acc += lo[-code]*x[own[-code]]
            end
        end
        out[c] = abs(b[c] - acc)/(dg[c]*max(x[c], xref))
    end
end
function _relres(sys::FaceSystem, x, m::PBMesh, out, xref; backend=CPU())
    _launch!(_k_relres!, backend, m.nc, out, sys.rhs, x, sys.diag, sys.upper, sys.lower, m.cface, m.own, m.nei, Int32(m.nf),
             Float64(xref))
    pb_maxabs(out; backend)
end
"""The relative residual normalised by the UNSPLIT diagonal (W6; the review of the integration, F1): max_P |b - A x|_P /
((a_P - s_P) max(x_P, xref)), s_P the split share -C_P/x_P^m the assembly put on the diagonal (0 where the cell was not split).
At the iterate the system was assembled at, the numerator is the unsplit system's (W1b: b - A x equals it), so this is the
unsplit relative residual. A cell held at the floor by the split (diagonal share 1e8, unsplit diagonal 1e-3) reads small under
the split diagonal and large here."""
@kernel function _k_relres_unsplit!(out, b, x, dg, shr, up, lo, cface, own, nei, nf::Int32, xref::Float64)
    c = @index(Global)
    @inbounds begin
        acc = dg[c]*x[c]
        for q in 1:4
            code = cface[q, c]
            if 0 < code <= nf
                acc += up[code]*x[nei[code]]
            elseif code < 0
                acc += lo[-code]*x[own[-code]]
            end
        end
        out[c] = abs(b[c] - acc)/((dg[c] - shr[c])*max(x[c], xref))
    end
end
function _relres_unsplit(sys::FaceSystem, x, shr, m::PBMesh, out, xref; backend=CPU())
    _launch!(_k_relres_unsplit!, backend, m.nc, out, sys.rhs, x, sys.diag, shr, sys.upper, sys.lower, m.cface, m.own, m.nei,
             Int32(m.nf), Float64(xref))
    pb_maxabs(out; backend)
end

"""
    komega_residual(w, mesh, st, bc, cfg; time_term=true) -> (k, omega)

The relative residuals of the k and omega systems assembled at the iterate (after turb_update!), each
max_P |r_P| / (a_P max(x_P, x_in)) with x_in the inflow value.
"""
function komega_residual(w, m::PBMesh, st::PBState, bc, cfg; time_term::Bool=true, unsplit::Bool=false)
    T = w.turb; be = cfg.backend
    assemble_komega!(w, m, st, bc, cfg; time_term)
    unsplit && return (_relres_unsplit(T.ksys, T.k, T.kshr, m, T.res, T.kin; backend=be),
                       _relres_unsplit(T.wsys, T.w, T.wshr, m, T.res, T.win; backend=be))
    (_relres(T.ksys, T.k, m, T.res, T.kin; backend=be), _relres(T.wsys, T.w, m, T.res, T.win; backend=be))
end

@kernel function _k_floor!(x, added, xmin::Float64, r::Float64, vol)
    c = @index(Global)
    @inbounds begin
        v = x[c]
        if !(v >= xmin) && isfinite(v)
            added[c] = (xmin - v)*r*vol[c]; x[c] = xmin
        else
            added[c] = 0.0
        end
    end
end
@kernel function _k_count_pos!(out, added)
    c = @index(Global)
    @inbounds out[c] = added[c] > 0 ? 1.0 : 0.0
end
@kernel function _k_kw_store!(rk, rw, k, w, r::Float64)
    c = @index(Global)
    @inbounds begin
        rk[c] = r*k[c]; rw[c] = r*w[c]
    end
end

"""The floor of a k-omega line solve's goal: a tenth of turb_tol, or of turb_step_tol under turb_coupling = :step when that
is smaller (W1b: a per-step tolerance below turb_tol could otherwise never be met; with turb_step_tol = Inf, the default,
this is W1's 0.1 turb_tol)."""
_kw_goal_floor(cfg) = 0.1*(cfg.turb_coupling === :step ? min(cfg.turb_tol, cfg.turb_step_tol) : cfg.turb_tol)

"""Line Gauss-Seidel on one system from x (the families loaded here), to the solve's stopping rule: (sweeps, converged, final,
retried). A solve that misses its goal within turb_sweeps_max sweeps (and stays finite) is continued from where it stopped for
3 turb_sweeps_max sweeps more, once (W1b, review N3: before, the miss rejected the whole step, an accepted flow iterate
included); `retried` says so. Below the first budget the arithmetic is W1's, bit for bit."""
function _kw_line_solve!(x, sys::FaceSystem, fj, fi, m::PBMesh, T, xref, cfg)
    be = cfg.backend
    line_tridiag!(fj, sys, m; backend=be); line_tridiag!(fi, sys, m; backend=be)
    r0 = _relres(sys, x, m, T.res, xref; backend=be)
    isfinite(r0) || return (0, false, r0, false)
    goal = max(cfg.turb_lin_rtol*r0, _kw_goal_floor(cfg))
    r0 <= goal && return (0, true, r0, false)
    rr = r0
    # W7 (opt-in): the line solves by cfg.line_solver, their corrections in Float32 under kw_precision = :f32 (the residual,
    # the iterate and the stopping test stay Float64); the defaults take the sweep below, bit for bit
    w7 = cfg.kw_precision === :f32 || cfg.line_solver !== :thomas
    if w7
        if cfg.kw_precision === :f32
            F = _kw_fast!(T, fj, fi, be)
            sj = F.fam[objectid(fj)]; si = F.fam[objectid(fi)]
            line_copy!(sj, fj; backend=be); line_copy!(si, fi; backend=be); zw = F.zw
        else
            sj = fj; si = fi; zw = T.zw
        end
    end
    for s in 1:4*cfg.turb_sweeps_max
        if w7                              # (the PCR eliminates in Float64: on fj, fi themselves, or with them as fam64)
            line_gs_sweep!(x, sys, m, sj, T.rw, zw, cfg.line_solver; backend=be, fam64=fj)
            line_gs_sweep!(x, sys, m, si, T.rw, zw, cfg.line_solver; backend=be, fam64=fi)
        else
            line_gs_sweep!(x, sys, m, fj, T.rw, T.zw; backend=be)
            line_gs_sweep!(x, sys, m, fi, T.rw, T.zw; backend=be)
        end
        rr = _relres(sys, x, m, T.res, xref; backend=be)
        isfinite(rr) || return (s, false, rr, s > cfg.turb_sweeps_max)
        rr <= goal && return (s, true, rr, s > cfg.turb_sweeps_max)
    end
    (4*cfg.turb_sweeps_max, false, rr, true)
end

"The Float32 copies of TurbWork's four line families and a Float32 correction vector (W7, kw_precision = :f32), made once."
function _kw_fast!(T, fj, fi, be)
    if T.w7 === nothing
        fam = Dict{UInt,Any}()
        for f in (T.fjk, T.fik, T.fjw, T.fiw); fam[objectid(f)] = line_family_like(f, Float32; backend=be); end
        T.w7 = (fam=fam, zw=KernelAbstractions.zeros(be, Float32, length(T.zw)))
    end
    T.w7
end

"""
    komega_solve!(w, mesh, st, bc, cfg; assemble=true) -> NamedTuple

One implicit k-omega update at the iterate: assemble_komega! (after turb_update!; `assemble = false` when the systems were
just assembled at this iterate, e.g. by komega_residual), the line solves of k and of omega from the iterate (each continued
once past its first budget when it misses its goal, counted in w.turb.retries; `_kw_line_solve!`), the floors (counted into
w.turb: cells and the added rho_l x V, per step), and rk = rho_l k, rw = rho_l omega written to level 1. Returns `converged`,
`finite`, the sweeps and final relative residuals, this solve's floored cells and `retried` (k, omega).
"""
function komega_solve!(w, m::PBMesh, st::PBState, bc, cfg; assemble::Bool=true, relax::Real=1.0)
    0 < relax <= 1 || throw(ArgumentError("komega_solve!: relaxation must lie in (0,1]"))
    be = cfg.backend; T = w.turb; rho = cfg.rho_l
    assemble && assemble_komega!(w, m, st, bc, cfg)
    sk, ck, rk, qk = _kw_line_solve!(T.k, T.ksys, T.fjk, T.fik, m, T, T.kin, cfg)
    sw, cw, rw, qw = _kw_line_solve!(T.w, T.wsys, T.fjw, T.fiw, m, T, T.win, cfg)
    push!(T.sweeps, sk, sw); T.retries += qk + qw
    fin = isfinite(pb_maxabs(T.k; backend=be)) && isfinite(pb_maxabs(T.w; backend=be))
    nk = 0; nw = 0
    if fin
        if relax < 1
            _launch!(_k_kw_relax!,be,m.nc,T.k,T.w,lev(st.rk,1),lev(st.rw,1),rho,Float64(relax))
            pb_sync(be)
        end
        for (x, xmin, isk) in ((T.k, T.kmin, true), (T.w, T.wmin, false))
            _launch!(_k_floor!, be, m.nc, x, T.fl, xmin, rho, m.vol)
            add = pb_sum(T.fl; backend=be)
            _launch!(_k_count_pos!, be, m.nc, T.res, T.fl)
            n = round(Int, pb_sum(T.res; backend=be))
            if isk
                nk = n; T.nfloor_k += n; T.added_k += add
            else
                nw = n; T.nfloor_w += n; T.added_w += add
            end
        end
        _launch!(_k_kw_store!, be, m.nc, lev(st.rk, 1), lev(st.rw, 1), T.k, T.w, rho)
        pb_sync(be)
    end
    (converged=ck && cw, finite=fin, sweeps=(sk, sw), res=(rk, rw), floored=(nk, nw), retried=(qk, qw))
end

@kernel function _k_kw_relax!(k,w,rk,rw,rho::Float64,alpha::Float64)
    i=@index(Global)
    @inbounds begin
        k[i]=(1-alpha)*rk[i]/rho+alpha*k[i]
        w[i]=(1-alpha)*rw[i]/rho+alpha*w[i]
    end
end

"Reset the step's turbulence counters (floors, sweeps, retries); pimple_step! calls it at the start of every attempt."
function turb_step_reset!(T::TurbWork)
    T.nfloor_k = 0; T.nfloor_w = 0; T.added_k = 0.0; T.added_w = 0.0; empty!(T.sweeps); T.retries = 0
    T
end

"""
    turbulence_steady_residual(st, mesh, bc, w, cfg) -> (k, omega)

The k and omega relative residuals of the state's level 1 with the time term removed (the steady residual of the turbulence
equations, as steady_residual is the momentum's): turb_update! and assemble_komega!(time_term = false) there. NORMALISED BY
THE UNSPLIT DIAGONAL since W6 (the review of the integration, F1; `_relres_unsplit`): the split diagonal hid a cell held at the
floor by W1b's positivity split (omega's steady residual read 2.9e-8 against the rule 1e-7 where the unsplit normalisation
reads 180, at the sharp trailing edge of the NACA 0012); before W6 this divided by the split diagonal (the re-review of W1b's
R3 measured the two within 1.01 on N60R and 1.74 in its cavity). The per-step residuals (turb_res, R_turb) and the line solves
keep the split diagonal, so no march's arithmetic changes; only the steady stop's reading does.
"""
function turbulence_steady_residual(st::PBState, m::PBMesh, bc, w, cfg)
    turb_update!(w, m, st, bc, cfg) || return (NaN, NaN)
    komega_residual(w, m, st, bc, cfg; time_term=false, unsplit=true)
end

# ---- the wall diagnostics (verification) -------------------------------------------------------------------------------
"""
    wall_shear(st, mesh, bc, cfg; L=2, form=:assembled, work=nothing) -> NamedTuple

On every no-slip face (host): the face centroid (x, y), the skin friction C_f = 2 tau_w/(rho_l U_ref^2), d1 = S_b.d_b/|S_b|,
the first-cell y+ = d1 sqrt(|tau_w|/rho)/nu and the face indices, sorted by x; level L of the state (2, the committed state,
by default).
  form = :assembled (THE DEFAULT since W1b, review N7): tau_w is the tangential part of the wall traction -F_b/|S_b|, F_b the
    viscous force the momentum's own boundary kernel applies through the face (`_k_mom_bface!`, or `_k_mom_bface_t!` under
    SST: the two-point part mu |S_b|^2/(S_b.d_b) (u_b - u_P), the non-orthogonal part mu bk.(grad u)_P, limited as the
    momentum's at cfg.nonorth, and under SST the transpose part mu (grad u)_P^T . S_b; mu_eff = mu at a no-slip wall), at the
    level-L velocity and its least-squares gradient (the solver's stencil and boundary values; `work`, a PBWork of this
    mesh and condition set, lends its least-squares geometry, else one is built); signed by the x component of the
    tangential traction. On a converged state it is the force the momentum applies, also on skewed wall cells.
  form = :two_point: W1's instrument, bit for bit (tau_w = mu u_t/d1, u_t the owner's velocity tangential to the face relative
    to the wall value, signed by the x component of that slip): the two-point part alone, which omits the non-orthogonal
    and the transpose parts (exact on wall-orthogonal cells with a negligible transpose part; W1's registered reading).
"""
function wall_shear(st::PBState, m::PBMesh, bc, cfg; L::Integer=2, form::Symbol=:assembled, work=nothing)
    form in (:assembled, :two_point) || throw(ArgumentError("wall_shear: form must be :assembled or :two_point (got :$form)"))
    bk = _host(bc.kind); q = findall(is_wall_bc, bk)
    own = _host(m.bown); u = _host(lev(st.ux, L)); v = _host(lev(st.uy, L)); ub = _host(bc.ub); vb = _host(bc.vb)
    Sx = _host(m.bSx); Sy = _host(m.bSy); Sm = _host(m.bSmag); dx = _host(m.bdx); dy = _host(m.bdy)
    xf = _host(m.bxf); yf = _host(m.byf); nu = cfg.mu/cfg.rho_l
    if form === :assembled
        bfx, bfy, bdg = _wall_flux_assembled(st, m, bc, cfg, L, work)
    end
    x = Float64[]; y = Float64[]; cf = Float64[]; d1 = Float64[]; yp = Float64[]
    for b in q
        P = own[b]; nx = Sx[b]/Sm[b]; ny = Sy[b]/Sm[b]
        d = (Sx[b]*dx[b] + Sy[b]*dy[b])/Sm[b]
        if form === :two_point
            du = u[P] - ub[b]; dv = v[P] - vb[b]; un = du*nx + dv*ny
            tx = du - un*nx; ty = dv - un*ny
            tau = cfg.mu*sign(tx)*hypot(tx, ty)/d          # signed by the x component of the tangential slip
        else
            # the force on the owner through the face, F = rhs - diag u_P; the wall's traction is -F/|S_b|
            Fx = bfx[b] - bdg[b]*u[P]; Fy = bfy[b] - bdg[b]*v[P]
            ax = -Fx/Sm[b]; ay = -Fy/Sm[b]; an = ax*nx + ay*ny
            tx = ax - an*nx; ty = ay - an*ny
            tau = sign(tx)*hypot(tx, ty)                   # signed by the x component of the tangential traction
        end
        push!(x, xf[b]); push!(y, yf[b]); push!(cf, 2tau/(cfg.rho_l*cfg.U_ref^2)); push!(d1, d)
        push!(yp, d*sqrt(abs(tau)/cfg.rho_l)/nu)
    end
    p = sortperm(x)
    (x=x[p], y=y[p], cf=cf[p], d1=d1[p], yplus=yp[p], faces=q[p])
end

"""The momentum's boundary-face terms (right sides bfx, bfy and diagonal bdg; host vectors) at level L's velocity: the
solver's boundary velocities (`_k_bvel!`) and least-squares velocity gradients (`_k_lsq_grad!` with the velocity weights, as
`velocity_gradients!`), then the boundary kernel the momentum uses (`_k_mom_bface!`, or `_k_mom_bface_t!` under SST with
mu_eff = mu on the faces: the wall's value), at the configuration's convection code and nonorth. For `wall_shear`."""
function _wall_flux_assembled(st::PBState, m::PBMesh, bc, cfg, L::Integer, work)
    be = cfg.backend; nc, nb = m.nc, m.nb
    G = work === nothing ? lsq_geom(m, bc; backend=be) : work.lsq
    z(n) = KernelAbstractions.zeros(be, Float64, n)
    u = lev(st.ux, L); v = lev(st.uy, L); Fb = lev(st.Fb, L)
    ubf, vbf = z(max(nb, 1)), z(max(nb, 1)); gux, guy, gvx, gvy = z(nc), z(nc), z(nc), z(nc)
    bfx, bfy, bdg = z(max(nb, 1)), z(max(nb, 1)), z(max(nb, 1))
    _launch!(_k_bvel!, be, nb, ubf, vbf, u, v, bc.kind, bc.ub, bc.vb, m.bown, m.bSx, m.bSy, m.bSmag, Fb)
    for (g1, g2, phi, phib) in ((gux, guy, u, ubf), (gvx, gvy, v, vbf))
        _launch!(_k_lsq_grad!, be, nc, g1, g2, phi, phib, G.u11, G.u12, G.u22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(0))
    end
    # the momentum's own boundary kernel (momentum_bface_terms!, shared with the free-air loads since the W1+W2 integration of
    # 22 Oct 2026; the same launch as before, bit for bit)
    mub = cfg.turbulence === :sst ? fill!(z(max(nb, 1)), cfg.mu) : nothing
    momentum_bface_terms!(bdg, bfx, bfy, m, bc, cfg, u, v, Fb, ubf, vbf, gux, guy, gvx, gvy; mub)
    _host(bfx), _host(bfy), _host(bdg)
end
