# =====================================================================================================================
# parity.jl -- the CPU-against-device parity of plan 5.5 at its three levels, and the device's bitwise properties (stage P3,
# work item W3 of plan Amendment 12, section 32; 21 Oct 2026). HOST DIAGNOSTICS, NEVER IN A STEP: every function here runs
# the solver's own functions on the reference backend (CPU()) and on a device backend from identical inputs and measures the
# difference. The rules (metrics, tolerances, readings) are registered before any CUDA datum: W3's in records
# air/w3_gpu/W3_REGISTRATION.md (PARITY_RULES; metric :w3, its registered reading of 22 Oct stands as read), and THE RULE FOR
# FUTURE GATES in records air/gpuint/PARITY_RULE_V2_REGISTRATION.md (PARITY_RULES_V2; metric :pl1, the default since the
# merge of W3 with the W1+W2 integration, 22 Oct 2026). The instrument is self-tested on the CPU (tools/test_pbcav.jl,
# testsets W3 and GPUINT) and on JLArrays' reference GPU array (records air/w3_gpu/device_discipline_20261021.jl,
# air/gpuint/jl_device_20261022.jl), where every level must agree BIT FOR BIT, with negative examples that must fail.
#
#   LEVEL 0, bitwise on each backend (`parity_bitwise`): a step repeated from the same state with a fresh work space gives the
#            same bits; a rejected step restores level 1 from level 2 bit for bit and leaves levels 2 and 3 untouched; a host
#            round trip of a state is bitwise, and a march from the round-tripped state equals one from the original.
#   LEVEL 1, kernels on identical inputs (`parity_kernels`): the work space and state of a CPU outer iteration (after a
#            coupled freeze at a perturbed iterate; under SST after turb_update! and the k-omega assembly) are loaded on both
#            backends, and each operation -- K1-K14 (W3: the face operator, the least-squares and Green-Gauss gradients, the
#            momentum assembly, H and the momentum interpolation, the pressure equation, the mass transport, the line
#            tridiagonals, Thomas and a coloured line sweep, a V-cycle, a numeric refresh, the reductions, the coupled freeze,
#            J and the SIMPLEC preconditioner) and, since the merge (review of W3, F7), K15-K17 under SST (turb_update!: the
#            SST terms and mu_t; the k-omega assembly; the k-omega residuals and a fixed-count line solve) and K18-K19 in free
#            air (the far-field values with a point vortex; the loads and the boundary fluxes) -- runs on each from that same
#            snapshot; every output is compared under the metric's rule.
#   LEVEL 2, parity mode (`parity_mode`): fixed counts (`parity_step!`: mode_solves coupled solves of exactly mode_iters FGMRES
#            iterations per step, the hierarchy refreshed at every outer iteration, under SST a k-omega update of exactly
#            mode_turb_sweeps line sweeps; both backends from one host-built hierarchy), mode_steps steps from one state, in
#            free air optionally the loads and the point vortex between steps on each backend: every committed field (rk, rw
#            and mut included since the merge) and the loads agree within the metric's band.
#   LEVEL 3, production mode (`parity_production`): along a CPU production march, from each committed CPU state, one device
#            production step (pimple_step!, its own hierarchy and policy); the device's committed iterate (every field since
#            the merge), re-evaluated by the CPU, must meet production x the stopping tolerances that end the step (under SST
#            also the k-omega solve's, and the committed mu_t must be the CPU's evaluation of mu_t at that iterate).
# =====================================================================================================================

"""The registered parity numbers of W3 (records air/w3_gpu/W3_REGISTRATION.md; plan 5.5 for the three levels): `kernel_pass`
16 Float64 ulps (plan 5.5's kernel level) and `kernel_near` the labelled NEAR band of a :value output; `resid` the bound on a
residual-type output's difference in units of the tolerance it is tested against; `mode` the parity mode's relative bound
(plan 5.5: ten steps to 1e-10) with its fixed counts `mode_solves` x `mode_iters` over `mode_steps` steps; `production` the
multiple of the stopping tolerances (plan 5.5: ten times). Metric :w3 (W3's registered reading of 22 Oct 2026)."""
const PARITY_RULES = (kernel_pass=16*eps(Float64), kernel_near=1e-12, resid=1e-6, mode=1e-10, mode_solves=6, mode_iters=8,
                      mode_steps=10, production=10.0)

"""THE PARITY RULE FOR FUTURE GATES (metric :pl1; registered in records air/gpuint/PARITY_RULE_V2_REGISTRATION.md before any
CUDA datum it scores; W3's proposal air/w3_gpu/W3_L1_METRIC_PROPOSAL.md with the conditions of the review of W3, F5/F8):
LEVEL 1, P-L1: (R1) a single-pass value output reldiff <= `kernel_pass` PASS, <= `kernel_near` NEAR; a reduction |a - b| /
sum|terms| likewise; (R2) a residual output per cell |a_c - b_c| <= `resid_eps` eps S_c (S_c the sum of the absolute values
of the terms it is computed from; a maximum: the largest S_c); (R3) a multi-stage value output reldiff <= max(`kernel_pass`,
`sigma_mult` sigma_cpu), sigma_cpu the CPU's own reldiff when every input of the operation is perturbed by a random +-1 ulp,
the largest over `sigma_draws` draws (seeds `sigma_seed` + 1, 2, ...); (R4) the device twice bit for bit (reproducibility),
and in a second device process with NVPTX FMA contraction off every output that depends on no transcendental function is the
CPU's bit for bit (the list: `parity_transcendental`), those that do meet R1-R3. LEVEL 2: every committed field and load
reldiff <= max(`mode`, `mode_sigma_mult` sigma2_cpu), sigma2_cpu the CPU's own parity-mode result from the start state perturbed
by +-1 ulp (seed `sigma_seed`); fixed counts `mode_solves` x `mode_iters`, `mode_turb_sweeps` k-omega sweeps, `mode_steps`
steps. LEVEL 3: R_mom, R_Theta <= `production` x the tolerances; under SST R_turb <= `production` x max(1, the CPU's own),
the k-omega solve's residual against its goal (:step), or <= `production` x turb_tol (:outer); the committed mu_t within
max(`production_mut_floor`, `production_mut_sigma_mult` sigma_mut) of the CPU's evaluation at the device's iterate.
`kw_sweeps`: level 1's fixed k-omega line sweeps (K17)."""
const PARITY_RULES_V2 = (kernel_pass=16*eps(Float64), kernel_near=1e-12, resid_eps=8.0, sigma_mult=10.0, sigma_draws=3,
                         sigma_seed=20261022, mode=1e-10, mode_sigma_mult=10.0, mode_solves=6, mode_iters=8,
                         mode_turb_sweeps=4, mode_steps=10, production=10.0, production_mut_floor=1e-12,
                         production_mut_sigma_mult=10.0, kw_sweeps=4)

"""
    reldiff(a, b) -> Float64

max_i |a_i - b_i| / max_i |b_i| (b the reference; arrays on any backend, or scalars): 0 when they are equal bit for bit
(NaNs equal to NaNs of the same position), Inf when a difference is not finite or the reference is all zero and a differs.
"""
function reldiff(a, b)
    A = a isa AbstractArray ? Array(a) : [a]; B = b isa AbstractArray ? Array(b) : [b]
    size(A) == size(B) || throw(DimensionMismatch("reldiff: sizes $(size(A)) and $(size(B))"))
    d = 0.0; s = 0.0
    for i in eachindex(A, B)
        x = Float64(A[i]); y = Float64(B[i])
        isfinite(y) && (s = max(s, abs(y)))
        isequal(x, y) && continue
        (isfinite(x) && isfinite(y)) || return Inf
        d = max(d, abs(x - y))
    end
    d == 0 ? 0.0 : s > 0 ? d/s : Inf
end
"max_i w_i |a_i - b_i| (a weighted absolute difference: a residual's difference in its tolerance's units)."
function wdiff(a, b, wt)
    A = a isa AbstractArray ? Array(a) : [a]; B = b isa AbstractArray ? Array(b) : [b]; W = wt isa AbstractArray ? Array(wt) : fill(Float64(wt), length(A))
    d = 0.0
    for i in eachindex(A, B)
        isequal(A[i], B[i]) && continue
        (isfinite(A[i]) && isfinite(B[i])) || return Inf
        d = max(d, abs(W[i])*abs(A[i] - B[i]))
    end
    d
end
"""max_i |a_i - b_i| / (eps S_i): a residual's difference in units of its own rounding scale (P-L1 R2); S a vector of the
entries' scales, or one number (a maximum's: the largest S_c). Equal entries count 0; a difference where S_i = 0, or a
non-finite one, is Inf."""
function epsdiff(a, b, S)
    A = a isa AbstractArray ? Array(a) : [a]; B = b isa AbstractArray ? Array(b) : [b]
    Sv = S isa AbstractArray ? Array(S) : fill(Float64(S), length(A))
    length(A) == length(B) == length(Sv) || throw(DimensionMismatch("epsdiff: lengths $(length(A)), $(length(B)), $(length(Sv))"))
    q = 0.0
    for i in eachindex(A, B, Sv)
        isequal(A[i], B[i]) && continue
        (isfinite(A[i]) && isfinite(B[i]) && Sv[i] > 0) || return Inf
        q = max(q, abs(A[i] - B[i])/(eps(Float64)*Sv[i]))
    end
    q
end

"Load every array (and every scalar of a mutable struct) of `src` into `dst`, recursively, across backends; the multigrid
hierarchy (`mg`) is skipped (a parity check builds it from the loaded operator). A host vector of numbers whose length differs
(a counter list, e.g. the k-omega sweeps of a TurbWork) is resized first."
function _sync!(dst, src)
    if dst isa AbstractArray
        if eltype(dst) <: Number
            dst isa Vector && length(dst) != length(src) && resize!(dst, length(src))
            copyto!(dst, src)
        else
            for i in eachindex(dst, src); _sync!(dst[i], src[i]); end
        end
        return dst
    end
    for f in fieldnames(typeof(dst))
        # (W7 review, finding 5: the W7 stores, CoupledWork.w7 and TurbWork.w7, hold backend arrays made and loaded by the
        # solver itself, cpl_freeze! and _kw_fast!; copying their references across would alias the snapshot's arrays)
        (f === :mg || f === :w7) && continue
        d = getfield(dst, f); s = getfield(src, f)
        if d isa AbstractArray || (isstructtype(typeof(d)) && !(d isa Number) && !(d isa Symbol) && !(d isa AbstractString) &&
                                   !(d isa KernelAbstractions.Backend) && !(d isa Dict) && !(d isa Tuple) && !(d isa NamedTuple))
            (s === nothing) && continue
            _sync!(d, s)
        elseif ismutable(dst) && !(d isa KernelAbstractions.Backend)
            setfield!(dst, f, s)
        end
    end
    dst
end

"""The +-1-ulp signs of the CPU sensitivity passes: splitmix64 from a fixed seed (no dependency beyond Base; the same signs
on every machine and Julia version)."""
mutable struct _UlpRNG
    s::UInt64
end
_UlpRNG(seed::Integer) = _UlpRNG(UInt64(seed))
function _nextsign!(r::_UlpRNG)
    r.s += 0x9e3779b97f4a7c15
    z = r.s; z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9; z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb; z = z ⊻ (z >> 31)
    isodd(z) ? 1.0 : -1.0
end

"""Every non-zero Float64 array entry of `x` (a host work space or state, recursively, as `_sync!` walks it; the hierarchy skipped)
moved by a random +-1 ulp (`rng`), each array once (`seen`; the momentum systems share their matrix). A face system whose
lower and upper parts are equal (the symmetric pressure operator) stays symmetric. P-L1 R3's and level 2's CPU sensitivity.
`unit = Float32` (W7 Amendment 2, the class R3f): each entry moved by +-1 ulp OF FLOAT32 at its value (x +- eps(Float32(x))),
the Float32 forms' working precision, instead of its own Float64 ulp."""
function _perturb_ulp!(x, rng, seen=IdDict{Any,Bool}(); unit::Type=Float64)
    if x isa AbstractArray
        haskey(seen, x) && return x
        seen[x] = true
        if eltype(x) === Float64
            for i in eachindex(x)                    # a zero stays zero (eps(0.0) is a subnormal, not an input's ulp)
                xi = x[i]; xi == 0 || (x[i] = xi + _nextsign!(rng)*(unit === Float32 ? Float64(eps(Float32(xi))) : eps(xi)))
            end
        elseif !(eltype(x) <: Number)
            for e in x; _perturb_ulp!(e, rng, seen; unit); end
        end
        return x
    end
    if x isa FaceSystem && !haskey(seen, x.upper) && x.lower !== x.upper && isequal(x.lower, x.upper)
        _perturb_ulp!(x.upper, rng, seen; unit); copyto!(x.lower, x.upper); seen[x.lower] = true
    end
    for f in fieldnames(typeof(x))
        f === :mg && continue
        d = getfield(x, f)
        if d isa AbstractArray || (isstructtype(typeof(d)) && !(d isa Number) && !(d isa Symbol) && !(d isa AbstractString) &&
                                   !(d isa KernelAbstractions.Backend) && !(d isa Dict) && !(d isa Tuple) && !(d isa NamedTuple) &&
                                   !(d isa Nothing))
            _perturb_ulp!(d, rng, seen; unit)
        end
    end
    x
end

"A host test vector (deterministic, smooth plus oscillating, no zero entries in practice)."
_ptestvec(n, k) = [sin(0.7i + k) + 0.3cos(1.3i*k + 0.1) + 0.05k for i in 1:n]

"""The class of a multi-stage output whose operation runs a form at the given precision setting (W7 Amendment 2): :R3f when
it is :f32 (the Float32 forms: K9 and K14 under prec_precision = :f32, K17's sweeps under kw_precision = :f32), else :R3."""
_r3cls(prec::Symbol) = prec === :f32 ? :R3f : :R3

"The hierarchy a parity check uses: a host build (with its numeric plan) of the work's current pressure operator, on the
work's backend; with identical operators on two backends the two hierarchies are identical (the build runs on the host).
With W7's speed settings its companion is made as the solver makes it (mg_update!: prec_precision, line_solver; the W7
review's finding 5: before, the parity checks tested the Float64 Thomas cycle whatever cfg said)."
function _parity_mg!(w, m, cfg)
    w.mg = build_schaffer_mg(w.psys, m; backend=cfg.backend, plan=true)
    mg_fast_update!(w.mg, cfg.prec_precision === :f32 ? Float32 : Float64, cfg.line_solver)
    copyto!(w.mgs.dref, w.psys.diag)
    w.mg
end

"""The parity configuration of a level-1 freeze and of the parity mode: `cfg` with the hierarchy refreshed at every outer
iteration (mg_diag_delta 0, mg_every_outer 1), so the refresh decision cannot differ between backends."""
_parity_cfg(cfg; kw...) = PBConfig(; (k => getfield(cfg, k) for k in fieldnames(PBConfig))..., mg_update=:refresh,
                                   mg_diag_delta=0.0, mg_every_steps=0, mg_every_outer=1, kw...)

"A copy of a condition set on its own backend (an operation that sets far-field values must not change the caller's)."
_bccopy(bc::BCSet) = BCSet(copy(bc.kind), copy(bc.ub), copy(bc.vb), copy(bc.pb), bc.npdir)

"""
    parity_prepare(m, bc, st, cfg) -> (w, st, snapw, snaps)

The CPU snapshot level 1 compares from: the committed state `st` (host) with level 1 perturbed deterministically (u, v, p
by 1e-3 relative smooth fields: an iterate distinct from level n; under SST also rk and rw, multiplicatively, so k and omega
stay positive), the SST terms at that iterate (turb_update!, as coupled_outer! takes them first), its momentum system
assembled, one coupled freeze (first outer iteration) and, under SST, the k-omega systems assembled, all on the CPU, so every
work array holds the values of a real outer iteration; returned with deep copies of the work and the state.
"""
function parity_prepare(m::PBMesh, bc, st0::PBState, cfg::PBConfig)
    cfg.backend isa CPU || throw(ArgumentError("parity_prepare runs on the CPU reference"))
    c = _parity_cfg(cfg); st = deepcopy(st0); restore_level!(st)
    for (fld, k) in ((:ux, 1), (:uy, 2), (:p, 3))
        A = lev(getfield(st, fld), 1); h = Array(A); s = maximum(abs, h; init=0.0)
        copyto!(A, h .+ 1e-3*max(s, 1e-30) .* _ptestvec(length(h), k))
    end
    sst = c.turbulence === :sst
    if sst
        for (fld, k) in ((:rk, 8), (:rw, 9))
            A = lev(getfield(st, fld), 1); h = Array(A); copyto!(A, h .* (1 .+ 1e-3 .* _ptestvec(length(h), k)))
        end
    end
    w = pbwork(m, bc, c); w.cpl = CoupledWork(m, w, c)
    t_new = st.t + st.dt
    _launch!(_k_theta!, CPU(), m.nc, w.theta, lev(st.rho, 2), c.rho_l)
    sst && (turb_update!(w, m, st, bc, c) || error("parity_prepare: turb_update! refused the snapshot's iterate"))
    assemble_momentum!(w, m, st, bc, c, t_new)
    cpl_freeze!(w.cpl, w, m, st, bc, c, t_new, true)
    sst && assemble_komega!(w, m, st, bc, c)
    (w, st, deepcopy(w), deepcopy(st))
end

# ---- the rounding scales of P-L1 R2 (host; from the CPU's arrays after the operation) -----------------------------------
"""The momentum residual's term scales per cell (raw units, x and y): the absolute values of every product the residual
(`_k_mom_resid!`, `_k_cpl_mom_res!`) and the arrays it reads (`_k_mom_cell!`, the face kernels `_k_mom_face!`/`_k_mom_face_t!`,
`_k_mom_bface!`/`_k_mom_bface_t!`, `_k_gg_grad!`) are computed from (PARITY_RULE_V2_REGISTRATION.md, section 3):
|b^s| (the MMS source with it), the diagonal's terms (|a^t|, every face's |coefficient| and every boundary |bdg|) times
|u_P|, the time term |a^t| |u^n|, every off-diagonal |coefficient u_N|, per internal face the deferred terms' parts -- the
convection correction |phi| (|g_x r_x| + |g_y r_y|), the non-orthogonal part mu_eff (|k_x| |g_x,f| + |k_y| |g_y,f|) and under
SST the transpose part mu_eff (|g_xu,f| |S_x| + |g_xv,f| |S_y|), the face gradients as the magnitudes of their two
interpolation terms -- per boundary face the same parts of its kernel, and the Green-Gauss pressure force (per face
max(|p_P|, |p_N|) |S| plus the skewness term's |grad p| |skew| |S|; boundary |p_b| |S_b|). mu_eff = cfg.mu laminar, w.muf /
w.mub under SST (W3 defined the laminar list, the probe's mom_scale; the review of W3, F8 (2), asked for the SST rows')."""
function _mom_terms(m::PBMesh, w, st::PBState, bc, cfg)
    H(x) = Array(x)
    cf = H(m.cface); own = H(m.own); nei = H(m.nei); nf = m.nf; V = H(m.vol); Sx = H(m.Sx); Sy = H(m.Sy); fw = H(m.fw)
    fa = H(m.fa); kx = H(m.kx); ky = H(m.ky); xf = H(m.xf); yf = H(m.yf); xc = H(m.xc); yc = H(m.yc); ftx = H(m.ftx); fty = H(m.fty)
    skx = H(m.skx); sky = H(m.sky); bSx = H(m.bSx); bSy = H(m.bSy); ba = H(m.ba); bkx = H(m.bkx); bky = H(m.bky); bown = H(m.bown)
    bkind = H(bc.kind)
    u = H(lev(st.ux, 1)); v = H(lev(st.uy, 1)); un = H(lev(st.ux, 2)); vn = H(lev(st.uy, 2)); p = H(lev(st.p, 1))
    F = H(lev(st.F, 1)); Fb = H(lev(st.Fb, 1))
    bsx = H(w.bsx); bsy = H(w.bsy); at = H(w.at); up = H(w.msx.upper); lo = H(w.msx.lower); bdg = H(w.bdg); pbf = H(w.pbf)
    gux = H(w.gux); guy = H(w.guy); gvx = H(w.gvx); gvy = H(w.gvy); gpx = H(w.gpx); gpy = H(w.gpy)
    cux = H(w.cux); cuy = H(w.cuy); cvx = H(w.cvx); cvy = H(w.cvy); ubf = H(w.ubf); vbf = H(w.vbf)
    sst = cfg.turbulence === :sst
    muf = sst ? H(w.muf) : fill(cfg.mu, nf); mub = sst ? H(w.mub) : fill(cfg.mu, max(m.nb, 1))
    lu = cfg.convection === :upwind ? 0 : 1
    if cfg.conv_gradient !== :unit
        cux, cuy, cvx, cvy = gux, guy, gvx, gvy
    end
    lamn = cfg.nonorth > 0
    rho = cfg.rho_l
    # per internal face: (x, y) magnitudes of the deferred terms (the same for the owner's and the neighbour's row)
    fdx = zeros(nf); fdy = zeros(nf)
    for f in 1:nf
        P = own[f]; N = nei[f]; phi = rho*F[f]; mu = muf[f]; a = fw[f]; a1 = 1 - a
        cu = 0.0; cv = 0.0
        if lu >= 1
            if phi >= 0
                rx = xf[f] - xc[P]; ry = yf[f] - yc[P]; Q = P
            else
                rx = xf[f] - (xc[N] + ftx[f]); ry = yf[f] - (yc[N] + fty[f]); Q = N
            end
            cu = abs(phi)*(abs(cux[Q]*rx) + abs(cuy[Q]*ry)); cv = abs(phi)*(abs(cvx[Q]*rx) + abs(cvy[Q]*ry))
        end
        ixu = abs(a*gux[P]) + abs(a1*gux[N]); iyu = abs(a*guy[P]) + abs(a1*guy[N])
        ixv = abs(a*gvx[P]) + abs(a1*gvx[N]); iyv = abs(a*gvy[P]) + abs(a1*gvy[N])
        nu_ = lamn ? mu*(abs(kx[f])*ixu + abs(ky[f])*iyu) : 0.0; nv_ = lamn ? mu*(abs(kx[f])*ixv + abs(ky[f])*iyv) : 0.0
        tx = sst ? mu*(ixu*abs(Sx[f]) + ixv*abs(Sy[f])) : 0.0; ty = sst ? mu*(iyu*abs(Sx[f]) + iyv*abs(Sy[f])) : 0.0
        fdx[f] = cu + nu_ + tx; fdy[f] = cv + nv_ + ty
    end
    tx = zeros(m.nc); ty = zeros(m.nc)
    for c in 1:m.nc
        sx = abs(bsx[c]) + abs(at[c])*(2abs(u[c]) + abs(un[c])); sy = abs(bsy[c]) + abs(at[c])*(2abs(v[c]) + abs(vn[c]))
        for k in 1:4
            code = cf[k, c]
            if code > nf
                b = code - nf; kd = bkind[b]; mu = mub[b]; G = mu*ba[b]; phi = rho*Fb[b]   # the owner is c
                sx += abs(bdg[b])*abs(u[c]); sy += abs(bdg[b])*abs(v[c])
                if is_velocity_bc(kd)
                    ain = max(-phi, 0.0)
                    sx += (G + ain)*abs(ubf[b]); sy += (G + ain)*abs(vbf[b])
                    if lamn
                        sx += mu*(abs(bkx[b]*gux[c]) + abs(bky[b]*guy[c])); sy += mu*(abs(bkx[b]*gvx[c]) + abs(bky[b]*gvy[c]))
                    end
                    if sst
                        sx += mu*(abs(gux[c]*bSx[b]) + abs(gvx[c]*bSy[b])); sy += mu*(abs(guy[c]*bSx[b]) + abs(gvy[c]*bSy[b]))
                    end
                    if lu >= 1 && phi > 0
                        sx += abs(phi)*(abs(ubf[b]) + abs(u[c])); sy += abs(phi)*(abs(vbf[b]) + abs(v[c]))
                    end
                elseif kd == UInt8(BC_SLIP)
                    sx += G*abs(ubf[b]); sy += G*abs(vbf[b])
                else
                    ain = max(-phi, 0.0); sx += ain*abs(u[c]); sy += ain*abs(v[c])
                end
                sx += abs(pbf[b]*bSx[b]); sy += abs(pbf[b]*bSy[b])
            else
                f = abs(code); co = code > 0 ? up[f] : lo[f]; Nb = code > 0 ? nei[f] : own[f]
                sx += abs(co)*(abs(u[c]) + abs(u[Nb])) + fdx[f]; sy += abs(co)*(abs(v[c]) + abs(v[Nb])) + fdy[f]
                pp = max(abs(p[own[f]]), abs(p[nei[f]])) + max(abs(gpx[own[f]]), abs(gpx[nei[f]]))*abs(skx[f]) +
                     max(abs(gpy[own[f]]), abs(gpy[nei[f]]))*abs(sky[f])
                sx += pp*abs(Sx[f]); sy += pp*abs(Sy[f])
            end
        end
        tx[c] = sx; ty[c] = sy
    end
    tx, ty, V
end
"Per cell: the scaled momentum residual's rounding scale, max(x, y) L/(rho U^2) / V (the units of w.res)."
function _mom_scale(m::PBMesh, w, st::PBState, bc, cfg)
    tx, ty, V = _mom_terms(m, w, st, bc, cfg); sc = cfg.L_ref/(cfg.rho_l*cfg.U_ref^2)
    max.(tx, ty) .* sc ./ V
end
"""Per cell: the continuity residual's (the pressure right side's) term scale: V |Theta^n|/dt + V |rho^n/rho_l|/dt + sum
|F*| over the faces (W3's cont_scale, probe_l1_20261022.jl)."""
function _cont_scale(m::PBMesh, w, st::PBState, cfg)
    cf = Array(m.cface); nf = m.nf; V = Array(m.vol); Fs = Array(w.Fs); Fbs = Array(w.Fbs); rhon = Array(lev(st.rho, 2))
    S = zeros(m.nc)
    for c in 1:m.nc
        s = V[c]*abs(rhon[c]/cfg.rho_l - 1)/st.dt + V[c]/st.dt*abs(rhon[c]/cfg.rho_l)
        for k in 1:4
            code = cf[k, c]; s += code > nf ? abs(Fbs[code-nf]) : abs(Fs[abs(code)])
        end
        S[c] = s
    end
    S
end
"""
    _rc_term_scale(m, w, st, bc, cfg; lam=1.0, pform=:v3) -> (Tf, Tb)

THE TERM SCALE OF THE RECOMPUTED RHIE-CHOW FLUX (parity amendment W6b-2, adopting the GPU merge's proposed A1,
../gpuint/PARITY_V2_READINGS.md section 3; records air/w6b_start2/PARITY_RULE_V2_AMENDMENT_W6B_2.md): per internal face T_f and
per boundary face T_b, the sum of the magnitudes of the terms `_k_rc_face!` and `_k_rc_bface!` form F* from,
F* = (H.S [+ the linear-weight and skewness terms] + a^t F^n + (1/lambda - 1) a F^(m-1) - V (|S|^2/(S.d)(p_N - p_P) + k.grad p))/A,
over |A|, with H's OWN terms in place of |H| (H_P = b^s_P - sum_N a_N u_N, `momentum_H!`: |b^s_P| + sum_N |a_N u_N|): the scale a
correct arithmetic's rounding of F* is relative to wherever the terms cancel (near a wall |F*| is the small cancelled result).
A velocity face's |u_b S_x| + |v_b S_y|; a slip face 0. T_f >= |F*_f|, so a scale built on it is never tighter than one on |F*|.
At the freeze's lam = 1 the Majumdar term vanishes. Host arithmetic on the work's arrays as the freeze left them.
PARITY RULE V3 (records air/w7_speed/PARITY_RULE_V3.md; the independent review of amendment W6b-2, C1): `pform = :v3` (alias
:ops, the review's "A1-ops") takes the pressure part as THE PRODUCT THE KERNEL FORMS, Vbar |fa_f (p_N - p_P)| on an internal
face and V_P |ba_b (p_b - p_P)| on a pressure outlet (p_N - p_P is one correctly rounded subtraction of inputs identical on both
sides), so the scale is invariant under a constant pressure datum; `pform = :a1` is W6b-2's fa (|p_N| + |p_P|), kept for the
record's readings (it is datum-dependent and lets the wake cut's dropped k.grad p through). Everything else is A1's.
THE DEFAULT IS :v3 (the merge review of cf838408, M9): the superseded, datum-dependent A1 scale is never a silent default;
a caller that wants it passes pform = :a1.
"""
function _rc_term_scale(m::PBMesh, w, st::PBState, bc, cfg; lam::Real=1.0, pform::Symbol=:v3)
    pform in (:a1, :v3, :ops) || throw(ArgumentError("_rc_term_scale: pform must be :v3 (:ops) or :a1 (got :$pform)"))
    pv3 = pform !== :a1
    H(x) = Array(x)
    cf = H(m.cface); own = H(m.own); nei = H(m.nei); nf = m.nf; nc = m.nc
    u = H(lev(st.ux, 1)); v = H(lev(st.uy, 1)); p = H(lev(st.p, 1)); Fn = H(lev(st.F, 2)); Fbn = H(lev(st.Fb, 2))
    bsx = H(w.bsx); bsy = H(w.bsy); up = H(w.msx.upper); lo = H(w.msx.lower); A = H(w.msx.diag); a = H(_arel(w, cfg)); at = H(w.at)
    as_ = H(w.as_); gpx = H(w.gpx); gpy = H(w.gpy); Fp = H(w.Fprev); Fbp = H(w.Fbprev)
    hxx = H(w.hxx); hxy = H(w.hxy); hyx = H(w.hyx); hyy = H(w.hyy)
    fw = H(m.fw); fa = H(m.fa); kx = H(m.kx); ky = H(m.ky); skx = H(m.skx); sky = H(m.sky); Sx = H(m.Sx); Sy = H(m.Sy); V = H(m.vol)
    bkind = H(bc.kind); ub = H(bc.ub); vb = H(bc.vb); pb = H(bc.pb); bown = H(m.bown); bSx = H(m.bSx); bSy = H(m.bSy); ba = H(m.ba)
    bkx = H(m.bkx); bky = H(m.bky)
    hx = abs.(bsx); hy = abs.(bsy)                            # H's own terms per cell
    for c in 1:nc, k in 1:4
        code = cf[k, c]
        if 0 < code <= nf
            hx[c] += abs(up[code]*u[nei[code]]); hy[c] += abs(up[code]*v[nei[code]])
        elseif code < 0
            hx[c] += abs(lo[-code]*u[own[-code]]); hy[c] += abs(lo[-code]*v[own[-code]])
        end
    end
    lamn = cfg.nonorth; klin = _rc_klin(cfg) != 0; krc = _rc_on(cfg) != 0; sk = cfg.skew; rel = abs((1 - lam)/lam)
    Tf = zeros(nf)
    for f in 1:nf
        P = own[f]; N = nei[f]; ww = fw[f]; w1 = 1 - ww; ax = abs(Sx[f]); ay = abs(Sy[f])
        if !krc
            Tf[f] = (ww*abs(u[P]) + w1*abs(u[N]))*ax + (ww*abs(v[P]) + w1*abs(v[N]))*ay
            continue
        end
        Ab = ww*A[P] + w1*A[N]; ab = ww*a[P] + w1*a[N]; atb = ww*at[P] + w1*at[N]; Vb = ww*V[P] + w1*V[N]
        t = (ww*hx[P] + w1*hx[N])*ax + (ww*hy[P] + w1*hy[N])*ay
        klin && (t += ww*w1*(abs(as_[P]) + abs(as_[N]))*((abs(u[N]) + abs(u[P]))*ax + (abs(v[N]) + abs(v[P]))*ay))
        if sk
            asb = ww*abs(as_[P]) + w1*abs(as_[N]); rx = abs(skx[f]); ry = abs(sky[f])
            t += asb*0.5*(((abs(hxx[P]) + abs(hxx[N]))*rx + (abs(hxy[P]) + abs(hxy[N]))*ry)*ax +
                          ((abs(hyx[P]) + abs(hyx[N]))*rx + (abs(hyy[P]) + abs(hyy[N]))*ry)*ay)
        end
        t += abs(atb*Fn[f]) + rel*abs(ab*Fp[f])
        np_ = lamn > 0 ? abs(kx[f])*(ww*abs(gpx[P]) + w1*abs(gpx[N])) + abs(ky[f])*(ww*abs(gpy[P]) + w1*abs(gpy[N])) : 0.0
        pp = pv3 ? abs(fa[f]*(p[N] - p[P])) : abs(fa[f])*(abs(p[N]) + abs(p[P]))   # v3: the product as formed
        Tf[f] = (t + abs(Vb)*(pp + np_))/abs(Ab)
    end
    Tb = zeros(max(m.nb, 1))
    for b in 1:m.nb
        k = bkind[b]; P = bown[b]
        if is_pressure_bc(k)
            if krc
                t = hx[P]*abs(bSx[b]) + hy[P]*abs(bSy[b]) + abs(at[P]*Fbn[b]) + rel*abs(a[P]*Fbp[b])
                np_ = lamn > 0 ? abs(bkx[b]*gpx[P]) + abs(bky[b]*gpy[P]) : 0.0
                ppb = pv3 ? abs(ba[b]*(pb[b] - p[P])) : abs(ba[b])*(abs(pb[b]) + abs(p[P]))
                Tb[b] = (t + abs(V[P])*(ppb + np_))/abs(A[P])
            else
                Tb[b] = abs(u[P]*bSx[b]) + abs(v[P]*bSy[b])
            end
        elseif is_velocity_bc(k)
            Tb[b] = abs(ub[b]*bSx[b]) + abs(vb[b]*bSy[b])
        end
    end
    Tf, Tb
end
"""Per cell: the continuity rows' term scale UNDER AMENDMENT W6b-2 (A1): `_cont_scale`'s with each face's |F*_f| replaced by
its term scale T_f (`_rc_term_scale`), for K12, whose freeze RECOMPUTES F* (K6, which reads the snapshot's F*, keeps
`_cont_scale`). Without Tf it forms T_f with `pform` (default :v3, the merge review's M9; :a1 for the record's readings)."""
function _cont_scale_a1(m::PBMesh, w, st::PBState, bc, cfg; Tf=nothing, Tb=nothing, pform::Symbol=:v3)
    Tf === nothing && ((Tf, Tb) = _rc_term_scale(m, w, st, bc, cfg; pform))
    cf = Array(m.cface); nf = m.nf; V = Array(m.vol); rhon = Array(lev(st.rho, 2))
    S = zeros(m.nc)
    for c in 1:m.nc
        s = V[c]*abs(rhon[c]/cfg.rho_l - 1)/st.dt + V[c]/st.dt*abs(rhon[c]/cfg.rho_l)
        for k in 1:4
            code = cf[k, c]; s += code > nf ? Tb[code-nf] : Tf[abs(code)]
        end
        S[c] = s
    end
    S
end
"""Per cell (x, y): the Newton correction's term list in K12's momentum rows UNDER AMENDMENT W6b-2 (A1): rho sum_f (T_f + |F_f|)
(|e_f| + the face increment's parts), F the convected flux (cw.Fconv), e the upwind face increment (`_k_cpl_conv_faces!`:
e = u_f - u_P with u_f = u_Q + grad u_Q . r; its parts |u_Q| + |du/dx_Q r_x| + |du/dy_Q r_y| + |u_P|), against the registered
rho sum_f (|F*_f| + |F_f|)|e_f|."""
function _newton_terms_a1(m::PBMesh, w, st::PBState, cfg, Tf)
    H(x) = Array(x); cw = w.cpl
    cf = H(m.cface); own = H(m.own); nei = H(m.nei); nf = m.nf
    Fc = H(cw.Fconv); eox = H(cw.eox); eoy = H(cw.eoy); enx = H(cw.enx); eny = H(cw.eny)
    u = H(lev(st.ux, 1)); v = H(lev(st.uy, 1)); xf = H(m.xf); yf = H(m.yf); xc = H(m.xc); yc = H(m.yc); ftx = H(m.ftx); fty = H(m.fty)
    lu = cfg.convection === :linear_upwind_unlimited
    gx, gy, hx, hy = (lu && cfg.conv_gradient === :unit) ? (H(w.cux), H(w.cuy), H(w.cvx), H(w.cvy)) : (H(w.gux), H(w.guy), H(w.gvx), H(w.gvy))
    px = zeros(nf); py = zeros(nf); qx = zeros(nf); qy = zeros(nf)     # the parts of (eo, en) per face, x and y
    for f in 1:nf
        P = own[f]; N = nei[f]
        Q, rx, ry = Fc[f] >= 0 ? (P, xf[f] - xc[P], yf[f] - yc[P]) : (N, xf[f] - (xc[N] + ftx[f]), yf[f] - (yc[N] + fty[f]))
        gu = lu ? abs(gx[Q]*rx) + abs(gy[Q]*ry) : 0.0; gv = lu ? abs(hx[Q]*rx) + abs(hy[Q]*ry) : 0.0
        px[f] = abs(u[Q]) + gu + abs(u[P]); py[f] = abs(v[Q]) + gv + abs(v[P])
        qx[f] = abs(u[Q]) + gu + abs(u[N]); qy[f] = abs(v[Q]) + gv + abs(v[N])
    end
    nx = zeros(m.nc); ny = zeros(m.nc)
    for c in 1:m.nc, q in 1:4
        code = cf[q, c]
        if 0 < code <= nf
            t = cfg.rho_l*(Tf[code] + abs(Fc[code])); nx[c] += t*(abs(eox[code]) + px[code]); ny[c] += t*(abs(eoy[code]) + py[code])
        elseif code < 0
            f = -code; t = cfg.rho_l*(Tf[f] + abs(Fc[f])); nx[c] += t*(abs(enx[f]) + qx[f]); ny[c] += t*(abs(eny[f]) + qy[f])
        end
    end
    nx, ny
end
"""Per cell: Theta = rho/rho_l - 1's term scale, |rho^n|/rho_l + (dt/V) sum_f |F_f| + 1 (the transport's flux sum included:
rho^{n+1} = rho^n - (dt/V) rho_l sum_f F_f)."""
function _theta_scale(m::PBMesh, st::PBState, cfg)
    cf = Array(m.cface); nf = m.nf; V = Array(m.vol); F = Array(lev(st.F, 1)); Fb = Array(lev(st.Fb, 1)); rhon = Array(lev(st.rho, 2))
    S = zeros(m.nc)
    for c in 1:m.nc
        s = 0.0
        for k in 1:4
            code = cf[k, c]; s += code > nf ? abs(Fb[code-nf]) : abs(F[abs(code)])
        end
        S[c] = abs(rhon[c])/cfg.rho_l + st.dt/V[c]*s + 1
    end
    S
end
"""Per cell: the k-omega relative residual's term scale, (|b| + |a_P x_P| + sum_N |a_N x_N|) / (a_P max(x_P, x_ref)), the
terms `_k_relres!` computes it from (the system as assembled; review of W3, F8 (2): the k-omega rows' list)."""
function _relres_scale(sys::FaceSystem, x, m::PBMesh, xref; shr=nothing)
    b = Array(sys.rhs); dg = Array(sys.diag); up = Array(sys.upper); lo = Array(sys.lower); X = Array(x)
    sh = shr === nothing ? zeros(m.nc) : Array(shr)               # W6: the unsplit normaliser's split shares (W6b)
    cf = Array(m.cface); own = Array(m.own); nei = Array(m.nei); nf = m.nf
    S = zeros(m.nc)
    for c in 1:m.nc
        s = abs(b[c]) + abs(dg[c]*X[c])
        for q in 1:4
            code = cf[q, c]
            if 0 < code <= nf
                s += abs(up[code]*X[nei[code]])
            elseif code < 0
                s += abs(lo[-code]*X[own[-code]])
            end
        end
        S[c] = s/((shr === nothing ? dg[c] : dg[c] - sh[c])*max(X[c], xref))
    end
    S
end

# ---- the level-1 operations -------------------------------------------------------------------------------------------
"""One compared output: `out` its label, `val` the array or number, `kind`/`scale` W3's classification (metric :w3),
`cls` P-L1's (:R1 a single-pass value, :R1s a reduction with `S` its sum of |terms|, :R2 a residual with `S` its rounding
scale per entry or for a maximum, :R3 a multi-stage value, :bit bitwise), `trans` whether it depends on a transcendental
function on this configuration (`parity_transcendental`; R4's FMA-off control)."""
_po(out, val, kind, scale; cls::Symbol=:R1, S=nothing, trans::Bool=false, sig::Real=0.0) = (out=out, val=val, kind=kind, scale=scale,
                                                                             cls=cls, S=S, trans=trans, sig=Float64(sig))

"""
    parity_transcendental(cfg; freeair=false) -> Vector of (operation prefix, output label)

THE OUTPUTS OF LEVEL 1 THAT DEPEND ON A TRANSCENDENTAL FUNCTION (P-L1 R4; the review of W3, F8 (1): enumerated from the code
before the data, registered in PARITY_RULE_V2_REGISTRATION.md section 4). In level 1 every operation starts from an identical
snapshot, so an output depends on one only when the operation itself evaluates it:
  * an MMS field with a momentum source (MMSTrig: sin, cos in `mms_source`, `_k_mom_cell!`; its mass source is identically 0):
    K4's b^s (bsx, bsy), the right sides msx.rhs, msy.rhs, the residual per cell and its maximum;
  * SST (`sst_terms`: tanh in F1 and F2, and F1's arg1^4, which CUDA.jl evaluates as libdevice pow(x, 4.0) while Julia's CPU
    `^(::Float64, ::Int)` is its compensated power): K15's mu_t, sigma_k, sigma_w, P_k, P_omega, beta, the cross diffusion,
    F1, F2, the cell flags and the face viscosities muf, mub (from mu_t).
Nothing else: sqrt and division are correctly rounded on both (IEEE), the far field's point vortex and the loads use only
+ - * / (cosd/sind of the incidence are host scalars), the wall distance's ^2 is a product, every mesh is the CPU's.
"""
function parity_transcendental(cfg::PBConfig; freeair::Bool=false)
    out = Tuple{String,String}[]
    if cfg.mms isa MMSTrig
        append!(out, [("K4", n) for n in ("bsx", "bsy", "msx.rhs", "msy.rhs", "res (per cell)", "momentum residual")])
    end
    if cfg.turbulence === :sst
        append!(out, [("K15", n) for n in ("mut", "sigk", "sigw", "Pk", "Pw", "bet", "cross", "F1", "F2", "stat", "muf", "mub")])
    end
    out
end
_istrans(list, name, out) = any(t -> startswith(name, t[1] * " ") && t[2] == string(out), list)   # out a String or a Symbol

"The level-1 operations: name => op(w, st, m, bc, cfg, t_new, ctx) returning `_po` outputs; ctx = (fs, ref)."
function _parity_ops(cfg::PBConfig; freeair::Bool=false)
    be(c) = c.backend
    dv(c, h) = _dev(c.backend, h)
    H(x) = Array(x)
    ops = Any[
    "K1 face operator psys*x" => function (w, st, m, bc, cfg, t, ctx)
        x = dv(cfg, _ptestvec(m.nc, 1)); y = similar(x); facesys_mul!(y, w.psys, x, m; backend=be(cfg))
        [_po("y", y, :value, nothing)]
    end,
    "K2 velocity gradients (least squares)" => function (w, st, m, bc, cfg, t, ctx)
        velocity_gradients!(w, m, st, bc, cfg)
        [_po(n, getfield(w, n), :value, nothing) for n in (:ubf, :vbf, :gux, :guy, :gvx, :gvy, :cux, :cuy, :cvx, :cvy)]
    end,
    "K3 pressure gradients (least squares, Green-Gauss)" => function (w, st, m, bc, cfg, t, ctx)
        pressure_gradients!(w.gpx, w.gpy, w.ggx, w.ggy, w.pbf, lev(st.p, 1), w, m, bc, cfg)
        [_po(n, getfield(w, n), :value, nothing) for n in (:gpx, :gpy, :ggx, :ggy, :pbf)]
    end,
    "K4 momentum assembly" => function (w, st, m, bc, cfg, t, ctx)
        r = assemble_momentum!(w, m, st, bc, cfg, t)
        S = ctx.ref ? _mom_scale(m, w, st, bc, cfg) : nothing
        out = Any[_po(n, getfield(w, n), :value, nothing) for n in (:a, :at, :as_, :bsx, :bsy, :fcx, :fcy, :bdg, :bfx, :bfy)]
        append!(out, [_po("msx.upper", w.msx.upper, :value, nothing), _po("msx.lower", w.msx.lower, :value, nothing),
                      _po("msx.rhs", w.msx.rhs, :value, nothing), _po("msy.rhs", w.msy.rhs, :value, nothing),
                      _po("res (per cell)", w.res, :resid_tol, cfg.mom_tol; cls=:R2, S=S),
                      _po("momentum residual", r, :resid_tol, cfg.mom_tol; cls=:R2, S=S === nothing ? nothing : maximum(S))])
        out
    end,
    "K5 H and the momentum interpolation (lambda 1)" => function (w, st, m, bc, cfg, t, ctx)
        momentum_H!(w, m, st, cfg); rhie_chow_flux!(w, m, st, bc, cfg, 1.0)
        [_po(n, getfield(w, n), :value, nothing) for n in (:Hx, :Hy, :hbx, :hby, :hxx, :hxy, :hyx, :hyy, :Fs, :Df, :Fbs, :Db)]
    end,
    "K6 pressure equation" => function (w, st, m, bc, cfg, t, ctx)
        assemble_pressure!(w, m, st, bc, cfg, t; tbar=w.tbar)
        wr = Array(m.vol); wr = (st.dt/cfg.theta_tol) ./ wr
        S = ctx.ref ? _cont_scale(m, w, st, cfg) : nothing
        [_po("psys.diag", w.psys.diag, :value, nothing), _po("psys.upper", w.psys.upper, :value, nothing),
         _po("psys.lower", w.psys.lower, :value, nothing),
         _po("psys.rhs (continuity residual)", w.psys.rhs, :resid_w, wr; cls=:R2, S=S)]
    end,
    "K7 mass transport" => function (w, st, m, bc, cfg, t, ctx)
        transport_mass!(w, m, st, cfg, t)
        S = ctx.ref ? _theta_scale(m, st, cfg) : nothing
        [_po("rho", lev(st.rho, 1), :value, nothing), _po("Theta", w.theta, :resid_tol, cfg.theta_tol; cls=:R2, S=S)]
    end,
    "K8 line tridiagonals, Thomas, a coloured line sweep (j then i)" => function (w, st, m, bc, cfg, t, ctx)
        line_tridiag!(w.fj, w.msx, m; backend=be(cfg)); line_tridiag!(w.fi, w.msx, m; backend=be(cfg))
        out = Any[_po("fj.$n", getfield(w.fj, n), :value, nothing) for n in (:ta, :td, :tc, :den, :cp)]
        append!(out, [_po("fi.$n", getfield(w.fi, n), :value, nothing) for n in (:ta, :td, :tc, :den, :cp)])
        r = dv(cfg, _ptestvec(m.nc, 2)); z = similar(r); fill!(z, 0.0); thomas_lines!(z, w.fj, r; backend=be(cfg))
        x = similar(r); fill!(x, 0.0)
        if cfg.line_solver === :thomas
            line_gs_sweep!(x, w.msx, m, w.fj, w.rw, w.zw; backend=be(cfg)); line_gs_sweep!(x, w.msx, m, w.fi, w.rw, w.zw; backend=be(cfg))
        else                               # W7 (review, finding 5): the sweep by cfg's line solver
            line_gs_sweep!(x, w.msx, m, w.fj, w.rw, w.zw, cfg.line_solver; backend=be(cfg))
            line_gs_sweep!(x, w.msx, m, w.fi, w.rw, w.zw, cfg.line_solver; backend=be(cfg))
        end
        append!(out, [_po("thomas z", z, :value, nothing; cls=:R3), _po("line sweep x", x, :value, nothing; cls=:R3)])
        out
    end,
    "K9 one V(1,1) cycle" => function (w, st, m, bc, cfg, t, ctx)
        h = _parity_mg!(w, m, cfg); r = dv(cfg, _ptestvec(m.nc, 3)); z = similar(r); fill!(z, 0.0); mg_vcycle!(z, h, r)
        [_po("z", z, :value, nothing; cls=_r3cls(cfg.prec_precision))]   # (R3f under the Float32 cycle: W7 Amendment 2)
    end,
    "K10 the hierarchy's numeric refresh" => function (w, st, m, bc, cfg, t, ctx)
        h = _parity_mg!(w, m, cfg)
        d2 = Array(w.psys.diag) .* (1 .+ 5e-4 .* (1 .+ sin.(1:m.nc)))      # raised only: a lowered diagonal makes the zero-row-sum rows indefinite
        s2 = FaceSystem(dv(cfg, d2), copy(w.psys.upper), copy(w.psys.lower), copy(w.psys.rhs))
        mg_refresh!(h, s2; coarse=cfg.mg_coarse)
        out = Any[]
        for (k, L) in enumerate(h.levels)
            for n in (:val, :la, :lden, :lcp, :Pval, :Rval, :Minv)
                A = getfield(L, n); length(A) > 0 && push!(out, _po("level $k $n", A, :value, nothing; cls=:R3))
            end
        end
        push!(out, _po("min line pivot", h.min_line_pivot, :value, nothing; cls=:R3),
              _po("min interpolation pivot", h.min_interp_pivot, :value, nothing; cls=:R3))
        out
    end,
    "K11 reductions (two-pass sum, max |.|, dot)" => function (w, st, m, bc, cfg, t, ctx)
        xh = _ptestvec(m.nc, 4); yh = _ptestvec(m.nc, 5); x = dv(cfg, xh); y = dv(cfg, yh)
        [_po("pb_sum", pb_sum(x; backend=be(cfg)), :scaled, sum(abs, xh); cls=:R1s, S=sum(abs, xh)),
         _po("pb_maxabs", pb_maxabs(x; backend=be(cfg)), :bitwise, nothing; cls=:bit),
         _po("pb_dot(x, y)", pb_dot(x, y; backend=be(cfg)), :scaled, sum(abs.(xh .* yh)); cls=:R1s, S=sum(abs.(xh .* yh))),
         _po("pb_dot(x, x)", pb_dot(x, x; backend=be(cfg)), :scaled, sum(abs2, xh); cls=:R1s, S=sum(abs2, xh))]
    end,
    "K12 the coupled freeze (first outer iteration)" => function (w, st, m, bc, cfg, t, ctx)
        _parity_mg!(w, m, cfg); cw = w.cpl
        cpl_freeze!(cw, w, m, st, bc, cfg, t, true)
        Wh = Array(cw.Wt)
        SR = nothing
        if ctx.ref
            tx, ty, _ = _mom_terms(m, w, st, bc, cfg); nc = m.nc
            # THE SCALE (PARITY RULE V3, records air/w7_speed/PARITY_RULE_V3.md): F* is RECOMPUTED here, so its rows' rounding is
            # relative to F*'s own terms as the kernel forms them (`_rc_term_scale`, pform :v3), not |F*|; ctx.k12 = :a1 is
            # amendment W6b-2's form and :v2 the registered V2 construction (|F*|), both kept for the record's readings
            k12 = get(ctx, :k12, :v3)
            a1 = k12 in (:v3, :ops, :a1)                 # (the term-scale forms; :v2 the |F*| form)
            Tf, Tb = a1 ? _rc_term_scale(m, w, st, bc, cfg; pform=k12) : (nothing, nothing)
            if cfg.cpl_newton && a1
                nx, ny = _newton_terms_a1(m, w, st, cfg, Tf); tx .+= nx; ty .+= ny
            elseif cfg.cpl_newton
                cf = Array(m.cface); nf = m.nf; Fs = Array(w.Fs); Fc = Array(cw.Fconv)
                eox = Array(cw.eox); eoy = Array(cw.eoy); enx = Array(cw.enx); eny = Array(cw.eny)
                for c in 1:nc, q in 1:4
                    code = cf[q, c]
                    if 0 < code <= nf
                        tx[c] += cfg.rho_l*(abs(Fs[code]) + abs(Fc[code]))*abs(eox[code]); ty[c] += cfg.rho_l*(abs(Fs[code]) + abs(Fc[code]))*abs(eoy[code])
                    elseif code < 0
                        tx[c] += cfg.rho_l*(abs(Fs[-code]) + abs(Fc[-code]))*abs(enx[-code]); ty[c] += cfg.rho_l*(abs(Fs[-code]) + abs(Fc[-code]))*abs(eny[-code])
                    end
                end
            end
            SR = vcat(tx, ty, a1 ? _cont_scale_a1(m, w, st, bc, cfg; Tf, Tb) : _cont_scale(m, w, st, cfg))
        end
        out = Any[_po("R (momentum and continuity residual)", cw.R, :resid_w, Wh; cls=:R2, S=SR)]
        append!(out, [_po(string(n), getfield(cw, n), :value, nothing; cls=(n === :dF0 ? :R3 : :R1))
                      for n in (:Wt, :Dc, :pdiag, :Fconv, :Fbconv, :eox, :eoy, :enx, :eny, :dF0)])
        append!(out, [_po(string(n), getfield(w, n), :value, nothing) for n in (:Fs, :Fbs, :Df, :Db)])
        append!(out, [_po("psys.diag", w.psys.diag, :value, nothing), _po("psys.upper", w.psys.upper, :value, nothing),
                      _po("fj.den", w.fj.den, :value, nothing), _po("fi.den", w.fi.den, :value, nothing),
                      _po("kref", cw.kref, :value, nothing)])
        for (k, L) in enumerate(w.mg.levels)
            push!(out, _po("refreshed level $k val", L.val, :value, nothing; cls=:R3))
        end
        out
    end,
    "K13 J x (matrix-free)" => function (w, st, m, bc, cfg, t, ctx)
        cw = w.cpl; x = dv(cfg, _ptestvec(cw.N, 6)); y = similar(x); fill!(y, 0.0); cpl_apply!(y, x, cw, w, m, bc, cfg)
        out = Any[_po("J x", y, :value, nothing), _po("F_lin", cw.Fl, :value, nothing), _po("Fb_lin", cw.Fbl, :value, nothing)]
        # W6b: J under the other cpl_visc_nonorth form (:deferred: without the viscous non-orthogonal part)
        cdf = _parity_cfg(cfg; cpl_visc_nonorth=cfg.cpl_visc_nonorth === :deferred ? :jacobian : :deferred)
        y2 = similar(x); fill!(y2, 0.0); cpl_apply!(y2, x, cw, w, m, bc, cdf)
        push!(out, _po("J x (cpl_visc_nonorth $(cdf.cpl_visc_nonorth))", y2, :value, nothing))
        out
    end,
    "K14 the SIMPLEC block preconditioner M r" => function (w, st, m, bc, cfg, t, ctx)
        _parity_mg!(w, m, cfg); cw = w.cpl; r = dv(cfg, _ptestvec(cw.N, 7)); z = similar(r); fill!(z, 0.0)
        cfg.prec_precision === :f32 && _cpl_f32_load!(cw, w, m, be(cfg))   # W7: the Float32 copies of the loaded freeze
        cpl_precond!(z, r, cw, w, m, bc, cfg)
        [_po("M r", z, :value, nothing; cls=_r3cls(cfg.prec_precision))]   # (R3f under the Float32 preconditioner)
    end,
    ]
    if cfg.turbulence === :sst
        append!(ops, Any[
        "K15 the SST terms at the iterate (turb_update!)" => function (w, st, m, bc, cfg, t, ctx)
            ok = turb_update!(w, m, st, bc, cfg); T = w.turb
            out = Any[_po("turb_update! accepted", ok ? 1.0 : 0.0, :bitwise, nothing; cls=:bit)]
            append!(out, [_po(string(n), getfield(T, n), :value, nothing) for n in (:kb, :wb, :gkx, :gky, :gwx, :gwy)])
            push!(out, _po("mut", lev(st.mut, 1), :value, nothing))
            append!(out, [_po(string(n), getfield(T, n), :value, nothing) for n in (:sigk, :sigw, :Pk, :Pw, :bet, :cross, :F1, :F2, :stat)])
            append!(out, [_po("muf", w.muf, :value, nothing), _po("mub", w.mub, :value, nothing)])
            out
        end,
        "K16 the k-omega assembly (assemble_komega!)" => function (w, st, m, bc, cfg, t, ctx)
            assemble_komega!(w, m, st, bc, cfg); T = w.turb
            out = Any[]
            for (s, sys) in (("ksys", T.ksys), ("wsys", T.wsys)), n in (:diag, :upper, :lower, :rhs)
                push!(out, _po("$s.$n", getfield(sys, n), :value, nothing))
            end
            append!(out, [_po(string(n), getfield(T, n), :value, nothing) for n in (:kfc, :wfc, :kbd, :kbr, :wbd, :wbr)])
            # W6: the split shares and the omega face mask; W6b: the bound's cut and (the review of W6b, F1) its budget Q
            append!(out, [_po(string(n), getfield(T, n), :value, nothing) for n in (:kshr, :wshr, :wcut, :wbq, :wfmask)])
            out
        end,
        "K17 the k-omega residuals and a fixed-count line solve" => function (w, st, m, bc, cfg, t, ctx)
            T = w.turb; b = be(cfg); out = Any[]
            for (s, sys, x, fj, fi, xref) in (("k", T.ksys, T.k, T.fjk, T.fik, T.kin), ("omega", T.wsys, T.w, T.fjw, T.fiw, T.win))
                r = _relres(sys, x, m, T.res, xref; backend=b); rc = copy(T.res)
                S = ctx.ref ? _relres_scale(sys, x, m, xref) : nothing
                push!(out, _po("$s relres (per cell)", rc, :value, nothing; cls=:R2, S=S),
                      _po("$s relres max", r, :value, nothing; cls=:R2, S=S === nothing ? nothing : maximum(S)))
                # W6: normalised by the unsplit diagonal (turbulence_steady_residual's reading)
                shr = s == "k" ? T.kshr : T.wshr
                ru = _relres_unsplit(sys, x, shr, m, T.res, xref; backend=b); rcu = copy(T.res)
                Su = ctx.ref ? _relres_scale(sys, x, m, xref; shr=shr) : nothing
                push!(out, _po("$s relres unsplit (per cell)", rcu, :value, nothing; cls=:R2, S=Su),
                      _po("$s relres unsplit max", ru, :value, nothing; cls=:R2, S=Su === nothing ? nothing : maximum(Su)))
                line_tridiag!(fj, sys, m; backend=b); line_tridiag!(fi, sys, m; backend=b)
                append!(out, [_po("$s fj.den", copy(fj.den), :value, nothing), _po("$s fi.den", copy(fi.den), :value, nothing)])
                y = copy(x)
                for _ in 1:PARITY_RULES_V2.kw_sweeps
                    _parity_kw_sweep!(y, sys, m, fj, fi, T, cfg)
                end
                push!(out, _po("$s after $(PARITY_RULES_V2.kw_sweeps) line sweeps", y, :value, nothing; cls=_r3cls(cfg.kw_precision)))
            end
            out
        end,
        "K20 the omega forms and the bound (W6 turb_omega_nonorth, W6b turb_omega_bound)" => function (w, st, m, bc, cfg, t, ctx)
            T = w.turb; out = Any[]
            for (fm, gb) in ((:full, 0.0), (:wall_off, 0.0), (:limited, 0.0), (:full, 0.25), (:wall_off, 0.25))
                cv = _parity_cfg(cfg; turb_omega_nonorth=fm, turb_omega_bound=gb)
                assemble_komega!(w, m, st, bc, cv)
                append!(out, [_po("$fm bound $gb $n", copy(getfield(T, n)), :value, nothing) for n in (:wfc, :wshr, :wcut, :wbq)])
                append!(out, [_po("$fm bound $gb wsys.$n", copy(getfield(T.wsys, n)), :value, nothing) for n in (:diag, :rhs)])
            end
            out
        end,
        ])
    end
    if freeair
        append!(ops, Any[
        "K18 the far-field values (set_farfield!, a point vortex G = 0.1)" => function (w, st, m, bc, cfg, t, ctx)
            bx = _bccopy(bc); set_farfield!(bx, m, ctx.fs, 0.1; backend=be(cfg))
            [_po("ub", bx.ub, :value, nothing), _po("vb", bx.vb, :value, nothing), _po("pb", bx.pb, :value, nothing)]
        end,
        "K21 the lifting potential start (W6: kutta_circulation, vortex_projection_start!; sharp-edged C-meshes)" => function (w, st, m, bc, cfg, t, ctx)
            # on fresh inputs (the start's own state); a sigma draw perturbs that state by +-1 ulp before each projection
            # (a baffle refuses the Kutta start too, kutta_circulation; the baffle review's F2, 27 Oct 2026)
            (m.family === :cmesh && isempty(H(m.base)) && isempty(H(m.baffle))) || return Any[]
            fs = ctx.fs; bx = _bccopy(bc)
            pre = ctx.rng === nothing ? nothing :
                  (s -> (for f in (:ux, :uy, :p, :rk, :rw); _perturb_ulp!(lev(getfield(s, f), 1), ctx.rng); end))
            G, j0, jG = kutta_circulation(m, bx, cfg, fs; work=w, prestart=pre)
            s1 = allocate_state(m; backend=be(cfg))
            vortex_projection_start!(s1, m, bx, cfg, fs, G; work=w, prestart=pre)
            # sigma floor: the projection's relative stopping tolerance (project_start!'s tol, 1e-12): two correct arithmetics
            # agree on an FCG solve stopped there to about that, not to the input's one-ulp sensitivity (amendment W6b, 1)
            sg = 1e-12
            Any[_po("G_K", G, :value, nothing; cls=:R3, sig=sg), _po("jump at 0", j0, :value, nothing; cls=:R3, sig=sg),
                _po("jump at G_K", jG, :value, nothing; cls=:R3, sig=sg), _po("ux", lev(s1.ux, 1), :value, nothing; cls=:R3, sig=sg),
                _po("uy", lev(s1.uy, 1), :value, nothing; cls=:R3, sig=sg), _po("p", lev(s1.p, 1), :value, nothing; cls=:R3, sig=sg),
                _po("F", lev(s1.F, 1), :value, nothing; cls=:R3, sig=sg), _po("far field ub", bx.ub, :value, nothing; cls=:R3, sig=sg)]
        end,
        "K19 the loads and the boundary fluxes (airfoil_loads!, boundary_fluxes)" => function (w, st, m, bc, cfg, t, ctx)
            b = be(cfg); fs = ctx.fs
            lw = LoadsWork(m; backend=b); L = airfoil_loads!(lw, st, m, bc, w, cfg, fs); fl = boundary_fluxes(lw, st, m; backend=b)
            out = Any[_po(string(n), getfield(lw, n), :value, nothing) for n in (:gux, :guy, :gvx, :gvy, :ubf, :vbf, :gpx, :gpy,
                                                                                    :ggx, :ggy, :pbf, :bdg, :bfx, :bfy)]
            append!(out, [_po(string(n), view(getfield(lw, n), 1:max(lw.nwall, 1)), :value, nothing) for n in (:fpx, :fpy, :fvx, :fvy, :mp, :mv)])
            Sc = Dict{Symbol,Float64}()
            if ctx.ref
                n = lw.nwall; fpx = H(lw.fpx)[1:n]; fpy = H(lw.fpy)[1:n]; fvx = H(lw.fvx)[1:n]; fvy = H(lw.fvy)[1:n]
                mp = H(lw.mp)[1:n]; mv = H(lw.mv)[1:n]; qc = qinf(fs)*fs.chord; ca = abs(cosd(fs.alpha)); sa = abs(sind(fs.alpha))
                ax = abs.(fpx) .+ abs.(fvx); ay = abs.(fpy) .+ abs.(fvy)
                Sc[:CL] = sum(ax .* sa .+ ay .* ca)/qc; Sc[:CD] = sum(ax .* ca .+ ay .* sa)/qc
                Sc[:CLp] = sum(abs.(fpx) .* sa .+ abs.(fpy) .* ca)/qc; Sc[:CDp] = sum(abs.(fpx) .* ca .+ abs.(fpy) .* sa)/qc
                Sc[:CDf] = sum(abs.(fvx) .* ca .+ abs.(fvy) .* sa)/qc; Sc[:CM] = sum(abs.(mp) .+ abs.(mv))/(qc*fs.chord)
                Fb = abs.(H(lev(st.Fb, 1)))
                Sc[:far] = sum(Fb[H(m.farfield)]; init=0.0); Sc[:outlet] = sum(Fb[H(m.outlets)]; init=0.0); Sc[:net] = sum(Fb)
            end
            sc(k) = ctx.ref ? Sc[k] : nothing
            for k in (:CL, :CD, :CLp, :CDp, :CDf, :CM)
                push!(out, _po(string(k), getfield(L, k), :scaled, sc(k); cls=:R1s, S=sc(k)))
            end
            for k in (:far, :outlet, :net)
                push!(out, _po("boundary flux $k", getfield(fl, k), :scaled, sc(k); cls=:R1s, S=sc(k)))
            end
            out
        end,
        ])
    end
    ops
end

"The verdict of one compared output under W3's PARITY_RULES (metric :w3): (:PASS | :NEAR | :FAIL, the measured number)."
function _parity_verdict(kind::Symbol, dev, ref, scale)
    R = PARITY_RULES
    if kind === :value
        d = reldiff(dev, ref); (d <= R.kernel_pass ? :PASS : d <= R.kernel_near ? :NEAR : :FAIL, d)
    elseif kind === :bitwise
        e = isequal(dev, ref); (e ? :PASS : :FAIL, e ? 0.0 : Inf)
    elseif kind === :scaled
        d = isequal(dev, ref) ? 0.0 : abs(dev - ref)/scale; (d <= R.kernel_pass ? :PASS : d <= R.kernel_near ? :NEAR : :FAIL, d)
    elseif kind === :resid_tol
        d = wdiff(dev, ref, 1/scale); (d <= R.resid ? :PASS : :FAIL, d)
    elseif kind === :resid_w
        d = wdiff(dev, ref, scale); (d <= R.resid ? :PASS : :FAIL, d)
    else
        throw(ArgumentError("unknown parity kind $kind"))
    end
end

"""The verdict of one compared output under P-L1 (metric :pl1, PARITY_RULES_V2): (verdict, number, the rule's bound). R1:
reldiff; R1s: |a - b|/S; R2: `epsdiff` in eps S units; R3: reldiff against max(16 eps, sigma_mult sigma); :bit: bitwise.
With `fmaoff` (R4's second process) an output that depends on no transcendental function must be bit for bit: its verdict is
:PASS only at exactly 0, else :FAIL (labelled by the caller)."""
function _parity_verdict_v2(cls::Symbol, dev, ref, S, sigma; fmaoff::Bool=false, trans::Bool=false)
    R = PARITY_RULES_V2
    d, bound, near = if cls === :R1
        (reldiff(dev, ref), R.kernel_pass, R.kernel_near)
    elseif cls === :R1s
        (isequal(dev, ref) ? 0.0 : abs(dev - ref)/S, R.kernel_pass, R.kernel_near)
    elseif cls === :R2
        (epsdiff(dev, ref, S), R.resid_eps, R.resid_eps)
    elseif cls === :R3
        b = max(R.kernel_pass, R.sigma_mult*sigma); (reldiff(dev, ref), b, b)
    elseif cls === :R3f                    # W7 Amendment 2: a multi-stage value of a Float32 form; sigma measured at Float32 ulps
        b = max(16*eps(Float32), R.sigma_mult*sigma); (reldiff(dev, ref), b, b)
    elseif cls === :bit
        (isequal(dev, ref) ? 0.0 : Inf, 0.0, 0.0)
    else
        throw(ArgumentError("unknown P-L1 class $cls"))
    end
    fmaoff && !trans && return (d == 0 ? :PASS : :FAIL, d, 0.0)
    (d <= bound ? :PASS : d <= near ? :NEAR : :FAIL, d, bound)
end

_hcopy(x) = x isa AbstractArray ? Array(x) : x

"""
    parity_kernels(m, bc, st, cfg, dev; mdev=to_backend(dev, m), bcdev=to_backend(dev, bc), perturb=0.0, only=nothing,
                   metric=:pl1, fs=nothing, fmaoff=false, k12_scale=:v3) -> (verdict, rows, reproducible)

LEVEL 1. `m, bc, st, cfg` the CPU case (a committed state); `dev` the device backend; `fs` the FreeStream of a free-air case
(adds K18-K19); SST configurations add K15-K17. Every operation of `_parity_ops` runs on the CPU and on `dev` from the same
snapshot (`parity_prepare`); `rows` holds (op, out, cls, trans, verdict, value, bound, sigma).
  metric = :w3 -- W3's registered rule (PARITY_RULES; the 22 Oct reading): :PASS, :NEAR_PASS (every :value/:scaled output
    within kernel_near and the rest PASS), :FAIL.
  metric = :pl1 -- THE RULE FOR FUTURE GATES (PARITY_RULES_V2; R1-R4): each operation runs a second time on the device and
    its outputs must equal the first bit for bit (`reproducible`, R4); each operation with an R3 output is also run on the CPU
    from the snapshot perturbed by +-1 ulp, sigma_draws times, for sigma_cpu; `fmaoff = true` (a device process with NVPTX
    FMA contraction off, R4) requires every output not listed by `parity_transcendental` to be bit for bit. :PASS (every
    output PASS and reproducible), :NEAR_PASS (an R1 output in the NEAR band, everything else PASS), :FAIL.
`perturb` (a negative example only) scales the device snapshot's level-1 velocity by (1 + perturb) before each operation.
`k12_scale`: K12's R2 rounding scale S_c. :v3 (THE DEFAULT: PARITY RULE V3, records air/w7_speed/PARITY_RULE_V3.md; alias :ops)
is the term scale with the pressure part as the kernel forms it, |fa (p_N - p_P)| (`_rc_term_scale`, pform :v3). :a1 (amendment
W6b-2, 23 Oct 2026; records air/w6b_start2/PARITY_RULE_V2_AMENDMENT_W6B_2.md, superseded by v3) builds its continuity rows on the recomputed flux's term scale T_f (`_cont_scale_a1`)
and its momentum rows' Newton term on (T_f + |F|)(|e| + the increment's parts) (`_newton_terms_a1`), the GPU merge's A1 with
fa (|p_N| + |p_P|) as its pressure part; :v2 is the registered V2 construction (|F*| in both), kept to read the record's earlier readings as they were read.
W7's speed settings of cfg are checked (for the device's configuration too) and their modes set for the check, then restored
(`with_speed_modes`; the W7 review's finding 5).
"""
parity_kernels(m::PBMesh, bc, st::PBState, cfg::PBConfig, dev; kw...) =
    with_speed_modes(() -> (check_speed_config(_parity_cfg(cfg; backend=dev)); cfg.prec_precision === :f32 && check_f32_state(st, cfg);
                            _parity_kernels(m, bc, st, cfg, dev; kw...)), cfg)   # (the f32 state rule: second W7 review, R2)
function _parity_kernels(m::PBMesh, bc, st::PBState, cfg::PBConfig, dev; mdev=to_backend(dev, m), bcdev=to_backend(dev, bc),
                         perturb::Real=0.0, only=nothing, metric::Symbol=:pl1, fs=nothing, fmaoff::Bool=false,
                         k12_scale::Symbol=:v3)
    metric in (:w3, :pl1) || throw(ArgumentError("parity_kernels: metric must be :w3 or :pl1 (got :$metric)"))
    k12_scale in (:v3, :ops, :a1, :v2) || throw(ArgumentError("parity_kernels: k12_scale must be :v3 (parity rule v3; alias :ops), :a1 (amendment W6b-2) or :v2 (got :$k12_scale)"))
    fmaoff && metric !== :pl1 && throw(ArgumentError("parity_kernels: the FMA-off control is P-L1's (metric :pl1)"))
    wc, stc, snapw, snaps = parity_prepare(m, bc, st, cfg)
    cc = _parity_cfg(cfg); cd = _parity_cfg(cfg; backend=dev)
    wd = pbwork(mdev, bcdev, cd); wd.cpl = CoupledWork(mdev, wd, cd)
    std = to_backend(dev, deepcopy(snaps))
    t_new = snaps.t + snaps.dt
    trl = parity_transcendental(cfg; freeair=fs !== nothing)
    rows = NamedTuple[]; repro = true
    function loadc!()
        _sync!(wc, snapw); copy_state!(stc, snaps); wc.mg = nothing
    end
    function loadd!()
        _sync!(wd, snapw); copy_state!(std, snaps); wd.mg = nothing
        if perturb != 0
            u = lev(std.ux, 1); copyto!(u, Array(u) .* (1 + perturb))
        end
    end
    for (name, op) in _parity_ops(cfg; freeair=fs !== nothing)
        only === nothing || any(s -> startswith(name, s), only) || continue
        loadc!(); oc = [merge(o, (val=_hcopy(o.val),)) for o in op(wc, stc, m, bc, cc, t_new, (fs=fs, ref=true, rng=nothing, k12=k12_scale))]
        loadd!(); od = [_hcopy(o.val) for o in op(wd, std, mdev, bcdev, cd, t_new, (fs=fs, ref=false, rng=nothing))]
        length(oc) == length(od) || error("parity_kernels: $name returned $(length(oc)) outputs on the CPU and $(length(od)) on the device")
        sig = [o.sig for o in oc]          # an output's own sigma floor (W6b: K21's projection tolerance), else 0
        if metric === :pl1
            loadd!(); od2 = [_hcopy(o.val) for o in op(wd, std, mdev, bcdev, cd, t_new, (fs=fs, ref=false, rng=nothing))]
            same = all(isequal(a, b) for (a, b) in zip(od, od2))
            same || (repro = false)
            # R3: the CPU's sensitivity to +-1 Float64 ulp; R3f (W7 Amendment 2, the Float32 forms): to +-1 Float32 ulp
            for (c3, unit) in ((:R3, Float64), (:R3f, Float32))
                any(o -> o.cls === c3, oc) || continue
                for k in 1:PARITY_RULES_V2.sigma_draws
                    loadc!(); rng = _UlpRNG(PARITY_RULES_V2.sigma_seed + k)
                    _perturb_ulp!(wc, rng; unit); _perturb_ulp!(stc, rng; unit)
                    op_ = op(wc, stc, m, bc, cc, t_new, (fs=fs, ref=false, rng=rng))     # rng: W6b's K21 perturbs its fresh inputs
                    for (i, o) in enumerate(oc)
                        o.cls === c3 && (sig[i] = max(sig[i], reldiff(_hcopy(op_[i].val), o.val)))
                    end
                end
            end
        end
        for (i, (a, b)) in enumerate(zip(oc, od))
            tr = _istrans(trl, name, a.out)
            if metric === :w3
                v, x = _parity_verdict(a.kind, b, a.val, a.scale); bd = NaN
            else
                v, x, bd = _parity_verdict_v2(a.cls, b, a.val, a.S, sig[i]; fmaoff, trans=tr)
            end
            push!(rows, (op=name, out=a.out, kind=a.kind, cls=a.cls, trans=tr, verdict=v, value=x, bound=bd, sigma=sig[i]))
        end
    end
    vs = [r.verdict for r in rows]
    verdict = (any(==(:FAIL), vs) || !repro) ? :FAIL : all(==(:PASS), vs) ? :PASS : :NEAR_PASS
    (verdict=verdict, rows=rows, reproducible=repro, transcendental=trl)
end

# ---- level 2: parity mode (fixed counts) --------------------------------------------------------------------------------
"""One alternating j-then-i k-omega line sweep of `sys` from x, in the form the solver's k-omega solve takes for cfg
(`_kw_line_solve!`; the W7 review's finding 5): the Thomas sweeps by default (bit for bit the parity tool before W7), else the
line solves by cfg.line_solver with the corrections in Float32 under kw_precision = :f32 (the solver's own Float32 copies,
loaded from fj and fi, the PCR on fj and fi themselves)."""
function _parity_kw_sweep!(x, sys::FaceSystem, m::PBMesh, fj, fi, T, cfg)
    be = cfg.backend
    if cfg.kw_precision === :f64 && cfg.line_solver === :thomas
        line_gs_sweep!(x, sys, m, fj, T.rw, T.zw; backend=be); line_gs_sweep!(x, sys, m, fi, T.rw, T.zw; backend=be)
        return x
    end
    if cfg.kw_precision === :f32
        F = _kw_fast!(T, fj, fi, be); sj = F.fam[objectid(fj)]; si = F.fam[objectid(fi)]
        line_copy!(sj, fj; backend=be); line_copy!(si, fi; backend=be); zw = F.zw
    else
        sj = fj; si = fi; zw = T.zw
    end
    line_gs_sweep!(x, sys, m, sj, T.rw, zw, cfg.line_solver; backend=be, fam64=fj)
    line_gs_sweep!(x, sys, m, si, T.rw, zw, cfg.line_solver; backend=be, fam64=fi)
    x
end

"""The k-omega update at fixed counts (the parity mode; never a production step): the systems assembled at the iterate (after
turb_update!), exactly `sweeps` alternating j-then-i line sweeps on k and on omega from the iterate (no stopping test), the
floors (`komega_solve!`'s, counted the same way) and rk = rho_l k, rw = rho_l omega written to level 1."""
function _parity_kw_fixed!(w, m::PBMesh, st::PBState, bc, cfg, sweeps::Integer)
    be = cfg.backend; T = w.turb; rho = cfg.rho_l
    assemble_komega!(w, m, st, bc, cfg)
    for (x, sys, fj, fi) in ((T.k, T.ksys, T.fjk, T.fik), (T.w, T.wsys, T.fjw, T.fiw))
        line_tridiag!(fj, sys, m; backend=be); line_tridiag!(fi, sys, m; backend=be)
        for _ in 1:sweeps
            _parity_kw_sweep!(x, sys, m, fj, fi, T, cfg)
        end
    end
    for (x, xmin, isk) in ((T.k, T.kmin, true), (T.w, T.wmin, false))
        _launch!(_k_floor!, be, m.nc, x, T.fl, xmin, rho, m.vol)
        add = pb_sum(T.fl; backend=be); _launch!(_k_count_pos!, be, m.nc, T.res, T.fl); n = round(Int, pb_sum(T.res; backend=be))
        isk ? (T.nfloor_k += n; T.added_k += add) : (T.nfloor_w += n; T.added_w += add)
    end
    _launch!(_k_kw_store!, be, m.nc, lev(st.rk, 1), lev(st.rw, 1), T.k, T.w, rho)
    KernelAbstractions.synchronize(be)
    nothing
end

"""
    parity_step!(st, m, bc, w, cfg; solves, iters, turb_sweeps=PARITY_RULES_V2.mode_turb_sweeps) -> st

ONE STEP AT FIXED COUNTS (plan 5.5's parity mode; never a production step): coupled_outer!'s outer iteration (under SST the
SST terms of the iterate first, turb_update!; the momentum assembly, the freeze, FGMRES, the update x += dx and F = F_RC +
F_lin, the mass transport; under SST with turb_coupling :outer the k-omega update after it) done exactly `solves` times,
each FGMRES run for exactly `iters` iterations (cpl_fgmres! at cpl_maxit = iters with an unreachable goal, its dx taken as
it is); under SST with :step the k-omega update on the final flow iterate (turb_update!, `_parity_kw_fixed!`, turb_update!
for the committed mu_t, as `_komega_step_update!` does); then committed whatever the residuals. The k-omega updates take
exactly `turb_sweeps` line sweeps each. `cfg` must be a `_parity_cfg` (the hierarchy refreshed at every outer iteration)
with cpl_maxit = iters, cpl_eta below any reachable ratio and cpl_abs 0 (`parity_mode` builds it). W7's modes are set from
cfg for the step and restored after (`with_speed_modes`; the W7 review's finding 5).
"""
parity_step!(st::PBState, m::PBMesh, bc, w, cfg::PBConfig; kw...) =
    with_speed_modes(() -> _parity_step!(st, m, bc, w, cfg; kw...), cfg)
function _parity_step!(st::PBState, m::PBMesh, bc, w, cfg::PBConfig; solves::Integer, iters::Integer,
                       turb_sweeps::Integer=PARITY_RULES_V2.mode_turb_sweeps)
    (cfg.cpl_maxit == iters && cfg.cpl_abs == 0 && cfg.cpl_restart >= iters && cfg.mg_diag_delta == 0 && cfg.mg_every_outer == 1) ||
        throw(ArgumentError("parity_step!: the configuration is not the parity mode's (see parity_mode)"))
    be = cfg.backend; nc = m.nc; t_new = st.t + st.dt
    sst = cfg.turbulence === :sst
    outer_turb = sst && cfg.turb_coupling === :outer && st.nstep >= cfg.turb_outer_ramp
    restore_level!(st)
    cfg.prec_precision === :f32 && check_f32_state(st, m, w, cfg)   # W7: Float32 only at a uniform density (second review, R2)
    if cfg.extrapolate && st.nstep >= 1 && st.dt_prev > 0
        r = st.dt/st.dt_prev
        for fld in (:ux, :uy, :p, :F, :Fb)
            A = getfield(st, fld); _launch!(_k_extrap!, be, size(A, 1), lev(A, 1), lev(A, 2), lev(A, 3), r)
        end
        KernelAbstractions.synchronize(be)
    end
    _launch!(_k_theta!, be, nc, w.theta, lev(st.rho, 2), cfg.rho_l)
    w.tbar = theta_drive_mean(w, m, bc, cfg)
    sst && turb_step_reset!(w.turb)
    cw = w.cpl === nothing ? (w.cpl = CoupledWork(m, w, cfg)) : w.cpl
    u = lev(st.ux, 1); v = lev(st.uy, 1); p = lev(st.p, 1)
    for s in 1:solves
        # mu_t as the production step forms it (W6b: turb_mut = :frozen, the default since W6, forms it at the first solve only
        # under the :step treatment; :iterate and :outer at every solve, the rule's registered form; turb_mut_base, 27 Oct 2026:
        # the first solve's from level n under :committed, coupled_outer!'s frozen_mut_level)
        sst && (cfg.turb_mut === :iterate || outer_turb || s == 1) &&
            (turb_update!(w, m, st, bc, cfg; level=s == 1 ? frozen_mut_level(cfg, outer_turb) : 1) ||
             error("parity_step!: turb_update! refused the iterate (solve $s)"))
        assemble_momentum!(w, m, st, bc, cfg, t_new)
        cpl_freeze!(cw, w, m, st, bc, cfg, t_new, s == 1)
        r = cpl_fgmres!(cw, w, m, bc, cfg)
        r.iters == iters || error("parity_step!: FGMRES ran $(r.iters) iterations, not $iters (a breakdown)")
        _launch!(_k_cpl_axpy!, be, nc, u, 1.0, cfield(cw.dx, 1, nc)); _launch!(_k_cpl_axpy!, be, nc, v, 1.0, cfield(cw.dx, 2, nc))
        _launch!(_k_cpl_axpy!, be, nc, p, 1.0, cfield(cw.dx, 3, nc))
        cpl_apply!(cw.Y, cw.dx, cw, w, m, bc, cfg)
        _launch!(_k_cpl_add2!, be, m.nf, lev(st.F, 1), w.Fs, cw.Fl)
        _launch!(_k_cpl_add2!, be, m.nb, lev(st.Fb, 1), w.Fbs, cw.Fbl)
        KernelAbstractions.synchronize(be)
        transport_mass!(w, m, st, cfg, t_new)
        if outer_turb
            turb_update!(w, m, st, bc, cfg) || error("parity_step!: turb_update! refused the iterate (solve $s, :outer)")
            _parity_kw_fixed!(w, m, st, bc, cfg, turb_sweeps)
        end
    end
    if sst && !outer_turb
        turb_update!(w, m, st, bc, cfg) || error("parity_step!: turb_update! refused the flow iterate (:step)")
        _parity_kw_fixed!(w, m, st, bc, cfg, turb_sweeps)
        turb_update!(w, m, st, bc, cfg) || error("parity_step!: turb_update! refused the k-omega iterate (:step)")
    end
    commit_level!(st)
end

"""
    parity_mode(m, bc, st, cfg, dev; steps, solves, iters, turb_sweeps, mdev, bcdev, devcfg=nothing, metric=:pl1, fs=nothing,
                vortex=nothing) -> (verdict, worst, band, sigma2, rows)

LEVEL 2. From the CPU state `st` (host; its committed level), `steps` fixed-count steps (`parity_step!`) on the CPU and on
`dev`, each backend from its own copy of the same state and its own copy of ONE hierarchy (a host build of the pressure
operator of a first CPU freeze at that state, so the symbolic structure is shared); after every step every field of
STATE_FIELDS (rk, rw and mut included: review of W3, F7) is compared at its committed level (`reldiff`). Free air (`fs`, a
FreeStream; `vortex = (relax, every)`): after every step each backend takes its own loads (airfoil_loads! of the committed
state) and, with `vortex`, its own point-vortex update of its own copy of the conditions (vortex_update!); C_L, C_D, C_M and
the circulation are compared too.
  metric = :w3 -- :PASS when the largest difference is at most PARITY_RULES.mode (W3's rule).
  metric = :pl1 -- the band is max(PARITY_RULES_V2.mode, mode_sigma_mult sigma2), sigma2 the CPU's own largest difference
    when the same fixed-count steps start from the state perturbed by +-1 ulp (every field, every level; seed sigma_seed),
    computed here on the CPU: :PASS within the band.
`devcfg` (a negative example only) replaces the device's configuration. W7's speed settings are checked (the device's
configuration too) and their modes set for the check and restored after (the W7 review's finding 5).
"""
parity_mode(m::PBMesh, bc, st0::PBState, cfg::PBConfig, dev; kw...) =
    with_speed_modes(() -> (check_speed_config(_parity_cfg(cfg; backend=dev)); cfg.prec_precision === :f32 && check_f32_state(st0, cfg);
                            _parity_mode(m, bc, st0, cfg, dev; kw...)), cfg)   # (the f32 state rule: second W7 review, R2)
function _parity_mode(m::PBMesh, bc, st0::PBState, cfg::PBConfig, dev; steps::Integer=PARITY_RULES.mode_steps,
                      solves::Integer=PARITY_RULES.mode_solves, iters::Integer=PARITY_RULES.mode_iters,
                      turb_sweeps::Integer=PARITY_RULES_V2.mode_turb_sweeps, mdev=to_backend(dev, m),
                      bcdev=to_backend(dev, bc), devcfg=nothing, metric::Symbol=:pl1, fs=nothing, vortex=nothing)
    metric in (:w3, :pl1) || throw(ArgumentError("parity_mode: metric must be :w3 or :pl1 (got :$metric)"))
    kw = (cpl_maxit=iters, cpl_restart=max(cfg.cpl_restart, iters), cpl_eta=1e-300, cpl_abs=0.0)
    cc = _parity_cfg(cfg; kw...); cd = devcfg === nothing ? _parity_cfg(cfg; kw..., backend=dev) : devcfg
    sst = cc.turbulence === :sst
    # the shared hierarchy: a CPU freeze at the start state gives the pressure operator both hierarchies are built from
    wc = pbwork(m, bc, cc); wc.cpl = CoupledWork(m, wc, cc); s0 = deepcopy(st0); restore_level!(s0)
    t1 = s0.t + s0.dt
    _launch!(_k_theta!, CPU(), m.nc, wc.theta, lev(s0.rho, 2), cc.rho_l)
    sst && (turb_update!(wc, m, s0, bc, cc) || error("parity_mode: turb_update! refused the start state"))
    assemble_momentum!(wc, m, s0, bc, cc, t1); cpl_freeze!(wc.cpl, wc, m, s0, bc, cc, t1, true)
    ps = deepcopy(wc.psys)
    function fresh(mm, bb, c)
        w = pbwork(mm, bb, c); w.cpl = CoupledWork(mm, w, c)
        copyto!(w.psys.diag, ps.diag); copyto!(w.psys.upper, ps.upper); copyto!(w.psys.lower, ps.lower); copyto!(w.psys.rhs, ps.rhs)
        _parity_mg!(w, mm, c); w
    end
    # one fixed-count march (its own work, conditions and vortex): the committed fields and loads after every step
    function march(mm, bb0, c, s)
        bb = _bccopy(bb0); w = fresh(mm, bb, c); out = Any[]
        pv = vortex === nothing ? nothing : PointVortex(; relax=vortex[1], every=vortex[2])
        lw = fs === nothing ? nothing : LoadsWork(mm; backend=c.backend)
        for k in 1:steps
            parity_step!(s, mm, bb, w, c; solves, iters, turb_sweeps)
            rec = Dict{Symbol,Any}(f => Array(lev(getfield(s, f), 2)) for f in STATE_FIELDS)
            if fs !== nothing
                L = airfoil_loads!(lw, s, mm, bb, w, c, fs)
                rec[:CL] = L.CL; rec[:CD] = L.CD; rec[:CM] = L.CM
                pv === nothing || vortex_update!(pv, L.CL, fs, bb, mm, s.nstep; backend=c.backend)
                rec[:G] = pv === nothing ? 0.0 : pv.G
            end
            push!(out, rec)
        end
        out
    end
    keys_ = fs === nothing ? collect(STATE_FIELDS) : vcat(collect(STATE_FIELDS), [:CL, :CD, :CM, :G])
    hc = march(m, bc, cc, deepcopy(st0)); hd = march(mdev, bcdev, cd, to_backend(dev, deepcopy(st0)))
    rows = NamedTuple[]; worst = 0.0
    for k in 1:steps, f in keys_
        d = reldiff(hd[k][f], hc[k][f]); worst = max(worst, d)
        push!(rows, (step=k, field=f, value=d))
    end
    sigma2 = NaN; band = PARITY_RULES.mode
    if metric === :pl1
        sp = deepcopy(st0); rng = _UlpRNG(PARITY_RULES_V2.sigma_seed)
        for f in STATE_FIELDS; _perturb_ulp!(getfield(sp, f), rng); end
        hp = march(m, bc, cc, sp)
        sigma2 = maximum(reldiff(hp[k][f], hc[k][f]) for k in 1:steps for f in keys_)
        band = max(PARITY_RULES_V2.mode, PARITY_RULES_V2.mode_sigma_mult*sigma2)
    end
    (verdict=worst <= band ? :PASS : :FAIL, worst=worst, band=band, sigma2=sigma2, rows=rows)
end

# ---- level 3: production mode -------------------------------------------------------------------------------------------
"""
    parity_production(m, bc, st, cfg, dev; steps, mdev, bcdev, devcfg=nothing, corrupt=0.0, corrupt_k=0.0, fs=nothing,
                      vortex=nothing) -> (verdict, rows)

LEVEL 3. A CPU production march of `steps` steps from the host state `st` (pbcav_march!'s step sequence: pimple_step! with
the halving rule); before each CPU step its committed state (and its conditions) is loaded on `dev` and ONE device production
step is taken there (pimple_step!, the device's own work space carried from step to step as a march carries it, its hierarchy
built and refreshed by the device's own policy); the device's iterate (every field: review of W3, F7) is brought back and
re-evaluated ON THE CPU: its momentum residual (assemble_momentum! at that iterate against the CPU's level n, time term on,
under SST with mu_t re-formed by turb_update! from the k and omega the momentum saw -- level n under turb_coupling :step, the
iterate's under :outer) and its max |Theta| (the mass transported on the CPU with the device's final flux); under SST also
R_turb (`turb_coupling` :step: the k-omega systems the step's update solved, assembled on the CPU at the device's flow
iterate, against the device's k and omega, in units of the solve's goal max(turb_lin_rtol r0, goal floor), over the CPU's
own value of the same quantity at its own step; :outer: komega_residual at the device's iterate over turb_tol) and the
committed mu_t against turb_update! on the CPU at the device's final iterate (reldiff; sigma_mut the CPU's own reldiff when
that iterate's level-1 fields are perturbed by +-1 ulp). A row PASSES (PARITY_RULES_V2) when the device step was accepted,
R_mom <= production x mom_tol, |Theta| <= production x theta_tol, and under SST R_turb <= production x max(1, R_turb of the
CPU) (:step) or <= production (:outer) and the mu_t difference <= max(production_mut_floor, production_mut_sigma_mult
sigma_mut). :PASS when every row passes. Each row also records the two steps' solve and Krylov counts.
Free air: `fs` and `vortex = (relax, every)` -- after each CPU commit the CPU's loads and its point-vortex update of `bc`
(the conditions the next device step is given). `corrupt` / `corrupt_k` (negative examples only) add that value to the device
iterate's u in cell 1, or multiply its largest k by (1 + corrupt_k), before the re-evaluation. A CPU step that is rejected is retried
at half dt, as the march does; the device is then compared at the dt the CPU accepts.
UNDER turb_mut = :frozen (W6's default; W6b) the momentum residual is re-evaluated with the mu_t the step's solves held: formed
by turb_update! at the step's START iterate (`_step_start!`, pimple_step!'s own prologue: level n restored and extrapolated, k^n, omega^n), as
coupled_outer! forms it, not at the device's final flow iterate (which is :iterate's mu_t); under turb_mut_base = :committed (27 Oct
2026) at that start state's level n, as coupled_outer! forms it then (`frozen_mut_level`).
W7's modes are set from cfg for the whole check and restored after (`with_speed_modes`; the W7 review's finding 5).
"""
parity_production(m::PBMesh, bc, st0::PBState, cfg::PBConfig, dev; kw...) =
    with_speed_modes(() -> _parity_production(m, bc, st0, cfg, dev; kw...), cfg)
function _parity_production(m::PBMesh, bc, st0::PBState, cfg::PBConfig, dev; steps::Integer, mdev=to_backend(dev, m),
                            bcdev=to_backend(dev, bc), devcfg=nothing, corrupt::Real=0.0, corrupt_k::Real=0.0, fs=nothing,
                            vortex=nothing, ramp=nothing)
    # THE RAMP START (27 Oct 2026, the baffle review's F1): with `ramp` (T, r0; needs fs), before every CPU attempt the far field
    # is set at the attempt's time t + dt as freeair_run!'s pre_step sets it (U r(t), the vortex's G), and the device step gets
    # those values with the rest of the conditions; the point vortex is not updated while t < T. `nothing`: the march as before
    ramp === nothing || (fs === nothing && throw(ArgumentError("parity_production: a ramp needs the freestream fs")))
    ramp === nothing || check_start_ramp(ramp)
    cd = devcfg === nothing ? PBConfig(; (k => getfield(cfg, k) for k in fieldnames(PBConfig))..., backend=dev) : devcfg
    sst = cfg.turbulence === :sst
    (sst && cfg.turb_coupling === :step && isfinite(cfg.turb_step_tol)) &&
        throw(ArgumentError("parity_production: under SST :step the rule is written for one k-omega update per step (turb_step_tol = Inf)"))
    bcl = _bccopy(bc); bcd = _bccopy(bcdev)                    # copies: the march's vortex updates the conditions
    sc = deepcopy(st0); sc.dt = sc.dt > 0 ? min(sc.dt, cfg.dt) : cfg.dt
    wc = pbwork(m, bcl, cfg); wd = pbwork(mdev, bcd, cd); we = pbwork(m, bcl, cfg)
    sd = to_backend(dev, deepcopy(sc)); se = deepcopy(sc); sp = deepcopy(sc)
    pv = vortex === nothing ? nothing : PointVortex(; relax=vortex[1], every=vortex[2])
    lw = fs === nothing ? nothing : LoadsWork(m)
    rows = NamedTuple[]; nrej = 0
    R = PARITY_RULES_V2
    rlast = NaN
    while length(rows) < steps
        if ramp !== nothing
            rf = start_ramp_factor(sc.t + sc.dt, ramp)
            rf == rlast || (set_farfield!(bcl, m, scaled_freestream(fs, rf), pv === nothing ? 0.0 : pv.G); rlast = rf)
        end
        copy_state!(sd, sc); copyto!(bcd.ub, bcl.ub); copyto!(bcd.vb, bcl.vb); copyto!(bcd.pb, bcl.pb)
        accd, reasond, infod = pimple_step!(sd, mdev, bcd, wd, cd)
        accc, reasonc, infoc = pimple_step!(sc, m, bcl, wc, cfg)
        if !accc
            nrej += 1; nrej > 10 && error("parity_production: the CPU march was refused (10 consecutive rejections)")
            sc.dt /= 2; continue
        end
        nrej = 0
        # the device iterate on the CPU: level 1 from the device (every field), levels 2-3 the CPU's (the same, loaded above)
        copy_state!(se, sc)
        for f in STATE_FIELDS; copyto!(lev(getfield(se, f), 1), Array(lev(getfield(sd, f), 1))); end
        if corrupt != 0
            u = lev(se.ux, 1); h = Array(u); h[1] += corrupt; copyto!(u, h)
        end
        if corrupt_k != 0
            k_ = lev(se.rk, 1); h = Array(k_); ck = argmax(h); h[ck] *= 1 + corrupt_k; copyto!(k_, h)
        end
        t_new = se.t + se.dt
        rturb = NaN; rturb_cpu = NaN; mutd = NaN; sigmut = NaN; okt = true
        if sst
            T = we.turb; outer = cfg.turb_coupling === :outer && sc.nstep >= cfg.turb_outer_ramp
            kd = Array(lev(se.rk, 1)) ./ cfg.rho_l; wdv = Array(lev(se.rw, 1)) ./ cfg.rho_l
            # (iii) mu_t: the CPU's evaluation at the device's final iterate, and its own one-ulp sensitivity there
            turb_update!(we, m, se, bcl, cfg) || error("parity_production: turb_update! refused the device iterate")
            mutc = Array(lev(se.mut, 1)); mutd = reldiff(Array(lev(sd.mut, 1)), mutc)
            copy_state!(sp, se); rng = _UlpRNG(R.sigma_seed + sc.nstep)
            for f in (:ux, :uy, :rk, :rw); _perturb_ulp!(lev(getfield(sp, f), 1), rng); end
            turb_update!(we, m, sp, bcl, cfg) || error("parity_production: turb_update! refused the perturbed iterate")
            sigmut = reldiff(Array(lev(sp.mut, 1)), mutc)
            if outer
                turb_update!(we, m, se, bcl, cfg)
                rk_, rw_ = komega_residual(we, m, se, bcl, cfg)
                rturb = max(rk_, rw_)/cfg.turb_tol; rturb_cpu = NaN
                okt = rturb <= R.production
            else
                # (ii) the k-omega systems the step's update solved: at the device's flow iterate with k^n, omega^n
                for f in (:rk, :rw); copyto!(lev(getfield(se, f), 1), lev(getfield(se, f), 2)); end
                turb_update!(we, m, se, bcl, cfg) || error("parity_production: turb_update! refused the device flow iterate")
                assemble_komega!(we, m, se, bcl, cfg)
                floor_ = _kw_goal_floor(cfg)
                qd = 0.0
                for (sys, x0, xd, xref) in ((T.ksys, Array(T.k), kd, T.kin), (T.wsys, Array(T.w), wdv, T.win))
                    r0 = _relres(sys, x0, m, T.res, xref); goal = max(cfg.turb_lin_rtol*r0, floor_)
                    qd = max(qd, _relres(sys, xd, m, T.res, xref)/goal)
                end
                # the CPU's own step against its own systems (sc's level 1 is the CPU's accepted iterate)
                sx = deepcopy(sc)
                kc = Array(lev(sx.rk, 1)) ./ cfg.rho_l; wcv = Array(lev(sx.rw, 1)) ./ cfg.rho_l
                for f in (:rk, :rw); copyto!(lev(getfield(sx, f), 1), lev(getfield(sx, f), 2)); end
                turb_update!(we, m, sx, bcl, cfg) || error("parity_production: turb_update! refused the CPU flow iterate")
                assemble_komega!(we, m, sx, bcl, cfg)
                qc = 0.0
                for (sys, x0, xc_, xref) in ((T.ksys, Array(T.k), kc, T.kin), (T.wsys, Array(T.w), wcv, T.win))
                    r0 = _relres(sys, x0, m, T.res, xref); goal = max(cfg.turb_lin_rtol*r0, floor_)
                    qc = max(qc, _relres(sys, xc_, m, T.res, xref)/goal)
                end
                rturb = qd; rturb_cpu = qc
                okt = rturb <= R.production*max(1.0, qc)
                # (i) the momentum saw k^n, omega^n: mu_t re-formed there (se's level 1 rk, rw are level n now); under
                # turb_mut = :frozen (W6's default) at the step's START iterate, as coupled_outer! formed it (W6b), or, under
                # turb_mut_base = :committed (27 Oct 2026), at that start state's level n (frozen_mut_level, coupled_outer!'s)
                if cfg.turb_mut === :frozen
                    s0 = deepcopy(se); _step_start!(s0, cfg)
                    turb_update!(we, m, s0, bcl, cfg; level=frozen_mut_level(cfg, outer)) ||
                        error("parity_production: turb_update! refused the step's start iterate")
                else
                    turb_update!(we, m, se, bcl, cfg)
                end
            end
            okt = okt && mutd <= max(R.production_mut_floor, R.production_mut_sigma_mult*sigmut)
        end
        rm = assemble_momentum!(we, m, se, bcl, cfg, t_new)
        transport_mass!(we, m, se, cfg, t_new); th = pb_maxabs(we.theta)
        ok = accd && rm <= R.production*cfg.mom_tol && th <= R.production*cfg.theta_tol && okt
        push!(rows, (step=sc.nstep + 1, t=t_new, dt=sc.dt, device_accepted=accd, device_reason=reasond, Rmom=rm/cfg.mom_tol,
                     Rtheta=th/cfg.theta_tol, Rturb=rturb, Rturb_cpu=rturb_cpu, mut_diff=mutd, sigma_mut=sigmut, ok=ok,
                     cpu_solves=infoc.outer, dev_solves=infod.outer, cpu_krylov=sum(infoc.fcg; init=0),
                     dev_krylov=sum(infod.fcg; init=0)))
        commit_level!(sc)
        if fs !== nothing
            L = airfoil_loads!(lw, sc, m, bcl, wc, cfg, fs)
            pv === nothing || (ramp !== nothing && sc.t < ramp.T) || vortex_update!(pv, L.CL, fs, bcl, m, sc.nstep)
        end
    end
    (verdict=all(r -> r.ok, rows) ? :PASS : :FAIL, rows=rows)
end

# ---- level 0: bitwise properties on one backend -------------------------------------------------------------------------
"""
    parity_bitwise(m, bc, st, cfg; steps=3) -> (verdict, checks)

LEVEL 0 on cfg.backend (mesh, conditions and the host state `st` moved there): (a) determinism: one step, and a `steps`-step
march, each repeated from the same state with a fresh work space, give the same state hash; (b) rollback: a step forced to
reject (n_outer_max 1, mom_tol 1e-30) restores level 1 from level 2 bit for bit and leaves levels 2-3 untouched; (c) the
host round trip of a state is bitwise, and a march from the round-tripped state equals one from the original. :PASS when
every check holds. Every field of STATE_FIELDS enters (mut included since the merge with W1).
"""
function parity_bitwise(m::PBMesh, bc, st0::PBState, cfg::PBConfig; steps::Integer=3)
    be = cfg.backend; md = to_backend(be, m); bd = to_backend(be, bc)
    newst() = to_backend(be, deepcopy(st0))
    # one time level of every field, in a PBState of its own (built from fieldnames: review of W3, F1)
    function lvfield(st, n, L)
        n === :NK && return st.NK
        n in STATE_FIELDS && return Array(view(getfield(st, n), :, :, L:L))
        n === :nstep ? 0 : 0.0
    end
    lvsha(st, L) = state_sha(PBState((lvfield(st, n, L) for n in fieldnames(PBState))...))
    checks = Pair{String,Bool}[]
    # (a) determinism
    s1 = newst(); s2 = newst()
    a1 = pimple_step!(s1, md, bd, pbwork(md, bd, cfg), cfg); a2 = pimple_step!(s2, md, bd, pbwork(md, bd, cfg), cfg)
    push!(checks, "one step twice: the same bits" => (a1[1] == a2[1] && state_sha(s1) == state_sha(s2)))
    s1 = newst(); s2 = newst()
    pbcav_march!(s1, md, bd, cfg; max_steps=steps); pbcav_march!(s2, md, bd, cfg; max_steps=steps)
    push!(checks, "a $steps-step march twice: the same bits" => (state_sha(s1) == state_sha(s2) && s1.nstep == st0.nstep + steps))
    # (b) rollback
    cr = PBConfig(; (k => getfield(cfg, k) for k in fieldnames(PBConfig))..., n_outer_max=1, mom_tol=1e-30, predict_reject=false)
    s3 = newst(); h2 = lvsha(s3, 2); h3 = lvsha(s3, 3)
    acc, reason, _ = pimple_step!(s3, md, bd, pbwork(md, bd, cr), cr)
    push!(checks, "a forced rejection is rejected" => !acc)
    push!(checks, "rollback: level 1 restored from level 2 bit for bit" => (lvsha(s3, 1) == lvsha(s3, 2)))
    push!(checks, "rollback: levels 2 and 3 untouched" => (lvsha(s3, 2) == h2 && lvsha(s3, 3) == h3))
    # (c) the host round trip
    s4 = newst(); pbcav_march!(s4, md, bd, cfg; max_steps=1)
    s5 = to_backend(be, to_host(s4))
    push!(checks, "host round trip: the same state hash" => (state_sha(s5) == state_sha(s4) && s5.nstep == s4.nstep))
    pbcav_march!(s4, md, bd, cfg; max_steps=steps - 1); pbcav_march!(s5, md, bd, cfg; max_steps=steps - 1)
    push!(checks, "a march from the round-tripped state equals one from the original" => (state_sha(s4) == state_sha(s5)))
    (verdict=all(last, checks) ? :PASS : :FAIL, checks=checks)
end
