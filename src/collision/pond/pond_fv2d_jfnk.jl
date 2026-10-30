# =====================================================================================================
# PHASE 2 (implicit roadmap) — JACOBIAN-FREE NEWTON-KRYLOV on the composite state. DRAFT PLUMBING
# (2026-07-11): not yet included by any tool; wired + gated after the Phase-1 LTS gate closes.
#
# Design per the roadmap + panel:
#  - Pseudo-transient continuation (PTC): solve  (V/Δτ)·δU + J·δU = −R(U)  per outer step, where
#    V/Δτ is the per-cell LTS diagonal (the dtf field) — PTC ≡ backward-Euler in pseudo-time; Δτ→∞
#    recovers pure Newton. NEVER raw Newton on transonic (globalization by construction).
#  - MATRIX-FREE: J·v by finite differences of the composite residual first (stand-up tier);
#    ε chosen FP32-aware. The AD-exact matvec (ForwardDiff duals / Enzyme forward through the KA
#    kernels — Gate-0 tier-ii) replaces `jfnk_matvec!` behind the same signature when de-risked.
#  - FGMRES(m≤15) with FP64 inner products (cdot/cnorm — Phase-0 step 4b), right-preconditioned.
#    Preconditioner tier 0: the PTC diagonal itself (dtf scaling). Tier 1 (roadmap): multicolor
#    point-block sweeps + geometric MG on the AMR ladder.
#  - HONESTY: R here is the Phase-1 steady operator's FLUX core (composite_residual!) — the same
#    frozen-closure caveats apply (cd/wm lagged, SVV/eddy/penalization outside R). The Newton loop
#    interleaves the Phase-1 smoother steps to carry the non-R passes (defect-correction style)
#    until the rate-ization completes them into R.
# =====================================================================================================

# FD-ε Jacobian-vector product: Jv ≈ [R(U + ε·v) − R(U)] / ε on the composite.
# U0/R0 are the base state/residual (cget!/crates! snapshots); v, out are CompositeVecs.
# scratch: one CompositeVec. ε per Knoll-Keyes: ε = sqrt(εmach)·(1 + ‖U‖)/‖v‖  (FP32 εmach).
# BC-consistent state set: cset! (sync of covered/halo images) + domain-BC re-imposition per level.
# Without the BC step, Newton increments accumulate uncorrected garbage in the L1 BC bands (masked
# from the NORM but read by flux stencils) — the step-8 blowup of the first bring-up run.
function _set_consistent!(C::Composite2T, v::CompositeVec)
    cset!(C, v)
    for d in C.L; pond_fv2dT_applyBC_steady!(d); end
    return C
end

function jfnk_matvec!(out::CompositeVec, C::Composite2T, U0::CompositeVec, R0::CompositeVec,
                      v::CompositeVec, scratch::CompositeVec; eps_mach::Float64=1.19e-7)
    nv = cnorm(C, v)
    nv == 0.0 && (ccopy!(out, R0); cscale!(0.0, out); return out)
    nu = cnorm(C, U0)
    eps = sqrt(eps_mach)*sqrt(1.0+nu)/nv     # PETSc/Knoll-Keyes form: sqrt(1+||U||) — the linear form
                                             # gave eps~0.4 on the raw composite norm (first-launch NaN)
    ccopy!(scratch, U0); caxpy!(eps, v, scratch)          # scratch = U0 + ε v  (v lives in the DOF subspace)
    _set_consistent!(C, scratch)
    composite_residual!(C)
    crates!(out, C); cmask!(out, C)                       # out = R(U0+εv), projected to the DOF subspace
    caxpy!(-1.0, R0, out); cscale!(1.0/eps, out)          # out = (R(U0+εv) − R0)/ε
    _set_consistent!(C, U0)                               # restore base state
    return out
end

# Right-preconditioned FGMRES(m) for  A·x = b  with A·v given by `matvec!(out,v)` and
# M⁻¹·v by `prec!(out,v)` (identity default). Returns (x, relres, iters).
# Basis: m+1 CompositeVecs (caller-allocated pool to control VRAM); Hessenberg on host FP64.
function jfnk_fgmres!(x::CompositeVec, C::Composite2T, b::CompositeVec, matvec!::Function;
                      prec!::Function=(o,v)->ccopy!(o,v), m::Int=10, tol::Float64=1e-2,
                      pool::Vector{<:CompositeVec})
    @assert length(pool) >= 2m+2 "fgmres pool: need ≥ 2m+2 vectors"
    V = pool[1:m+1]; Z = pool[m+2:2m+2]                   # Arnoldi basis + preconditioned directions
    H = zeros(Float64, m+1, m); g = zeros(Float64, m+1)
    cscale!(0.0, x)
    r0 = cnorm(C, b); r0 == 0.0 && return (x, 0.0, 0)
    ccopy!(V[1], b); cscale!(1.0/r0, V[1]); g[1] = r0
    k = 0
    for j in 1:m
        k = j
        prec!(Z[j], V[j])
        matvec!(V[j+1], Z[j])
        for i in 1:j                                       # modified Gram-Schmidt (FP64 dots)
            H[i,j] = cdot(C, V[j+1], V[i])
            caxpy!(-H[i,j], V[i], V[j+1])
        end
        H[j+1,j] = cnorm(C, V[j+1])
        isfinite(H[j+1,j]) || return (x, NaN, j)          # matvec blew up — report, don't propagate
        H[j+1,j] > 1e-300 && cscale!(1.0/H[j+1,j], V[j+1])
        # solve the (j+1)×j least squares on host (tiny) for the current residual
        Hj = H[1:j+1,1:j]; gj = g[1:j+1]
        y = Hj \ gj
        res = sqrt(sum(abs2, gj .- Hj*y))/r0
        if res < tol || j == m
            for i in 1:j; caxpy!(y[i], Z[i], x); end
            return (x, res, j)
        end
    end
    (x, 1.0, k)
end

# One PTC-Newton outer step:  (diag(1/dtf) + J)·δU = −R(U);  U ← U + δU.
# `dt_scale!` applies the PTC diagonal as the tier-0 right preconditioner (δU ~ dtf·v).
# TIER-1 PRECONDITIONER (engineering block C, 2026-07-12): psweeps>0 upgrades M⁻¹ from the bare dtf
# diagonal to ν damped-Jacobi sweeps on the SAME PTC-shifted operator:  z₀ = dtf·v;
# z ← z + ω·dtf·(v − A·z).  Matrix-free (each sweep = 1 FD matvec), +1 pool vector. The point is
# MEMORY, not per-matvec compute: a stronger M⁻¹ holds the FGMRES restart m down (m=4 ⇒ 15+1 pool
# vectors ≈ 5 GB at chord-1800 — the production-Newton enabler; m=10 needed ~11 GB and did not fit).
# psweeps=0 (default) is bitwise the tier-0 path.
function jfnk_ptc_step!(C::Composite2T, dtfs::Vector, pool::Vector{<:CompositeVec};
                        m::Int=10, lintol::Float64=1e-2, trust::Float64=0.02,
                        psweeps::Int=0, pomega::Float64=0.8)
    need = 2m+7 + (psweeps>0 ? 1 : 0)
    @assert length(pool) >= need "ptc pool: need ≥ 2m+7 (+1 if psweeps>0) vectors"
    U0=pool[end-3]; R0=pool[end-2]; rhs=pool[end-1]; scratch=pool[end]
    cget!(U0, C)
    # REVIEW C1: the base evaluation MUST see the same BC-applied bands as every matvec/trial —
    # the L1 sponge is NOT idempotent, so an un-sponged base biases (R(U+eps v)-R0)/eps by a
    # constant/eps and makes the rejection-restore norm inconsistent with r_old.
    _set_consistent!(C, U0)
    composite_residual!(C); crates!(R0, C); cmask!(R0, C)  # base residual in the DOF subspace
    r_old = cnorm(C, R0)
    ccopy!(rhs, R0); cscale!(-1.0, rhs)                    # b = −R (masked)
    # A·v = v/dtf + J·v  (PTC-shifted operator), with J·v matrix-free
    mv!(o,v) = begin
        jfnk_matvec!(o, C, U0, R0, v, scratch)
        for l in 1:length(C.L), c in 1:4
            o.f[l][c] .= o.f[l][c] .+ v.f[l][c] ./ dtfs[l]
        end
        o
    end
    zr = psweeps>0 ? pool[end-5] : nothing                 # Jacobi-sweep residual scratch
    prec!(o,v) = begin                                     # tier-0 seed: o = dtf .* v (per-level fields)
        for l in 1:length(C.L), c in 1:4
            o.f[l][c] .= v.f[l][c] .* dtfs[l]
        end
        for _ in 1:psweeps                                 # tier-1: z ← z + ω·dtf·(v − A·z)
            mv!(zr, o)
            for l in 1:length(C.L), c in 1:4
                o.f[l][c] .= o.f[l][c] .+ Float32(pomega).*dtfs[l].*(v.f[l][c] .- zr.f[l][c])
            end
        end
        o
    end
    dU = pool[end-4]
    _,relres,kits = jfnk_fgmres!(dU, C, rhs, mv!; prec! = prec!, m=m, tol=lintol, pool=pool[1:2m+2])
    cmask!(dU, C)
    # trust region + BACKTRACKING with step rejection (globalization — the first bring-up run showed
    # partially-converged directions + unmasked updates walking the state off a cliff)
    nd_ = cnorm(C,dU)/sqrt(4*C.ndof)
    s = isfinite(nd_) && nd_>0 ? min(1.0, trust/nd_) : 0.0
    @printf("        [NWT-lin] relres=%.2e kits=%d  |dU|_rms=%.3e  damp0=%.3f
", relres, kits, nd_, s)
    r_new = r_old; accepted = false
    trial = scratch
    for bt in 0:3
        s <= 0 && break
        ccopy!(trial, U0); caxpy!(s, dU, trial)
        _set_consistent!(C, trial)
        composite_residual!(C); crates!(R0, C); cmask!(R0, C)
        r_new = cnorm(C, R0)
        if isfinite(r_new) && r_new < r_old
            accepted = true; caxpy!(s, dU, U0)            # commit: U0 ← U0 + s·δU
            break
        end
        @printf("        [NWT-bt ] s=%.4f rejected (r=%.3e ≥ %.3e)
", s, r_new/sqrt(4*C.ndof), r_old/sqrt(4*C.ndof))
        s *= 0.25
    end
    if !accepted
        _set_consistent!(C, U0)                            # restore the base state exactly
        composite_residual!(C); crates!(R0, C); cmask!(R0, C)
        return (cnorm(C, R0), false)
    end
    (r_new, true)
end
