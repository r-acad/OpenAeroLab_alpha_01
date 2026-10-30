# ==============================================================================
# Particles-on-Demand — gauge (reference-frame) transform   [path A, Phase 1]
# ==============================================================================
# Re-express the D3Q39 populations `f`, known in reference frame U_src, into frame
# U_dst, preserving ALL 13 hydrodynamic moments (mass, momentum, full 2nd-order
# stress, 3rd-order energy/heat flux) INCLUDING the non-equilibrium part. This is
# the operator the per-block binary map was missing — the moment-exact gauge
# transform f^{λ'} = 𝓜_{λ'}^{-1} 𝓜_λ f^λ (Dorschner-Bösch-Karlin, PRL 121,
# 130602 (2018), Eq. 4; LELBM Noh-Lee arXiv:2508.06911).
#
# Realized through the solver's existing entropic 13-moment machinery rather than a
# precomputed transfer matrix (the papers only give the closed-form matrix for
# D2Q9; for higher lattices they use the generic 𝓜_{λ'}^{-1}𝓜_λ — which IS the
# measure-moments → solve-λ → evaluate route below).
#
# CONVENTION: the lattice arrays store the BASE velocities c_i (the frame offset U
# is applied at streaming, not baked into cx_arr). So the peculiar velocity in any
# frame is c_i (⇒ pass shift = 0 to the moment/solve helpers), and the frame
# difference enters ONLY through the target moments: the peculiar velocity in frame
# U is u_lab − U, so M[2]/ρ from a frame-U population set equals (u_lab − U).
# Lab velocity u_lab = (peculiar velocity) + U is therefore preserved by construction.
# ==============================================================================

"""
    pond_gauge_transform(f_src, Usrc..., Udst..., T, cx_arr, cy_arr, cz_arr, w_arr,
                         cs2, eos_R, max_iter) -> (g::MVector{39}, converged::Bool)

Transform populations `f_src` (frame `U_src`) into frame `U_dst`. Preserves the 13
hydrodynamic moments (incl. non-equilibrium stress + heat flux); higher (ghost)
moments are projected to their max-entropy values — the intended Grad/entropic
projection. `converged=false` signals a non-realizable solve; the caller should
fall back (e.g. keep the source populations or use the equilibrium-only transform).
"""
@inline function pond_gauge_transform(
    f_src::MVector{39, Float32},
    Usrc_x::Float32, Usrc_y::Float32, Usrc_z::Float32,
    Udst_x::Float32, Udst_y::Float32, Udst_z::Float32,
    T::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32, eos_R::Float32, max_iter::Int32
)
    # 1. Moments of the source populations EXPRESSED IN THE DESTINATION FRAME.
    #    A population f[k] has lab velocity c_i + U_src; its velocity relative to the
    #    destination frame is (c_i + U_src) − U_dst = c_i − ΔU with ΔU = U_dst − U_src.
    #    `compute_13_moments_of_populations_d3q39(f, shift=ΔU)` evaluates exactly the
    #    13 peculiar-about-U_dst moments (the moments g MUST have), with the full
    #    non-equilibrium content carried automatically — no equilibrium split, no
    #    per-order binomial bookkeeping (which is where the naive carry went wrong on
    #    the 3rd-order heat flux). This IS 𝓜_λ f^λ measured in the λ' basis.
    dux = Udst_x - Usrc_x
    duy = Udst_y - Usrc_y
    duz = Udst_z - Usrc_z
    M = compute_13_moments_of_populations_d3q39(
        f_src, cx_arr, cy_arr, cz_arr, cs2, dux, duy, duz)
    rho = max(M[1], 1.0f-6)

    # 2. Reconstruct the destination populations reproducing those 13 moments exactly
    #    (𝓜_{λ'}^{-1}). The destination frame's peculiar velocity is c_i ⇒ solve/eval
    #    with shift 0; the frame difference is already baked into the target moments M.
    lam = solve_13mom_lambdas_explicit_d3q39(
        rho, M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9], M[10], M[11], M[12], M[13],
        cx_arr, cy_arr, cz_arr, w_arr, cs2, max_iter, 0.0f0, 0.0f0, 0.0f0, 1.0f0)
    converged = lam[14] > 0.5f0

    # 6. Evaluate the destination populations (basis c_i).
    g = MVector{39, Float32}(undef)
    @inbounds for k in Int32(1):Int32(39)
        g[k] = eval_13mom_feq_single(
            rho, lam,
            Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]),
            Float32(w_arr[k]), cs2)
    end
    return (g, converged)
end
