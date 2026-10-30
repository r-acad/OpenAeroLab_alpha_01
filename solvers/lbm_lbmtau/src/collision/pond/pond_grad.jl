# ==============================================================================
# Particles-on-Demand — closed-form Grad (Hermite) reconstruction   [shock-robust]
# ==============================================================================
# Replaces the entropic 13-moment NEWTON solve (which DIVERGES at a shock front,
# where adjacent cells' frame-velocity jump approaches the lattice manifold edge)
# with a NON-ITERATIVE truncated-Hermite (Grad) projection — always finite.
# Kallikounis-Dorschner-Karlin, PRE 106, 015301 (2022), arXiv:2203.05477, Eqs. 16,
# 19–22; shock schemes in Reyhanian et al. arXiv:2207.09404.
#
# 2nd-order Grad (mass + momentum + stress) — the order appropriate for D3Q39, which
# is NOT a Hermite tensor-product set (the 3rd-order a3 needs D3Q64; K22 §II C). The
# 2nd-order projection + a TVD-limited departure read is the recommended robust path
# for moderate-to-strong shocks on a non-product lattice.
#
# compute_13_moments_of_populations_d3q39(f, shift) already returns the HERMITE
# moments: M[1]=ρ, M[2..4]=momentum, M[5..10]=2nd-order Hermite stress
# (M5=Σf(cx²−cs²), M8=Σf cx cy, …). So the Grad coefficients ARE those moments and
# the reconstruction is a direct polynomial evaluation — no solve.
# ==============================================================================

"""
    pond_grad_eval(rho, j1,j2,j3, s5,s6,s7,s8,s9,s10, cx,cy,cz, w, cs2)

Closed-form 2nd-order Grad population for one direction:
  f_i = w_i [ ρ + (j·c_i)/cs² + (1/2cs⁴)·Σ s_αβ (c_iα c_iβ − cs²δ_αβ) ]
where (ρ, j, s) are the Hermite moments (M[1], M[2..4], M[5..10]). Floored positive.
"""
@inline function pond_grad_eval(rho::Float32, j1::Float32, j2::Float32, j3::Float32,
                                s5::Float32, s6::Float32, s7::Float32,
                                s8::Float32, s9::Float32, s10::Float32,
                                cx::Float32, cy::Float32, cz::Float32,
                                w::Float32, cs2::Float32)
    inv_cs2  = 1.0f0 / cs2
    inv_2cs4 = 0.5f0 / (cs2 * cs2)
    h1 = j1*cx + j2*cy + j3*cz
    h2 = s5*(cx*cx - cs2) + s6*(cy*cy - cs2) + s7*(cz*cz - cs2) +
         2.0f0*(s8*cx*cy + s9*cx*cz + s10*cy*cz)
    return max(w * (rho + inv_cs2*h1 + inv_2cs4*h2), 1.0f-12)
end

"""
    pond_grad_transform(f_src, Usrc..., Udst..., cx,cy,cz,w, cs2) -> MVector{39}

Shock-robust gauge transform: re-express f_src (frame U_src) in frame U_dst by
measuring its Hermite moments in the destination frame (shift ΔU = U_dst − U_src)
and reconstructing with the closed-form 2nd-order Grad. Non-iterative, never diverges.
"""
@inline function pond_grad_transform(
    f_src::MVector{39, Float32},
    Usrc_x::Float32, Usrc_y::Float32, Usrc_z::Float32,
    Udst_x::Float32, Udst_y::Float32, Udst_z::Float32,
    cx_arr, cy_arr, cz_arr, w_arr, cs2::Float32)
    dux = Udst_x - Usrc_x; duy = Udst_y - Usrc_y; duz = Udst_z - Usrc_z
    M = compute_13_moments_of_populations_d3q39(f_src, cx_arr, cy_arr, cz_arr, cs2, dux, duy, duz)
    g = MVector{39, Float32}(undef)
    @inbounds for k in Int32(1):Int32(39)
        g[k] = pond_grad_eval(M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9], M[10],
                              Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]),
                              Float32(w_arr[k]), cs2)
    end
    return g
end

"""
    pond_grad_collide(f, ω, cx,cy,cz,w, cs2) -> MVector{39}

Shock-robust co-moving collision: relax the non-equilibrium stress (the deviation of
the Hermite stress from its equilibrium value ρ u_α u_β) by (1−ω), conserve mass+momentum,
and reconstruct with the closed-form 2nd-order Grad. Single ω (BGK-class) here; per-order
splitting can be layered as in pond_collide. Non-iterative.
"""
@inline function pond_grad_collide(
    f::MVector{39, Float32}, omega::Float32,
    cx_arr, cy_arr, cz_arr, w_arr, cs2::Float32)
    M = compute_13_moments_of_populations_d3q39(f, cx_arr, cy_arr, cz_arr, cs2, 0.0f0, 0.0f0, 0.0f0)
    rho = max(M[1], 1.0f-6); inv = 1.0f0/rho
    ux = M[2]*inv; uy = M[3]*inv; uz = M[4]*inv
    # equilibrium Hermite stress = ρ u_α u_β (convective); non-eq = M − that
    e5 = rho*ux*ux; e6 = rho*uy*uy; e7 = rho*uz*uz
    e8 = rho*ux*uy; e9 = rho*ux*uz; e10 = rho*uy*uz
    om1 = 1.0f0 - omega
    s5  = e5  + om1*(M[5]  - e5);  s6  = e6  + om1*(M[6]  - e6);  s7  = e7  + om1*(M[7]  - e7)
    s8  = e8  + om1*(M[8]  - e8);  s9  = e9  + om1*(M[9]  - e9);  s10 = e10 + om1*(M[10] - e10)
    g = MVector{39, Float32}(undef)
    @inbounds for k in Int32(1):Int32(39)
        g[k] = pond_grad_eval(rho, M[2], M[3], M[4], s5, s6, s7, s8, s9, s10,
                              Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]),
                              Float32(w_arr[k]), cs2)
    end
    return g
end
