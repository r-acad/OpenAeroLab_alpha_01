"""
AMR_RECOMPOSITION_PREVIEW.JL

Diagnostic-only preview of Palabos-style AMR recomposition.

Given staged decomposed transfer buffers, reconstruct the populations that a
future applied C2F/F2C operator would create from macro state plus Pi_neq stress
projection. The preview checks conservation, stress recovery, and positivity,
but it never writes into live solver arrays.
"""

@inline function _amr_recomp_state_matrix(buffers::AMRStagedTransferBuffers,
                                          state::Symbol)
    state == :t0 && return buffers.coarse_t0
    state == :t1 && return buffers.coarse_t1
    state == :t12 && return buffers.coarse_t12
    error("unknown AMR recomposition preview state: $state")
end

@inline function _amr_recomp_state_to_macro(state_matrix::Matrix{Float32},
                                            idx::Int)
    rho = max(state_matrix[AMR_DECOMP_RHO, idx], Float32(1e-12))
    inv_rho = 1.0f0 / rho
    ux = state_matrix[AMR_DECOMP_JX, idx] * inv_rho
    uy = state_matrix[AMR_DECOMP_JY, idx] * inv_rho
    uz = state_matrix[AMR_DECOMP_JZ, idx] * inv_rho
    T = max(state_matrix[AMR_DECOMP_T, idx], Float32(1e-6))
    return rho, ux, uy, uz, T
end

function _amr_recomp_preview_populations!(pop::Vector{Float32},
                                          state_matrix::Matrix{Float32},
                                          idx::Int;
                                          eq_pop=nothing)
    rho, ux, uy, uz, T = _amr_recomp_state_to_macro(state_matrix, idx)
    pxx = state_matrix[AMR_DECOMP_PIXX, idx]
    pyy = state_matrix[AMR_DECOMP_PIYY, idx]
    pzz = state_matrix[AMR_DECOMP_PIZZ, idx]
    pxy = state_matrix[AMR_DECOMP_PIXY, idx]
    pxz = state_matrix[AMR_DECOMP_PIXZ, idx]
    pyz = state_matrix[AMR_DECOMP_PIYZ, idx]

    n_dirs = min(Int(N_DIRS), length(pop))
    shift_x = Float32(LATTICE_SHIFT_X)
    shift_y = Float32(LATTICE_SHIFT_Y)
    shift_z = Float32(LATTICE_SHIFT_Z)
    cs2 = Float32(LATTICE_TYPE == :d3q39 ? 2.0f0 / 3.0f0 : 1.0f0 / 3.0f0)
    eps_pop = Float32(1e-10)
    compressible = Int32((COMPRESSIBILITY_CORRECTION || COMPRESSIBLE_MODE) ? 1 : 0)

    lambda_eq = nothing
    mass_corr_eq = 1.0f0
    if LATTICE_TYPE == :d3q39 && n_dirs == 39
        lambda_eq = solve_13mom_lambdas_d3q39(
            rho,
            ux - shift_x,
            uy - shift_y,
            uz - shift_z,
            C_X,
            C_Y,
            C_Z,
            W,
            cs2,
            Int32(80),
            T,
            shift_x,
            shift_y,
            shift_z,
            100.0f0
        )
        mass_corr_eq = compute_13mom_mass_correction_d3q39(
            rho,
            lambda_eq,
            C_X,
            C_Y,
            C_Z,
            W,
            cs2,
            shift_x,
            shift_y,
            shift_z
        )
    end

    min_pop = Inf32
    n_floored = 0

    @inbounds for k in 1:n_dirs
        cx_shifted = Float32(C_X[k])
        cy_shifted = Float32(C_Y[k])
        cz_shifted = Float32(C_Z[k])
        cx_rel = cx_shifted - shift_x
        cy_rel = cy_shifted - shift_y
        cz_rel = cz_shifted - shift_z
        wk = Float32(W[k])

        feq = if LATTICE_TYPE == :d3q39 && n_dirs == 39
            mass_corr_eq * eval_13mom_feq_single(
                rho,
                lambda_eq,
                cx_rel,
                cy_rel,
                cz_rel,
                wk,
                cs2
            )
        else
            calculate_equilibrium_auto(
                rho,
                ux,
                uy,
                uz,
                wk,
                cx_shifted,
                cy_shifted,
                cz_shifted,
                compressible
            )
        end

        stress_pop = project_pi_neq_stress_mode(
            wk,
            cx_rel,
            cy_rel,
            cz_rel,
            cs2,
            pxx,
            pyy,
            pzz,
            pxy,
            pxz,
            pyz
        )
        raw_pop = feq + stress_pop
        if raw_pop < eps_pop
            n_floored += 1
        end
        if eq_pop !== nothing
            eq_pop[k] = feq
        end
        pop[k] = max(raw_pop, eps_pop)
        min_pop = min(min_pop, raw_pop)
    end

    return Float64(min_pop), n_floored
end

function _amr_recomp_preview_entity(state_matrix::Matrix{Float32}, idx::Int)
    rho, _, _, _, _ = _amr_recomp_state_to_macro(state_matrix, idx)
    pxx = state_matrix[AMR_DECOMP_PIXX, idx]
    pyy = state_matrix[AMR_DECOMP_PIYY, idx]
    pzz = state_matrix[AMR_DECOMP_PIZZ, idx]
    pxy = state_matrix[AMR_DECOMP_PIXY, idx]
    pxz = state_matrix[AMR_DECOMP_PIXZ, idx]
    pyz = state_matrix[AMR_DECOMP_PIYZ, idx]

    n_dirs = Int(N_DIRS)
    shift_x = Float32(LATTICE_SHIFT_X)
    shift_y = Float32(LATTICE_SHIFT_Y)
    shift_z = Float32(LATTICE_SHIFT_Z)
    cs2 = Float32(LATTICE_TYPE == :d3q39 ? 2.0f0 / 3.0f0 : 1.0f0 / 3.0f0)

    pop = zeros(Float32, n_dirs)
    eq_pop = zeros(Float32, n_dirs)
    min_pop, n_floored = _amr_recomp_preview_populations!(
        pop, state_matrix, idx; eq_pop=eq_pop)
    rho_rec = 0.0f0
    jx_rec = 0.0f0
    jy_rec = 0.0f0
    jz_rec = 0.0f0
    pxx_rec = 0.0f0
    pyy_rec = 0.0f0
    pzz_rec = 0.0f0
    pxy_rec = 0.0f0
    pxz_rec = 0.0f0
    pyz_rec = 0.0f0

    @inbounds for k in 1:n_dirs
        cx_shifted = Float32(C_X[k])
        cy_shifted = Float32(C_Y[k])
        cz_shifted = Float32(C_Z[k])
        cx_rel = cx_shifted - shift_x
        cy_rel = cy_shifted - shift_y
        cz_rel = cz_shifted - shift_z
        f = pop[k]
        feq = eq_pop[k]

        rho_rec += f
        jx_rec += f * cx_shifted
        jy_rec += f * cy_shifted
        jz_rec += f * cz_shifted

        fneq = f - feq
        qxx = cx_rel * cx_rel - cs2
        qyy = cy_rel * cy_rel - cs2
        qzz = cz_rel * cz_rel - cs2
        pxx_rec += qxx * fneq
        pyy_rec += qyy * fneq
        pzz_rec += qzz * fneq
        pxy_rec += cx_rel * cy_rel * fneq
        pxz_rec += cx_rel * cz_rel * fneq
        pyz_rec += cy_rel * cz_rel * fneq
    end

    drho = Float64(rho_rec - rho)
    djx = Float64(jx_rec - state_matrix[AMR_DECOMP_JX, idx])
    djy = Float64(jy_rec - state_matrix[AMR_DECOMP_JY, idx])
    djz = Float64(jz_rec - state_matrix[AMR_DECOMP_JZ, idx])
    dpi_xx = Float64(pxx_rec - pxx)
    dpi_yy = Float64(pyy_rec - pyy)
    dpi_zz = Float64(pzz_rec - pzz)
    dpi_xy = Float64(pxy_rec - pxy)
    dpi_xz = Float64(pxz_rec - pxz)
    dpi_yz = Float64(pyz_rec - pyz)

    return (
        abs_drho = abs(drho),
        abs_dj = sqrt(djx*djx + djy*djy + djz*djz),
        abs_dpi = sqrt(dpi_xx*dpi_xx + dpi_yy*dpi_yy + dpi_zz*dpi_zz +
                       2.0 * (dpi_xy*dpi_xy + dpi_xz*dpi_xz + dpi_yz*dpi_yz)),
        min_pop = Float64(min_pop),
        n_floored = n_floored
    )
end

function summarize_amr_recomposition_preview(registry::AMRInterfaceRegistry,
                                             buffers::AMRStagedTransferBuffers;
                                             state::Symbol=:t12)
    state_matrix = _amr_recomp_state_matrix(buffers, state)

    n_pairs = 0
    abs_drho_sum = 0.0
    abs_dj_sum = 0.0
    abs_dpi_sum = 0.0
    max_abs_drho = 0.0
    max_abs_dj = 0.0
    max_abs_dpi = 0.0
    min_population = Inf
    n_floored_populations = 0

    n_face = 0
    n_edge = 0
    n_corner = 0
    abs_dpi_face_sum = 0.0
    abs_dpi_edge_sum = 0.0
    abs_dpi_corner_sum = 0.0

    @inbounds for idx in eachindex(registry.entities)
        (buffers.coarse_valid[idx] && buffers.child_valid[idx]) || continue
        entity = registry.entities[idx]
        p = _amr_recomp_preview_entity(state_matrix, idx)

        n_pairs += 1
        abs_drho_sum += p.abs_drho
        abs_dj_sum += p.abs_dj
        abs_dpi_sum += p.abs_dpi
        max_abs_drho = max(max_abs_drho, p.abs_drho)
        max_abs_dj = max(max_abs_dj, p.abs_dj)
        max_abs_dpi = max(max_abs_dpi, p.abs_dpi)
        min_population = min(min_population, p.min_pop)
        n_floored_populations += p.n_floored

        if entity.topology == Int8(1)
            n_face += 1
            abs_dpi_face_sum += p.abs_dpi
        elseif entity.topology == Int8(2)
            n_edge += 1
            abs_dpi_edge_sum += p.abs_dpi
        elseif entity.topology == Int8(3)
            n_corner += 1
            abs_dpi_corner_sum += p.abs_dpi
        end
    end

    return (
        state = state,
        n_pairs = n_pairs,
        mean_abs_drho = n_pairs > 0 ? abs_drho_sum / n_pairs : 0.0,
        mean_abs_dj = n_pairs > 0 ? abs_dj_sum / n_pairs : 0.0,
        mean_abs_dpi = n_pairs > 0 ? abs_dpi_sum / n_pairs : 0.0,
        max_abs_drho = max_abs_drho,
        max_abs_dj = max_abs_dj,
        max_abs_dpi = max_abs_dpi,
        min_population = n_pairs > 0 ? min_population : 0.0,
        n_floored_populations = n_floored_populations,
        n_face_pairs = n_face,
        n_edge_pairs = n_edge,
        n_corner_pairs = n_corner,
        mean_abs_dpi_face = n_face > 0 ? abs_dpi_face_sum / n_face : 0.0,
        mean_abs_dpi_edge = n_edge > 0 ? abs_dpi_edge_sum / n_edge : 0.0,
        mean_abs_dpi_corner = n_corner > 0 ? abs_dpi_corner_sum / n_corner : 0.0
    )
end

function write_amr_recomposition_preview_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,state,n_pairs," *
            "mean_abs_drho,mean_abs_dj,mean_abs_dpi," *
            "max_abs_drho,max_abs_dj,max_abs_dpi," *
            "min_population,n_floored_populations," *
            "n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "mean_abs_dpi_face,mean_abs_dpi_edge,mean_abs_dpi_corner")
    end
end

function append_amr_recomposition_preview_csv_row(filepath::String, step::Int,
                                                  parent_level::Int, child_level::Int,
                                                  summary)
    if !isfile(filepath)
        write_amr_recomposition_preview_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%s,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%d,%d,%.9e,%.9e,%.9e\n",
            step,
            parent_level,
            child_level,
            String(summary.state),
            summary.n_pairs,
            summary.mean_abs_drho,
            summary.mean_abs_dj,
            summary.mean_abs_dpi,
            summary.max_abs_drho,
            summary.max_abs_dj,
            summary.max_abs_dpi,
            summary.min_population,
            summary.n_floored_populations,
            summary.n_face_pairs,
            summary.n_edge_pairs,
            summary.n_corner_pairs,
            summary.mean_abs_dpi_face,
            summary.mean_abs_dpi_edge,
            summary.mean_abs_dpi_corner)
    end
end

function amr_recomposition_preview_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_recomposition_preview_history.csv")
end
