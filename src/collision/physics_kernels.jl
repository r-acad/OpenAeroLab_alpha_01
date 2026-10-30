# ==============================================================================
# COLLISION OPERATOR REGISTRY
# ==============================================================================
# This file includes all available collision operator kernels and provides
# the dispatch function used by the timestep driver.
#
# To add a new collision operator:
#   1. Create a new file: kernel_<name>.jl  (copy an existing one as template)
#   2. Define a kernel function: stream_collide_<name>!(...)
#      with either the legacy shared signature or a purpose-built C4 variant
#      tuple wired in timestep.jl
#   3. Include the file below
#   4. Add a dispatch entry in `get_collision_kernel`
#   5. Add the operator symbol to VALID_COLLISION_OPERATORS in config_loader.jl
# ==============================================================================

using KernelAbstractions

# Tier 1 Item 4 (param-packing scaffold, 2026-05-28). Provides immutable
# `isbits` structs for bundling related kernel scalar args. Live wiring is the
# documented next step — see kernel_param_packs.jl header for the rollout plan.
include("kernel_param_packs.jl")

# --- Include all operator kernels ---
# (2026-06-24 consolidation: D3Q27 legacy entropic + regularized_bgk kernels removed;
#  cumulant is the sole D3Q27 operator, entropic_d3q39 is the D3Q39 operator.)
include("kernel_dynamic_sgs.jl")              # Dynamic Smagorinsky (Germano 1991, Lilly 1992) — used by cumulant + entropic_d3q39
include("kernel_cumulant.jl")
include("kernel_regularization_d3q39.jl")   # RR-LBM Hermite projection (Latt-Chopard 2006, Coreixas 2017)
include("kernel_regularization_d3q19.jl")   # RR-LBM D3Q19 projection — KEPT as the D3Q19 seed (2026-06-24); rebuild-from-here
# (9 D3Q19 RR-BGK / AA-pattern implementation kernels removed 2026-06-24 with the D3Q19 retirement.
#  Seed kept: build_d3q19_static() in lattice.jl + kernel_regularization_d3q19.jl above.)
include("kernel_svv_yz_d3q39.jl")            # SVV Y-Z bi-Laplacian (Tadmor 1989, Karamanos 2000); also hosts the D3Q27 SVV launch
include("kernel_nscbc_outlet.jl")             # NSCBC outflow BC (Poinsot-Lele 1992, Heubes 2014)
include("kernel_larsson_pgr.jl")              # Larsson 2010 wall-model PG correction (B6 ext)
include("kernel_quintic_prolongation.jl")     # B1 Pillar 1: cell-averaged quartic mortar prolongation scaffold
include("kernel_gendre_normal_gradient.jl")   # Gendre 2017: AMR-interface normal-gradient correction (spurious-acoustic fix)
include("kernel_entropic_d3q39.jl")
include("kernel_thermal.jl")

# --- Kernel dispatch ---
"""
    get_collision_kernel(operator::Symbol, backend, is_l1::Bool=false)

Return the compiled kernel function for the given collision operator.

Supported operators (2026-06-24 consolidation — two solvers):
  - :cumulant          — Cumulant collision, the sole D3Q27 operator (Ma ≤ 0.6)
  - :entropic (D3Q39)  — KBC with numerical equilibrium, the D3Q39 operator (Ma > 0.6)
"""
function get_collision_kernel(operator::Symbol, backend, is_l1::Bool=false)
    # (D3Q19 dispatch removed 2026-06-24 with the D3Q19 retirement. Two active solvers:
    #  D3Q27 + cumulant, D3Q39 + entropic.)
    if operator == :cumulant
        if LATTICE_TYPE == :d3q39
            error("[Collision] Cumulant operator is not compatible with D3Q39 lattice. " *
                  "The Chimera transform requires 3×3×3 (D3Q27) structure. " *
                  "D3Q39 requires collision_operator: entropic")
        end
        # D3Q27 cumulant — the sole D3Q27 collision operator (2026-06-24 consolidation).
        return stream_collide_cumulant!(backend)
    elseif operator == :entropic
        # :entropic is the D3Q39 (compressible) operator. The legacy D3Q27 entropic
        # kernel was removed in the 2026-06-24 consolidation (D3Q27 uses cumulant only).
        if LATTICE_TYPE == :d3q39
            return stream_collide_entropic_d3q39!(backend)
        else
            error("[Collision] collision_operator: entropic is only valid with lattice: d3q39. " *
                  "For D3Q27 use collision_operator: cumulant.")
        end
    else
        error("[Collision] Unknown or retired operator: :$operator. " *
              "Valid options: :cumulant (D3Q27), :entropic (D3Q39).")
    end
end
# (get_d3q19_* kernel factory functions removed 2026-06-24 with the D3Q19 retirement.)
