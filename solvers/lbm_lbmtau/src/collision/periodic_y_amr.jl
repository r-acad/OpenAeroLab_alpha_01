# ==============================================================================
# PERIODIC-Y AMR — compile-time gate and device helpers (27 Oct 2026)
# ==============================================================================
# Spanwise (Y) periodicity on EVERY AMR level (`advanced.boundary.periodic_y: true`
# together with `advanced.boundary.periodic_y_amr: true`).
#
# Two halves:
#   * CPU domain build (domain_topology.jl / domain.jl / domain_generation.jl):
#     runtime flag `periodic_y_amr_active()` — every level's neighbour table, the
#     halo / standoff / donor / grading / wake closures, the feather band, the
#     flood fill, the wall distances and the Bouzidi solid-neighbour test wrap Y.
#   * GPU/CPU kernels (streaming fallback into the parent, every coarse→fine
#     prolongation stencil, the B-spline prefilter, the Π_neq filter, the force
#     probe stencil): COMPILE-TIME const `_PERIODIC_Y_AMR`, read from the env var
#     LUDWIG_PERIODIC_Y_AMR at module load. src/preset_bootstrap.jl sets it from
#     the queued cases' `advanced.boundary.periodic_y_amr` BEFORE dependencies.jl
#     is included (the same mechanism as the solver_preset seam gates), and
#     config_loader.jl refuses a case whose config disagrees with the compiled
#     gate. With the gate off every helper below is the identity (the branches
#     constant-fold away), so the compiled kernels are the pre-change kernels:
#     existing cases are byte-identical by construction.
#
# The wrap: a level's Y period in cells is its full grid height
# (by_max · BLOCK_SIZE = ny_global at that level). The coarsest level's Y extent
# is a whole number of blocks by construction (physics_scaling.jl), so every
# finer level tiles the same period exactly (by_max doubles per level).
# ==============================================================================

const _PERIODIC_Y_AMR = get(ENV, "LUDWIG_PERIODIC_Y_AMR", "0") == "1"
_PERIODIC_Y_AMR && println("[Init] Periodic-Y AMR kernels COMPILED IN (LUDWIG_PERIODIC_Y_AMR=1)")

"""
    _pwrap_y(g, n) -> Int32

Wrap a 1-based global cell (or block) Y index into [1, n] when the periodic-Y
AMR gate is compiled in; identity otherwise. Works for any offset (fld-based),
so multi-cell stencils (B-spline ±3, prefilter ±24) wrap correctly.
"""
@inline function _pwrap_y(g::T, n::Integer) where {T<:Integer}
    if _PERIODIC_Y_AMR
        nn = T(n)
        nn > zero(T) && return g - nn * fld(g - one(T), nn)
    end
    return g
end

"""
    _pbracket_y(p_cont) -> (i0, i1, w)

Unclamped linear bracket for the periodic Y axis: i0 = floor(p_cont) may be 0
and i1 = i0 + 1 may be n + 1 — the stencil lookups wrap them. Only called when
the gate is compiled in (callers select it with `_PERIODIC_Y_AMR ? … : legacy`).
"""
@inline function _pbracket_y(p_cont::Float32)
    i0 = floor(Int32, p_cont)
    return i0, i0 + Int32(1), p_cont - Float32(i0)
end
