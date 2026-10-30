# ==============================================================================
# REGIME SELECTOR (Phase C, 2026-06-24)
# ==============================================================================
# Maps the user's physical inputs (velocity, characteristic length, fluid) to the
# best solver + the gold-standard feature set NEEDED for that flow regime — "no
# more, no less" — with per-setting user override. See the locked plan in
# docs/2026-06-24_solver_architecture_and_regime_selector.md.
#
# Two solvers (D3Q19 retired 2026-06-24):
#   Ma ≤ MA_REGIME_THRESHOLD (0.6) → D3Q27 + cumulant   (incompressible high-Re)
#   Ma >  MA_REGIME_THRESHOLD       → D3Q39 + entropic   (compressible/transonic/supersonic)
#
# OVERRIDE MODEL: the selector only sets a config value when the user did NOT set
# that key explicitly in their case YAML (detected via config_has_path). Any
# explicitly-set key wins. So the user can pin the lattice, the operator, or any
# individual feature flag and the selector respects it.
# ==============================================================================

# The single tunable boundary between the incompressible (D3Q27) and compressible
# (D3Q39) regimes. 0.6 = the upper bound of D3Q27 + compressibility_correction
# validity (the O(u⁴) equilibrium). User decision 2026-06-24.
const MA_REGIME_THRESHOLD = 0.6

# Reynolds number above which the gold-standard Π_neq adaptive SGS (variable_nu_sgs +
# dynamic Germano) is auto-enabled on D3Q27 (when the user didn't set it). 1e5 ≈ the
# onset of genuinely turbulent, under-resolvable wakes/BLs where WALE-only under-
# dissipates. Below this, laminar/transitional cases keep the leaner default.
const REGIME_RE_HIGH_RE_THRESHOLD = 1.0e5

"""
    classify_regime(Ma::Real) -> Symbol

Return the flow regime for a physical Mach number:
  :incompressible  (Ma ≤ 0.6)  → D3Q27 + cumulant
  :compressible    (Ma > 0.6)  → D3Q39 + entropic (transonic / supersonic)
"""
function classify_regime(Ma::Real)
    return Ma <= MA_REGIME_THRESHOLD ? :incompressible : :compressible
end

"""
    select_solver(regime::Symbol) -> (lattice::Symbol, operator::Symbol)

Map a regime to its (lattice, collision_operator) pair.
"""
function select_solver(regime::Symbol)
    if regime === :incompressible
        return (:d3q27, :cumulant)
    elseif regime === :compressible
        return (:d3q39, :entropic)
    else
        error("[RegimeSelector] unknown regime :$regime")
    end
end

"""
    apply_regime_selection!(CFG; Ma, Re) -> nothing

The top-level entry, called from load_case_configuration AFTER FLOW_VELOCITY and
the fluid speed-of-sound are known but BEFORE LATTICE_TYPE / COLLISION_OPERATOR are
finalized. Resolves the lattice + operator from (Ma) when the user left them unset,
and records the auto-selection for the per-feature defaults that follow.

Returns a NamedTuple `(regime, lattice, operator, lattice_auto, operator_auto)` so
the caller can apply regime-appropriate feature defaults (each gated on its own
explicit-key check). `*_auto` = true when the selector chose it (user didn't pin it).
"""
function resolve_regime_solver(CFG, Ma::Real, Re::Real)
    regime = classify_regime(Ma)
    sel_lattice, sel_operator = select_solver(regime)

    lattice_explicit  = config_has_path(CFG, "advanced", "numerics", "lattice")
    operator_explicit = config_has_path(CFG, "advanced", "numerics", "collision_operator")

    lattice  = lattice_explicit  ? Symbol(safe_get(CFG, "advanced", "numerics", "lattice")) : sel_lattice
    operator = operator_explicit ? Symbol(safe_get(CFG, "advanced", "numerics", "collision_operator")) : sel_operator

    # If the user pinned the LATTICE but not the operator, derive the operator from
    # the pinned lattice (each lattice has exactly one valid operator post-2026-06-24).
    if lattice_explicit && !operator_explicit
        operator = lattice === :d3q39 ? :entropic : :cumulant
    end

    @printf("[Regime] Ma_phys = %.3f, Re = %.3g → regime = %s\n", Float64(Ma), Float64(Re), String(regime))
    if lattice_explicit
        @printf("[Regime]   lattice = %s (user-pinned)\n", String(lattice))
    else
        @printf("[Regime]   lattice = %s (auto: Ma %s %.2f)\n", String(lattice),
                Ma <= MA_REGIME_THRESHOLD ? "≤" : ">", MA_REGIME_THRESHOLD)
    end
    @printf("[Regime]   collision_operator = %s (%s)\n", String(operator),
            operator_explicit ? "user-pinned" : "auto from lattice")

    return (regime = regime, lattice = lattice, operator = operator,
            lattice_auto = !lattice_explicit, operator_auto = !operator_explicit)
end

"""
    regime_feature_default(CFG, regime, keys...; incompressible, compressible) -> value

Helper for per-feature regime defaults with user-override. Returns the user's
explicitly-set value when `keys` are present in CFG; otherwise the regime-appropriate
default (`incompressible` or `compressible`). Used by load_case_configuration to set
the ~12 user-facing feature flags per the matrix in the plan doc.
"""
function regime_feature_default(CFG, regime::Symbol, keys::Vararg{String};
                                incompressible, compressible)
    if config_has_path(CFG, keys...)
        return safe_get(CFG, keys...)
    end
    return regime === :compressible ? compressible : incompressible
end
