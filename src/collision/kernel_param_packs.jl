# ==============================================================================
# Kernel parameter packs — Tier 1 Item 4 (C4 2026-05-28)
# ==============================================================================
# Reduces kernel argument count by bundling related static config scalars into
# `isbitstype` immutable structs passed as a single arg. The GPU's kernel-arg
# shadow registers are limited (~85 args at the cap); bundling frees register
# slots and reduces kernel launch overhead.
#
# Each pack:
#  - Is `isbits` (no heap allocation)
#  - Uses primitive Float32/Int32 fields only
#  - Stays small enough to be loaded by a single GPU vector instruction
# ==============================================================================

# ── SgsAndFeaturesPack — 11 SGS-related scalars + feature flags ────────────
# Packs:
#   c_pi_stress         — coupling const for |Π_neq^pre|/(ρ·cs²)
#   nu_sgs_floor        — minimum ν_sgs in smooth regions
#   sigma_min_p         — adaptive σ minimum (h-damping)
#   sigma_max_p         — adaptive σ maximum
#   sigma_pi_ref_p      — |Π_neq^pre| normalization threshold
#   feather_nu_max      — AMR feather-band extra ν cap
#   variable_nu_sgs_i   — 1 ⇒ per-cell ν_sgs from Π_neq; 0 ⇒ constant floor
#   variable_sigma_i    — 1 ⇒ per-cell σ from |Π_neq|; 0 ⇒ constant
#   sponge_xu_sagaut_i  — 1 ⇒ post-collision Xu-Sagaut sponge; 0 ⇒ legacy
#   knudsen_adaptive_i  — 1 ⇒ Latt 2020 Knudsen-adaptive τ; 0 ⇒ off
#   feather_active_i    — 1 ⇒ feather band active on this level
#
# Size: 11 × 4 = 44 bytes. Fits in a single GPU 16-byte load × 3.
#
# Live wiring (case 031+):
#   1. Add `sgs_cfg::SgsAndFeaturesPack` as a new LAST positional arg to all 4
#      collision kernel signatures (alongside the existing 11 individual args).
#   2. In kernel bodies, derive: `c_pi_stress = sgs_cfg.c_pi_stress`, etc.,
#      mirroring the Val pattern from the IS_L1 refactor. Comment-out the
#      direct uses of the existing per-scalar kernel args.
#   3. Verify LLVM dead-arg elim removes the 11 now-unused kernel args from
#      the compiled PTX (check via `CUDA.code_ptx`).
#   4. Once verified, remove the 11 unused args from kernel signatures and
#      the matching call-site positions in timestep.jl.
#
# Expected gain: ~3-5% on kernel-launch overhead (10 fewer arg-shadow regs).
# Not done now because the full refactor needs a clean validation cycle —
# done blind during a live production run would risk regressing case 030.
struct SgsAndFeaturesPack
    c_pi_stress::Float32
    nu_sgs_floor::Float32
    sigma_min_p::Float32
    sigma_max_p::Float32
    sigma_pi_ref_p::Float32
    feather_nu_max::Float32
    variable_nu_sgs_i::Int32
    variable_sigma_i::Int32
    sponge_xu_sagaut_i::Int32
    knudsen_adaptive_i::Int32
    feather_active_i::Int32
end

# ── MacroBoundsPack — configurable macroscopic realizability clamp bounds ──
# AUDIT FIX (2026-06-03): the D3Q39 macro realizability clamp had HARD-CODED
# density bounds (far-bulk floor 0.42, ceiling 2.10; near-wall 0.35 / 2.40).
# Those were hand-tuned for the transonic cylinder and are FATAL to any flow
# with legitimately low density: the Sod shock-tube right state (ρ=0.125) is
# clamped UP to 0.42 on the first step, destroying the benchmark. Strong
# expansions and supersonic pockets hit the same wall. These bounds are now
# config-driven (advanced.numerics.macro_density_floor / _ceil and the
# near-wall variants), defaulting to the legacy 0.42/2.10/0.35/2.40 so every
# existing aero case is byte-identical. Canonical low-density cases (Sod, Lax,
# Riemann) set physical bounds (e.g. floor 0.05).
# Size: 4 × 4 = 16 bytes — a single GPU vector load.
struct MacroBoundsPack
    rho_floor_far::Float32   # far-bulk density floor (legacy 0.42)
    rho_ceil_far::Float32    # far-bulk density ceiling (legacy 2.10)
    rho_floor_near::Float32  # near-wall density floor (legacy 0.35)
    rho_ceil_near::Float32   # near-wall density ceiling (legacy 2.40)
end
