# Starter case template (Phase D, 2026-06-24)

The inputs-only workflow. **You specify the physics; the solver picks itself.**

## Use it
1. `cp -r CASES/TEMPLATE_starter CASES/Z<next>_<name>` (next sequential Z number).
2. Drop your STL into the new folder; set `basic.stl_file`.
3. Edit the physics: `flow.velocity`, `fluid.{kinematic_viscosity, speed_of_sound}`,
   `reference_chord`. That's what drives the auto-selection.
4. Point `cases_to_run.yaml` at your case and run `run.bat`.

## What gets auto-selected (from Ma & Re)
- **Lattice + operator**: Ma ≤ 0.6 → D3Q27+cumulant; Ma > 0.6 → D3Q39+entropic.
- **SGS**: D3Q27 at Re ≥ 1e5 → variable_nu_sgs + dynamic Germano (gold-standard) auto-on.
- The startup log prints the decision: `[Regime] Ma_phys = ... → regime = ... lattice = ...`.

## Overriding
Any explicitly-set key wins. Set `advanced.numerics.lattice` / `collision_operator` /
`variable_nu_sgs.enabled` to pin a choice. See the comments in `config.yaml`.

Full design: `docs/2026-06-24_solver_architecture_and_regime_selector.md`.
