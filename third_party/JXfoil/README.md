# JXfoil

`JXfoil` is a Julia reimplementation of the core inviscid XFoil workflow:

- Airfoil loading from coordinate files
- NACA 4-digit geometry generation
- 2D XFoil-style node-based linear-vorticity inviscid panel solve
- XFoil polar parsing and cross-validation harness
- Viscous closure scaffolding (ported correlations from `xblsys.f`)
- BL interval residual scaffolding (`BLDIF`-style residuals for similarity/laminar/turbulent/wake)
- First Julia viscous marching solver (interval Newton solve on each surface using inviscid `Cp -> Ue`)
- Quasi-coupled viscous iteration with displacement-thickness feedback to edge-velocity profiles
- Coupled XFoil-style VISCAL Newton solver with wake coupling
- Transition-interval coupling with a TRDIF-inspired mixed residual solve (and split-march fallback), plus `DAMPL2` support in amplification-rate setup
- Algorithmic force/moment integration using an XFoil `CLCALC`-style pressure integration form
- Algorithmic viscous drag assembly from BL skin-friction integration and Squire-Young wake extrapolation form
- XFoil viscous-state dump parsing plus Julia-vs-XFoil internal BL state comparison coverage
- Transition location reported directly from marched BL transition solve (`xtr=1.0` if no trip)
- Public viscous/coupled result structs include solver provenance so exact-solver fallback selections are explicit when a case does not converge cleanly, parity output now flags XFoil-nonconverged / Julia-converged cases as the ideal outcome instead of treating them as bug targets, and detailed validation now marks XFoil public-output artifacts as `XF-PUB-ROUND(...)` / `XF-PUB-TRUNC(...)` instead of treating them as solver mismatches
- Solver entry points accept `xfoil_load_normalize_digits` when parity work needs Julia to mimic XFoil's text-file `LOAD` precision before repaneling
- `Pkg.test()` now supports a fast tier via `JXFOIL_TEST_TIER=fast`, so CI can keep PR coverage quick while reserving the heaviest parity/state-comparison blocks for nightly runs

Current status:
- Automated cross-validation runs both `build/bin/xfoil.exe` and the Julia solver.
- Public `solve_viscous_case` and `solve_coupled_case` now both default to the exact wake-coupled VISCAL solver path.
- First viscous baseline case is automated through XFoil (`e387`, `Re=100k`, `alpha=5`) with tolerance checks.
- Julia viscous mode predicts `CL/CD/CM` plus transition `xtr` directly from marched BL states with no post-fit parity calibration.
- The coupled `solve_coupled_case_viscal` path now matches XFoil to near dump-level precision on the exact VISCAL parity suite.
- `CL/CM` are integrated from pressure using an XFoil `CLCALC`-style expression on the node-based linear-vorticity solve.
- `CD` uses BL skin-friction integration plus XFoil-style wake-coupled Squire-Young extrapolation from the converged BL/wake state.
- Public inviscid, viscous, and viscous polar outputs are now at near-executable parity on the bundled validation suites; the remaining meaningful non-parity is concentrated in flagged frontier cases where XFoil itself is unhealthy rather than the default exact-coupled public solver.

Latest validated parity snapshot (2026-03-29):
- Full validation status: `PASS`
- Inviscid default suite: max `|dCL| = 4.7e-5`, max `|dCM| = 4.9e-5`
- Inviscid extended suite: max `|dCL| = 4.7e-5`, max `|dCM| = 4.9e-5`
- Viscous comparison default: max `|dCL| = 4.5e-5`, max `|dCD| = 4.0e-6`, max `|dCM| = 3.5e-5`
- Viscous comparison extended: max `|dCL| = 4.5e-5`, max `|dCD| = 5.0e-6`, max `|dCM| = 4.0e-5`
- Viscous comparison coupled: max `|dCL| = 4.5e-5`, max `|dCD| = 4.0e-6`, max `|dCM| = 3.5e-5`
- Viscous comparison coupled extended: max `|dCL| = 4.5e-5`, max `|dCD| = 5.0e-6`, max `|dCM| = 4.0e-5`
- Polar sweep validation: max `rmse(dCL) = 3.4e-5`, max `dCL = 4.7e-5`, max `rmse(dCD) = 4.0e-6`
- Last full extreme parity report: Julia converged on `158/158` cases, including `112/112` healthy XFoil references and `46/46` frontier cases; `41` cases are explicitly flagged `IDEAL-XF-NCV`, and `XF-NCV/JX-BEST` is now `0`
- The extreme harness has since been expanded with harsher low-`Ncrit` and flap corners, plus frontier sanity scoring and machine-readable summaries; the larger matrix is intended to be tracked by the nightly CI lane
- Report: `JXfoil/reports/VALIDATION_REPORT.md`

## Run

1. Build headless XFoil executable from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_xfoil_headless.ps1
```

2. Run Julia tests/regression:

```powershell
julia --project=JXfoil -e "using Pkg; Pkg.test()"
```

3. Run the fast validation lane used by PR CI:

```powershell
julia --project=JXfoil JXfoil/scripts/run_fast_validation.jl
```

4. Or run only regression comparison:

```powershell
julia --project=JXfoil JXfoil/scripts/run_regression.jl
```

5. Run viscous baseline check:

```powershell
julia --project=JXfoil JXfoil/scripts/run_viscous_baseline.jl
```

6. Run viscous Julia-vs-XFoil comparison suite:

```powershell
julia --project=JXfoil JXfoil/scripts/run_viscous_comparison.jl
```

7. Run full validation and generate report:

```powershell
julia --project=JXfoil JXfoil/scripts/run_full_validation.jl
```

Report path:

- `JXfoil/reports/VALIDATION_REPORT.md`
- `JXfoil/reports/VALIDATION_SUMMARY.json`
- `JXfoil/reports/VALIDATION_SUMMARY.tsv`

## Front end

Launch the local web front end from the repository root:

```powershell
julia --project=JXfoil JXfoil/scripts/run_frontend.jl
```

Optional environment overrides:

```powershell
$env:JXFOIL_FRONTEND_PORT = "8057"
$env:JXFOIL_FRONTEND_OPEN_BROWSER = "0"
julia --project=JXfoil JXfoil/scripts/run_frontend.jl
```

The app includes:

- searchable UIUC / Selig airfoil selection sourced from `https://m-selig.ae.illinois.edu/ads/coord_database.html`
- optional built-in NACA 4-digit generation
- Julia-only, XFoil-only, or compare mode
- geometry, operating-point coefficients, Cp, transition, boundary-layer development, and multi-Re polar plots
- optional flap geometry generation through XFoil's `GDES -> FLAP -> EXEC -> SAVE` path, with both solvers run on the same resulting coordinates

8. Run dedicated polar-sweep validation:

```powershell
julia --project=JXfoil JXfoil/scripts/run_polar_validation.jl
```

9. Run coupled viscous comparison suite:

```powershell
julia --project=JXfoil JXfoil/scripts/run_coupled_viscous_comparison.jl
```

10. Run coupled default + extended validation suites:

```powershell
julia --project=JXfoil JXfoil/scripts/run_coupled_validation_suite.jl
```
