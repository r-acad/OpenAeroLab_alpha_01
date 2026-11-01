# OPEN Ludwig

GPU accelerated CFD solvers and the local Ludwig Workbench web app. The repository
contains the LBM and C mesh solver source, the workbenches, their runtime tools, and the
versioned pressure based and LBM solver sources used by the app.

## Run Ludwig Workbench

Install Julia 1.13, then instantiate the bundled JXfoil environment:

```powershell
julia --project=third_party/JXfoil -e 'using Pkg; Pkg.instantiate()'
apps\ludwig_web\run.cmd
```

The launcher normally opens at `http://127.0.0.1:8058/`. Open Aero Lab for 3D
LBM preparation and results, or Airfoil Lab for section analyses. Airfoil Lab
also has its standalone `apps\airfoil_web\run.cmd` launcher. The workbenches
keep generated runs, uploaded geometry, and downloaded airfoil coordinates in a sibling
`02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE` directory. Set
`LUDWIG_PRIVATE_ROOT` to use another private location. The pressure based and
LBM modes use the bundled, SHA checked solver sources under `solvers/`.

Experimental and published reference datasets are stored separately and are
optional. Matched experimental comparisons and the two cavitation reference
presets require those datasets. The XFOIL executable is also optional; set
`AIRFOIL_XFOIL_EXE` to its installed path. JXfoil runs from the bundled source.

## Run the command line solver

Instantiate `src/Project.toml`. Copy `CASES/TEMPLATE_starter` to a new case,
provide an STL file, and edit its `config.yaml`. Copy
`cases_to_run.example.yaml` to `cases_to_run.yaml`, set the case name, then run
`run.bat` on Windows or `julia --project=src src/main.jl` elsewhere.

The app's default solver modes are development stage models. Inspect each
result's convergence and mesh diagnostics before drawing physical conclusions.

## Project layout

- `src/`: solver source and command line entry point. See `src/README.md` for the solver map; compressible C-mesh and Species cavitation now have separate source directories.
- `apps/ludwig_web/`: top-level launcher and Aero Lab.
- `apps/airfoil_web/`: Airfoil Lab app and static assets, also hosted by the launcher.
- `tools/`: tools required by the production cavitation driver.
- `solvers/`: checksum pinned LBM and pressure-solver snapshots used by app backends; the other solvers live under `src/` and `third_party/`.
- `third_party/JXfoil/`: bundled JXfoil package.
- `CASES/TEMPLATE_starter/`: case input template.

Agent notes, tests, validation runs, papers, external reference data, and
temporary output are kept outside this repository in the sibling
`02_PRIVATE_PROJECT_DEVELOPMENT` directory. See `LICENSE` for AGPL-3.0 terms.
