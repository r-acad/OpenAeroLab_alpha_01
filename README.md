# OpenAeroLab

GPU accelerated CFD solvers and the local Ludwig Workbench web app. The repository
contains the LBM and C mesh solver source, the workbenches, their runtime tools, and the
versioned pressure based and LBM solver sources used by the app.

## Install and launch

Download the **complete repository** using **Code → Download ZIP**, extract it to a
writable folder, then run setup at the repository root. Git, Python and Node.js are
not required. Keep the setup console open until it reports **Setup complete**.

| Operating system | One-time setup | Launch | Guide |
| --- | --- | --- | --- |
| Windows 64-bit | Double-click `setup.cmd` | `__Start here!__/Windows/OpenAeroLab.cmd` | [Windows instructions](__Start%20here!__/Windows/README.md) |
| Linux x86-64 / ARM64 | `bash setup.sh` | `bash "__Start here!__/Linux/OpenAeroLab.sh"` | [Linux instructions](__Start%20here!__/Linux/README.md) |
| macOS Intel / Apple Silicon | `bash setup.command` | `bash "__Start here!__/Mac/OpenAeroLab.command"` | [Mac instructions](__Start%20here!__/Mac/README.md) |

Setup reuses the required **Julia 1.13.2**, or downloads an official archive and
verifies its SHA-256 before installing it privately. It automatically downloads
the Julia packages and binary artifacts for the web workbenches, JXfoil,
supported solver backends, and CairoMakie reports/renders. It then precompiles
packages and verifies each runtime. Subsequent setup runs reuse downloaded packages;
run setup again after updating the repository. No shared system image is required.

The current CUDA solvers require an NVIDIA GPU and a compatible installed NVIDIA
driver on **Windows or Linux**. The pinned solver environments select CUDA 13.0;
their toolkit libraries download automatically, but system GPU drivers are not
installed by setup. On Windows, setup also installs the
[Microsoft Visual C++ x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe)
when missing, after checking Microsoft's signature; Windows may request administrator authorization.
macOS setup installs the workbenches, JXfoil and Makie;
CUDA solver environments are skipped. Model placement and saved-result viewing
remain available on Mac; Apple/AMD GPU solver support is not implemented.

See [setup options, storage and troubleshooting](tools/install/README.md).

The launcher normally opens at `http://127.0.0.1:8058/` in your default browser. Open Aero Lab for 3D
LBM preparation and results, or Airfoil Lab for section analyses. Airfoil Lab
also has its standalone `apps\airfoil_web\run.cmd` launcher. The workbenches
keep runs, uploaded geometry and result imports in the sibling
`02_PRIVATE_PROJECT_DEVELOPMENT/results` directory; studies stay in `studies`.
Disposable previews, downloads, staging files and logs use
`02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE`. Set
`LUDWIG_PRIVATE_ROOT` to use another private location. The pressure based and
LBM modes use the bundled, SHA checked solver sources under `solvers/`.

For compressible CFD and the sharp-mesh pressure solver, **Outputs → Save field
snapshots every, minutes** controls elapsed real time, independently of iteration
history updates. The default is 10 minutes; zero disables periodic snapshots.
The timer starts when marching begins and saves at the next committed step after
the interval. Final fields are always saved. **Results → Flow field → Saved
snapshot** opens any saved frame, including after reopening a study. These are
visualization fields, not restart checkpoints. The low-speed C+H adapter remains
final-field-only; LBM and species output retain their existing iteration controls.
Snapshot writes report their elapsed cost in the log, retain all successfully
saved frames, and refuse a write that would leave less than 1 GiB free.

Never put useful source, validation evidence, references, studies or results in
`TEMP_DELETEABLE`. Stop the workbenches and Telegram bridge before clearing it.
The Julia utility `tools/workspace/clean_temp.jl` previews cleanup; `--delete`
performs it and protects registered worktrees and running services. Run it with
`julia --project=third_party/JXfoil`. Auxiliary checkouts must be created with
`tools/workspace/new_aux_worktree.jl` and live inside that same scratch tree.
Development and maintenance scripts use Julia, with browser JavaScript and native
installation/launcher scripts as the application-specific exceptions.

Experimental and published reference datasets are stored separately and are
optional. Matched experimental comparisons and the two cavitation reference
presets require those datasets. The XFOIL executable is also optional; set
`AIRFOIL_XFOIL_EXE` to its installed path. JXfoil runs from the bundled source.

## Workbench workflow

The apps share compact Boxer-style Project, Model, Mesh, Analysis, Results and
Display menus, a parameter pane, and a large viewport. Aero Lab also has a Domain
menu. **Help → Quick start and workflow** explains the sequence in each app.

In Aero Lab, open a study or import geometry, place it in the domain, create and
inspect the mesh, choose the output folder and run the analysis. **Results** opens
surface fields, recorded planes and images/files, including results from a saved
study. **Display** changes guides, mesh visibility and contour palette without
discarding loaded results. The colour choices follow JFEM-Boxer: spectrum,
Viridis, Inferno, cool–warm and grayscale. PNGs retain their saved colours.

## Run the command line solver

Run the root setup first. Copy `CASES/TEMPLATE_starter` to a new case,
provide an STL file, and edit its `config.yaml`. Copy
`cases_to_run.example.yaml` to `cases_to_run.yaml`, set the case name, then run
`__Start here!__/Windows/LBM-solver.cmd` or
`bash "__Start here!__/Linux/LBM-solver.sh"`. An installed Julia can also run
`julia --project=src src/main.jl --case /path/to/case`.

The app's default solver modes are development stage models. Inspect each
result's convergence and mesh diagnostics before drawing physical conclusions.

## Project layout

- `src/`: solver source and command line entry point. See `src/README.md` for the solver map; compressible C-mesh and Species cavitation now have separate source directories.
- `apps/ludwig_web/`: top-level launcher and Aero Lab.
- `apps/airfoil_web/`: Airfoil Lab app and static assets, also hosted by the launcher.
- `tools/`: tools required by the production cavitation driver.
- `tools/install/`: shared setup, verified runtime downloads and launcher helpers.
- `apps/common/`: shared browser integration, native dialogs and workbench styling.
- `__Start here!__/`: Windows, Linux and Mac instructions and launchers.
- `solvers/`: checksum pinned LBM and pressure-solver snapshots used by app backends; the other solvers live under `src/` and `third_party/`.
- `third_party/JXfoil/`: bundled JXfoil package.
- `CASES/TEMPLATE_starter/`: case input template.

Agent notes, tests, validation runs, papers, external reference data, and
temporary output are kept outside this repository in the sibling
`02_PRIVATE_PROJECT_DEVELOPMENT` directory. See `LICENSE` for AGPL-3.0 terms.
