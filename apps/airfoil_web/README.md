# Airfoil Lab

Run the [root setup](../../README.md#install-and-launch) first. Launch with
`run.cmd` on Windows, `bash run.sh` on Linux or `bash run.command` on Mac.
The app uses the repository's solver `src/` and bundled `solvers/` sources. Output
goes to the sibling private development directory (or `LUDWIG_PRIVATE_ROOT`).

Project, Model, Mesh, Analysis, Results and Display use the shared Boxer-style
menus. Menu choices select the parameter pane without changing the current result.
Help explains the workflow; Display provides five contour palettes for numeric
fields. Mac supports JXfoil and result viewing; CUDA backends require Windows
or Linux with compatible NVIDIA hardware.

The app includes JXfoil, compressible C and C+H meshes, the conservative
Species path, low speed pressure based CFD, and LBM 3D. XFOIL executable runs
require `AIRFOIL_XFOIL_EXE`. Published reference comparisons require optional
datasets in the private directory.

Airfoil Lab's LBM 3D path offers D3Q27 + cumulant or D3Q39 + entropic from its
pinned solver snapshot. D3Q27 uses the fixed low-Mach carrier and supports
periodic-Y AMR. D3Q39 takes the Mach input (0.01 to 1.6), enables compressible
thermal flow, accepts up to 7 periodic-Y AMR levels when domain and memory allow, and uses legacy mirror-slip top
and bottom walls. Both paths are development only and are not validated for
airfoils. D3Q19 remains retired pending a separate solver review.
The optional VRAM target mode chooses the densest 8-cell increment of finest
cells per chord whose conservative pre-build estimate fits the selected
percentage of currently available VRAM. The GUI shows the resolved GB value. Create
mesh reports the resolved geometry estimate and refuses it if it exceeds the
  target; the ceiling follows detected free VRAM with a device reserve.

Outputs and reports selects applicable VTK/native fields, images and CSV files,
an export destination, HTML reports, Makie PNG/SVG/PDF charts, and LBM 3D PNG
renders. A unique job folder preserves existing files. The display interval
controls solver output in iterations. LBM follows the exact requested cadence;
the pinned low-speed C+H path exposes its final field while history respects the
interval. JXfoil and XFOIL publish results per completed operating point. The GUI
explains the selected backend's capabilities. Makie uses its own environment in
`tools/postprocess`.

The default C+H Standard mesh uses curvature aware leading edge spacing. Mesh
quality and solver convergence are reported with each result.
The sharp C mesh also concentrates surface columns near the leading edge in
Standard, Fine, and Very Fine. A separate low speed `fine` study profile uses
384 surface cells, 128 wall-normal cells, and a 7e-6 chord first wall spacing.
It passes the PBCav reference mesh gates for NACA 0012; the app still reports
the audit for every generated section. Grid independence and stall accuracy
are not established by a mesh-quality pass.
The pressure worker writes `rejection_history.csv` with the accepted step,
time step, and cumulative rejection counts by reason. A checked warm start
that changes incidence by at least 1.5 degrees begins at a small time step;
normal step growth resumes after accepted steps. The full steady residuals,
not stable-looking force coefficients, decide whether a polar point converged.

## Source layout

| Directory | Contents |
| --- | --- |
| `src/api/` | Job and campaign status helpers. |
| `src/backends/` | CFD, JXfoil, and XFOIL adapters and orchestration. |
| `src/diagnostics/` | Convergence, residual, and time-history analysis. |
| `src/lbm/` | LBM case construction, fields, and backend. |
| `src/lowspeed/` | Pressure-solver backend configuration and result handling. |
| `manifests/` | Checksums for pinned LBM and pressure-solver snapshots. |
| `src/mesh/` | Mesh generation, transforms, audits, and streamline alignment. |
| `src/references/` | Optional experimental reference integration. |
| `src/runtime/` | Shared source receipts and compressible worker components. |
| `src/workers/` | Isolated CFD, pressure, LBM, and underwater worker entry points. |
| `static/` | Browser page and icon, with styles in `styles/`, scripts in `scripts/`, and Babylon.js in `vendor/`. |

The app root contains the launch scripts, static assets, manifests, and this
README. `src/server.jl` is the HTTP entry point; `src/` also holds the shared
wire, field, and GPU-lock modules. Retired Gurney insertion prototypes are
kept in the private development archive.

The app offers six backend choices: JXfoil, optional external XFOIL,
compressible C-mesh, cavitation/Species, pressure-based low-speed CFD, and
LUDWIG LBM. `solvers/` at the repository root contains only the two pinned
copies used by the LBM and pressure backends. The primary C-mesh and LBM
source is in `src/`; the compressible and Species cavitation entry modules are in
`src/compressible/` and `src/cavitation/`. JXfoil is in `third_party/JXfoil/`. The `solvers/`
directory is a snapshot store, not a catalog of all solver families.
