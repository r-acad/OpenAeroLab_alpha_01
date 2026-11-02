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
require `AIRFOIL_XFOIL_EXE`. RAE 2822 and ONERA OAT15A geometry and measured Cp
are bundled; they work offline in a fresh installation.

## Studies

Use **Project → New study / Open study / Save study / Save study as**. Open
offers recent Airfoil studies and a native system file picker. Save reuses the
opened filename; Save As chooses a different folder or name. Ctrl+S, Ctrl+Shift+S
and Ctrl+O provide the corresponding shortcuts. Unsaved changes are marked `*`;
the app offers Save, Discard or Cancel before replacing the current setup.

An Airfoil `.ludwig-study` file contains the source coordinates, complete solver
request, editable controls, current results tab and literature selection. UIUC
coordinates travel with the study, so reopening does not require downloading
them again. Results are **links to job folders**, not copies of large fields.
Save after starting/opening an analysis to retain that link. Opening a study
loads available linked results without starting a solver or replacing its setup.
**Results → Load study results** reopens them explicitly. Missing result folders
are reported while the setup remains usable. Linked job folders now live in
private `results/airfoil_runs`, outside disposable storage. Keep those folders
and selected export directories to retain results.

Outputs provides **Save field snapshots every, minutes** for the compressible
solver and sharp-mesh pressure solver. Default: 10 real-time minutes; 0: disabled.
This is independent of iteration history and does not change final-field saving.
In **Results → Flow field**, use **Saved snapshot** to reopen any recorded frame.
The fields are visualization snapshots, not restart files. Writes report their
duration and reject insufficient disk space without announcing an empty file.

## Published comparison cases

Selecting RAE 2822 applies **AGARD Case 9 corrected free-air conditions**:
M=0.734, Re=6.5 million, alpha=2.79°, trips at 0.03c, constant dynamic viscosity.
Selecting ONERA OAT15A applies the measured **2.5° steady comparison**:
M=0.73, Re=3 million, static T=271 K, trips at 0.07c, Sutherland viscosity.
The source menu and Results both offer the other bundled measured angles and
RAE Case 6. OAT15A angles 3.25°, 3.5° and 3.9° select physical time, for comparison
of the post-burn-in mean with measured mean Cp. The numerical duration and step
limit are editable starting choices, not claims of converged buffet statistics.

Presets clear previous sweeps, flaps and trailing-edge overrides. They explicitly
select the A-plus GPU path because the current C+H driver uses fully turbulent
SST and constant kinematic viscosity. Experimental curves appear immediately in
Results, independently of CFD jobs. Edited conditions are labelled as different;
Apply published case restores them. CFD convergence and mesh/time independence
remain separate checks. Sources, tunnel corrections and ONERA redistribution
conditions are documented in [data/references](data/references/README.md) and
linked next to the plots.

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

## Plots and validation studies

Results, Polar and Diagnostics expand their charts to the available viewport.
Each chart offers **Export SVG**, including its current zoom and legend. The
shared **Grid** and **Probe values** controls enable coordinate lines and
nearest-sample interrogation. Results displays CL, CD and CM about `(0.25c, 0)`;
this specified reference point is not an inferred aerodynamic center.
Incomplete CFD surface distributions and polar points remain visible with their
convergence status, including experimental overlays.

The **Validation** menu loads RAE 2822 Case 9, ONERA OAT15A steady conditions, or
the NACA 0012 pressure-solver stall sweep (10–18 degrees, every 2 degrees).
References appear without requiring a solver run. The NACA comparison uses
Ladson's tripped CL/CD data; its incompressible, fully turbulent approximation
and geometry differences are documented with the data. Save each setup as a
study to retain its conditions, result link, palette and field limits.

Numeric flow fields use the Boxer palettes and editable automatic/manual
limits. The legend offers numeric limits, sliders, palette selection and
collapse. **Mach = 1** draws a black sonic contour on compressible 2D fields.

The compressible A-plus mesher reduces tangential drift in near-body columns,
fading smoothly into the outer mesh. It preserves the wall, first eight cell
rows, wake, trailing-edge neighborhood and outer boundary. Quality checks
reduce the correction or retain the parent mesh if necessary. Metadata records
the applied correction; an improved shape alone does not establish accuracy.

The A-plus mesh also adds smoothly graded normal rows through the off-wall
shock envelope. Fine uses a nominal 0.004-chord height cap through 0.65 chord
from the surface, followed by gradual growth to the far field. The first twelve
wall layers and the wall/outer contours remain exact. This increases the total
cell count (about 2.5–2.7 times for the Fine RAE/OAT examples). Mesh metadata
records the actual row count, cap and any quality fallback. Both upper and lower
sides share one row map to keep the nose and trailing-edge joins continuous.

Field colors and sonic contours use bilinear interpolation on the original
sample quadrilaterals. This removes the artificial display diagonal without
averaging CFD samples or changing the saved solution. It cannot repair an
under-resolved or unconverged simulation; existing studies retain their old grids.
The Mach-one contour waits for the full-resolution field instead of using the
decimated preview. Loading failures appear beside the field with a retry button.

Steady GPU runs use Float64 damping filters by default. After the loads settle,
damping is tapered off and convergence windows restart on the undamped
equations. Force stationarity is reported separately from continuity success.
An already small initial residual need not decrease by an arbitrary additional
ratio; absolute, plateau, spike, mass-balance and force checks remain active.
Snapshot output is controlled by wall time (default ten minutes), independently
of iteration-history updates. Final fields are retained even without convergence.

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
