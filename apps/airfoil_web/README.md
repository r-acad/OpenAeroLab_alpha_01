# Airfoil Lab

Run the [root setup](../../README.md#install-and-launch) first. Launch with
`run.cmd` on Windows, `bash run.sh` on Linux or `bash run.command` on Mac.
The app uses the repository's solver `src/` and bundled `solvers/` sources. Output
goes to the sibling private development directory (or `LUDWIG_PRIVATE_ROOT`).

The initial **Setup** pane keeps the main workflow together: open an XFOIL
`.dat` airfoil, enter Reynolds number and angle of attack, optionally add a
Gurney flap, then **Run analysis**. Low-speed 2D CFD is the default. Mesh,
starting flow, turbulence and numerical settings use the recommended automatic
configuration, with matching clean/flap settings. Explicit comparisons and
other solvers are under **Setup → Solver and numerical settings**.

The menus follow Project, Setup, Mesh, Results, Display, Validation and Help.
Project manages studies; Mesh opens inspection; results actions share one saved
job list. **Create mesh** is the optional preview in Geometry, while Run creates
the mesh automatically. Resolution is selected once in Advanced setup for
low-speed runs.

The small **i** buttons open contextual help using the mouse or keyboard;
controls also have hover tooltips. Help provides a low-speed quick start.
Display controls contour palettes. Mac supports JXfoil and result viewing;
CUDA backends require Windows or Linux with compatible NVIDIA hardware.

The app includes JXfoil, compressible C and C+H meshes, the conservative
Species path, low speed pressure based CFD, and LBM 3D. XFOIL executable runs
require `AIRFOIL_XFOIL_EXE`. RAE 2822 and ONERA OAT15A geometry and measured Cp
are bundled; they work offline in a fresh installation.

## Studies

Choose **Setup → Airfoil and flow conditions → XFOIL coordinate file (.dat)** to open a local
coordinate file with the system file picker. Plain or titled XFOIL contours and
the two-surface count format are supported. Points are ordered from the trailing
edge around the leading edge and back. Import normalizes the chord to one and
places the leading edge at the origin. Invalid rows are reported with their line
number. Save the study to embed the coordinates; reopening needs no original
file or network connection.

Automatic topology retains the imported section: a finite trailing-edge gap
uses native C+H and a closed edge uses sharp C. A low-speed Gurney flap requires
finite trailing-edge coordinates. A closed-edge flap case is refused; provide
the measured finite-edge section rather than adding an artificial gap.

**Cancel run** requests a stop for the active job. The status stays Cancelling
while the current operation finishes and saved fields are written, then becomes
Cancelled. Completed points, recorded fields, checkpoints and logs remain
available. Reopening a running or cancelling job reattaches its progress. The
HTTP equivalent is `POST /api/jobs/{id}/cancel`; polling the normal job endpoint
continues until `cancelled`.

API clients may set `flow.lowspeed_initial_job_id` to a terminal low-speed job's
UUID to offer its saved flow as a **checked initial guess** for the first angle.
The server chooses the closest compatible saved point within 2.5 degrees and
checks source geometry, mesh/baffle settings, Reynolds number, freestream
turbulence and solver source. Only checkpoints confined to that job's generated
results are accepted. Missing or incompatible states are refused; a worker's
cold fallback is not accepted. Results record the source job and checkpoint hash.
Pseudo-time and step count reset under the ordinary warm-start policy. This is
separate from exact native `.pbrestart` continuation in the command-line driver.
API revision 25 advertises this capability as `lowspeed_initial_flow.checked_job_seed`.

Use **Project → New study / Open study / Save study / Save study as**. Open
offers recent Airfoil studies and a native system file picker. Save reuses the
opened filename; Save As chooses a different folder or name. Ctrl+S, Ctrl+Shift+S
and Ctrl+O provide the corresponding shortcuts. Unsaved changes are marked `*`;
the app offers Save, Discard or Cancel before replacing the current setup.

API-only settings from a saved study are retained and listed as saved request
overrides in Advanced setup. **Clear saved overrides** returns only these
additional settings to automatic defaults for the next request. It keeps the
current airfoil, visible flow conditions and flap controls, marks the study as
changed, and leaves any running job untouched.

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
solver and both sharp-C and C+H low-speed solvers. Default: 10 real-time minutes;
0: disabled.
This is independent of iteration history and does not change final-field saving.
In **Results → Flow field**, use **Saved snapshot** to reopen any recorded frame.
The recorded frames are visualization snapshots. Writes report their
duration and reject insufficient disk space without announcing an empty file.

Both low-speed workers also save final **VTU volume fields** (`flow_fields.vtu`)
and a native **restart checkpoint** (`state_checkpoint.pbrestart`). Download
them from **Flow field → Saved solver files** for the selected operating point.
The native driver supports continuing a compatible
checkpoint with `PB_RESTART`; it verifies source, mesh, physics and numerical
settings before restoring the state. See the [native free-air driver guide](../../solvers/pbcav_s1030/tools/README_FREEAIR.md).

## Steady RANS acceleration

The automatic low-speed steady path damps each nonlinear SST update with a
factor of 0.5. This removes a diagnosed alternating turbulence iteration while
retaining the same steady fixed point. Momentum and SST stopping limits remain
1e-6 and 1e-7; nearly constant force coefficients alone do not pass the stop.
The native reference driver retains its historical setting unless
`PB_TURB_STEP_RELAX=0.5` is selected explicitly for a steady run.

C+H runs use SIMPLEC preconditioner factor 0.95. A matched GPU continuation
from the same NACA4412 state reduced linear work and rejected steps compared
with 0.8 while preserving matched-time coefficients to roundoff. This is a
measured convergence setting, not an aerodynamic validation. The advanced
API key `lowspeed_cpl_au` permits controlled comparisons.

Automatic clean and Gurney C+H runs use FGMRES restart 60 with at most 300
iterations per linear solve. App sharp C and the native reference driver retain
restart 30 and maximum 300. Matched frozen-system CPU and one-step GPU
comparisons support the larger C+H subspace; each airfoil run still requires
its own convergence assessment. The linear true-residual target, outer checks
and strict steady limits remain unchanged. The restart uses more memory: at
about 175,000 cells, the two Krylov basis arrays alone use about 485 MiB instead
of 244 MiB.

Advanced API keys `lowspeed_cpl_restart` (integer 1–120) and
`lowspeed_cpl_maxit` (integer 1–10000) override these defaults. They are recorded
in `input.toml` and the actual solver receipt, and retained as saved request
overrides. API revision 26 advertises their bounds and per-route defaults in
`lowspeed.krylov`; the current web client requires that revision for new runs.
Exact native continuation requires matching settings and solver source.

The Advanced setup pane offers **Steady solution method** for compressible CFD and
**Try steady RANS acceleration (experimental)** for pressure/Species cases.
Acceleration is off by default and does not appear for physical-time,
phase-change water, panel, or LBM runs. The legacy case flow key is
`rans_acceleration=true`; the selected method and its enabled state are saved
in the solver's `input.toml` and result diagnostics.

| RANS route | Accelerated setting |
| --- | --- |
| Compressible C/A-plus, CPU or GPU | Existing j-line implicit residual smoothing (IRS-J); steady local pseudo-time only. |
| Compressible C+H, CPU or GPU | IRS-J on both blocks. |
| Low-speed sharp C and C+H | The pressure solver already uses implicit backward Euler. The option raises its requested pseudo-time ceiling from 0.25 to 0.5 c/U; nonlinear rejected steps are reduced and later regrown by the solver. |
| Steady Species air | Existing wall-normal line-implicit relaxation. |

The setting changes convergence behavior, not the physical-time integrator or
the convergence criteria. One-step and three-step smoke cases verify method
selection and worker execution; they do not establish faster time to a
converged solution. Compare residuals, force stability, rejected steps, wall
time and final loads against the standard setting on the same mesh before
using a result. On a low-speed C+H startup smoke case the 0.5 c/U request was
reduced to 0.015625 c/U after five rejected trials, one more rejection than
the 0.25 c/U control; the setting is particularly case dependent there.

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
both low-speed paths save periodic full fields by elapsed wall time and final
fields when they stop, while history respects its own interval. JXfoil and
XFOIL publish results per completed operating point. The GUI
explains the selected backend's capabilities. Makie uses its own environment in
`tools/postprocess`.

The default C+H Standard mesh uses curvature aware leading edge spacing. Mesh
quality and solver convergence are reported with each result.
The sharp C mesh also concentrates surface columns near the leading edge in
Standard, Fine, and Very Fine. The low-speed sharp Standard profile uses 384
surface cells, 128 remapped wall-normal cells, and a 7e-6 chord first wall
spacing. Automatic uses the recommended profile for the detected trailing
edge. The low-speed native C+H Standard profile has about 175,000 cells for NACA
0012/4412; Fine subdivides it once more, with four times as many cells. The
native C+H route retains the same host mesh with and without a flap;
flap-tip refinement is an explicit Advanced comparison. The app reports the
audit for every generated section. A quality pass does not establish grid
independence or stall accuracy.
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
ratio. Absolute median and maximum residual limits, signed mass balance and
force checks remain active; relative reduction and plateau ratios are diagnostics.
Snapshot output is controlled by wall time (default ten minutes), independently
of iteration-history updates. Final fields are retained even without convergence.

## Experimental steady methods

For **Compressible CFD**, choose **Steady pseudo-time**, the single-block
**A+** trailing-edge topology, then **Steady solution method**. The choices are
standard explicit iteration, j-line residual smoothing, line-implicit, native
Newton–Krylov, and Newton–Krylov using Krylov.jl GMRES. The coupled C+H mesh
supports standard iteration and residual smoothing. These methods run on CPU
or GPU; the advanced settings are stored in the study.

The three new implicit methods are research tools. CPU and GPU integration
checks pass, but dependable full-airfoil convergence and a speed advantage
have not been demonstrated. An optional bounded explicit recovery burst can
help Newton–Krylov escape rejected updates; it is disabled by default and does
not weaken convergence checks. Krylov settings do not apply to line-implicit.

Diagnostics separates stable loads from settled equations. Implicit runs write
`implicit_history.csv` and `steady_assessment_history.csv`, including four
flow-equation errors, turbulence update defects, signed global mass balance,
CL, CD and CM. Turbulence update defects are not raw SST residuals; passing
these monitored checks does not certify complete equation convergence.
Full fields and surface Cp remain available after an incomplete or rejected run.
Checked restart files require matching mesh, physical settings and solver source.

The pressure-based low-speed solver is already implicit. Its acceleration
checkbox raises the requested pseudo-time ceiling while retaining automatic
rejection and retry. A native steady stop is accepted only after final equation
and mass-balance checks; stable coefficients alone do not make a run converged.

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
