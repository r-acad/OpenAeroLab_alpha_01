# Airfoil Lab

Run the [root setup](../../README.md#install-and-launch) first. Launch with
`run.cmd` on Windows, `bash run.sh` on Linux or `bash run.command` on Mac.
The app uses the repository's solver `src/` and bundled `solvers/` sources. Saved
studies use their own folder for new results; unsaved studies and lookup records
use the sibling private development directory (or `LUDWIG_PRIVATE_ROOT`).

## Workflow

Choose the **Solver** first. Its selector stays above the setup steps, and every
step then shows only the inputs that solver uses. Low-speed 2D CFD is the
default; the others are Compressible CFD, Species / cavitation CFD, LBM 3D,
JXfoil, XFOIL and four comparison runs.

The tab row is the order of work:

1. **Airfoil**: open an XFOIL `.dat` file or pick a built-in section. The
   optional Gurney flap is here (low-speed CFD and LBM 3D).
2. **Flow**: Reynolds number and angle, Mach where the solver uses it, and the
   alpha sweep.
3. **Mesh**: the mesh resolution, then **Create mesh**, which builds and shows
   the mesh that Run will use. A panel method has no Mesh step.
4. **Solver settings**: the method, the GPU switch and the stopping limits.
5. **Run and outputs**: the saved files, then **Run analysis**. Run, Cancel and
   the progress also stay below every step.
6. **Results**: the saved result and its exported files. The result views
   appear when the corresponding fields, surface data, histories or logs
   exist; a polar needs more than one angle.

**Study** (new, open, save and the saved-run list), **Display** (palette, grid
and probe), **Validation** (published comparison cases) and **Help** follow a
separator. There are no top-menu flyouts.

Each control has one location. Flow, Mesh, Solver settings and Run and outputs
each have one **Advanced** group for rarely needed settings: freestream
turbulence and air viscosity; trailing-edge thickness, flap-tip refinement and
streamline alignment; steady limits, statistics block, RK weights, mass ceiling
and SFD precision; history cadence, snapshot interval, reports and renders. A group
is closed by default and opens when a loaded study, saved job or preset uses a
setting in it. There is one mesh resolution control per solver family, in Mesh;
Create mesh and Run both send it.

The mesh topology follows the section and is not a choice. A finite (blunt)
trailing edge, native or set by the trailing-edge thickness, is meshed C+H; a
closed (sharp) one is a single C block. The section is never closed or given a
base fan to fit a mesh. The Species free-fluid domain (water or air) on a finite
trailing edge runs the two-block Species pair on the same C+H mesh, in physical
time; water in the tunnel is C+H too. Two Species requests still have one C
block only and are refused on a finite trailing edge with that reason: air in
the tunnel, and the steady air iteration (select Physical time, or the
Compressible solver for a steady result).

Options that earlier versions offered are gone from the page and the server:
the mesh profile choice, the single-block base-fan (A+) topology, the base-face
count, the uniform A+ profiles, the three research implicit methods and their
settings, the coarse low-speed mesh and the explicit low-speed topology. A
saved study or request that names one of them with the
value the app now always uses (`cfd_te_topology="auto"`,
`cfd_mesh="aplus_nested_refined"`, `lowspeed_topology="auto"`, a `false`
switch) loads and the key is dropped. Any other value is refused with a message
that names the key and the value; nothing is run as something else.

The small **i** buttons open contextual help using the mouse or keyboard;
controls also have hover tooltips. Help lists the steps above and opens the
workflow help. Mac supports JXfoil and result viewing; CUDA backends require
Windows or Linux with compatible NVIDIA hardware.

The app includes JXfoil, compressible C and C+H meshes, the conservative
Species path, low speed pressure based CFD, and LBM 3D. XFOIL executable runs
require `AIRFOIL_XFOIL_EXE`. RAE 2822 and ONERA OAT15A geometry and measured Cp
are bundled; they work offline in a fresh installation.

## Studies

Choose **Airfoil → XFOIL coordinate file (.dat)** to open a local
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

Use **Study → New study / Open study / Save study / Save study as**. Open
offers recent Airfoil studies and a native system file picker. Save reuses the
opened filename; Save As chooses a different folder or name. Ctrl+S, Ctrl+Shift+S
and Ctrl+O provide the corresponding shortcuts. Unsaved changes are marked `*`;
the app offers Save, Discard or Cancel before replacing the current setup.

A study stores the request the page built and a snapshot of its controls. On
opening, a saved value that a selector no longer offers leaves that selector at
its default; the page adds no option for it and sends no key that it has no
control for.

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

The Solver pane offers **Steady solution method** for compressible CFD and
**Try steady RANS acceleration (experimental)** for pressure/Species cases.
Acceleration is off by default and does not appear for physical-time,
phase-change water, panel, or LBM runs. The case flow key is
`rans_acceleration=true`; the selected method and its enabled state are saved
in the solver's `input.toml` and result diagnostics.

| RANS route | Accelerated setting |
| --- | --- |
| Compressible single C block, CPU or GPU | j-line implicit residual smoothing (IRS-J); steady local pseudo-time only. |
| Compressible C+H, CPU or GPU | IRS-J on both blocks. |
| Low-speed sharp C and C+H | The pressure solver already uses implicit backward Euler. The option raises its requested pseudo-time ceiling from 0.25 to 0.5 c/U; nonlinear rejected steps are reduced and later regrown by the solver. |
| Steady Species air | Wall-normal line-implicit relaxation. |

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
Results also offers RAE Case 6. The RAE 2822 section has a closed trailing edge,
so both cases run on the single C block with their prescribed trips and
viscosity law.

Selecting ONERA OAT15A applies the DPW conditions: M=0.73, Re=3 million, trips
at 0.07c on both sides, Sutherland viscosity at 271 K, at alpha 2.5°, 3.0° and
3.1° (steady) and 3.25°, 3.5° and 3.9° (physical time, 100 CTU with 20 CTU of
burn-in). The section has a finite trailing edge, so these cases run on the
two-block C+H mesh.

The compressible C+H driver has three options beside its default (steady,
fully turbulent SST, constant kinematic viscosity):

- Prescribed trips (`trip_upper`, `trip_lower`, per side; 0 or the Native
  transition setting: none). Both SST productions are zero ahead of the trip
  station of each side, rise linearly over 0.02c and are unchanged behind it,
  as in the single-block solver. The strip block behind the base has no trip.
  The result records the trips applied and the surface faces they made laminar.
- Viscosity law (`cfd_viscosity_model`): Sutherland or constant dynamic in the
  gas stresses and heat flux and in the SST diffusion, blending functions and
  wall omega. Residual smoothing (IRS-J) is available with constant kinematic
  viscosity only.
- Physical time (`cfd_time_mode="physical"`): one global timestep for both
  blocks, started from the freestream. The run ends at `cfd_duration_ctu` or at
  the step limit; forces are sampled at least every `cfd_force_sample_dt_ctu`
  and their means after `cfd_burnin_ctu` are reported with their sampling
  status. The Cp and the fields are those of the final instant; no time-mean
  Cp is computed. The timestep is the explicit stability limit of the smallest
  cell: on a wall-resolved mesh at the OAT15A conditions it is of the order of
  1e-7 CTU, so the step limit ends such a run long before its duration.

These options have unit tests and short CPU runs behind them, not a converged
or a GPU validation run.

Presets clear previous sweeps, flaps and trailing-edge overrides. Experimental curves appear immediately in
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

Run and outputs selects applicable VTK/native fields, images and CSV files,
an optional results parent folder, HTML reports, Makie PNG/SVG/PDF charts, and LBM 3D PNG
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
spacing. The low-speed mesh choice is Standard, the recommended profile for the
detected trailing edge. The low-speed native C+H Standard profile has about
175,000 cells for NACA 0012/4412; Fine subdivides it once more, with four times
as many cells. On a sharp trailing edge Fine uses the same generator settings as
Standard, so the page offers Fine on C+H only. The native C+H route retains the
same host mesh with and without a flap; flap-tip refinement is off unless
Standard or Fine is chosen in the Advanced group of Mesh. The C+H starting flow
and base turbulence treatment (Advanced group of Solver settings) are sent only
when the section runs on C+H. The app reports the
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

The **Validation** tab loads RAE 2822 Case 9 or
the NACA 0012 pressure-solver stall sweep (10–18 degrees, every 2 degrees).
References appear without requiring a solver run. The NACA comparison uses
Ladson's tripped CL/CD data; its incompressible, fully turbulent approximation
and geometry differences are documented with the data. Save each setup as a
study to retain its conditions, result link, palette and field limits.

Numeric flow fields use the Boxer palettes and editable automatic/manual
limits. The legend offers numeric limits, sliders, palette selection and
collapse. **Mach = 1** draws a black sonic contour on compressible 2D fields.

The compressible single-block mesher reduces tangential drift in near-body columns,
fading smoothly into the outer mesh. It preserves the wall, first eight cell
rows, wake, trailing-edge neighborhood and outer boundary. Quality checks
reduce the correction or retain the parent mesh if necessary. Metadata records
the applied correction; an improved shape alone does not establish accuracy.

The single-block mesh also adds smoothly graded normal rows through the off-wall
shock envelope. Fine uses a nominal 0.004-chord height cap through 0.65 chord
from the surface, followed by gradual growth to the far field. The first twelve
wall layers and the wall/outer contours remain exact. This increases the total
cell count (about 2.5–2.7 times for the Fine RAE example). Mesh metadata
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

## Steady solution methods

For **Compressible CFD**, choose **Steady pseudo-time** and the **Steady
solution method** in Solver settings. There are three: the standard explicit
iteration, implicit residual smoothing (IRS-J, experimental; single C block and
C+H, CPU or GPU) and Implicit Newton (CPU, development, below). The request key
is `rans_method` with `standard`, `irsj` or `newton`; any other name is refused.

Diagnostics separates stable loads from settled equations. Full fields and
surface Cp remain available after an incomplete run. Checked restart files
require matching mesh, physical settings and solver source, including
reconstruction, metrics and the velocity lattice; checkpoints without those
source identities are refused and start cold.

The pressure-based low-speed solver is already implicit. Its acceleration
checkbox raises the requested pseudo-time ceiling while retaining automatic
rejection and retry. A native steady stop is accepted only after final equation
and mass-balance checks; stable coefficients alone do not make a run converged.

### Implicit Newton (CPU) on the marched mesh

Development result, not validated. For **Compressible CFD** in **Steady
pseudo-time**, the **Steady solution method** entry **Implicit Newton (CPU)**
(`rans_method="newton"`) runs the pseudo-transient Newton–Krylov march of
`src/compressible/cmesh_steady_newton_ka.jl` with SST, on the CPU in Float64,
whatever the GPU switch says.

- Mesh: always the marched regular mesh (`src/cmesh/cmesh_marched.jl`), which no
  other method uses; the request names the method only. Sizes (surface, wake per side, wall-normal cells):
  standard 768 × 128 × 96, fine 1152 × 192 × 128, very fine 1536 × 256 × 160.
  The wall spacing follows the Reynolds number as in the other compressible
  meshes. Create mesh shows the mesh of the run.
- Mesh report: the page states the geometric check that passed (positive cell
  areas; for one block also convex cells) and names every CMeshQualityV1 air
  threshold the mesh lies outside, with the measured value and the limit. These
  thresholds are the ones reported for the app's single C block and
  C+H mesh; for the marched mesh they are comparative and refuse nothing.
- Page: the method is chosen in Solver settings. The GPU switch, smart stop, RK
  weights, scaled mass ceiling, SFD precision and streamline alignment do not
  apply to it; they are hidden and not sent.
- Topology follows the section after the usual conditioning (including
  `cfd_te_thickness`): a closed trailing edge runs one C block, a finite one
  runs two blocks (C+H, the base is a wall of the strip block). The equation
  NACA four-digit sections have a finite trailing edge.
- Start: cold at every angle, three mesh levels (every fourth node, every
  second node, all nodes). No checkpoint is read or written.
- Steps: `cfd_steps` limits the fine level, at most 600 (the page sets 300);
  each coarse level takes at most 200.
- Transition: forced trips on the surface (`trip_upper`, `trip_lower`, the
  app's defaults when absent); the strip behind a finite base has no trip.
  Viscosity: Sutherland, constant dynamic or constant kinematic. Constant
  kinematic is covered by the method's small tests only.
- History: one row every second step with a single counter over the three
  levels. The plotted residual is the RMS continuity residual divided by the
  freestream density; `residual_l2` is the largest of the four flow-equation
  RMS residuals and `residual_linf` the larger of the k and omega RMS
  residuals (the march computes no maximum norm). `newton_history.csv` holds
  all six, the mesh level, the CFL and the linear iterations of every row.
  The history interval of the output settings does not thin these rows.
- Status: a point is **completed** only when the march reports that CL and CD
  settled on the fine level (relative span below 1e-4 over 20 steps); its stop
  guard then reads `early_stop_eligible`. Otherwise the guard reads
  `loads_not_settled` and the point is **not_converged**; its fields and
  coefficients remain a saved diagnostic state. The six equation residuals are
  recorded without qualified limits and do not enter the status.
- Files: one block writes the files of the single-block compressible points;
  two blocks write the files of the C+H points (`fields_blocks*.msgpack`,
  `field_C_full`, `field_H_full`, `wall_force_faces.csv`).
- `LUDWIG_NEWTON_THREADS` limits the worker's threads (default: all).

## Source layout

| Directory | Contents |
| --- | --- |
| `src/api/` | Job progress and study helpers. |
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

### Native low-speed host compilation

The bundled low-speed environment is also the `PBCavNativeCache` Julia package.
Its host-only precompile workload exercises projection, a nonzero coupled SST
implicit step, and wall/turbulence diagnostics. Normal setup precompilation
prepares this code; the first worker can compile it when its cache is absent.
The workload imports no CUDA module and creates no device/context/graph. The
worker stores new compiled code under the private development folder's
`TEMP_DELETEABLE/airfoil_native_host_cache` depot, retaining installed dependency
depots for reads; clearing that disposable cache only causes recompilation.

Each run still starts a fresh owned process and takes the existing GPU lock
before CUDA initialization. Before use, the package checks its loaded native
and generator sources plus locked environment against their recorded bundle.
The app's approved snapshot manifest remains the admission check. Older pinned
snapshots without the package retain their direct-include worker path. A broken
or changed named package is refused rather than silently treated as legacy.

Run `julia tools/tests/lowspeed_native_cache.jl` for CPU component checks of all
state arrays and diagnostics against direct/legacy loading, source invalidation,
both actual worker loading prefixes and absence of CUDA loading. The small
host workload is not an airfoil convergence or full GPU performance benchmark.
