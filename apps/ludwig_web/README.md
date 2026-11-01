# Ludwig Workbench

Run the [root setup](../../README.md#install-and-launch) first; it installs the
packages for both workbenches, supported solver workers and Makie. Launch
`run.cmd` on Windows, `bash run.sh` on Linux or `bash run.command` on Mac.
The launcher opens or reuses the current local server, trying port 8058 first
when it needs a new one. It opens Aero Lab and Airfoil Lab.
Airfoil Lab also keeps its standalone launcher in `../airfoil_web/`.

The launcher and Aero Lab API live here; Airfoil Lab is included from its own
app source and served at `/airfoil-lab`. Its existing `/api/*` and `/static/*`
URLs remain available. Aero Lab uses `/api/aero/*`, with runs and uploads in
the sibling private `02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE` directory.
The server binds only to loopback.

Project / Model / Domain / Mesh / Analysis / Results / Display menus use the
shared Boxer-style interface in `../common/static/`. Help explains the workflow.
Display offers six palettes for numeric fields; changing palette retains the
loaded surface/plane and does not change saved PNGs. Mac supports study/result
viewing and JXfoil; CUDA runs require Windows or Linux with an NVIDIA GPU.

Aero Lab offers **D3Q27 + cumulant** (Mach at most 0.6) from either solver
source and **D3Q39 + entropic** (Mach at most 1.6) from the main solver. D3Q39
cases accept 1 to 7 AMR levels and enable compressible mode, the thermal energy
model, and thermodynamic pressure. The D3Q39 GPU allowance is deliberately
conservative because only short web cases have been run so far.
In VRAM target mode, the app runs CPU build-only probes and selects the finest
cell size whose conservative run-memory estimate fits the requested percentage
of currently available VRAM. The GUI shows the corresponding GB value and any
device reserve cap; the final
built mesh is checked again before a GPU run.
Build and inspect the mesh before a run; these web cases are development cases,
not validated flow predictions. D3Q19 remains retired pending a separate
solver review.

In Aero Lab, import an STL, OBJ, or GLB from **File**, set its scale, rotation,
and offset under **Model**, then choose domain clearances, boundary conditions,
the backend, and the finest cell size. **Run → Create mesh** runs the selected
solver's CPU mesh path.
Its AMR blocks, resolved dimensions, cell counts, and GPU memory estimate
appear before **Run solver** is enabled. Runs export surface Cp, pressure and
wall shear as well as velocity, pressure and vorticity on configured cut
planes. Recorded cut locations are selected before a run; the default is three
midplanes. With full-volume VTK output enabled, the inspector can sample other
axis-aligned positions from the latest saved snapshot. The case link resumes the job after a browser reload. The mesh
result also reports finest-level wall cells; a severely sparse surface mesh
must be rebuilt before a flow run.

The top bar provides **Project, Model, Domain, Mesh, Analysis, Results,
Display, and Help** dropdowns. Menu choices select a scrollable parameter
panel on the left; meshing is in **Mesh**, and running and cancellation are in **Analysis**.
Display contains preparation controls: geometry and AMR display, camera, guides
and fitting. Results groups surface fields, vectors, cut-plane fields, imported
result sources and downloads. It stays grey until data are available, then turns
green; it always opens so saved files can be imported. The permanent results
toolbar provides **Surface**, **Recorded planes**, **Images & files**, and
**Load saved results**, so view changes never strand a loaded field. The status
beside these controls identifies saved results and warns when draft settings
need a new mesh; editing a draft preserves the last completed analysis.
Reopening an edited study also retains its historical local results, clearly
labelled as belonging to the saved run. Build a new mesh before running the
edited setup. A missing result location does not prevent opening the study.
**File → Read native / VTK results** is also available for import.
Triangle and cell counts, VRAM estimate and GPU budget
share a slim status strip. **Mesh details** sits immediately left of **Messages**
in the header; **Solver log** sits immediately left of **Show mesh / Hide mesh**.
The i panel lists domain dimensions, axes, boundary
conditions and inspection help without labels covering the model. Important
messages open a centered, closable dialog; **Above GPU budget** immediately
focuses the warning text. Dismissed warnings stay closed during repeated polls.
Use **Messages** or **Mesh details** to inspect details again.
Both labs' top-left icons return to the workbench launcher, and
their System/Light/Dark theme setting is shared.

During mesh creation, a separate canvas overlay shows an estimated percentage
based on solver build stages, the current activity, and a live log. It is
orange while creating, magenta on error and green when ready. The terminal
card retains its log and can be closed; **Build log** reopens it. Ready actions
inspect the mesh or show its details. It is independent of warning dialogs.
When a build cannot meet its constraints and a
valid alternative is available, the warning offers a proposed surface cell size
and AMR level count. Its action applies that proposal and creates the mesh in
one click; changed setups require a fresh proposal.

The solver has a separate progress card with phase, iteration count, elapsed
time, available diagnostics and a retained log. GPU and system RAM readings
update while running; process readings can be live or retained samples and are
labelled accordingly. Memory is shown in GiB. Unexported residuals and
unreliable ETA values stay explicitly unavailable. Terminal cards can be closed
and reopened without losing their logs. Reopening a saved study does not open
a completed solver card or replay an old mesh-sizing proposal; **Solver log**
remains available for deliberate inspection.

Choose **Model → Appearance and GLB textures**, then **GLB materials and
textures**, to display the original GLB appearance instead of the neutral
surface. Surface-field mode displays computed quantities. Under **Model → Move /
rotate widget**, choose the **None**, **Move**, or **Rotate** radio button.
Move shows draggable translation axes; Rotate shows rotation rings. Numeric
placement fields stay synchronized. New cases archive the original GLB along
with the solver geometry so textures survive a case reload. Older cases can recover the original from a matching retained study; otherwise
load the original GLB to recover its appearance.
Under **Model → Appearance and GLB textures**, **Placement projections** can
show the model's silhouette on the ground, inlet and lateral walls, separately
or together, to help check clearances. All three projections are on by default
for new studies; saved view preferences are retained.

**Model → Surface triangle refinement** sets the maximum sampling triangle edge
in millimetres (default **50 mm**). New meshes subdivide the placed model's
triangles without changing its shape, and the solver samples pressure and shear
on those smaller faces. This refines surface mapping; fluid resolution still
depends on the finest AMR cell size. A triangle-count limit prevents accidental
unbounded subdivision, and its memory allowance is included in the estimate.
Existing studies retain the geometry used for their saved results until rebuilt.
Surface colours use full brightness, with a visible legend. **High contrast**
uses the 2nd–98th percentiles and reports true extrema; **Full data range**
includes every field value in the colour scale. **Results → Field scale limits**
provides minimum/maximum numbers and sliders, **Apply limits** and **Automatic**.
Select surface, interactive section or recorded numeric cut; each quantity keeps
independent limits in the study. The signed Boxer palette uses red for negative,
blue for positive and a neutral colour at actual zero, including asymmetric limits.
Colour clipping never changes stored data; saved PNGs retain their original colours.

**Model → Reference dimensions** accepts total wing area, reference chord,
optional span, and the moment reference point in metres. The point is measured
from the centred/placed construction frame (X aft, Y right, Z up), before the
flow angles; its Y coordinate is fixed to the wall for symmetric cases.
**Analysis → Fluid properties** sets alpha and beta; beta is zero in symmetry.
Angles rotate the geometry into the solver's +X freestream and require a new
mesh. Manual placement rotation aligns the imported model before these angles.
Optional Boxer body, wind and stability axes and angle arcs appear in the view;
the stability axes use the entered reference alpha, and the moment origin is optional.

The reference panel computes CL, CD, CY, body roll/pitch/yaw Cl/Cm/Cn and L/D
from the latest saved dimensional force and moment history. It shifts moments
to the requested origin using `(old_origin - new_origin) × force`. With
`q = rho U² / 2`, forces use `q S`; pitch uses `q S c`, and roll/yaw use
`q S b` when span is entered (otherwise chord, explicitly labelled).
For symmetric cases the effective solved area is half the total area. Native
histories already mirror the half-model loads, so full loads / full area is used,
equivalent to half loads / half area. Enter actual dimensions: the app does not
guess wing area from a bounding box. These are instantaneous saved loads;
a completed run is not evidence of convergence.

The surface-pressure sampler now finalizes pressure after nearest-fluid recovery,
and checks the reverse normal when a closed-surface face has no valid forward
probe. The recovered sign is used consistently in forces and exported normals.
The two-sided force API retains separate sampling sides for thin surfaces. Unavailable samples
are explicitly missing (grey), rather than fabricated zero pressure. Old surface
files are unchanged: rerun the analysis to replace affected window/facet samples.

Under **Display → Move a cutting plane**, enable **Move plane: edge handle** to
show a small draggable sphere at a domain corner. Drag it or the plane border
along the plane normal; hold Shift for tenfold finer movement, or enter an exact
position. Mesh overlays hide during dragging. With surface data loaded,
**Surface field section curves** shows the intersected face values against both
in-plane coordinates, around the model in 3D and along the top/right of the 2D
section. The curves use the currently selected surface quantity and retain its
actual values; they do not infer new flow information between saved samples.

Click a model surface to put the orbit centre on that exact point. Right drag
keeps the view pinned to the cursor; double-click the background to fit the
domain. **Hide mesh / Show mesh** controls AMR visibility independently of the
model. The mesh is temporarily hidden while dragging a section plane and is
restored on release according to that choice. Prescribed velocity arrows sit outside and touch the inlet. Inlet and
outlet faces are translucent yellow and blue; the lower domain face is green,
including in free air. Symmetry is blue. Actual boundary conditions are listed
under **i**; shading does not change them.
**View → AMR levels** independently shows or hides each refinement level and
switches between block outlines, true 12-edge cell outlines, solid voxels and
translucent voxels. GPU thin instances draw a coherent exposed-cell envelope;
individual cells are not decimated into striped gaps. Interior cell edges appear
in the section. Solid faces write depth; translucency uses Babylon depth peeling
on supported devices and sorted face instances otherwise. The plane clips the
model and mesh by default and is dragged by its border. Exact body intersections
appear in 3D and in the 2D section. Wheel zoom is pinned to the cursor; drag to
pan, and double-click or Fit section to reset. Studies retain zoom and pan.
The 2D section shows actual cell edges and AMR ownership;
coarse cells covered by finer blocks are omitted. Saved field colours use the
latest full-volume VTU, with finest-cell ownership, or an exactly matching
recorded plane. Missing fields stay explicitly unavailable; no neighbouring
plane is substituted. The interactive VTU reader accepts snapshots up to 512 MB.

Recorded cut planes support cursor-anchored wheel zoom, drag panning, zoom
buttons and **Fit plane**; double-click also fits. The movable mesh section and
recorded solver plane have independent zoom settings.
To obtain a recorded field, move the inspection plane, choose **Record plane**,
then **Run solver**. Under **Results → View cut-plane fields**, choose the
recorded plane and iteration. Moving the plane alone does not record solver
data. Inspecting a new position after a run requires existing volume VTK or a
future run recording that plane. Editing only the recorded-plane list retains
the mesh and existing results; the changed planes apply to the next analysis.

Surface rendering retains every imported triangle and lights both sides. The
solver import repairs consistent winding and orients closed components outward;
open and nonmanifold surfaces retain explicit geometry audit notes. Under
**Model → Surface handling**, the default **Closed solid** checks orientation.
Select **Two-sided / intentional open surface** (`two_sided_open`) to accept
an intentional open or ambiguously oriented surface and allow an existing mesh
past that geometry admission check. The solver already intersects triangles
from both sides; duplicating coincident backfaces is unnecessary. This option
does not repair topology or bypass wall-coverage and GPU-memory checks. Open
surfaces can change the computed flow and remain identified in the audit.

Use Aero Lab's **Outputs** menu (Outputs and reports in Airfoil Lab) to select
VTK, native surface and cut data, PNG images, and CSV histories. Choose an
explicit output root before **Run solver**; a blank destination opens Outputs
instead of starting a job. Each analysis creates a dated study subfolder with
the complete native output, a study snapshot, `run_manifest.json`, and typed
mirrors of selected formats. Old implicit `TEMP_DELETEABLE/exports` defaults
are cleared when restored; explicitly selected destinations are preserved.
Internal GUI files also remain in private runtime storage.
On Windows, folder Browse buttons for outputs, studies and imported results
open the native folder-selection dialog. Paths can also be entered directly.
Set Display results every to the requested number of solver iterations. Makie
can generate HTML reports, PNG/SVG/PDF charts and PNG 3D renders with a selected
quantity, camera and image size. These run in the separate `tools/postprocess`
environment. Output choices and cadence can be changed before Run without
rebuilding the Aero lattice.

**Read/display results as they appear** in Run and Results controls automatic
snapshot reading. Turn it off to retain the current display; **Read latest
results now** still reads available surface and recorded-cut snapshots on
demand. Results explicitly identifies missing surface, volume or cut data.
Under **Results → Files, generated images and downloads**, file groups start
collapsed and retain user expansion choices while polling. The image gallery
comes first. **Follow latest image** advances to the newest flow image while
live reading is enabled; choosing an image manually stops following and retains
that choice. Recheck Follow to resume it.

**File → Open study** offers a recent-studies list with one-click loading, dates,
paths and saved-analysis hints. The list persists across sessions and includes
run snapshots. **Browse other study** opens the native Windows dialog.
Opening associates the source path and file hash with the study. **Save study** writes back to that file;
**Save study as** chooses a new path. An externally changed file requires a
fresh overwrite decision. Cancelled dialogs preserve the previous association.
The portable `.ludwig-study` file contains
the complete editable setup, canonical simulation geometry, original upload when
available, view settings, and pointers to existing results. A retained copy lives
in `02_PRIVATE_PROJECT_DEVELOPMENT/studies/aero`, outside temporary storage.
Opening restores geometry and settings, including retained GLB materials and
textures. Compatible local results attach without replacing the editable study
cut list or starting another solve. Large results remain in their referenced
folders and are not embedded.

**File → Read native / VTK results** loads a file, results folder, or multiple
uploaded files without running a solver. It supports native LBMP surfaces and LBMSLICE
planes, inline ASCII/binary/zlib VTU/VTP, and ASCII legacy VTK surfaces. Appended
XML and binary legacy VTK are reported as unsupported. Matching AMR metadata
supplies native placement and level outlines; explicit coordinate mode and
X/Y/Z offset can override it. Native surface pressure/shear retain Pa; external
VTK values remain in their stored units unless case scaling is known. Surface
quantities colour the model; volume quantities appear on the movable plane.
Study result references do not automatically follow foreign paths.

One retained 1,000-iteration development case has 20 surface snapshots, 33
recorded cuts and 108 PNG images, with volume output disabled. Its saved fields
and images can be viewed without another run; it cannot supply arbitrary new
volume sections.

The VRAM percentage stays editable in manual and automatic sizing. Manual mode
retains the chosen cell size and mesh when only its budget changes. Oversized
active jobs are cancelled, built mesh data are retained when present, and Surface
cell size turns red. Run admission rechecks the percentage before allocation;
running-job checks use the admission allowance, not free memory after the
solver's own allocation.

**Domain → Dimensions and clearances → Domain sizing** defaults to **Minimum
clearances**. The mesher can enlarge the outer domain when the requested cell
size and AMR level count need more room, while preserving a ground or symmetry
plane and the model's clearance from it. **Fixed closed section** disables
automatic domain growth for AMR; native lattice snapping is reported. Boundary
conditions are selected separately. A combination that cannot retain all requested
levels requires a different cell size, level count, or section. The summary
retains the requested envelope and reports resolved dimensions. Requested levels
are not silently removed.

Choose **Domain → Domain sizing → Domain proportions** to enter positive
X:Y:Z target ratios. The mesher scales the domain to retain the requested
minimum clearances and all AMR levels. Native coarse-block rounding is reported.
With symmetry, Y is automatic: half the shorter resolved X or Z dimension.
This Y rule also applies to the default Minimum clearances mode; X and Z may
grow and round to compatible coarse-block counts to keep the AMR hierarchy
safe. Fixed closed section is exempt from that automatic proportion rule.
The finest shell uses the native one-block minimum instead of two blocks;
wall coverage, donor halos, parent coverage and grading remain enforced.

The Free air choice uses distant transverse far-field boundaries and slip on
the top and bottom faces. Increase vertical clearance when those faces could
affect the flow. For **Domain → Boundary conditions → Symmetry plane**, the
Model parameters show two placement radios. **Move to plane of symmetry**
places a full model's Y centre at Y = 0 and uses its positive half. **Adjust
model to wall** places the minimum Y edge of a prepared half model on that
plane. Access these under **Model → Scale, rotate and place**.
Ground places the model's minimum Z on the wall; a positive Z
offset adds clearance. The pinned lbmtau solver has no arbitrary placement;
use the main LBM solver for model offsets.

The GPU run uses the same private shared lock as Airfoil Lab. If another run
holds it, Aero Lab shows the owner, allows cancellation, and reports a timeout
after three minutes. Uploads, initial mesh builds and runtime metadata use
`02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE/aero_lab`. Accepted runs retain
their geometry, mesh and complete results in the selected persistent output
folder. Open its study snapshot to restore the case even if disposable runtime
metadata has been removed. Keep an active case's runtime and output folders
until the job has stopped.

The Aero Lab browser viewer uses the bundled Babylon.js core from Airfoil Lab
and the matching Babylon loaders in `static/vendor/` for STL, OBJ, and GLB.
The loader's Apache-2.0 license is kept beside the bundle.

Completed mesh summaries are validated from complete in-memory snapshots.
Job-state replacement is atomic, and brief Windows file locks are retried.
The browser retains the last valid model and results through refresh failures;
the first two failed polls retry without an interrupting dialog. Persistent
errors open a warning with **Open diagnostic report** when the server provides
a report. A successful refresh clears that matching warning automatically and
resets the retry count without dismissing unrelated notices.

The launcher records the current server and reuses it after checking its API
revision and repository identity. Older servers are not reused; a new GUI opened
on an obsolete backend identifies the mismatch before allowing a mesh request.
On Windows, `run.cmd` prints a unique startup log path under
`02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE/ludwig_web`. It checks Julia and
the project, retains stdout/stderr, and keeps the console open on failure. A
successful reuse waits for the browser handoff, displays the existing server's
URL and confirms that it is online, then waits for a keypress before closing
the launcher. Closing this helper does not stop the existing server. A newly
started server keeps its console open while it hosts the app. Automated callers
can set `LUDWIG_WEB_NO_PAUSE=1` to skip the success and failure prompts, and
`LUDWIG_WEB_NO_BROWSER=1` to suppress browser opening.

PonD (Particles-on-Demand) remains a research solver family. GPU finite-volume,
thermal, DUGKS, STL immersed-boundary and sparse/AMR modules exist, with canonical
benchmark and kernel checks. The current top-level PonD dispatch still uses a
separate uniform-grid sphere driver; general geometry and AMR are not connected
to Aero Lab. It is therefore described in the solver guide, not offered as a
selectable validated backend.
