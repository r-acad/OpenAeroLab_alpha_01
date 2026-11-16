# Ludwig Workbench

Run the [root setup](../../README.md#install-and-launch) first; it installs the
packages for both workbenches, supported solver workers and Makie. Launch
`run.cmd` on Windows, `bash run.sh` on Linux or `bash run.command` on Mac.
The launcher opens or reuses the current local server, trying port 8058 first
when it needs a new one. It opens Aero Lab and Airfoil Lab.
Airfoil Lab also keeps its standalone launcher in `../airfoil_web/`.

The launcher and Aero Lab API live here; Airfoil Lab is included from its own
app source and served at `/airfoil-lab`. Its `/api/*` and `/static/*` URLs
remain available. Aero Lab uses `/api/aero/*`, with runs and uploads in the
sibling private `02_PRIVATE_PROJECT_DEVELOPMENT/results/aero_lab` directory.
The server binds only to loopback.

## Aero Lab

Aero Lab runs 3D external flow around an imported model. The page has no
menus. The **Solver** is chosen first, above the setup pane; the row of
numbered steps then shows, step by step, only what that solver uses:

**1 Model, 2 Domain, 3 Flow, 4 Mesh, 5 Run and outputs, 6 Results**, then
**Study, Display, Help**.

Each step ends with its action (**Create mesh** in step 4, **Run solver** in
step 5). Rarely needed settings wait in one closed **Advanced** group per
step. Changing steps 1 to 4 after a mesh exists needs a new mesh; step 5 does
not.

### Solvers

| Solver | Device | Range | Status |
|---|---|---|---|
| GPU lattice Boltzmann · D3Q27 cumulant | NVIDIA GPU | Mach at most 0.6, 1 to 7 AMR levels | development |
| GPU lattice Boltzmann · D3Q39 entropic, compressible | NVIDIA GPU | Mach at most 1.6, 1 to 7 AMR levels, thermal energy | development; only short web cases have been run |
| GPU thermal kinetic finite volume | NVIDIA GPU | Mach 0.05 to 1.5, slip or laminar no-slip walls, no turbulence model | experimental |
| CPU octree lattice Boltzmann · D3Q19 | CPU, system RAM | Mach at most 0.3, 1 to 10 octree levels | development |

None of them is a validated flow prediction. A completed run means the
requested steps were reached, not that the flow converged. Mac supports study
and result viewing and the CPU octree solver; the GPU solvers need Windows or
Linux with an NVIDIA GPU. The solver guide at `/solvers` describes the methods.

The pinned `lbmtau` snapshot of the lattice Boltzmann solver is no longer an
Aero Lab solver (Airfoil Lab still uses it). A study saved with it opens with
the solver selected on the page, and says so.

### 1 Model

Load an STL, OBJ or GLB file. Scale, rotate and offset it with the numbers, or
drag it in the 3D view (**Move**, **Rotate**). Model X is the flow direction,
Y lateral, Z up; rotation is about the model centre.

**Surface handling** defaults to **Closed solid**, which checks orientation.
**Two-sided / intentionally open surface** accepts an open or ambiguously
oriented surface and lets an existing mesh past that geometry check. The
solvers intersect triangles from both sides; the option repairs nothing and
does not bypass the wall-coverage and memory checks.

Advanced (GPU solvers): **Maximum surface triangle edge**, default 50 mm. The
placed model's triangles are subdivided without changing its shape and the
solver samples pressure and shear on the smaller faces. Fluid resolution still
depends on the cell size. The CPU octree solver keeps the source facets and
samples the wall per fluid cell, so it has no such setting.

### 2 Domain

Choose **Free air**, **Ground** or **Symmetry plane**, and the upstream,
downstream, lateral and vertical clearances. Free air uses far-field side
faces and slip on top and bottom. Ground puts the model's lowest point on the
wall; a positive Z offset lifts it. With a symmetry plane only the positive-Y
domain is solved, sideslip is zero, and three placements are offered: leave
the model where it is (a manual Y offset stays possible and is kept in a
study), centre a full model on the plane, or put a prepared half model against
it.

Advanced: **Domain sizing** (minimum clearances, which may grow to keep the
mesh levels; a fixed closed section; or target X:Y:Z proportions) and the
**X domain offset**, which shifts the domain relative to the model.

### 3 Flow

Angle of attack and sideslip rotate the model into the +X free stream. Set
speed, sound speed, density and kinematic viscosity. For the thermal solver the
free stream is set by Mach number and temperature (ideal air, gamma 1.4,
R = 287.05 J/(kg K)); speed and sound speed are shown as derived values, the
wall choice is slip or laminar no-slip, and the CFL number is under Advanced.

**Reference dimensions**: total wing area, chord, optional span, and the
moment reference point in model axes (X aft, Y right, Z up, from the model
centre after placement, before the flow angles). They give the coefficients
shown under Results and can be edited at any time.

### 4 Mesh

Set the **surface cell size** and the number of **mesh levels**, then
**Create mesh**. The mesh, its cell counts, resolved domain and memory
estimate appear before a run is possible; **Mesh details** in the header lists
them. A severely sparse surface mesh must be rebuilt. When a build cannot meet
its limits and an alternative exists, the message offers a proposed cell size
and level count that one click applies.

GPU solvers: the cell size is used as given, or found from a **GPU memory
target** (a percentage of the VRAM free now): build-only probes on the CPU
select the finest cell size whose run-memory estimate fits. The target is
checked again before the run.

CPU octree solver: the **finest band** and **coarser bands** (cells kept
around the surface per level), the **wake** box (length in body lengths, level
below the finest), **Exclude interior cells** for external flow around a leaky
model, the **wall function** (experimental, not validated; off by default)
and the number of **CPU threads**. Advanced: the inlet lattice speed (it sets
the time step), the hybrid collision number (0.98 by default; 1 means no
hybrid part), the checkpoints kept and the sample limit of a saved plane. The
solver has one collision, starts from the uniform free stream and absorbs at
every far-field face; these are not choices.

### 5 Run and outputs

Set the **solver steps**, the output folder (required), the saved formats and
the result interval, and the **planes recorded during the run** (three
midplanes by default). **Run solver** starts the run; **Cancel** stops the
active mesh build or run. Each run creates a dated subfolder with the complete
native output, a study snapshot, `run_manifest.json` and the selected export
formats. For the CPU octree solver the page forecasts the storage of the run
and refuses a run that cannot hold its surface frames, final checkpoint and
reserve.

A GPU run takes the same shared lock as Airfoil Lab; if another run holds it
the page shows the owner and reports a timeout after three minutes. The CPU
octree solver takes no lock and is refused only when the run would not fit in
80 % of the available RAM.

### 6 Results

**Show results as they appear** follows new snapshots; **Read latest results
now** reads them on demand. Choose the 3D view, the recorded planes or the
images and files. The surface field can show Cp, pressure or shear, sampled
per wall cell (**wall samples**, CPU octree solver) or per model triangle
(**facet averages**); grey marks geometry without a sample. **High contrast**
spans the 2nd to 98th percentiles and reports the true extremes.

**Coefficients** shows CL, CD, CY, the body moments and L/D from the latest
saved loads, the reference dimensions of step 3 and `q = rho U² / 2`. With a
symmetry plane the full loads are divided by the full area.

Advanced: colour limits per quantity, **Save 3D view PNG**, and **Read other
result files**: a file, a folder or uploaded files (LBMP surfaces, LBMSLICE
planes, inline VTU/VTP, ASCII legacy VTK) are read without a run.

The bar above the view (**Surface, Recorded planes, Images & files, Load saved
results, Open other results**) stays in place for every step, so a loaded
field is never stranded by a change of step.

### Study, Display, Help

**Study** opens, saves and renames studies. A `.ludwig-study` file holds the
setup, the simulation geometry, the original upload when available, the view
settings and links to results; large results stay in their folders. Opening
lists recent studies and run snapshots. Results that match the opened setup
attach automatically; results of a different setup are shown as historical.

A study saved by an earlier version opens with the current default for any
setting that no longer exists. Two cases are refused with a message: a study
with a symmetry plane that names no model placement, and a request for a
solver the backend does not have.

**Display** holds the view (geometry, mesh, surface field), camera, guides,
fit buttons, palette, model opacity and GLB appearance, placement projections,
the visibility and drawing of each mesh level, and the **movable section**: a
plane through mesh and fields that can be dragged, clipped, shown in 2D and
added to the recorded planes of the next run. Its Advanced group shows the
aerodynamic axes. Display settings never change the case.

**Help** repeats the steps and links to the solver guide.

### Notices

The header carries **Solver log**, **Hide mesh / Show mesh**, **Mesh
details**, **Messages** and the **i** panel (domain, axes, boundary
conditions, file paths). Mesh creation and the solver each have a progress
card with a retained log. Important messages open a closable dialog; a
dismissed warning stays closed while the job is polled. A failed refresh keeps
the last valid model and results and retries.

## Storage

Uploads, mesh builds and job metadata use
`02_PRIVATE_PROJECT_DEVELOPMENT/results/aero_lab`. Accepted runs keep their
geometry, mesh and results in the output folder chosen in step 5; open the
study snapshot there to restore the case. Saved studies also keep a copy in
`02_PRIVATE_PROJECT_DEVELOPMENT/studies/aero`. These records are durable; only
previews, staging files and launcher logs belong in `TEMP_DELETEABLE`.

## Case files

`src/aero/case.jl` writes one `config.yaml` per mesh build and per run, with
only the keys the selected solver reads:

- every solver: geometry file, reference values, fluid and flow speed, run
  length and result intervals, domain clearances and placement, symmetry mode,
  far-field flag, forces, diagnostics, recorded planes;
- the GPU solvers (`src/io/config_loader.jl`, `src/io/lbm_io_opts.jl`):
  surface subdivision, lattice and collision, mesher and wall schemes, inlet
  ramp, volume fields, slice images, GPU queue depth; the thermal solver adds
  its `pond_thermal` block and D3Q39 its thermal and shifted-lattice keys;
- the CPU octree solver (`src/octree_q19/src/config.jl`): the `octree_q19`
  block with precision, threads, mesh bands and wake, the hybrid collision
  number, the wall function, the sponge strength, outputs and checkpoints.

`tools/tests/aero_lab_markup.jl` builds the request of every solver from the
page's default values with the page's own builder (`static/aero_request.js`)
and checks that each case file names only keys its solver reads.

## Viewer and launcher

The 3D view uses the bundled Babylon.js core from Airfoil Lab and the matching
loaders in `static/vendor/` for STL, OBJ and GLB; the loader's Apache-2.0
license is kept beside the bundle. Click a model surface to put the orbit
centre there; right drag pans; double-click the background fits the domain.

The launcher records the current server and reuses it after checking its API
revision and repository identity. On Windows, `run.cmd` prints a startup log
path under `02_PRIVATE_PROJECT_DEVELOPMENT/TEMP_DELETEABLE/ludwig_web`, keeps
the console open on failure, and waits for a keypress after handing over to an
existing server. `LUDWIG_WEB_NO_PAUSE=1` skips the prompts and
`LUDWIG_WEB_NO_BROWSER=1` suppresses browser opening.

A reporting client can request the exact saved surface frame of an open tab
(**Provide this view for Telegram reports**, under Results → Advanced, or the
link `/aero-lab?job=CASE_UUID&reporting=1`); the server verifies the job,
frame and geometry hashes before accepting an image.

The original PonD (Particles-on-Demand) modules remain a research solver
family described in the solver guide; they are not an Aero Lab solver. The
thermal finite-volume solver offered here is a separate path.
