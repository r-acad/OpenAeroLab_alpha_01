# Native low-speed driver outputs and restart

For normal airfoil work, AirfoilLab reads the coordinate file, prepares the mesh,
and records its settings. The native `pbcav_driver.jl` is also available for
controlled command-line studies in this solver's own Julia environment. It takes
registered `PB_*` environment variables, with no positional arguments.

`PB_FOIL` names a normalized XFOIL/Selig contour ordered from the trailing edge
around the leading edge and back. Use the same mesh, flow, turbulence and start
settings for a clean/flap comparison. A Gurney flap is an interior baffle on a
finite-trailing-edge C+H mesh; it requires the strip topology and matching base
resolution. AirfoilLab supplies those settings from the imported contour.

Run Julia with `--project=solvers/pbcav_s1030/src/pbcav` from the repository root.
The existing driver's mesh/physics settings remain `PB_*` values. For example,
add these output and startup settings to a native free-air case:

```julia
withenv("PB_DOMAIN"=>"freeair", "PB_STOP"=>"run",
        "PB_FOIL"=>raw"C:\airfoils\section.dat",
        "PB_RE"=>"3000000", "PB_ALPHA"=>"12",
        "PB_TURBULENCE"=>"sst", "PB_START"=>"ramp",
        "PB_START_RAMP"=>"2,0.05", "PB_TURB_MUT_BASE"=>"committed",
        "PB_TURB_STEP_RELAX"=>"0.5",
        "PB_DT"=>"0.25", "PB_START_DT"=>"0.005",
        "PB_CPU_LAUNCH"=>"balanced",
        "PB_OUT_DIR"=>raw"C:\studies\section_run_01") do
    run(`$(Base.julia_cmd()) --startup-file=no --threads=6
        --project=solvers/pbcav_s1030/src/pbcav
        solvers/pbcav_s1030/tools/pbcav_driver.jl`)
end
```

Set the appropriate `PB_MESH`, `PB_MESH_TE_TOPOLOGY`, base-cell, and remap knobs
for the contour before using this example. The driver's default mesh is not an
automatic airfoil mesh policy. The app's `input.toml` and solver log retain the
resolved settings of each web run.

`PB_START_DT=0.005` caps a cold start's first requested step. Accepted steps grow
toward `PB_DT` under the ordinary adaptive controller. Rejected attempts are
still reduced and reported. A restart uses its saved timestep and controller;
the cold-start cap is not reapplied.

AirfoilLab's steady defaults also use `PB_TURB_STEP_RELAX=0.5` to damp the
diagnosed alternating SST iteration. The native reference default remains 1.
This setting damps the nonlinear update; it does not relax either steady
stopping tolerance. A finite `PB_TURB_STEP_TOL` requests extra inner iterations
and can be expensive during a cold start; it is not required for the app's
ordinary steady workflow.

For the app's C+H preconditioner setting, use `PB_CPL_AU=0.95`. The native
reference default remains 0.8. This factor changes only the SIMPLEC block
preconditioner; the linear solve still has to meet its true-residual target.

`PB_CPL_RESTART` controls the FGMRES subspace size (integer 1–120, native
default 30). `PB_CPL_MAXIT` limits iterations per linear solve (integer
1–10000, native default 300). AirfoilLab uses restart 60 for both clean and
Gurney C+H runs, with maximum 300; its sharp C route retains 30/300. For the
same C+H settings through this driver, set `PB_CPL_RESTART=60`. The larger
restart uses more basis memory, while the verified residual goal and all
outer/steady acceptance criteria stay unchanged. The actual configuration
receipt prints both settings. Exact continuation requires both to match the
checkpoint, along with the loaded solver source; an older-source checkpoint
remains refused after a solver update.

The free-air boundary transports outgoing momentum from the upwind interior
cell while retaining the prescribed far-field normal flux and viscous value.
Its pressure outlets use Bernoulli pressure and a known freestream plus
point-vortex state if flow enters the domain. Velocity gradients, SST inflow
and the coupled boundary Jacobian use the same branch. Generic pressure
outlets used by other native problems retain their existing behavior.

The following files are written in `PB_OUT_DIR` after the march:

- `loads_history.csv`, `surface.csv`, and applicable baffle/turbulence histories.
- `fields_stepNNNNNNNNN.vtu`: the final committed cell field, including velocity,
  speed, pressure, Cp, density, k, omega, and eddy-viscosity ratio. Open it in a
  VTU-capable viewer. `PB_OUTPUT_FIELDS=off` disables this file.
- `state_stepNNNNNNNNN.pbrestart`: a portable native checkpoint containing all
  three state time levels, boundary values, circulation history, timestep
  controller and conservation ledger. `PB_OUTPUT_CHECKPOINT=off` disables it.

To continue a native checkpoint, keep the same mesh, geometry, physics and
numerical settings, set `PB_RESTART` to its path, and choose a **new output
directory**. `PB_MAX_STEPS` counts additional committed steps for that invocation.
The wall-time budget, stopping tolerances, output switches and output paths may
change. The checkpoint reader checks solver-source and mesh signatures, settings,
array types/sizes, finite physical state, time counters and continuation metadata
before changing a destination state. Incompatible input is rejected rather than
used as a partial warm start.

The checkpoint data itself round-trips exactly. Resuming rebuilds work arrays and
linear-solver caches, so subsequent arithmetic can differ at Float64 roundoff.
The native regression compares uninterrupted and split continuation, including
timestep regrowth, vortex history, conservation counters and nonzero wall loads;
it does not claim a bitwise-identical future trajectory on every mesh or backend.
Saved fields and small residuals do not establish mesh independence or validation
against experimental airfoil data.
