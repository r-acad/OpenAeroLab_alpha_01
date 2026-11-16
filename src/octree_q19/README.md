# CPU octree D3Q19 solver

This experimental CPU backend uses cell-level 2:1 octree refinement, a D3Q19 hybrid regularised collision with a subgrid model, interpolated Bouzidi walls with an optional wall function (experimental) and absorbing far-field layers. It is separate from the retired GPU D3Q19 implementation. It is not validated for aircraft performance or stall prediction.

Run the repository installer to download its Julia dependencies. In Aero Lab, choose **Solver source → D3Q19 CPU octree (development)**, set a manual surface cell size and 1–10 levels, create the mesh, choose a persistent output folder, then Run. No NVIDIA GPU or GPU lock is required. The app limits this athermal backend to Mach 0.3.

The command-line equivalent is:

```sh
julia --project=src/octree_q19 -t auto src/octree_q19/main.jl --case /path/to/case
```

The case directory contains `config.yaml` and its STL. `--build-only` creates `mesh.oct`, AMR inspection JSON and a build summary. Normal runs reuse a matching mesh and write native surface and plane data, optional surface/volume VTK, force and diagnostic histories, and checkpoints. `--resume` continues a matching checkpoint. A mesh file that does not match the case is rebuilt.

Native planes are limited to four million samples each by default; larger domains increase the export sampling spacing without changing the solver mesh. `octree_q19.outputs.max_plane_samples` may specify 10,000–16,000,000 samples. The exported header records the actual spacing and sampling stride. The app can read older, larger local native planes with bounded sampling.

The displayed RAM allowance includes mesh storage, state and AMR transfer buffers, per-thread force arrays and export work with a reserve. Completed requested steps are explicitly reported separately from convergence. Symmetric cases report full-model loads. A pressure/viscous force decomposition is not available; legacy `_p`/`_v` history columns must not be interpreted as a measured split. Surface pressure sampling on thin, open or poorly resolved features still needs inspection.

## Numerical method and settings

The `octree_q19:` block of `config.yaml` holds the solver's own settings; nothing else selects a method.

```yaml
octree_q19:
  precision: f32                # f32 | f64
  mesh: {levels: 6, finest_dx_m: 0.0192, band_fine: 8, band: 6, seal_cavities: true, wake: {length: 0, level_offset: 1, margin: 0.1}}
  numerics: {hrr_sigma: 0.98, sgs: wale, u_lattice: 0.05}
  walls: {wall_function: {enabled: false, sampling: normal_1, filter_steps: 50, pressure_gradient: false}}
  sponge: {strength: 0.05}
  outputs: {surface_vtk: false, surface_lbmp: true, max_plane_samples: 4000000, averaging_start_ctu: 0}
  checkpoint: {every_minutes: 30, keep: 2}
```

- **Collision.** Hybrid recursive regularisation: a share `1 - hrr_sigma` of the second-order moments comes from finite-difference velocity gradients; `hrr_sigma: 1` is plain recursive regularisation. The default 0.98 is a stability choice: on an aircraft start at Reynolds number 4.7 million the plain form left the stable range within 70 coarse steps, 0.98 kept the density within 1.6 % with no limiter call. The hybrid part adds dissipation (shear-wave viscosity error +7.6 % at sigma 0.8 and 16 cells per wavelength; not measured at 0.98). Subgrid model `sgs`: `wale` (default), `smagorinsky`, `vreman` or `none`.
- **Kernel.** The step works on the eight cells of an oct at a time (one SIMD vector per population; `src/kernel.jl`, package SIMD.jl), with the collision in closed form from sums and differences of opposite populations. Cells cut by the surface gather their arriving populations one by one and join their oct's collision; cells beside a domain face are stepped one by one. The transfers between levels work oct by oct as well. `tools/tests/q19_kernel.jl` checks the closed form against the per-direction definitions (9e-16 in double precision) and the vector step and transfers against the cell-by-cell forms, bit for bit. Measured on a 16-core Ryzen 9 5950X, 16 threads, single precision, C212 half model: about 30 million cell updates per second (12 to 13.5 with the kernel before 14 November 2026). The solver is limited by memory bandwidth on that machine: 32 threads give no more than 16, and the level transfers are half of the time.
- **Walls.** Interpolated bounce-back (Bouzidi) on the cut links. With `wall_function.enabled`, an equilibrium wall function (Reichardt's law) acts through those links as a slip velocity of the wall; `src/walls.jl` describes the steps. On a resolved wall it vanishes (a Poiseuille channel is unchanged to four digits). `sampling` is where the speed is read: the wall cell (`cell`) or the fluid cell one or two cells out along the normal (`normal_1`, `normal_2`). `pressure_gradient: true` adds the adverse-pressure-gradient term of the law (the friction falls as the gradient grows and vanishes at the half-power profile of separation). **The wall function is experimental and not validated; it is off by default.** In a turbulent channel at friction Reynolds number 2000 (half-width 20 cells, wall cells at y+ 50; reference bulk velocity 21.7 friction velocities) it gives 16.7, and no-slip walls give 19.1: the wall takes about three times the friction the law asks for. In the coarse channel of the test (8.6 cells across, wall cells at y+ 5 to 17) the run is stable, keeps its mass, and the wall receives 1.2 to 1.3 times the modelled friction, against 3.5 to 4.4 times with no-slip walls. `src/walls.jl` lists the forms tried to close that budget and why each was dropped. The pressure-gradient term is checked on the law only, not on a flow. Cells where opposing surfaces meet keep the no-slip rule. `tools/tests/q19_wall_model.jl`.
- **Far field.** Cells within `advanced.domain.sponge_thickness` reference lengths of an inlet, outlet, freestream or slip face are blended toward the freestream with strength `sponge.strength` (0 turns the layers off). Slip faces reflect sound perfectly; without a layer in front of them a start rings for the whole run. The layers force the freestream near those faces: keep the body well clear of them.
- **Start.** A run starts from uniform freestream. The start still sends a pressure wave through the domain; forces from the first flow-through times are not usable. The running force means start `outputs.averaging_start_ctu` chord flow times into the run.
- **Transfers between levels.** Moments on cubic stencils, rebuilt at the receiving level's relaxation rate, linear in time. They are not flux-conserving: a flow crossing the interfaces leaves a one-time mass change of order 1e-4 with no drift; `mass_control: true` rescales the mass of closed or periodic cases.

In Aero Lab the D3Q19 settings show the same choices.

An **experimental staged startup** can transfer a completed checkpoint to a separately built, finer mesh:

```sh
julia --project=src/octree_q19 -t auto src/octree_q19/main.jl --case /path/to/new-fine-case --from-case /path/to/completed-coarse-case
```

The target must have a prebuilt matching mesh, more levels and a larger total step limit. Geometry, physical conditions, boundaries, force references, coarse spacing and coarse time step must agree. Both states must fit in RAM with a reserve. The source files are read-only; physical time continues from the source checkpoint, and the target writes `RESULTS/refinement_start.toml`. The transfer is a startup approximation, not conservative dynamic AMR. Near-wall cells with no visible donor can be initialized at no-slip reference-density equilibrium only under a bounded, recorded fallback. Geometry voxelization changes, reconstructed stresses and unavailable half-time history can cause a transient. Review the receipt and early flow fields; this method does not establish solution accuracy or convergence. Newly exposed target cells with no visible source donor start at rest and reference density: within a quarter cell of the surface as before, and (since 14 November 2026) also farther away, where they lie in pockets the coarser mesh did not resolve as fluid; both kinds are counted, their coordinates recorded in `refinement_start.toml`, and their total is limited. Before that change the second kind stopped the run.

Numerical checks are in `tools/tests/q19_*.jl`; run them with this project environment, except the Aero Lab API tests (`q19_octree_app.jl`, `q19_output_cadence.jl`), which use `--project=third_party/JXfoil`.

## Native wall-pressure samples

Surface output also writes `RESULTS/VTK/surface_NNNNNN.lbmw` and a reusable `surface_wall_geometry_<sha256>.lbmwg`. These retain the pressure variation between adjacent fluid wall cells, including variation within one original STL triangle. Intersections sharing the same fluid cell and original facet are coalesced because they use exactly the same density. The sample position is their mean intersection, on that facet. Normals come from triangle geometry and face the adjacent fluid cell, independently of the STL winding.

The pressure is the adjacent cell estimate `rho0 * (Uref/u_lattice)^2 / 3 * (rho_lattice - 1)`; `Cp = pressure / (0.5*rho0*Uref^2)`. It is not an extrapolated wall value. Opposite fluid sides remain separate samples. Unmapped links, unsampled faces and the excluded symmetry half are not filled or mirrored. No local shear value is claimed. Original-facet `.lbmp`/VTK output and force integration are retained for comparison; they still average pressure over each original facet.

Fine output follows the field cadence independently of force recording. Native YAML `octree_q19.outputs.wall_samples: false` disables it. It is generated when wall samples are enabled and either native surface output or surface VTK is requested. Static geometry is written once, and each native frame needs only four bytes per sample plus a 128-byte header. For seven million samples this is about 252 MB of static geometry and 28 MB per native frame. Disk admission must include the selected number of frames; the full dataset is never silently decimated. Browser display sampling, when used, is separate from this scientific output.

With `surface_vtk: true` and `wall_samples: true`, each fine frame also has an uncompressed `surface_wall_samples_NNNNNN.vtp` and the directory has a `surface_wall_samples.pvd` time collection for ParaView. The VTP has one vertex per measured wall-cell/facet sample, with `Pressure`, `Cp`, `LocalCellSize`, `OriginalFacetId`, `AMRLevel` and `SupportFlags` point arrays. Use a points or point-Gaussian representation; this point cloud does not assert watertight surface triangles or provide force quadrature. It adds 39 bytes per sample per frame, about 273 MB for seven million samples. Even when native surface output is disabled, the matching compact native wall samples are retained as the source for this VTP export. Disable surface VTK or wall samples to avoid these VTP files.

To convert an already recorded frame without running the solver or altering its native input:

```sh
julia --project=src/octree_q19 src/octree_q19/export_wall_vtp.jl --frame /path/to/surface_000012.lbmw
```

The converter uses bounded working memory, verifies the geometry and frame identities, and refuses to replace a different existing VTP.

A stored checkpoint can be exported without advancing the solver or rebuilding its mesh:

```sh
julia --project=src/octree_q19 src/octree_q19/export_wall_checkpoint.jl --case /path/to/case --checkpoint /path/to/checkpoint_000000012.bin
```

`--mesh` selects an existing `mesh.oct`; `--output` selects a RESULTS directory. Mesh, configuration and checkpoint identities must agree. The command reads only the saved wall-cell densities, never creates population buffers, and records the checkpoint SHA256 in the frame sidecar. It preserves old results and refuses to replace different existing fine frames. Reading and hashing a large checkpoint can take several minutes.

### Binary format version 1

Both formats are little endian. `.lbmwg` starts with the eight bytes `LBMWGEO\0`, version `UInt32(1)`, record size `UInt32(36)`, and sample count `UInt64`. Each 36-byte record contains position `Float32[3]`, unit normal `Float32[3]`, local cell size `Float32`, one-based original facet ID `UInt32`, AMR level `UInt8`, contributing link count `UInt8`, and flags `UInt16`. Flag bit 0 means a supported intersection; bit 1 marks a grazing/ambiguous fluid-facing normal. No implicit struct padding is written.

`.lbmw` starts with `LBMWFLD\0`, version `UInt32(1)`, field mask `UInt32(1)` (pressure), sample count `UInt64`, the static geometry's SHA256 as 64 ASCII bytes, step `Int64`, then physical time, reference density, reference speed and reference dynamic pressure as four `Float64` values. This is exactly 128 bytes, followed by one pressure `Float32` per geometry record; NaN remains missing. Its `.toml` sidecar identifies the geometry filename, units, source hashes, coverage and within-facet pressure variation. Readers must verify geometry count and hash before pairing data. Geometry and sidecars are published before the atomic field file, so incomplete frames are not discoverable.
