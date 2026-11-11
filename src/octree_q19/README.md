# CPU octree D3Q19 solver

This experimental CPU backend uses cell-level 2:1 octree refinement, D3Q19 regularised collision, optional subgrid viscosity and interpolated Bouzidi walls. It is separate from the retired GPU D3Q19 implementation. It is not validated for aircraft performance or stall prediction.

Run the repository installer to download its Julia dependencies. In Aero Lab, choose **Solver source → D3Q19 CPU octree (development)**, set a manual surface cell size and 1–10 levels, create the mesh, choose a persistent output folder, then Run. No NVIDIA GPU or GPU lock is required. The app limits this athermal backend to Mach 0.3.

The command-line equivalent is:

```sh
julia --project=src/octree_q19 -t auto src/octree_q19/main.jl --case /path/to/case
```

The case directory contains `config.yaml` and its STL. `--build-only` creates `mesh.oct`, AMR inspection JSON and a build summary. Normal runs reuse a matching mesh and write native surface and plane data, optional surface/volume VTK, force and diagnostic histories, and checkpoints. `--resume` continues a matching checkpoint.

Native planes are limited to four million samples each by default; larger domains increase the export sampling spacing without changing the solver mesh. `octree_q19.outputs.max_plane_samples` may specify 10,000–16,000,000 samples. The exported header records the actual spacing and sampling stride. The app can read older, larger local native planes with bounded sampling.

The displayed RAM allowance includes mesh storage, state and AMR transfer buffers, per-thread force arrays and export work with a reserve. Completed requested steps are explicitly reported separately from convergence. Symmetric cases report full-model loads. A pressure/viscous force decomposition is not available; legacy `_p`/`_v` history columns must not be interpreted as a measured split. Surface pressure sampling on thin, open or poorly resolved features still needs inspection.

The wall-law functions in the source are research components and are not applied in the time step. A run explicitly requesting a wall function or quadratic AMR time interpolation is rejected rather than silently using another method. Coarse-to-fine staged startup is not yet implemented in production. Meshes made with the older cavity-sealing classifier must be rebuilt before a new cavity-sealed run; old result and mesh files remain readable.

Numerical checks are in `tools/tests/q19_octree_*.jl`; run them with this project environment, except the Aero Lab API test, which uses `--project=third_party/JXfoil`.

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
