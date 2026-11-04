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

Numerical checks are in `tools/tests/q19_octree_*.jl`; run them with this project environment, except the Aero Lab API test, which uses `--project=third_party/JXfoil`.
