# Solver source map

| Solver | Source | Entry point |
| --- | --- | --- |
| Compressible C-mesh | `compressible/` plus shared `cmesh/` | `apps/airfoil_web/src/backends/cfd_backend.jl` |
| Conservative Species cavitation | `cavitation/` plus shared `cmesh/` | `tools/cmesh_cav_movie.jl` or the app backend |
| LUDWIG LBM | `main.jl`, `collision/`, `geometry/`, `forces/`, `io/`, `solver/` | `main.jl`; the app uses the pinned `../solvers/lbm_lbmtau/` snapshot |
| Pressure-based low-speed/cavitation (PBCav) | `../solvers/pbcav_s1030/src/pbcav/` | `../solvers/pbcav_s1030/tools/pbcav_driver.jl` |
| JXfoil | `../third_party/JXfoil/` | Airfoil Lab's JXfoil backend |

Additional research paths remain in `barotropic/`, `collision/pond/`, and the free-surface files in `cmesh/`. They share data structures and callers with the existing source tree, so this layout change leaves their file paths intact.

`cmesh/` keeps mesh generation, common C-mesh physics, and supporting modules used by more than one solver. Its four short compatibility files forward old include paths to `compressible/` and `cavitation/`. The pinned solver snapshots retain their own layout and checksums.
