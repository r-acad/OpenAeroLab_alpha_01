# Airfoil Lab

Launch with `run.cmd` on Windows, or run `run.jl` from the bundled JXfoil
project. The app uses the public `src/` and bundled `solvers/` sources. Output
goes to the sibling private development directory (or `LUDWIG_PRIVATE_ROOT`).

The app includes JXfoil, compressible C and C+H meshes, the conservative
Species path, low speed pressure based CFD, and LBM 3D. XFOIL executable runs
require `AIRFOIL_XFOIL_EXE`. Published reference comparisons require optional
datasets in the private directory.

The default C+H Standard mesh uses curvature aware leading edge spacing. Mesh
quality and solver convergence are reported with each result.
The sharp C mesh also concentrates surface columns near the leading edge in
Standard, Fine, and Very Fine. A separate low speed `fine` study profile uses
smaller nose and wall spacings; its outer mesh audit flags remain visible in
the result and it is not a validated stall prediction.

## Source layout

| Directory | Contents |
| --- | --- |
| `api/` | Job and campaign status helpers. |
| `backends/` | CFD backend adapters and orchestration. |
| `diagnostics/` | Time-history analysis helpers. |
| `lbm/` | LBM case construction, fields, and backend. |
| `lowspeed/` | Pressure-solver backend configuration and result handling. |
| `mesh/` | Mesh transforms, audits, and streamline alignment. |
| `references/` | Optional experimental reference integration. |
| `runtime/` | Runtime pieces used by compressible workers. |
| `static/` | Browser assets. |

The root keeps `run.*`, `server.jl`, `cfd_backend.jl`, worker entry points,
shared wire and field formats, and solver snapshot manifests. These are stable
paths used by launched processes and source receipts. Retired Gurney insertion
prototypes are kept in the private development archive.

The app offers six backend choices: JXfoil, optional external XFOIL,
compressible C-mesh, cavitation/Species, pressure-based low-speed CFD, and
LUDWIG LBM. `solvers/` at the repository root contains only the two pinned
copies used by the LBM and pressure backends. The primary C-mesh and LBM
source is in `src/`; JXfoil is in `third_party/JXfoil/`. The `solvers/`
directory is a snapshot store, not a catalog of all solver families.
