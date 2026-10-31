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
384 surface cells, 128 wall-normal cells, and a 7e-6 chord first wall spacing.
It passes the PBCav reference mesh gates for NACA 0012; the app still reports
the audit for every generated section. Grid independence and stall accuracy
are not established by a mesh-quality pass.
The pressure worker writes `rejection_history.csv` with the accepted step,
time step, and cumulative rejection counts by reason. A checked warm start
that changes incidence by at least 1.5 degrees begins at a small time step;
normal step growth resumes after accepted steps. The full steady residuals,
not stable-looking force coefficients, decide whether a polar point converged.

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
