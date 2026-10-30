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
