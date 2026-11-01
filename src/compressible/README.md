# Compressible C-mesh solver

`cmesh_solver_ka.jl` is the active CPU/GPU carrier. `cmesh_solver.jl` is the legacy CPU version. Mesh generation and the shared flux, EOS, wall, and outlet modules remain in `../cmesh/`; the cavitation solver also uses the carrier.

Airfoil Lab loads the active solver from `apps/airfoil_web/src/backends/cfd_backend.jl`. The old paths in `../cmesh/` are compatibility includes for existing scripts.
