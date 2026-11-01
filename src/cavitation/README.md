# Conservative Species cavitation solver

`cmesh_cav_species.jl` is the production Species transport module. `cmesh_cav_tem.jl` supplies its configuration and the older TEM stepping path. The modules use the carrier in `../compressible/` and supporting code in `../cmesh/`.

The production tunnel driver is `../../tools/cmesh_cav_movie.jl`. The old paths in `../cmesh/` are compatibility includes for existing scripts. The separate pressure-based PBCav solver is under `../../solvers/pbcav_s1030/`.
