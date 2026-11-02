# Thermal kinetic finite volumes on the LBM mesh

Aero Lab's `pond_thermal` backend is **Thermal kinetic FV (experimental)**.
It uses the native LBM Cartesian mesh builder and advances five conservative
variables: density, three momentum components, and total energy. Its face flux
is analytic Maxwellian kinetic flux-vector splitting (KFVS), evaluated through
half-space moments. It does not use the historical PonD gauge evolution,
discrete-population clipping, or a DUGKS collision flux.

The historical `pond_fv*`, `pond_gauge`, and `pond_case_runner` modules remain
separate research implementations. Selecting the new backend does not select
that legacy uniform-grid sphere driver.

## Mesh ownership and conservation

`PondLBMMesh.build_pond_lbm_mesh(grids)` consumes the actual `BlockLevel`
hierarchy from `setup_multilevel_domain`. The block allocation, cell positions,
refinement levels, and obstacle masks remain the native LBM mesh. Covered
parent cells and solid cells are excluded from the evolved state; their
ownership remains recorded in a block-local lookup.

Each fluid/fluid face is stored once. A coarse face beside a 2:1 refinement
interface is split into its four fine face patches. Both cells use the same
computed face flux with opposite signs, so coarse/fine conservation follows
from the shared face construction. All levels advance on one global physical
clock; there is no temporal subcycling or interpolation between level times.

The adapter checks nested ownership against `amr_covered`, rejects holes and
unbalanced faces, verifies that fluid and solid leaf volumes partition the
domain, and verifies each fluid cell's area-normal closure. This preserves the
constant-pressure geometric balance. It retains the original level, block,
and local cell index for every fluid cell. `build_point_index(mesh)` returns
the existing block-local lookup, and `find_leaf(index,x,y,z)` returns the
fluid leaf cell ID or zero for solid/outside points.

Immersed boundaries are **staircase voxel faces**. Their normals and areas close
the voxel fluid volumes. Original triangle normals are not substituted into
these flux faces, and the LBM Bouzidi link algorithm is not used by this solver.
The LBM wall-distance field is retained, but this is not a geometrically exact
cut-cell or boundary-layer mesh.

## Equations and stepping

`PondThermalFV.ThermalFV` uses limited least-squares primitive reconstruction
(MUSCL), analytic KFVS face fluxes, and SSP-RK2 updates. Limiting, walls, and
refinement interfaces can reduce local reconstruction order; no global
second-order accuracy claim is implied. If a stage produces inadmissible
density or pressure, the whole step is rejected and retried with a smaller
time step. Accepted conserved states are not repaired by density/energy floors.

The baseline wall choices are inviscid slip and laminar no-slip. The latter
adds Newtonian viscous stresses and heat conduction. There is currently no
RANS or LES closure in this backend. Farfield faces use characteristic data;
symmetry uses a slip condition. The lower-level topology and solver also
support explicitly paired periodic faces.

Mesh lengths remain in metres for native cases. Density and velocity are
normalized by the configured reference density and freestream speed of sound
`a_ref = sqrt(gamma * R * T_ref)`. Pressure scales with `rho_ref * a_ref^2`;
physical time is the solver time divided by `a_ref`. Native outputs convert
these quantities back to physical units. Laminar viscosity is converted to
the same normalization, rather than borrowing an LBM relaxation time.

## Native configuration and outputs

The native case selects `advanced.numerics.pond_thermal.enabled: true` and uses
`lattice: d3q27` for the common CPU mesh construction. That lattice choice does
not determine the thermal finite-volume equations. The legacy PonD option must
remain disabled. Additional keys under `pond_thermal` are `gamma` (default
1.4), `temperature_k` (288.15), `gas_constant` (287.05), `prandtl` (0.72), `cfl` (0.25), and
`wall_model` (`slip` or `noslip`). The current case runner admits freestream
Mach 0.05 to 1.5. `POND_THERMAL_DEVICE` selects `auto`, `cpu`, or `cuda`.
Native initialization is uniform freestream, or initially stationary fluid with
an incoming-velocity ramp when `ramp_steps > 0`. Custom native initial states
and nonadiabatic walls are rejected; verification scripts may provide custom
initial states through the lower-level `ThermalFV` API.

Results use the existing output folder and recorded-plane workflow: surface
LBMP data, native plane fields, optional plane PNGs and volume VTK, diagnostic
CSV files, a thermal summary, and a final conserved-state inspection checkpoint.
The checkpoint records the mesh and reference scales; automatic restart from
that file is a separate capability. Surface triangle pressure is sampled from
fluid candidates along each original triangle normal. Only if all those probes
fail is the opposite side tried; sidecars count these fallbacks, and exported
normals retain their original orientation. Voxel membership does not prove the
geometric exterior, especially for thin or underresolved components. Missing
samples remain missing, and triangle skin friction is not fabricated from the
voxel loads. Integrated forces and moments come from the solver's voxel wall
fluxes.

Completing the requested number of steps is **not** a declaration of steady
convergence. Inspect physical time, residual histories, loads, admissibility,
and mesh sensitivity. The summary records this distinction explicitly.

## Verification scope and current limits

Run the portable CPU regression suite from the repository root with
`julia --startup-file=no --project=src tools/tests/thermal_fv.jl`.
It checks analytic Euler flux consistency, face orientation, leaf topology,
periodic AMR conservation, uniform-state preservation, and invalid covered masks.
It creates no files and requires neither private study data nor a GPU.

Mesh checks cover uniform and partially refined blocks, three-level ownership,
solid walls, periodic refinement boundaries, face closure, and point sampling.
Tests using the native LBM mesher confirm that its covered masks and leaf-cell
ownership agree with the adapter. CPU evolution checks preserve a uniform
transonic state across 2:1 interfaces and conserve all five integral quantities
for periodic inviscid and laminar perturbations. These checks establish
discrete consistency and conservation, not aerodynamic predictive accuracy.

The mesh is the existing static LBM refinement hierarchy. It does not yet add
shock-driven refinement, so a shock extending away from the body may require
additional resolved volume. Voxel walls and the absence of turbulence closure
limit high-Reynolds-number aircraft predictions, particularly boundary layers,
drag and separation. A transonic benchmark, wall-resolution study, and mesh
convergence assessment are needed before treating a new geometry's loads as
validated results.
