# Compressible C-mesh solver

`cmesh_solver_ka.jl` is the active CPU/GPU carrier. `cmesh_solver.jl` is the legacy CPU version. Mesh generation and the shared flux, EOS, wall, and outlet modules remain in `../cmesh/`; the cavitation solver also uses the carrier.

Airfoil Lab loads the active solver from `apps/airfoil_web/src/backends/cfd_backend.jl`. The old paths in `../cmesh/` are compatibility includes for existing scripts.

The Julia controller API also accepts the experimental `linear_solver=:native_cgs2`.
It currently requires a Float64 free-air thermal state and original-node MUSCL
reconstruction, `fullvisc=1`, Float64 blocks,
`preconditioner=:kinetic_muscl_ls_fullvisc_block_adi`, and explicit
`lagged_component_limiter=true`. Explicit recovery must stay disabled
(`recovery_steps=0`) for this initial scope. For an already prepared state:

```julia
controller = build_jfnk_airfoil_controller(s;
    m=32, linear_solver=:native_cgs2,
    preconditioner=:kinetic_muscl_ls_fullvisc_block_adi,
    block_scalar_type=Float64, lagged_component_limiter=true,
    globalization=:pseudo_transient)
```

For a GPU state, also pass a fresh `available_bytes=CUDA.free_memory()` observation.
The requested basis uses packed storage; the existing operator keeps a basis of
two. Admission counts state, reconstruction, operator, coupled backups and packed
buffers under both a 2 GiB numeric-storage limit and half the observed free memory.
This bound does not establish the CUDA allocator or memory-pool peak.

CGS2 recomputes the true residual against the original linear right-hand side and
rejects a missed requested accuracy before physical or SST updates. Physical
residuals, positivity checks, backward-Euler admission and coupled rollback retain
their existing role. `airfoil_linear_solver_contract(controller)` returns portable
method, storage and source metadata; rebuild workspaces after restoring a state
instead of serializing device views. Existing `:native` and `:krylov` choices and
web defaults are unchanged. Full-airfoil convergence and performance remain
unqualified for this optional method.
