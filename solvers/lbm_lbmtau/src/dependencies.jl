using LinearAlgebra
using Printf
using WriteVTK
using Dates
using StaticArrays
using KernelAbstractions
using CUDA
using Adapt
using Base.Threads
using YAML
using Atomix
using Plots
isdefined(@__MODULE__, :configure_gr_windows_path!) || include(joinpath(@__DIR__,"..","..","..","tools","install","gr_runtime.jl"))
configure_gr_windows_path!(Plots)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL PRECISION SELECTION (compile-time)
# ──────────────────────────────────────────────────────────────────────────────
# Selects whether LBM populations, macros, and lattice arrays use Float32 or
# Float64. Read from the LUDWIG_PRECISION environment variable at module-load
# time (so it's effectively a compile-time constant for the whole Julia session
# — same as Palabos/OpenLB/waLBerla's `T=double` template parameter pattern).
#
# Usage (Windows PowerShell):
#   $env:LUDWIG_PRECISION = "double"; .\run.bat
# Usage (Bash):
#   LUDWIG_PRECISION=double julia --project=src src/main.jl
#
# Default = "single" (Float32). Use "double" (Float64) on datacenter GPUs
# (H100, A100, MI300, RTX A6000, W7900) where FP64 throughput is 1/2 or 1/4
# of FP32. AVOID on consumer GPUs (RTX 4060/4070/4080/4090) where FP64 is
# 1/32 — the slowdown is 5-30× depending on bandwidth vs compute mix.
#
# Note on scope: this scaffold defines the type alias and propagates it into
# the foundational allocation + lattice spec files. The vast majority of
# kernel code still uses explicit `Float32` and `0.5f0` literals, which work
# under both precisions (they promote to Float64 in mixed arithmetic but the
# stored values are still single-precision representations of those Float32
# constants). A full propagation of `LBM_F` through all kernel literals is
# the documented follow-up (~1-2 weeks of focused work).
const _LUDWIG_PRECISION_STR = lowercase(get(ENV, "LUDWIG_PRECISION", "single"))
const LBM_F = _LUDWIG_PRECISION_STR == "double" ? Float64 :
              _LUDWIG_PRECISION_STR == "single" ? Float32 :
              error("LUDWIG_PRECISION must be \"single\" or \"double\" (got \"$_LUDWIG_PRECISION_STR\")")
println("[Init] LBM precision: $LBM_F (set via LUDWIG_PRECISION env var; default = single/Float32)")

include("collision/periodic_y_amr.jl")   # periodic-Y AMR compile-time gate + device helpers (27 Oct 2026)
include("io/config_loader.jl")
include("solver/regime_selector.jl")   # Phase C: Ma/Re → lattice+operator+features auto-selection
include("solver/physics_scaling.jl")

println("[Init] Modules loaded")

include("collision/lattice.jl")

include("geometry/geometry.jl")       # Geometry module (STL loading, mesh structs)
using .Geometry

include("geometry/blocks.jl")         # BlockLevel, SolverMesh, BLOCK_SIZE
include("geometry/bouzidi_common.jl") # BouzidiDataSparse, helpers
include("geometry/bouzidi_math.jl")   # Ray-triangle intersection
include("geometry/bouzidi_setup.jl")  # Q-map computation
include("geometry/bouzidi_kernel.jl") # Bouzidi boundary-condition kernel (legacy linear BFL 2001)
include("geometry/bouzidi_ymls_kernel.jl") # B7 — YMLS-quadratic + Marson 2021 enriched BB (gold-standard wall BC ladder)
include("geometry/bouzidi_diagnostic.jl") # CPU one-shot wall-link symmetry report
include("geometry/bouzidi_ff_diagnostic.jl") # Pass 192: per-cell q<0.5 f_ff fallback audit
include("geometry/bouzidi_missed_link_diagnostic.jl") # Pass 193: skipped q-link solid-upstream audit
include("geometry/intersection_utils.jl") # Shared triangle/AABB helpers
include("geometry/domain_topology.jl")# Active blocks, halo, neighbor tables
include("geometry/amr_interface_registry.jl") # CPU AMR face/edge/corner topology registry
include("geometry/amr_decomposed_transfer.jl") # Diagnostic decomposed AMR transfer buffers
include("geometry/amr_staged_transfer.jl") # Diagnostic Palabos-style staged AMR transfer buffers
include("geometry/amr_recomposition_preview.jl") # Diagnostic staged AMR recomposition preview
include("geometry/amr_live_preview_compare.jl") # Diagnostic live-vs-staged AMR population comparison
include("geometry/domain_generation.jl") # Voxelization, flood fill, sponge, wall distances
include("geometry/domain.jl")         # Multi-level domain orchestration

const KAPPA        = 0.41f0
# Speed of sound squared — set at startup based on LATTICE_TYPE
# D3Q27: cs² = 1/3, D3Q39: cs² = 2/3
global CS2_PHYSICS::Float32  = 1.0f0 / 3.0f0
global CS4_PHYSICS::Float32  = CS2_PHYSICS * CS2_PHYSICS

include("collision/physics_utils.jl")          # Equilibrium, gradient helpers, noise
include("geometry/grad_wall_kernel.jl")        # Grad-reconstruction (regularized) wall BC — needs calculate_equilibrium_auto from physics_utils
include("collision/physics_interpolation.jl")  # Coarse→fine rescaling interpolation
include("collision/amr_coupling.jl")           # Fine→parent restriction after subcycling
include("collision/physics_kernels.jl")        # Stream-collide WALE kernel
include("collision/kernel_pi_neq.jl")           # Diagnostic Π_neq stress tensor
include("collision/kernel_thermal_filter.jl")    # Post-recovery T laplacian filter (AMR banding fix)
include("collision/timestep.jl")               # perform_timestep_v2!, build_lattice_arrays_gpu

include("io/diagnostics.jl")
include("io/diagnostics_vram.jl")
include("io/diagnostics_thermal_interface.jl")

include("forces/structs.jl")         # ForceData
include("forces/surface.jl")         # Surface stress mapping & integration
include("forces/io.jl")              # Force CSV / VTK output

include("io/io_vtk.jl")
include("io/io_slice.jl")
include("io/raw_state_dump.jl")            # opt-in raw per-level state dump (periodic-Y AMR tests, 27 Oct 2026)
include("io/io_amr_validation_gates.jl")  # Gates 1+3 (13-mom conservation + H-theorem) and gate 2 (vortex metrics)
include("io/amr_viz.jl")
include("io/analysis_summary.jl")
include("io/provenance.jl")  # Solver version, git hash, host, GPU, full config dump
include("io/lbm_io_opts.jl")  # lbm-io-opts (27 Oct 2026): numeric slices, plane/field selection, build-only

include("solver/multi_gpu.jl")
include("solver/solver_control.jl")
