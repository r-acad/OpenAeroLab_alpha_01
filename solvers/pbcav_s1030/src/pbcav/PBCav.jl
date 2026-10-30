# =====================================================================================================================
# PBCav -- version 2: the pressure-based real-water cavitation solver (stage P0b, begun 14 Oct 2026).
#
# Plan: docs/2026-10-13_PRESSURE_BASED_SOLVER_PLAN.md (5750d50d, amended bc79814f), section 6 (module and driver layout).
# Records: docs/_records/2026-10-13_pressure_based/ (P0a, the multigrid decision: p0a/; P0b: p0b/).
#
# WHAT THIS MODULE IS. A separate module beside version 1 (src/cmesh/, tools/cmesh_cav_movie.jl), with its own environment
# (src/pbcav/Project.toml and Manifest.toml at version 1's pins: KernelAbstractions 0.9.40, CUDA 5.9.6, Adapt 4.4.0,
# Atomix 1.1.2, StaticArrays 1.9.16; plan 6.1, 6.3). Version 1 is not edited and cannot reach this module
# (tools/check_code_map.jl verifies it). The only version-1 file this module includes is src/cmesh/cmesh_gen.jl, the
# mesh generator, read-only (mesh_faces.jl), so the meshes are version 1's bit for bit.
#
# ONE CODE BASE FOR CPU AND GPU. Every floating-point loop over cells or faces is a KernelAbstractions kernel launched on
# the backend the caller names (CPU(), or CUDA.jl's CUDABackend() since W3 of plan Amendment 12, 21 Oct 2026: the module
# never loads CUDA itself; the driver loads it for PB_BACKEND=cuda, builds the mesh on the CPU and moves mesh, conditions
# and state with to_backend, backend.jl). Host-side loops are allowed only in one-time
# setup that is integer index arithmetic (the face topology of a mesh, the line colours), in the multigrid hierarchy's
# host build (once per march with its numeric plan since review F5; its values are then refreshed by kernels, and only the
# coarsest level's few hundred cells are inverted on the host, linsolve_mg.jl), and in host reads of a few scalars
# (convergence tests, ledgers), never over cells in a step.
# NO @Const IN A NEW KERNEL (measured 14 Oct 2026). KernelAbstractions 0.9.40 wraps a kernel that has an @Const argument in
# an alias scope that, on Julia 1.12, lets the compiler reorder a load of a written array past a store to it (upstream
# issue 652): a Thomas recurrence through z[...] in such a kernel returned wrong values on the CPU (reproduced on a
# six-cell line). The kernels of P0b's core (from the solver files on) carry no @Const; the skeleton's mesh kernels read only
# their @Const inputs and write disjoint outputs, which the issue does not affect (their results are checked bitwise).
#
# THE FILES, in include order (plan 6.1; names may change in P0b's review, the structure may not):
#   knobs.jl            the PB_ knob registry (every knob the driver reads, its default and parser; unknown PB_ refused)
#   mesh_faces.jl       Float64 metrics, true centroids, owner-neighbour face lists, special faces, line maps, wall
#                       distance, box and periodic meshes, bilinear subdivision for nested levels      [IMPLEMENTED]
#   mesh_remap.jl       the opt-in wall-resolved remap of a free-air C-mesh's columns (W4, 22 Oct 2026; PB_MESH_REMAP=common,
#                       default none): wall rows kept, one shared spacing law, the wake's first spacing grown along the cut,
#                       the optional stall layer                                                         [IMPLEMENTED, host]
#   state.jl            structure-of-arrays state, three time levels, span index reserved (NK = 1)     [IMPLEMENTED]
#   phase_laws.jl       Tait, isothermal vapour, incompressible limits, gas term, v1-fluid bridge law   [stub, P1]
#   source.jl           Schnerr-Sauer, Merkle, Zwart behind one interface; dGamma/dp                    [stub, P5a]
#   turbulence.jl       SST-2003m point functions, wall omega W-a, freestream k and omega              [IMPLEMENTED,
#                       W1 (plan 32), opt-in PBConfig.turbulence = :sst; W-b, the trip, Reboud V3 and V2: stubs, P2]
#   assemble_system.jl  the face-based linear system; least-squares and Green-Gauss gradients           [IMPLEMENTED]
#   assemble_momentum.jl  momentum predictor; Rhie-Chow face flux (Bartholomew form, Choi, Majumdar, D) [IMPLEMENTED]
#   assemble_phase.jl   phase predictor and Zalesak limiter                                              [stub, P5a]
#   assemble_pressure.jl  the volumetric pressure equation (psi = 0; source hook zero); corrections;
#                       flux-form mass transport                                                         [IMPLEMENTED]
#   assemble_komega.jl  k-omega transport (conservative, upwind, destruction implicit, production lagged),
#                       line Gauss-Seidel solves, floors counted; mu_t into the momentum (W1, plan 32)   [IMPLEMENTED, opt-in]
#                       W1b (review of W1): the full non-orthogonal correction with a positivity split, the
#                       :step coupling by default, SST refused unless the density is uniform (fail-closed)
#   linsolve_lines.jl   coloured line Gauss-Seidel, Thomas (one work item per line)                     [IMPLEMENTED;
#                       W7 (plan 33, 22 Oct 2026): the workgroup PCR of plan 5.5 and Float32 line copies, opt-in]
#   linsolve_mg.jl      Schaffer V(1,1) j-line semicoarsening multigrid, Float64 (the P0a winner)       [IMPLEMENTED;
#                       W7: the cycle's Float32 and PCR companion MGFast, opt-in]
#   linsolve_krylov.jl  FCG with device scalars, two-pass reductions                                    [IMPLEMENTED;
#                       FGMRES: P1; BiCGStab: fallback]
#   bc.jl               face boundary conditions of section 9 (per-face kinds and values)               [IMPLEMENTED]
#   backend.jl          the data on a backend: to_backend / to_host / copy_state! / state_sha; the mesh of a device
#                       run is built on the CPU and moved (W3, plan 32)                                 [IMPLEMENTED]
#   ledger.jl           section 8 (mixture mass, Theta, continuity term, D, rejections)                  [IMPLEMENTED;
#                       vapour rows: P5a]
#   mms.jl              the manufactured-solution forcing hook (u, v, p fields)                          [IMPLEMENTED]
#   pimple.jl           the step of 4.7 (psi = 0); rollback; regrowth; typed rejections; the march      [IMPLEMENTED]
#   linsolve_coupled.jl the coupled fallback of plan 5.3: (u, v, p) by FGMRES(30) with C's SIMPLEC block
#                       preconditioner (PBConfig.coupling = :coupled; plan 27; the DEFAULT since plan 28.3) [IMPLEMENTED;
#                       backend-agnostic since W3 (21 Oct 2026): CUDA parity registered, run when the GPU is free]
#   parity.jl           HOST DIAGNOSTICS: the CPU-against-device parity of plan 5.5 at its three levels and the device's
#                       bitwise properties (W3; its rules in records air/w3_gpu/W3_REGISTRATION.md; SST, free air and
#                       the rule for future gates since the merge, air/gpuint/PARITY_RULE_V2_REGISTRATION.md)  [IMPLEMENTED]
#   pbrestart.jl        checkpoint schema PBRestart/1 (TOML header + raw little-endian arrays; stdlib only) [IMPLEMENTED]
#   io.jl               receipts (MESH COORDINATE RECEIPT, face census), cavmov_ CSVs, progress lines   [receipts]
#   freeair.jl          free air (work item W2, plan section 32, 21 Oct 2026; opt-in, PB_DOMAIN=freeair): the far-field
#                       tags, the freestream and point-vortex far field, the loads and surface C_p, the free-air run, the
#                       mesh audit                                                                       [IMPLEMENTED, CPU]
#                       THE W1+W2 INTEGRATION (branch pbcav-int, 22 Oct 2026; records air/int_w1w2/): the loads are the
#                       momentum's own boundary terms (momentum_bface_terms!, under SST with mu_eff and the transpose term),
#                       the free-air run takes PB_TURBULENCE and the PB_TURB_* knobs, an SST steady stop tests the k and omega
#                       steady residuals against a registered rule (pbcav_march!), the default turb_tol is 1e-8
#                       W6, A ROBUST START FOR STEADY SST IN FREE AIR (branch pbcav-start, 22 Oct 2026; records air/w6_start/):
#                       the SST defaults turb_mut = :frozen (mu_t formed once per step, not at every outer iterate) and
#                       turb_omega_nonorth = :wall_off (no omega non-orthogonal correction on wall-adjacent cells' faces, which
#                       trapped omega at the floor behind the sharp trailing edge); the k-omega steady rule measures the
#                       unsplit residual; PB_DEFAULTS_BEFORE_W6 restores the integration's arithmetic; the opt-in lifting
#                       potential start (PB_START=kutta: kutta_circulation, vortex_projection_start!)
#                       THE FROZEN mu_t'S STATE (branch pbcav-mutbase, 27 Oct 2026; records air/mutbase/; the A2F_a15
#                       diagnosis): opt-in turb_mut_base = :committed (PB_TURB_MUT_BASE=committed) forms the frozen mu_t from
#                       the committed level n instead of the extrapolated start iterate; the default :extrapolated is W6's,
#                       bit for bit, and its receipts are unchanged
#   baffle.jl           INTERIOR WALLS (branch pbcav-baffle, 24 Oct 2026; records air/baffle/; opt-in, PB_BAFFLE, default empty):
#                       a Gurney plate on the C column from the trailing-edge corner, its internal faces cut into two no-slip
#                       wall faces each (mesh_faces.jl's BaffleCut), the i-lines split there; the geometric rule (the node
#                       nearest h, the optional straightening onto the chord's normal and the exact height, smoothed into the
#                       neighbourhood), its receipt PB BAFFLE, and the plate's outputs (baffle.csv, PB BAFFLE LOADS)
# A STUB IS LOUD. Every function not yet built throws PBNotImplemented naming itself and the stage that builds it, and
# tools/test_pbcav.jl checks that each one does, so nothing can pass by doing nothing.
# =====================================================================================================================
module PBCav

using KernelAbstractions
using Adapt
using LinearAlgebra
using Printf
using SHA
using TOML

"Version string of the module, written into every receipt and checkpoint header."
const PBCAV_VERSION = "0.1.0-p0b"

"""
    PBNotImplemented(name, stage)

Thrown by every interface whose body is not built yet: `name` is the function, `stage` the plan stage that builds it.
"""
struct PBNotImplemented <: Exception
    name::Symbol
    stage::String
end
Base.showerror(io::IO, e::PBNotImplemented) =
    print(io, "PBNotImplemented: PBCav.", e.name, " is not built yet (", e.stage,
          "; docs/2026-10-13_PRESSURE_BASED_SOLVER_PLAN.md section 12)")
_nyi(name::Symbol, stage::AbstractString="stage P0b, second half") = throw(PBNotImplemented(name, String(stage)))

include("knobs.jl")
include("mesh_faces.jl")
include("mesh_remap.jl")
include("state.jl")
include("phase_laws.jl")
include("source.jl")
include("turbulence.jl")
include("assemble_system.jl")
include("assemble_momentum.jl")
include("assemble_phase.jl")
include("assemble_pressure.jl")
include("assemble_komega.jl")
include("linsolve_lines.jl")
include("linsolve_mg.jl")
include("linsolve_krylov.jl")
include("bc.jl")
include("backend.jl")
include("ledger.jl")
include("mms.jl")
include("pimple.jl")
include("linsolve_coupled.jl")
include("parity.jl")
include("pbrestart.jl")
include("io.jl")
include("freeair.jl")
include("baffle.jl")

end # module PBCav
