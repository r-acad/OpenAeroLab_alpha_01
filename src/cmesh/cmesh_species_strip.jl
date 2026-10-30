# ---------------------------------------------------------------------------------------------
# PATH STATUS: **PRODUCTION**, opt-in and default off. The strip trailing-edge topology's second block (5 Oct 2026,
# memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md): built only by build_species_ka(...; strip=(solver=..., tem=...)); with
# `strip=nothing` (every existing caller) nothing in this file is reached, the cache's `strip` field is nothing, and every
# kernel receives `ctx = nothing`, the default path, unchanged. Map: memory/project_docs/CAVITATION_CODE_MAP.md
# (checked by tools/check_code_map.jl).
# ---------------------------------------------------------------------------------------------
# Included inside CMeshCavSpecies (after the steady mode). Stage 2 part 1 of the plan (section 10): the data model of
# sections 3.1-3.3 and the coupling of section 4.1, with the moments of section 4.2 at which the exchange runs. Stage 2
# part 2: the pair's interleaved stage residual and its immediate step (section 4.2; the clocks of section 4.6 on the
# immediate path), and the distance-weighted Green-Gauss face value at the interface rows (section 4.4, in the kernels,
# through the context's static heights h_nb). Stage 2 part 3: the deferred attempt of the pair (section 4.2's interleave
# inside _enqueue_species_deferred!, through _enqueue_pair_deferred!), the accumulate admission of section 4.6 (the strip's
# own banks and specs, its batches reduced by atomic twins of the two batch kernels into the pair's one admission vector),
# and section 4.7's persistence (the 18-array geometry, CMeshSpeciesKA/3 with its :strip entry, written and restored).
# Version 1.5 (7 Oct 2026, section 4.8): the conservation ledger over both blocks, the strip's launch of the ledger kernel
# accumulating into the pair's one ledger after the C block's, in both attempts, after the S6 fallback's copies.
# Stage 4 (7 Oct 2026, section 5): the line-implicit step's wake line through the strip (the section at the end of this file:
# the wake-line twin of the block sweep, the strip's interface linearisations, the strip's face corrections, the scalar SST
# twin, the pair's clocks and its line-implicit attempt _enqueue_pair_imex!, reached from _enqueue_species_imex!).
#
# THE TWO BLOCKS. The C block is the sharp construction around foil plus strip (NIF = n_foil): lower leg i = 1..NIW,
# surface NIW+1..NIW+NIF, upper leg NIW+NIF+1..NI, its j = 1 row the foil wall and, on the leg columns, the interfaces.
# The strip is an H block of NIW x NS cells behind the base (NS = 2 n_base): i = 1 beside the base, j = 1 beside the lower
# leg, j = NS beside the upper leg. Strip column st (a "station") meets the lower leg's column NIW+1-st and the upper leg's
# column NIW+NIF+st (plan section 2).
#
# THE STRIP'S CARRIER is a second KSol built by build_sol_ka from build_strip_metrics: KSol gains no field (critique B
# 4.1). Its identity is outer_mode 2 (every outer tag OUTER_STRIP_INTERFACE = 4; cmesh_solver_ka.jl), NIW = NI and NIF = 0
# (every j = 1 face an "interface" for the existing index tests, no face a foil wall), and the kernel argument `ctx` below.
#
# THE KERNEL CONTEXT `ctx` (one NamedTuple type for both blocks, so each changed kernel compiles once for a strip pair):
#   block     0 the C block, 1 the strip
#   iwall     1 on the strip: its i = 1 faces are the base, a no-slip wall on an i face (the i-face wall family)
#   sides     the pressure-outlet sides: 3 on the C block (both i ends), 2 on the strip (its i = NI+1 end only)
#   wallmode  the pair's wall mode (2, no-slip; the only one version 1 admits with a strip)
#   NS        the strip's NJ
#   c_nb      Float32 (NIW, 2, 2) [station, (x, y), interface]: the centroid of the cell across each interface face
#             (static, built once): the strip's edge cells for the C block, the legs' row-1 cells for the strip
#   sig_nb    Float32 (NIW, 2, 2) [station, (sigma_k, sigma_w), interface]: that cell's SST coefficients, written by
#             the SST exchange after the coefficient kernels (the pair's SST preparation, plan 4.2)
#   U_nb      Float64 (NIW, g, 4, 2) [station, layer, component, interface]: that cell's (and its g-1 neighbours')
#             committed conserved state, written by the exchange: the donor of the interface ghosts for the check and
#             the trio's corner threads, and the authoritative Y of the species j flux (plan 4.1 item 3)
#   h_nb      Float32 (NIW, 2): that cell's height vol/|S| at the interface face (static): the distance-weighted
#             Green-Gauss face value of section 4.4 at the interface rows
#   vol       the block's own cell volumes (the base wall's d1 = 0.5 vol[1,j]/|S_i1| on the strip)
#   NIW, NIF  the block's own column ranges (C block: the legs' NIW and the surface's NIF; strip: NI and 0), for the
#             kernels that do not take them as arguments (the Green-Gauss gradients)
#
# THE EXCHANGE (plan 4.1 item 1; critique B 2.2): after both blocks' projections it writes the interface ghost rows of
# both blocks directly from the neighbour's committed and projected interior (rho, ux, uy, T, pexc, am, alpha, p, and
# tk, tom, nut with SST), the strip's four base-wall corner blocks (critique A 2.1(a)) and U_nb. The ghosts then hold the
# neighbour's published values, so the check's identities hold by construction. The fused row fill of the C block is
# unchanged: on the leg columns its partner copy (the other leg, now h away) is dead, overwritten before any read; the
# trio skips those ghosts (the block-aware donor's src > 0); the strip has no row fill.
#
# THE S6 FALLBACK (plan section 4.3, invoked 6 Oct 2026 after the first CUDA run; record
# memory/project_docs/_records/2026-10-05_strip_topology/stage2/gpu/README.md): each interface face's flux is the strip's evaluation of
# it, which _k_strip_copy_flux! writes into the C block's leg entry Fj[i_C, 1] (negated at the lower interface) for the
# carrier, species and SST j-flux arrays, after both blocks' flux kernels and before anything reads them. Interface
# conservation is then structural on every backend (each face pair sums to exactly zero), whatever arithmetic the two
# blocks' evaluations round differently.
# ---------------------------------------------------------------------------------------------

const STRIP_TOPOLOGY_TAG = "CMeshSpeciesStripTopologyV1"

"""The interface state of a strip pair (plan section 4.1): the two Float64 conserved-donor buffers (`U_legs`: the strip's
cells across the legs' interface faces, read by the C block; `U_strip`: the legs' cells across the strip's, read by the
strip), the two per-stage SST coefficient copies, the static neighbour tables (centroids and heights) and the two kernel
contexts built from them. Layouts: `U_*` (NIW, g, 4, 2) [station, layer, component, interface 1 lower / 2 upper],
`sig_*` and `c_*` (NIW, 2, 2) [station, (sigma_k, sigma_w) or (x, y), interface], `h_*` (NIW, 2)."""
struct StripInterface{A2,A3,A4,CX}
    U_legs::A4
    U_strip::A4
    sig_legs::A3
    sig_strip::A3
    c_legs::A3
    c_strip::A3
    h_legs::A2
    h_strip::A2
    ctx_legs::CX
    ctx_strip::CX
end

"""The strip block of a strip pair (plan section 3.1): its carrier `solver` (a KSol of outer_mode 2), TEM, SST state,
reconstruction, conserved arrays, species fluxes, source, pressure and status, its share of the pair's rollback backup,
its geometry copies and the host mirrors of its column 1 (the base faces: `Six1_h`, `Siy1_h`, `voli1_h`, for
`loads_strip_ka` and the base sidecar), and the pair's `interface`. The settings a per-block helper of the species step
reads (`wallmode`, `eos_single`, `source_beta`, `source_model` and `ss` (the phase-change source, 10 Oct 2026), `reboud_n`,
`reboud_form` and `reboud_prod` (the Reboud treatment, 10 and 13 Oct 2026), `trip` (inert: the strip holds no foil column), `sst_climit`,
`stage_margin`, `inlet_present` (false), `kinetic` (nothing: refused with a strip)) are the pair's, under the names of
SpeciesKA, so those helpers launch on the strip unchanged (plan section 3.1, "per-block atomic"). The deferred admission
(plan 4.6): `admission` is the pair's one slot vector, the C block's own object (the strip's batches accumulate into it),
and `banks` the strip's own status and clock banks, batch specs and reduction workspace (`_build_banks` on the strip's
carrier), so the per-block slot helpers (`_status_slice`, `_clock_slice`, `_batch_specs`) serve the strip unchanged.
`steps` mirrors the pair's accepted steps for the failure reports. Do not edit its arrays."""
mutable struct SpeciesStripKA{S,T,A,Q,C,R,B,BD,G,M,I,H,V,BB}
    const solver::S
    const tem::T
    const wallmode::Int32
    const turbulence::Union{Nothing,SpeciesSSTKA{A}}
    const q::Q
    const q0::Q
    const qtrial::Q
    const conserved::C
    const Fi::A
    const Fj::A
    const source::A
    const source_dt::A
    const pressure::A
    const status::A
    const reconstruction::R
    const kinetic::Nothing
    const backup::B
    const admission::V # the pair's one admission vector (the C block's object, plan 4.6): scratch, never rollback or checkpoint state
    const banks::BB    # the strip's own banks, specs and reduction workspace (plan 4.6): scratch
    const bindings::BD
    const geometry::G
    const boundary_kinds::M
    const interface::I
    const Six1_h::H
    const Siy1_h::H
    const voli1_h::H
    const n_base::Int
    const eos_single::Bool
    const source_beta::Bool
    const source_model::Int32           # the pair's phase-change source model (10 Oct 2026): 0 Merkle, 1 Schnerr-Sauer
    const ss::NTuple{4,Float64}         # its Schnerr-Sauer parameters (N0, D, c3, alpha_nuc); zeros with Merkle
    const reboud_n::Float32
    const reboud_form::Int32            # the pair's Reboud form (10 Oct 2026): 0 off, 1 the density form V1, 2 the composition form V2, 3 the void-fraction form V3 (13 Oct)
    const reboud_prod::Int32            # the pair's production-consistent Reboud form (13 Oct 2026): 1 the k production from the corrected nu_t, 0 not
    const trip::NTuple{2,Float32}
    const sst_climit::Float32
    const stage_margin::Float32
    const inlet_present::Bool
    steps::Int
end

# The kernel context of a block (see the header): nothing on a cache without a strip, which every kernel compiles away.
# The untyped methods serve a SpeciesKA and any field-for-field stand-in of one (the suite's owned-status checks pass a
# NamedTuple projection of a cache to the helpers); the strip block has its own.
_strip_ctx(c)=c.strip===nothing ? nothing : c.strip.interface.ctx_legs
_strip_ctx(x::SpeciesStripKA)=x.interface.ctx_strip
# The pressure-outlet sides of a block: both i ends, or the strip's i = NI+1 end only (its i = 1 side is the base wall).
_outlet_sides(c)=Int32(3)
_outlet_sides(x::SpeciesStripKA)=Int32(2)
# The block named in a failure report (critique A 6.2, version 1's form): nothing on a cache without a strip, bitwise
# today's message.
_block_label(c)=c.strip===nothing ? "" : " in the C block"
_block_label(x::SpeciesStripKA)=" in the strip block"
# The strip's seven metric arrays (its outer_ref_* stubs are never read on the species path).
_strip_geometry(s)=(s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.xc,s.yc)
# The pair's static geometry (plan section 4.7): the C block's eleven arrays of _geometry, then the strip's seven, an 18-tuple;
# a cache without a strip keeps the 11-tuple _geometry(s). The pair's hash (_geometry_hash_pair below) covers these 18 arrays
# and both blocks' boundary kinds; each block compares its own share on its own backend at every owner check (the C block's
# 11 in c.geometry, the strip's 7 in x.geometry).
_geometry_pair(s,ss)=(_geometry(s)...,_strip_geometry(ss)...)
# The pair's configuration (plan section 4.7, critique A 7.2): the C block's tuple, then the strip carrier's and its TEM's
# _scalars and the topology tag; a cache without a strip keeps the C block's tuple as it is. Used by build_species_ka and the
# restart validator (tools/cmesh_species_diagnostics.jl), so both name the same tuple.
_pair_configuration(configuration,ss,ts)=(configuration...,_scalars(ss),_scalars(ts),_strip_tag(ss))
# The strip's published arrays and the pair's two U_nb buffers, in the order _rollback_arrays(c) appends them.
function _strip_rollback_arrays(x::SpeciesStripKA)
    (_rollback_arrays(x.solver,x.tem,x.conserved,x.pressure,x.turbulence)...,x.interface.U_legs,x.interface.U_strip)
end
_strip_tag(ss)=Symbol(STRIP_TOPOLOGY_TAG,"/nb=",ss.NJ÷2,"/ns=",ss.NJ,"/niw=",ss.NI)

# Every Number field of a strip carrier must equal the C block's but the topology (NI, NJ, NIW, NIF, outer_mode) and the
# channel reference the strip has no use for (outer_ref_*; last_global_dt is excluded everywhere): "the same scalars as
# the C block" of plan section 3.2.
const _STRIP_OWN_SCALARS=(:NI,:NJ,:NIW,:NIF,:outer_mode,:outer_ref_rho,:outer_ref_ux,:outer_ref_uy,:outer_ref_T,
                          :outer_ref_ready,:outer_ref_signature,:last_global_dt)
function _strip_scalar_mismatch(s,ss)
    for k in fieldnames(typeof(s))
        k in _STRIP_OWN_SCALARS && continue
        a=getfield(s,k);b=getfield(ss,k)
        a isa Number || continue
        isequal(a,b) || return k
    end
    isequal(s.outlet_eos,ss.outlet_eos) || return :outlet_eos
    nothing
end
function _tem_scalar_mismatch(t,ts)
    for k in fieldnames(typeof(t))
        a=getfield(t,k);b=getfield(ts,k)
        a isa Number || continue
        isequal(a,b) || return k
    end
    nothing
end

"""The face-vector identities of plan section 2, asserted on the two carriers (critique A 7.3): the strip's lower edge
faces are exactly minus the lower leg's j = 1 faces (the same two nodes in reversed order: the exact negative of a
Float32 subtraction, `==`, which is bitwise but for the sign of a zero component, -0.0 against +0.0 on a face parallel to
an axis) and its upper edge faces are the upper leg's bit for bit (the same nodes in the same order, `===`). A
receipt-regenerated mesh that failed them is refused here, at the solver, not trusted on the generator's promise."""
function _strip_face_identities(s,ss)
    NIW,NIF,NS=s.NIW,s.NIF,ss.NJ
    SjxC,SjyC,SjxS,SjyS=map(Array,(s.Sjx,s.Sjy,ss.Sjx,ss.Sjy))
    for k in 1:NIW
        iL=NIW+1-k;iU=NIW+NIF+k
        SjxS[k,1]==-SjxC[iL,1] && SjyS[k,1]==-SjyC[iL,1] || throw(ArgumentError(
            "strip topology face-vector identity fails at the lower interface, station $k: the strip's j = 1 face " *
            "($(SjxS[k,1]), $(SjyS[k,1])) is not minus the lower leg's face $iL ($(SjxC[iL,1]), $(SjyC[iL,1])); the two blocks' nodes do not coincide"))
        SjxS[k,NS+1]===SjxC[iU,1] && SjyS[k,NS+1]===SjyC[iU,1] || throw(ArgumentError(
            "strip topology face-vector identity fails at the upper interface, station $k: the strip's j = NS+1 face " *
            "($(SjxS[k,NS+1]), $(SjyS[k,NS+1])) is not bitwise the upper leg's face $iU ($(SjxC[iU,1]), $(SjyC[iU,1])); the two blocks' nodes do not coincide"))
    end
    nothing
end

# The refusals of plan section 3.3 that depend on a global switch, checked at build and at every owner check. Version 1.5
# (7 Oct 2026, plan section 4.8) lifts version 1's refusal of the conservation ledger: LEDGER_ENABLED[] now fills the pair's
# one ledger over both blocks (_ledger_stage_pair!, _ledger_post_commit_pair! below).
function _strip_switch_refusals()
    CMeshSolverKA.FUSED_PRESSURE_BC[] || throw(ArgumentError(
        "the strip topology does not support the unfused pressure path (FUSED_PRESSURE_BC[] = false) in version 1 ($STRIP_TOPOLOGY_TAG): it has no block-aware donor; use the fused trio"))
    nothing
end

"""Admit a strip carrier and its TEM beside the C block's (plan sections 3.2-3.3): the switch refusal (the unfused
pressure path; the conservation ledger is admitted since version 1.5), the all-y+ wall function and every wall mode but
no-slip refused by name, the
carrier admission with outer_mode 2 (no channel reference), the layout (`NI_s = NIW`, `NIW_s = NI_s`, `NIF_s = 0`, `NS =
2 n_base >= 4`, at least g rows across and g stations), the same backend and array types, every scalar of the carrier
and of its TEM the C block's (but the topology), and the face-vector identities. Kinetic faces and a checkpoint are
refused by build_species_ka itself."""
function _admit_strip(s,t,ss,ts,wallmode::Int32)
    _strip_switch_refusals()
    s.wallfun==2.0f0 && throw(ArgumentError(
        "the strip topology does not support the all-y+ wall function (wallfun 2, TOPO_SPECIES_WALL_FUNCTION=reichardt) in version 1 ($STRIP_TOPOLOGY_TAG): the base cells would need the wall-law strain and omega pin on an i face"))
    wallmode==Int32(2) || throw(ArgumentError(
        "the strip topology supports the no-slip wall (wallmode 2) only in version 1 ($STRIP_TOPOLOGY_TAG); got wallmode $wallmode"))
    ss isa CMeshSolverKA.KSol && ts isa CMeshCavTEM.TEMState || throw(ArgumentError(
        "strip=(solver=..., tem=...) must hold the strip's KSol and TEMState"))
    ss.outer_mode==Int32(2) || throw(ArgumentError(
        "the strip carrier must be built from build_strip_metrics (every outer tag OUTER_STRIP_INTERFACE, outer_mode 2); got outer_mode $(ss.outer_mode)"))
    s.outer_mode==Int32(1) || throw(ArgumentError("the C block of a strip pair must be a segmented channel (outer_mode 1)"))
    ss!==s && ts!==t || throw(ArgumentError("the strip's carrier and TEM must be their own objects, not the C block's"))
    s.NI==2s.NIW+s.NIF && s.NIW>=s.g || throw(ArgumentError(
        "the C block of a strip pair needs the C-mesh layout NI = 2 NIW + NIF with at least g = $(s.g) wake stations"))
    ss.NI==s.NIW && ss.NIW==ss.NI && ss.NIF==0 && ss.g==s.g || throw(ArgumentError(
        "the strip carrier's layout does not match the C block: it needs NI = NIW = $(s.NIW) (the legs' stations) and NIF = 0 (got NI $(ss.NI), NIW $(ss.NIW), NIF $(ss.NIF))"))
    iseven(ss.NJ) && ss.NJ>=4 && ss.NJ>=ss.g || throw(ArgumentError(
        "the strip needs NS = 2 n_base cells across with n_base >= 2 (at least g = $(ss.g) rows; got NS $(ss.NJ))"))
    ss.backend==s.backend && typeof(ss.rho)===typeof(s.rho) && typeof(ts.alpha)===typeof(t.alpha) || throw(ArgumentError(
        "the strip's carrier and TEM must share the C block's backend and array types"))
    k=_strip_scalar_mismatch(s,ss)
    k===nothing || throw(ArgumentError(
        "the strip carrier's scalar $k differs from the C block's ($(getfield(s,k)) against $(getfield(ss,k))): build it with the same scalars (plan section 3.2)"))
    k=_tem_scalar_mismatch(t,ts)
    k===nothing || throw(ArgumentError(
        "the strip TEM's scalar $k differs from the C block's ($(getfield(t,k)) against $(getfield(ts,k))): build it with the same physics"))
    _admit(ss,ts,wallmode;strip_carrier=true)
    _strip_face_identities(s,ss)
    nothing
end

# The static neighbour tables (plan 4.1 item 4), built once from the two carriers' metrics: the centroid and the height
# vol/|S| (Float32 arithmetic, as a kernel forms its own cell's) of the cell across each interface face.
function _strip_tables(s,ss)
    NIW,NIF,NS=s.NIW,s.NIF,ss.NJ
    xC,yC,vC,sxC,syC=map(Array,(s.xc,s.yc,s.vol,s.Sjx,s.Sjy))
    xS,yS,vS,sxS,syS=map(Array,(ss.xc,ss.yc,ss.vol,ss.Sjx,ss.Sjy))
    hgt(v,sx,sy)=v/sqrt(sx*sx+sy*sy)
    c_legs=zeros(Float32,NIW,2,2);c_strip=zeros(Float32,NIW,2,2)
    h_legs=zeros(Float32,NIW,2);h_strip=zeros(Float32,NIW,2)
    for st in 1:NIW
        iL=NIW+1-st;iU=NIW+NIF+st
        c_legs[st,1,1]=xS[st,1];c_legs[st,2,1]=yS[st,1];c_legs[st,1,2]=xS[st,NS];c_legs[st,2,2]=yS[st,NS]
        h_legs[st,1]=hgt(vS[st,1],sxS[st,1],syS[st,1]);h_legs[st,2]=hgt(vS[st,NS],sxS[st,NS+1],syS[st,NS+1])
        c_strip[st,1,1]=xC[iL,1];c_strip[st,2,1]=yC[iL,1];c_strip[st,1,2]=xC[iU,1];c_strip[st,2,2]=yC[iU,1]
        h_strip[st,1]=hgt(vC[iL,1],sxC[iL,1],syC[iL,1]);h_strip[st,2]=hgt(vC[iU,1],sxC[iU,1],syC[iU,1])
    end
    dev(a)=copyto!(similar(s.rho,Float32,size(a)...),a)
    map(dev,(c_legs,c_strip,h_legs,h_strip))
end

function _strip_interface(s,ss,wallmode::Int32)
    NIW,g=s.NIW,s.g
    c_legs,c_strip,h_legs,h_strip=_strip_tables(s,ss)
    U()=fill!(similar(s.rho,Float64,NIW,g,4,2),0.0)
    S()=fill!(similar(s.rho,Float32,NIW,2,2),0.0f0)
    U_legs=U();U_strip=U();sig_legs=S();sig_strip=S()
    ctx(block,iwall,sides,c_nb,sig_nb,U_nb,h_nb,vol,niw,nif)=(block=Int32(block),iwall=Int32(iwall),sides=Int32(sides),
        wallmode=wallmode,NS=Int32(ss.NJ),c_nb=c_nb,sig_nb=sig_nb,U_nb=U_nb,h_nb=h_nb,vol=vol,NIW=Int32(niw),NIF=Int32(nif))
    StripInterface(U_legs,U_strip,sig_legs,sig_strip,c_legs,c_strip,h_legs,h_strip,
        ctx(0,0,3,c_legs,sig_legs,U_legs,h_legs,s.vol,s.NIW,s.NIF),ctx(1,1,2,c_strip,sig_strip,U_strip,h_strip,ss.vol,ss.NIW,ss.NIF))
end

"""Build the strip block of a strip pair (plan section 3.1) beside the C block's carrier `s` and TEM `t`: admit the strip
carrier `ss` and TEM `ts` (`_admit_strip`), check its static geometry, and allocate its SST state, conserved arrays,
species fluxes, source, pressure, status, reconstruction workspace and its share of the rollback backup, the static
neighbour tables and the interface buffers, and its own admission banks (plan 4.6: `_build_banks` on the strip's carrier,
with the C block's reduction tile) beside the pair's one admission vector `admission` (the C block's object). No state is
initialised here: build_species_ka runs the pair's build sequence (`_strip_build_sequence!`) or, with a CMeshSpeciesKA/3
`checkpoint`, its restore (`_strip_restore!`); the strip's SST conserved state comes from the checkpoint's `:strip` entry.
The pair's settings (`eos_single`, ...) are passed by build_species_ka."""
function build_species_strip(s,t,ss,ts;wallmode::Int32,reconstruction::Bool,eos_single::Bool,source_beta::Bool,
                             reboud_n::Float32,sst_climit::Float32,stage_margin::Float32,admission,reduction_tile::Int,
                             checkpoint=nothing,source_model::Int32,ssp::NTuple{4,Float64},reboud_form::Int32,reboud_prod::Int32,
                             recon_retreat::Int32=Int32(0))
    _admit_strip(s,t,ss,ts,wallmode)
    CMeshSolverKA.CMeshPressureOutlet.check_pressure_end_geometry(ss;sides=Int32(2))
    all(isfinite,ss.vol) && minimum(ss.vol)>0 || throw(ArgumentError("the strip's cell volumes must be finite and positive"))
    z(m,n)=fill!(similar(ss.rho,m,n),0.0f0)
    turbulence=_build_sst(ss,checkpoint===nothing ? nothing : checkpoint.strip)
    packed()=fill!(similar(ss.rho,Float64,ss.NI,ss.NJ,4),0.0)
    conserved=(U=packed(),U0=packed(),Utrial=packed())
    q=@view conserved.U[:,:,4];q0=@view conserved.U0[:,:,4];qtrial=@view conserved.Utrial[:,:,4]
    pressure=z(size(ss.rho)...)
    # the pair's reconstruction retreat (15 Oct 2026): the strip's workspace retreats as the C block's
    workspace=reconstruction ? _build_reconstruction(ss;retreat=recon_retreat) : nothing
    iface=_strip_interface(s,ss,wallmode)
    backup=map(similar,(_rollback_arrays(ss,ts,conserved,pressure,turbulence)...,iface.U_legs,iface.U_strip))
    Six1=Array(ss.Six)[1,:];Siy1=Array(ss.Siy)[1,:];vol1=Array(ss.vol)[1,:]
    # the strip's own banks (plan 4.6), on its own carrier's dimensions; the admission vector is the pair's (the C block's)
    admission isa AbstractVector{UInt32} && length(admission)==2*ADMISSION_SLOTS && KA.get_backend(admission)==ss.backend ||
        throw(ArgumentError("the strip's admission must be the pair's UInt32 slot vector (2 ADMISSION_SLOTS) on its backend"))
    banks=_build_banks(ss;reduction_tile)
    SpeciesStripKA(ss,ts,wallmode,turbulence,q,q0,qtrial,conserved,z(ss.NI+1,ss.NJ),z(ss.NI,ss.NJ+1),
        z(ss.NI,ss.NJ),z(ss.NI,ss.NJ),pressure,z(size(ss.rho)...),workspace,nothing,backup,admission,banks,_bindings(ss,ts),
        map(copy,_strip_geometry(ss)),copy(ss.outer_face_kind),iface,Six1,Siy1,vol1,ss.NJ÷2,
        eos_single,source_beta,source_model,ssp,reboud_n,reboud_form,reboud_prod,(0.0f0,0.0f0),sst_climit,stage_margin,false,0)
end

# The geometry hash of a strip pair: the pair's 18 arrays (_geometry_pair) with each block's kinds after its own share, the
# C block's eleven arrays and kinds, then the strip's seven metric arrays and kinds, in that order (plan section 4.7); a
# cache without a strip keeps _geometry_hash(s), byte for byte.
function _geometry_hash_pair(s,ss)
    gp=_geometry_pair(s,ss)
    io=IOBuffer()
    for a in (gp[1:11]...,s.outer_face_kind,gp[12:18]...,ss.outer_face_kind)
        host=Array(a)
        print(io,eltype(host),':',size(host),'\n')
        write(io,reinterpret(UInt8,vec(host)))
    end
    bytes2hex(sha256(take!(io)))
end

# ---- the exchange (plan section 4.1 items 1 and 3) -------------------------------------------------------------------
# One thread per (station st, layer k): the C block's leg ghosts of layer k from the strip's rows k (lower) and NS+1-k
# (upper), the strip's ghosts of layer k from the legs' rows k, U_nb of both directions, and, for st <= g, the strip's
# base-wall corners of ghost column g+1-st from the lower / upper leg's row-k cell with the velocity reflected (the
# interface ghost of the mirrored station, critique A 2.1(a)). Every thread reads interior cells only and writes its own
# cells, so the launch has no read-after-write hazard on either backend. A pure copy: bitwise the neighbour's values.
@inline function _strip_copy_cell!(rd,ud,vd,Td,xd,ad,yd,pd,i,j,rs,us,vs,Ts,xs,as,ys,ps,ii,jj,reflect::Bool)
    @inbounds begin
        rd[i,j]=rs[ii,jj];Td[i,j]=Ts[ii,jj];xd[i,j]=xs[ii,jj];ad[i,j]=as[ii,jj];yd[i,j]=ys[ii,jj];pd[i,j]=ps[ii,jj]
        if reflect
            ud[i,j]=-us[ii,jj];vd[i,j]=-vs[ii,jj]
        else
            ud[i,j]=us[ii,jj];vd[i,j]=vs[ii,jj]
        end
    end
    nothing
end
@kernel function _k_strip_exchange!(rC,uC,vC,TC,xC,aC,yC,pC,kC,wC,nC,@Const(UC),
                                    rS,uS,vS,TS,xS,aS,yS,pS,kS,wS,nS,@Const(US),
                                    U_legs,U_strip,primitives::Int32,sst_on::Int32,
                                    NIW::Int32,NIF::Int32,NS::Int32,g::Int32)
    st,k=@index(Global,NTuple)
    @inbounds if st<=NIW && k<=g
        s32=Int32(st);k32=Int32(k)
        iL=NIW+Int32(1)-s32;iU=NIW+NIF+s32           # the lower / upper leg's column at this station
        for m in 1:4
            U_legs[st,k,m,1]=US[st,k,m];U_legs[st,k,m,2]=US[st,NS+Int32(1)-k32,m]
            U_strip[st,k,m,1]=UC[iL,k,m];U_strip[st,k,m,2]=UC[iU,k,m]
        end
        if primitives==Int32(1)
            jg=g+Int32(1)-k32                           # ghost row of layer k below a j = 1 row
            jl=g+k32                                    # interior row k
            ju=NS+g+Int32(1)-k32                        # the strip's interior row NS+1-k
            jt=NS+g+k32                                 # the strip's ghost row of layer k above its row NS
            iS=s32+g;iLa=iL+g;iUa=iU+g
            # the legs' interface ghosts <- the strip's edge rows
            _strip_copy_cell!(rC,uC,vC,TC,xC,aC,yC,pC,iLa,jg,rS,uS,vS,TS,xS,aS,yS,pS,iS,jl,false)
            _strip_copy_cell!(rC,uC,vC,TC,xC,aC,yC,pC,iUa,jg,rS,uS,vS,TS,xS,aS,yS,pS,iS,ju,false)
            # the strip's interface ghosts <- the legs' row k
            _strip_copy_cell!(rS,uS,vS,TS,xS,aS,yS,pS,iS,jg,rC,uC,vC,TC,xC,aC,yC,pC,iLa,jl,false)
            _strip_copy_cell!(rS,uS,vS,TS,xS,aS,yS,pS,iS,jt,rC,uC,vC,TC,xC,aC,yC,pC,iUa,jl,false)
            if sst_on==Int32(1)
                kC[iLa,jg]=kS[iS,jl];wC[iLa,jg]=wS[iS,jl];nC[iLa,jg]=nS[iS,jl]
                kC[iUa,jg]=kS[iS,ju];wC[iUa,jg]=wS[iS,ju];nC[iUa,jg]=nS[iS,ju]
                kS[iS,jg]=kC[iLa,jl];wS[iS,jg]=wC[iLa,jl];nS[iS,jg]=nC[iLa,jl]
                kS[iS,jt]=kC[iUa,jl];wS[iS,jt]=wC[iUa,jl];nS[iS,jt]=nC[iUa,jl]
            end
            if s32<=g
                ic=g+Int32(1)-s32                       # the strip's base-wall ghost column of layer st
                _strip_copy_cell!(rS,uS,vS,TS,xS,aS,yS,pS,ic,jg,rC,uC,vC,TC,xC,aC,yC,pC,iLa,jl,true)
                _strip_copy_cell!(rS,uS,vS,TS,xS,aS,yS,pS,ic,jt,rC,uC,vC,TC,xC,aC,yC,pC,iUa,jl,true)
            end
        end
    end
end
"""The pair exchange (plan section 4.1): the interface ghosts of both blocks, the strip's base-wall corners and U_nb from
the two blocks' committed and projected interiors. It runs after both projections (`_project_pair!`), and once with
`primitives=false` (U_nb only) at build, before the first projections, whose outlet-corner threads read U_nb."""
function _strip_exchange!(c;primitives::Bool=true)
    x=c.strip;s,ss=c.solver,x.solver;f=x.interface
    sst_on=Int32(s.sst)
    _k_strip_exchange!(s.backend)(s.rho,s.ux,s.uy,s.T,s.pexc,s.am,c.tem.alpha,c.pressure,s.tk,s.tom,s.nut,c.conserved.U,
        ss.rho,ss.ux,ss.uy,ss.T,ss.pexc,ss.am,x.tem.alpha,x.pressure,ss.tk,ss.tom,ss.nut,x.conserved.U,
        f.U_legs,f.U_strip,Int32(primitives),sst_on,Int32(s.NIW),Int32(s.NIF),Int32(ss.NJ),Int32(s.g);ndrange=(s.NIW,s.g))
    nothing
end

# The SST exchange (plan section 4.2 steps 2 and 4): mode 0 the tk/tom interface rows of both blocks from the neighbours'
# recovered interiors (after _sst_bc! on both), mode 1 the nut interface rows and sig_nb (after the coefficient kernels
# on both). The fills leave the interface rows alone (_k_sst_ghosts! with a context), so the order of the fills and this
# launch does not matter for the rows (only the strip's corner ghosts, read by nothing, follow the order). The pair's SST
# preparation (_sst_prepare_pair! below) interleaves it into every stage residual.
@kernel function _k_strip_exchange_sst!(kC,wC,nC,@Const(sigkC),@Const(sigwC),kS,wS,nS,@Const(sigkS),@Const(sigwS),
                                        sig_legs,sig_strip,mode::Int32,NIW::Int32,NIF::Int32,NS::Int32,g::Int32)
    st,k=@index(Global,NTuple)
    @inbounds if st<=NIW && k<=g
        s32=Int32(st);k32=Int32(k)
        iL=NIW+Int32(1)-s32;iU=NIW+NIF+s32;iS=s32+g;iLa=iL+g;iUa=iU+g
        jg=g+Int32(1)-k32;jl=g+k32;ju=NS+g+Int32(1)-k32;jt=NS+g+k32
        if mode==Int32(0)
            kC[iLa,jg]=kS[iS,jl];wC[iLa,jg]=wS[iS,jl];kC[iUa,jg]=kS[iS,ju];wC[iUa,jg]=wS[iS,ju]
            kS[iS,jg]=kC[iLa,jl];wS[iS,jg]=wC[iLa,jl];kS[iS,jt]=kC[iUa,jl];wS[iS,jt]=wC[iUa,jl]
        else
            nC[iLa,jg]=nS[iS,jl];nC[iUa,jg]=nS[iS,ju];nS[iS,jg]=nC[iLa,jl];nS[iS,jt]=nC[iUa,jl]
            if k32==Int32(1)
                sig_legs[st,1,1]=sigkS[st,1];sig_legs[st,2,1]=sigwS[st,1]
                sig_legs[st,1,2]=sigkS[st,NS];sig_legs[st,2,2]=sigwS[st,NS]
                sig_strip[st,1,1]=sigkC[iL,1];sig_strip[st,2,1]=sigwC[iL,1]
                sig_strip[st,1,2]=sigkC[iU,1];sig_strip[st,2,2]=sigwC[iU,1]
            end
        end
    end
end
function _strip_exchange_sst!(c;mode::Int)
    mode in (0,1) || throw(ArgumentError("the SST exchange mode is 0 (tk, tom) or 1 (nut, sigma_k, sigma_w)"))
    x=c.strip;s,ss=c.solver,x.solver;a,b=c.turbulence,x.turbulence;f=x.interface
    a===nothing && return nothing
    _k_strip_exchange_sst!(s.backend)(s.tk,s.tom,s.nut,a.sigk,a.sigw,ss.tk,ss.tom,ss.nut,b.sigk,b.sigw,
        f.sig_legs,f.sig_strip,Int32(mode),Int32(s.NIW),Int32(s.NIF),Int32(ss.NJ),Int32(s.g);ndrange=(s.NIW,s.g))
    nothing
end

# ---- the S6 fallback: each interface flux evaluated once, on the strip's side (plan section 4.3) ---------------------
# Invoked 6 Oct 2026 by the first CUDA run of the strip code (S0903_RUNS/strip_gpu/species_gpu.out): S6 measured the lower
# interface's Fj_mx and Fj_my up to 8 and 2 ulps of the face apart between the two representations (173 and 219 of 952 face
# evaluations; r, e, the species, k and omega exact; the upper interface exact; the CPU 0 everywhere), which plan 4.3 says
# fails S6 and invokes this fallback. The cause (the record memory/project_docs/_records/2026-10-05_strip_topology/stage2/gpu/README.md):
# CUDA.jl's compiler contracts multiply-adds (its __init__ sets LLVM's -nvptx-fma-level=1; ptxas runs with its default
# --fmad), and the AUSM+-up interface pressure's sum Pp pL + Pm pR becomes fma(pL, Pp, round(pR Pm)). The lower interface's
# two representations evaluate one face with L and R exchanged and the normal reversed, so they round the other product,
# and pf, hence Fmx and Fmy, can differ in the last bit; the upper interface's two have the same operands in the same order.
#
# The strip's representation of every interface face is the face's one value: one thread per station st writes, in every
# array of the set (a tuple, unrolled at compile time), the lower leg's entry (NIW+1-st, 1) as -F_strip[st, 1] (the two face
# vectors are exact negatives) and the upper leg's entry (NIW+NIF+st, 1) as F_strip[st, NS+1] (the same face vector). A copy
# and a sign flip: nothing is rounded, so it is the same bits on every backend, and every interface face pair then sums to
# exactly zero. The C block's own evaluation of those faces is still made by its flux kernels (S6 measures it, before the
# copy), and its flux entries are overwritten before any reader of a flux array reads them (the fallback testset poisons them
# right after the C block's flux kernels, STRIP_FLUX_PROBE below). One by-product of that evaluation is not a flux and is not
# copied: the interface conductance of the leg's row-1 cells, which the C block's SST clock reads (_k_sst_species_residual!,
# cmesh_species_sst.jl:738 from the SST flux kernel's terms, :735 recomputed without a reconstruction). It bounds the step and
# moves nothing across the interface. On the CPU the copy left every march measured bit for bit unchanged (record section 5),
# but it is not the identity in general there either: an exactly-zero lower-interface flux takes the strip's sign of zero (r
# and e at rest), and where an interface cell's floored k rounds below the floor in rk/rho the two blocks' k donors differ
# (_sst_donor: the own cell's rk/rho, the ghost's published floored tk), so the strip's k flux of that face replaces the C
# block's (record section 8).
@inline _strip_copy_faces!(::Tuple{},::Tuple{},st,iL,iU,jt)=nothing
@inline function _strip_copy_faces!(FC::Tuple,FS::Tuple,st,iL,iU,jt)
    @inbounds begin
        a=first(FC);b=first(FS)
        a[iL,1]=-b[st,1];a[iU,1]=b[st,jt]
    end
    _strip_copy_faces!(Base.tail(FC),Base.tail(FS),st,iL,iU,jt)
end
@kernel function _k_strip_copy_flux!(FC,FS,NIW::Int32,NIF::Int32,NS::Int32)
    st=@index(Global,Linear)
    @inbounds if st<=NIW
        s32=Int32(st)
        _strip_copy_faces!(FC,FS,s32,NIW+Int32(1)-s32,NIW+NIF+s32,NS+Int32(1))
    end
end
# S6's instrument before the copy (a test instrument, nothing in production): STRIP_COPY_OBSERVER[] = f(c, set) is called by
# _strip_copy_flux! just before its launch, while the C block's interface entries of `set` still hold its own evaluation. It
# must only read. It is refused with a device executor (step_species_ka!), as STRIP_STAGE_OBSERVER is: a captured graph would
# not replay it.
const STRIP_COPY_OBSERVER=Ref{Any}(nothing)
# The fallback's precedence probe (a test instrument, nothing in production; added after the independent review of 6 Oct
# 2026, record stage2/gpu/README.md section 8): STRIP_FLUX_PROBE[] = f(c, set, phase) is called by the pair helpers right
# after the C block's own flux kernels of `set` (phase :produced: before the strip's flux kernels of the set and before the
# copy) and by _strip_copy_flux! right after the copy's launch (phase :copied). The fallback testset writes NaN into the C
# block's interface entries at :produced, which no reader may consume before the copy overwrites them, and, as its negative
# control, at :copied, where the first reader must consume them. It may write. It is refused with a device executor
# (step_species_ka!), as the observers are: a captured graph would not replay it.
const STRIP_FLUX_PROBE=Ref{Any}(nothing)
function _strip_flux_probe(c,set::Symbol,phase::Symbol)
    f=STRIP_FLUX_PROBE[]
    f===nothing || f(c,set,phase)
    nothing
end
# The three sets of j-flux arrays of a pair, the C block's then the strip's: the carrier (r, mx, my, e; _k_flux_j!), the
# species (_k_species_flux_j!) and the SST (k, omega; _k_sst_species_flux_j!).
function _strip_copy_arrays(c,set::Symbol)
    x=c.strip;s,ss=c.solver,x.solver
    set===:carrier && return ((s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e),(ss.Fj_r,ss.Fj_mx,ss.Fj_my,ss.Fj_e))
    set===:species && return ((c.Fj,),(x.Fj,))
    set===:sst && return ((c.turbulence.Fjk,c.turbulence.Fjw),(x.turbulence.Fjk,x.turbulence.Fjw))
    throw(ArgumentError("the strip pair's interface flux copy takes the set :carrier, :species or :sst (got :$set)"))
end
"""The S6 fallback's copy (plan section 4.3): the strip's j flux of every interface face into the C block's leg entries,
negated at the lower interface, for one `set` of arrays (`:carrier`: Fj_r, Fj_mx, Fj_my, Fj_e; `:species`: the species
j flux; `:sst`: Fjk, Fjw). Called by the pair's sequences right after both blocks' flux kernels of that set and before any
clock, flux, trial or diagnostic reads them: `_stage_residual_pair!` (the carrier), `_species_flux_pair!` (the species),
`_sst_residual_pair!` (the SST, between the flux and the residual kernels). One device launch, no host read."""
function _strip_copy_flux!(c,set::Symbol)
    FC,FS=_strip_copy_arrays(c,set)
    f=STRIP_COPY_OBSERVER[]
    f===nothing || f(c,set)
    s=c.solver
    _k_strip_copy_flux!(s.backend)(FC,FS,Int32(s.NIW),Int32(s.NIF),Int32(c.strip.solver.NJ);ndrange=s.NIW)
    _strip_flux_probe(c,set,:copied)
    nothing
end

# ---- the strip's own fills and projection (plan section 4.1, "the strip's own fills") --------------------------------
# The base-wall side's interior ghost rows as a TRUE mirror (critique A 2.2): layer k (column g+1-k) from cell k (column
# g+k) of the same row, rho, T, pexc, am, alpha and p copied, the velocity reflected (no-slip): the donor map's
# (g+1-i32, jj) cell exactly, as the foil wall's row fill is today. Its corners are the exchange's. It runs after the
# strip's trio, so it copies the fresh p, T, pexc, am.
@kernel function _k_strip_bc_cols!(rho,ux,uy,T,pexc,am,alpha,p,NJ::Int32,g::Int32)
    j,k=@index(Global,NTuple)
    @inbounds if j<=NJ && k<=g
        jj=Int32(j)+g;il=g+Int32(1)-Int32(k);is=g+Int32(k)
        rho[il,jj]=rho[is,jj];T[il,jj]=T[is,jj];pexc[il,jj]=pexc[is,jj];am[il,jj]=am[is,jj]
        alpha[il,jj]=alpha[is,jj];p[il,jj]=p[is,jj]
        ux[il,jj]=-ux[is,jj];uy[il,jj]=-uy[is,jj]
    end
end
"""The strip block's projection (plan section 4.2): the fused pressure trio with the strip's context (the interface rows
skipped, the outlet threads on its i = NI+1 side only), then its base-wall column fill. It has no row fill: the exchange
is its row fill. Never applyBC_ka! (refused on the strip carrier)."""
function _project_strip!(x::SpeciesStripKA;batched::Bool=false)
    ss=x.solver
    CMeshSolverKA._mixture_pressure_bc!(ss,x.pressure;conservative_sst=ss.sst,conserved=x.conserved.U,eos_single=x.eos_single,
        inlet_present=false,pressure_slot=batched ? :batched : nothing,status_slot=batched ? :batched : nothing,
        strip_ctx=_strip_ctx(x))
    _k_strip_bc_cols!(ss.backend)(ss.rho,ss.ux,ss.uy,ss.T,ss.pexc,ss.am,x.tem.alpha,x.pressure,Int32(ss.NJ),Int32(ss.g);
        ndrange=(ss.NJ,ss.g))
    nothing
end
"""The pair's projection (plan section 4.2): the C block's (row fill and trio, the interface ghosts skipped), the strip's
(trio with sides 2 and the base-wall column fill), then the exchange: every interface ghost of both blocks, the strip's
base-wall corners and U_nb from the two committed and projected interiors."""
function _project_pair!(c;batched::Bool=false)
    _project!(c;batched)
    _project_strip!(c.strip;batched)
    _strip_exchange!(c)
end

# The pair's build sequence (plan section 4.2, "the same once at build"; critique A 9): both blocks' conserved state from
# their published primitives, U_nb from it (the trios' outlet-corner threads read it), a first projection of each block
# with an exchange after each, so that the strip carrier's ghosts are admissible and coherent before the pair's first
# check, then the initial checks of both blocks, the commit of each block's q/r-derived alpha cache (the alpha that the
# first EOS projection of today's build follows), and the pair projection whose state build_species_ka checks. The first
# projections run strip first: the C block's immediate inlet check reads its whole pressure array, the leg columns'
# interface ghosts included, which its trio skips and which hold a value only once the strip has published one (the
# strip has no inlet, so its own unwritten interface pressures are not read before the exchange writes them). From then
# on every interface ghost holds the previous exchange's state, so the pair projection of the step is C, S, exchange.
# No applyBC_ka! on the strip carrier.
function _strip_build_sequence!(c)
    s,t=c.solver,c.tem;x=c.strip;ss,ts=x.solver,x.tem
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    _k_species_init!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,t.rho_l_sat,t.rho_v_sat,ni,nj,g;ndrange=(s.NI,s.NJ))
    _k_species_init!(ss.backend)(x.conserved.U,ss.rho,ss.ux,ss.uy,ts.alpha,ts.rho_l_sat,ts.rho_v_sat,nis,njs,g;ndrange=(ss.NI,ss.NJ))
    _strip_exchange!(c;primitives=false)
    _project_strip!(x)
    _strip_exchange!(c)
    _project!(c)
    _strip_exchange!(c)
    _check_state!(c;initial=true)
    _check_state!(x;initial=true)
    copyto!(c.conserved.Utrial,c.conserved.U)
    _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
        Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
    copyto!(x.conserved.Utrial,x.conserved.U)
    _k_species_commit!(ss.backend)(x.conserved.U,ss.rho,ss.ux,ss.uy,ts.alpha,x.conserved.Utrial,
        Float64(ts.rho_l_sat),Float64(ts.rho_v_sat),nis,njs,g;ndrange=(ss.NI,ss.NJ))
    _project_pair!(c)
    nothing
end

# ---- persistence (plan section 4.7): CMeshSpeciesKA/3 ----------------------------------------------------------------
# A strip pair's checkpoint is the C block's /2 record with the format CMeshSpeciesKA/3 and one more entry, `strip =
# (conserved, turbulence)`: the strip's authoritative Float64 U and its SST conserved state (CMeshSpeciesSST/1, bound to the
# strip's own wall-distance hash). The configuration and the geometry hash are the pair's (they already name the strip's
# scalars, the topology tag and its 7 + kinds arrays). A cache without a strip writes CMeshSpeciesKA/2, byte for byte; the
# two formats are mutually refused, never converted.
const STRIP_CHECKPOINT_FORMAT="CMeshSpeciesKA/3"
const _STRIP_CHECKPOINT_KEYS=(:format,:configuration,:geometry_sha256,:wallmode,:conserved,:steps,:elapsed,:turbulence,:strip)
"""The CMeshSpeciesKA/3 checkpoint of a strip pair (plan section 4.7): both blocks' owner and state checks first, then the C
block's /2 entries (format /3) and `strip = (conserved = the strip's U, turbulence = its SST state)`. Save the complete
solver/TEM state of both carriers too (`write_cmesh_species_restart` does)."""
function _species_checkpoint_pair(c)
    x=c.strip
    _check_owner(c); _check_state!(c); _check_state!(x)
    (format=STRIP_CHECKPOINT_FORMAT,configuration=c.configuration,geometry_sha256=c.geometry_sha256,
        wallmode=c.wallmode,conserved=Array(c.conserved.U),steps=c.steps,elapsed=c.elapsed,turbulence=_sst_checkpoint(c.turbulence),
        strip=(conserved=Array(x.conserved.U),turbulence=_sst_checkpoint(x.turbulence)))
end
# The topology of a checkpoint against the cache being built (build_species_ka, before any other check): a /3 record restores
# only into a strip pair and a strip pair restores only a /3 record, each refusal naming the cause.
function _checkpoint_topology(checkpoint,strip)
    format=checkpoint isa NamedTuple && hasproperty(checkpoint,:format) ? checkpoint.format : nothing
    strip===nothing && format==STRIP_CHECKPOINT_FORMAT && throw(ArgumentError(
        "a $STRIP_CHECKPOINT_FORMAT checkpoint ($STRIP_TOPOLOGY_TAG) restores only into a strip pair: build with strip=(solver=..., tem=...)"))
    strip!==nothing && format!=STRIP_CHECKPOINT_FORMAT && throw(ArgumentError(
        "a strip-bound cache ($STRIP_TOPOLOGY_TAG) restores only a $STRIP_CHECKPOINT_FORMAT checkpoint; this one is $(format===nothing ? "unformatted" : format) (a cache without a strip writes CMeshSpeciesKA/2); old formats are not converted"))
    nothing
end
# The /3 schema and both blocks' conserved state (the twin of build_species_ka's /2 checks, plus the strip's).
function _strip_checkpoint_schema(checkpoint,s,ss,wallmode::Int32)
    checkpoint isa NamedTuple && keys(checkpoint)==_STRIP_CHECKPOINT_KEYS && checkpoint.format==STRIP_CHECKPOINT_FORMAT &&
        checkpoint.strip isa NamedTuple && keys(checkpoint.strip)==(:conserved,:turbulence) ||
        throw(ArgumentError("species checkpoint format/schema mismatch; a strip pair's checkpoint is $STRIP_CHECKPOINT_FORMAT; old formats are not converted"))
    for (U,q,label) in ((checkpoint.conserved,s,"C block"),(checkpoint.strip.conserved,ss,"strip"))
        U isa Array{Float64,3} && size(U)==(q.NI,q.NJ,4) && all(isfinite,U) ||
            throw(ArgumentError("species checkpoint conserved state/metadata mismatch (the $label's conserved state)"))
        all(U[i,j,1]>0 && 0<=U[i,j,4]<=U[i,j,1] for j in 1:q.NJ,i in 1:q.NI) ||
            throw(ArgumentError("species checkpoint conserved phase bounds (the $label)"))
    end
    checkpoint.wallmode===wallmode && checkpoint.steps isa Int && checkpoint.steps>=0 && checkpoint.elapsed isa Float64 &&
        isfinite(checkpoint.elapsed) && checkpoint.elapsed>=0 ||
        throw(ArgumentError("species checkpoint conserved state/metadata mismatch"))
    (ss.sst ? checkpoint.strip.turbulence!==nothing : checkpoint.strip.turbulence===nothing) ||
        throw(ArgumentError("species checkpoint: the strip's SST state does not match the carrier's turbulence setting"))
    nothing
end
# The pair's restore from a /3 checkpoint (build_species_ka; the twin of its /2 branch): the geometry hash, both blocks'
# authoritative U, the clock, then the pair's projections in the build sequence's order (U_nb; the strip first, whose
# projection reads no interface pressure; the C block, whose inlet check reads its whole pressure array, the interface
# ghosts included, only once the strip has published them), so the restored pair enters its first residual published. The
# restart tool then overwrites it with the saved rollback payload, which is the same state.
function _strip_restore!(c,checkpoint)
    x=c.strip
    checkpoint.geometry_sha256==c.geometry_sha256 ||
        throw(ArgumentError("species checkpoint geometry/boundary/reference mismatch (the strip pair's 18 arrays and kinds)"))
    copyto!(c.conserved.U,checkpoint.conserved);copyto!(x.conserved.U,checkpoint.strip.conserved)
    c.steps=checkpoint.steps;c.elapsed=checkpoint.elapsed;x.steps=checkpoint.steps
    _strip_exchange!(c;primitives=false)
    _project_strip!(x)
    _strip_exchange!(c)
    _project!(c)
    _strip_exchange!(c)
    nothing
end

# ---- the pair's stage residual and its immediate step (plan section 4.2; critique A 1.4, 6.1, 6.5) ------------------
# Per-block atomic and interleaved helpers (plan 3.1, the table): the clocks, the species face fluxes, the trials, the
# commits, the snapshots, the reconstruction, the residual bodies and the state checks are launched on the C block, then
# on the strip, each with its own block context; the SST preparation and the stage residual are interleaved across the
# pair, never applied to a block as units, because `nut` and `sigma_k, sigma_w` are formed by the coefficient kernel at
# the start of each residual and the interface faces must see the neighbour's fresh values (critique A 1.4: a "C then S"
# preparation would hand the C block the strip's previous-stage nut, and the interface viscous and k/omega diffusive
# fluxes would not be single-valued). The single-block functions are untouched; these sequences live here.

# One block's SST coefficient launch with its context (the coefficient kernel of _sst_prepare!, written out for a block of
# either kind), with the immediate host check of _sst_prepare!, the failing block named. `slot`: the deferred attempt's
# flag-slot bank slice of that block (no host read, no fill), exactly as _sst_prepare!(c; slot) treats its own.
function _sst_coeff_block!(q;slot=nothing)
    a=q.turbulence;s=q.solver
    status=slot===nothing ? q.status : slot        # deferred: the block's flag-slot bank slice
    slot===nothing && fill!(q.status,0.0f0)
    _k_sst_species_coeff!(s.backend)(s.nut,a.sigk,a.sigw,a.sk,a.sw,a.rate,status,s.rho,q.tem.alpha,q.pressure,s.tk,s.tom,
        s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,q.reboud_n,q.reboud_form,q.reboud_prod,
        q.tem.rho_l_sat,q.tem.rho_v_sat,_reboud_eos(s),
        Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(s.NIW),Int32(s.NIF),_wall_function(s),_strip_ctx(q);ndrange=(s.NI,s.NJ))
    slot===nothing && (maximum(q.status)==0 || throw(ArgumentError("nonfinite conservative SST coefficient or source$(_block_label(q))")))
    nothing
end

"""The pair's SST preparation (plan section 4.2 steps 1-4): recover `tk, tom` of both blocks from their conserved state;
both blocks' SST ghost fills of their own boundaries (the interface rows are left to the exchange), then the tk/tom
exchange of the interface rows from the neighbours' recovered interiors; the coefficient kernels of both blocks, which
therefore see the fresh interface tk/tom; the nut and sigma_k, sigma_w exchange (sig_nb); both blocks' nut fills of their
own boundaries. Every interface ghost the stage residual reads is then the neighbour's value of this stage. `slots`:
nothing (the immediate attempt: each coefficient kernel's status checked on the host) or the deferred attempt's two flag-
slot bank slices `(C block's, strip's)`, the same launches in the same order with no host read."""
function _sst_prepare_pair!(c;slots=nothing)
    x=c.strip;s,ss=c.solver,x.solver;a,b=c.turbulence,x.turbulence
    _k_sst_recover!(s.backend)(s.tk,s.tom,a.rk,a.rw,s.rho,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    _k_sst_recover!(ss.backend)(ss.tk,ss.tom,b.rk,b.rw,ss.rho,Int32(ss.NI),Int32(ss.NJ),Int32(ss.g);ndrange=(ss.NI,ss.NJ))
    _sst_bc!(c);_sst_bc!(x)                        # tk/tom ghosts of each block's own boundaries
    _strip_exchange_sst!(c;mode=0)                 # the interface tk/tom rows of both blocks
    _sst_coeff_block!(c;slot=slots===nothing ? nothing : slots[1])   # nut, sigma_k, sigma_w, sources and rates
    _sst_coeff_block!(x;slot=slots===nothing ? nothing : slots[2])   # of both blocks
    _strip_exchange_sst!(c;mode=1)                 # the interface nut rows and sig_nb
    _sst_bc!(c;nut_only=true);_sst_bc!(x;nut_only=true)
    nothing
end

"""The pair's stage residual (plan section 4.2 steps 1-6; the twin of _stage_residual! for a strip pair): the interleaved SST
preparation, then the reconstruction of both blocks (the least-squares stencil uses in-block neighbours only, so both
interfaces stay first-order on both sides, as the wake cut is today), then the residual bodies of both blocks with their
contexts (the base wall, the interface conductances, the top interface, the one-sided outlet), then the carrier copy of the
S6 fallback (plan 4.3): the strip's interface fluxes r, mx, my, e into the C block's leg entries, before the clocks, the
species and SST fluxes and the trials read them. `slots`: nothing (the immediate attempt, host checks) or
`(_residual_slots(c, base), _residual_slots(c.strip, base))`, the deferred attempt's device slots of each block, as
_stage_residual!(c; slots) takes one block's. The callers run the state checks of both blocks after it."""
function _stage_residual_pair!(c,wallmode::Int32;slots=nothing)
    x=c.strip;s,ss=c.solver,x.solver;r,rs=c.reconstruction,x.reconstruction
    s.sst && _sst_prepare_pair!(c;slots=slots===nothing ? nothing : (slots[1].sst_prepare,slots[2].sst_prepare))
    r===nothing || _reconstruct!(c,r;slot=slots===nothing ? nothing : slots[1].recon)
    rs===nothing || _reconstruct!(x,rs;slot=slots===nothing ? nothing : slots[2].recon)
    CMeshSolverKA._raw_residual_body_ka!(s,wallmode,Int32(s.wall_flux),true,c.pressure;
        face_states=r===nothing ? nothing : r.states,kinetic=c.kinetic,inlet_present=c.inlet_present,
        pressure_slot=slots===nothing ? nothing : slots[1].flux_pressure,
        store_divergence=false,strip_ctx=_strip_ctx(c),outlet_sides=_outlet_sides(c))
    _strip_flux_probe(c,:carrier,:produced)        # a test instrument (the fallback's precedence probe); nothing in production
    CMeshSolverKA._raw_residual_body_ka!(ss,wallmode,Int32(ss.wall_flux),true,x.pressure;
        face_states=rs===nothing ? nothing : rs.states,kinetic=x.kinetic,inlet_present=x.inlet_present,
        pressure_slot=slots===nothing ? nothing : slots[2].flux_pressure,
        store_divergence=false,strip_ctx=_strip_ctx(x),outlet_sides=_outlet_sides(x))
    _strip_copy_flux!(c,:carrier)                  # the S6 fallback (plan 4.3): the strip's interface fluxes, both blocks
    nothing
end

"""Both blocks' species fluxes (the C block's i and j faces, then the strip's) on their carriers' mass fluxes, then the species
copy of the S6 fallback (plan 4.3): the strip's interface species fluxes into the C block's leg entries, before the trials
read them. The launches are those both attempts of a pair made in turn before the fallback; the carrier copy before them
(in `_stage_residual_pair!`) gives the C block's interface faces the strip's mass flux as well."""
function _species_flux_pair!(c)
    x=c.strip;s,t=c.solver,c.tem;ss,ts=x.solver,x.tem
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    fc=c.reconstruction===nothing ? nothing : c.reconstruction.states
    fs=x.reconstruction===nothing ? nothing : x.reconstruction.states
    _k_species_flux_i!(s.backend)(c.Fi,s.Fi_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
        fc,nothing,ni,nj,g;ndrange=(s.NI+1,s.NJ))
    _k_species_flux_j!(s.backend)(c.Fj,s.Fj_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
        fc,nothing,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
    _strip_flux_probe(c,:species,:produced)        # a test instrument (the fallback's precedence probe); nothing in production
    _k_species_flux_i!(ss.backend)(x.Fi,ss.Fi_r,x.conserved.U,ss.rho,ts.alpha,ts.rho_l_sat,ts.rho_v_sat,
        fs,nothing,nis,njs,g;ndrange=(ss.NI+1,ss.NJ))
    _k_species_flux_j!(ss.backend)(x.Fj,ss.Fj_r,x.conserved.U,ss.rho,ts.alpha,ts.rho_l_sat,ts.rho_v_sat,
        fs,nothing,nis,njs,Int32(ss.NIW),Int32(ss.NIF),g,_strip_ctx(x);ndrange=(ss.NI,ss.NJ+1))
    _strip_copy_flux!(c,:species)
    nothing
end

# One block's SST transport split at the S6 fallback's copy: its face reconstruction and k, omega flux launches, then its
# residual-and-clock launch, exactly the three launches of _sst_residual! (cmesh_species_sst.jl), written out for a block of
# either kind as _sst_coeff_block! is the coefficient launch; the single-block function stays the default path's, untouched
# (plan 4.2). `faces_slot`/`slot`: the deferred attempt's SST-face flag and clock bank slices of that block, or nothing.
function _sst_flux_block!(q;faces_slot=nothing)
    a=q.turbulence;s=q.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    faces=_sst_reconstruct_faces!(q;slot=faces_slot)
    r=q.reconstruction
    diffusion=r===nothing ? nothing : r.sst_diffusion
    gradients=r===nothing ? nothing : r.sst_gradients
    terms=r===nothing ? nothing : r.sst_diffusion_terms
    ctx=_strip_ctx(q)
    _k_sst_species_flux_i!(s.backend)(a.Fik,a.Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,s.nu_lam,s.k_inf,s.w_inf,faces,diffusion,gradients,terms,ni,nj,g,q.sst_climit,ctx;ndrange=(s.NI+1,s.NJ))
    _k_sst_species_flux_j!(s.backend)(a.Fjk,a.Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,s.nu_lam,s.k_inf,s.w_inf,q.wallmode,faces,diffusion,gradients,terms,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s),q.sst_climit,ctx;ndrange=(s.NI,s.NJ+1))
    faces
end
function _sst_cap_block!(q,faces;slot=nothing)
    a=q.turbulence;s=q.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    r=q.reconstruction
    terms=r===nothing ? nothing : r.sst_diffusion_terms
    target=slot===nothing ? q.source_dt : slot     # deferred: the clock slot's bank slice
    _k_sst_species_residual!(s.backend)(a.dck,a.dcw,target,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
        a.sk,a.sw,a.rate,a.rk,a.rw,faces,terms,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,q.wallmode,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s),q.sst_climit,_strip_ctx(q);ndrange=(s.NI,s.NJ))
    nothing
end
"""The pair's SST transport and clock (plan sections 4.2-4.3; the twin of `min(_sst_residual!(c), _sst_residual!(c.strip))`):
both blocks' SST face reconstructions and k, omega fluxes, then the SST copy of the S6 fallback (the strip's interface k,
omega fluxes into the C block's leg entries), then both blocks' SST residuals and clocks, which read them. `slots` and
`faces_slots`: nothing (the immediate attempt: each block's host checks, and the smaller of the two caps is returned) or the
deferred attempt's clock and SST-face flag bank slices `(C block's, strip's)`, the same launches with no host read."""
function _sst_residual_pair!(c;slots=nothing,faces_slots=nothing)
    x=c.strip
    fC=_sst_flux_block!(c;faces_slot=faces_slots===nothing ? nothing : faces_slots[1])
    _strip_flux_probe(c,:sst,:produced)            # a test instrument (the fallback's precedence probe); nothing in production
    fS=_sst_flux_block!(x;faces_slot=faces_slots===nothing ? nothing : faces_slots[2])
    _strip_copy_flux!(c,:sst)
    _sst_cap_block!(c,fC;slot=slots===nothing ? nothing : slots[1])
    _sst_cap_block!(x,fS;slot=slots===nothing ? nothing : slots[2])
    slots===nothing || return nothing
    capC=minimum(c.source_dt);capS=minimum(x.source_dt)
    capC>0.0f0 && capS>0.0f0 || throw(ArgumentError("invalid or unresolved conservative SST transport/source clock"))
    min(capC,capS)
end

# The pair's one global step in both blocks (plan 4.6, the immediate path; critique A 6.1): both CFL fields and both
# carriers' last_global_dt hold it, so the strip's trial, its stage checks and its rejections use the pair's dt.
function _pair_dt!(c,dt::Float32)
    s,ss=c.solver,c.strip.solver
    fill!(s.dtf,dt);s.last_global_dt=dt
    fill!(ss.dtf,dt);ss.last_global_dt=dt
    nothing
end

"""One immediate attempt of a strip pair (the twin of _step_species_once!; plan sections 4.2 and 4.6): the interleaved
stage residual; both blocks' CFL fields (`prepare_dt_ka!` on each), the pair's step the smaller of the two minima, then at
each of the four clock sites (the mixture, source, viscous and SST clocks) the smaller of the two blocks' caps, both CFL
fields refilled with the pair's step every time (critique A 6.1). Per stage: the stage residual and both state checks
(stage > 1), the two mixture clocks, both blocks' species fluxes, the stage clocks of both, the trial of the C block then
the strip's (each with its own status and maximum, the same SpeciesNumericalRejection with the pair's dt; critique A 6.5),
the SST trials (since version 1.5 through `_trials_pair!`: every check of a kind evaluated before either is reported, and a
rejection the strip's own trial raised named `:phase_trial_strip` or `:sst_trial_strip`, plan 4.6), both commits, the SST
commits, and the pair's projection (C, strip, exchange). After the stages, the
pair's SST preparation and both blocks' final checks. The S6 fallback's three copies (plan 4.3) sit inside the pair
helpers, each right after the flux kernels of its set and before every reader: the carrier copy ends
`_stage_residual_pair!` (before the mixture, source, viscous and SST clocks, the species and SST fluxes and the trials),
the species copy ends `_species_flux_pair!` (before the trials), the SST copy sits inside `_sst_residual_pair!` between
the k, omega flux kernels and the SST residual-and-clock kernels. With `LEDGER_ENABLED[]` (version 1.5, plan 4.8) the
pair's one ledger is filled where the single-block attempt fills its own, so after all three copies of the stage: the
stage's entries right before the trials (`_ledger_stage_pair!`), the k-floor entry after the SST commits
(`_ledger_post_commit_pair!`), each the C block's launch (`=`) and then the strip's (`+=`), and published once the step
is admitted (`_publish_ledger!`, unchanged); without it the ledger scalars are NaN, as on every cache."""
function _step_pair_once!(c,cfl::Float32;max_dt::Float32,wallmode::Int32)
    x=c.strip;s,t=c.solver,c.tem;ss,ts=x.solver,x.tem
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    try
        viscous=s.nu_lam>0.0f0
        # All clocks see the published ghosts and EOS of the last pair projection (build, previous step or restore).
        _stage_residual_pair!(c,wallmode)
        _check_state!(c;coherent=false);_check_state!(x;coherent=false)
        CMeshSolverKA.prepare_dt_ka!(s,cfl;max_dt=max_dt)
        CMeshSolverKA.prepare_dt_ka!(ss,cfl;max_dt=max_dt)
        _pair_dt!(c,min(s.last_global_dt,ss.last_global_dt))
        cap=min(_mixture_cap!(c,cfl),_mixture_cap!(x,cfl))
        dt=c.reconstruction===nothing ? min(s.last_global_dt,cap) : min(s.last_global_dt,CLOCK_MARGIN[]*cap)
        _pair_dt!(c,dt)
        reactive=_reactive(c)
        if reactive
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,min(_source_terms!(c),_source_terms!(x))))
            _pair_dt!(c,dt)
        end
        if viscous
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,min(_viscous_cap!(c),_viscous_cap!(x))))
            _pair_dt!(c,dt)
        end
        if s.sst
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,_sst_residual_pair!(c)))
            _pair_dt!(c,dt)
        end
        copyto!(c.conserved.U0,c.conserved.U);copyto!(x.conserved.U0,x.conserved.U)
        if s.sst
            _sst_snapshot!(c);_sst_snapshot!(x)
        end
        LEDGER_ENABLED[] && fill!(c.ledger,0.0)   # the pair's one ledger (version 1.5): rebuilt for every attempt, never restored
        for (stage,(w,b)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
            if stage>1
                _stage_residual_pair!(c,wallmode)
                _check_state!(c;coherent=false);_check_state!(x;coherent=false)
                cap=min(_mixture_cap!(c,cfl;stage),_mixture_cap!(x,cfl;stage))
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:mixture_clock,s.last_global_dt))
            end
            _species_flux_pair!(c)                    # both blocks' species fluxes, then the S6 fallback's species copy
            if stage>1 && reactive
                cap=min(_source_terms!(c),_source_terms!(x))
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:source_clock,s.last_global_dt))
            end
            if stage>1 && viscous
                cap=min(_viscous_cap!(c),_viscous_cap!(x))
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:viscous_clock,s.last_global_dt))
            end
            if stage>1 && s.sst
                cap=_sst_residual_pair!(c)
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:sst_clock,s.last_global_dt))
            end
            LEDGER_ENABLED[] && _ledger_stage_pair!(c,stage,reactive)   # after the stage's three S6 copies (plan 4.8)
            _trials_pair!(c,w,stage)                  # both blocks' trials; a rejection names the strip's (plan 4.6, version 1.5)
            _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
                Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
            _k_species_commit!(ss.backend)(x.conserved.U,ss.rho,ss.ux,ss.uy,ts.alpha,x.conserved.Utrial,
                Float64(ts.rho_l_sat),Float64(ts.rho_v_sat),nis,njs,g;ndrange=(ss.NI,ss.NJ))
            if s.sst
                _sst_commit!(c);_sst_commit!(x)
                LEDGER_ENABLED[] && _ledger_post_commit_pair!(c,stage)
            end
            _project_pair!(c)
            _strip_stage_observe(c,stage)
        end
        s.sst && _sst_prepare_pair!(c)
        _check_state!(c);_check_state!(x)
        next_time=c.elapsed+Float64(s.last_global_dt)
        isfinite(next_time) && next_time>c.elapsed || error("species physical clock cannot advance")
        LEDGER_ENABLED[] ? _publish_ledger!(c,reactive) : _publish_no_ledger!(c)
        c.steps+=1;c.elapsed=next_time;x.steps=c.steps
    catch
        c.valid=false
        rethrow()
    end
    nothing
end

# ---- the trial-flag attribution (version 1.5, plan section 4.6 and S10; 7 Oct 2026) --------------------------------------
# A rejection carries (stage, reason, dt) and no block. From version 1.5 a TRIAL rejection of a strip pair names the strip when
# the strip's own trial of that kind failed at that stage: `:phase_trial_strip`, `:sst_trial_strip` (the driver's rejection tally
# prints the symbol; no driver change). Clock rejections stay unattributed: the clocks are per-cell minima in shared slots.
# - The deferred attempt (_enqueue_pair_deferred!): both blocks' trial statuses reach the shared slots through the accumulate
#   batches as before, so the pair's admission is unchanged and _deferred_decision is not touched; in addition, at the same
#   batch point, the strip's own phase-trial status is reduced (_slot_max!) into the free flag slot at offset 0 of its stage
#   block (ids 4, 20, 36, the plan's slot) and its own SST-trial status into the free ids 58, 59, 60. The plan named one slot
#   per stage; one slot cannot tell the strip's phase trial from its SST trial (a C-block phase failure beside a strip SST
#   failure would be named the strip's), hence the second (record memory/project_docs/_records/2026-10-05_strip_topology/stage15/README.md,
#   part 2). _step_species_replayed! names a trial rejection of the decision through _strip_attribute, after the decision.
# - The immediate attempt (_step_pair_once!, through _trials_pair!): each block's phase trial, then (with SST) each block's SST
#   trial, as before; the checks of one kind are now all evaluated before either is reported, so that both attempts give one
#   name: the strip's when its own trial of the failing kind failed (whether or not the C block's did too), the plain reason
#   when only the C block's did. Running the strip's trial after a C-block failure changes no published state (the wrapper
#   restores the attempt's base state on every rejection).
# Nothing here is reached without a strip: a strip-free cache never writes these slots, and _step_species_replayed!'s call is
# behind `c.strip===nothing ||`.
const _S_STRIP_TRIAL=0                       # stage block offset of the strip's phase-trial slot (ids 4, 20, 36)
_strip_sst_trial_slot(stage)=57+stage        # the strip's SST-trial slot of a stage (ids 58, 59, 60)
"""Name a trial rejection of the deferred decision (plan 4.6, version 1.5): `r` as it is unless it is a trial rejection
whose stage's strip slot is flagged in the attempt's flag vector `h` (1 = flagged, as the decision reads it)."""
function _strip_attribute(h,r::SpeciesNumericalRejection)
    b=_stage_base(r.stage)
    r.reason===:phase_trial && h[b+_S_STRIP_TRIAL]!=0x00000000 && return SpeciesNumericalRejection(r.stage,:phase_trial_strip,r.dt)
    r.reason===:sst_trial && h[_strip_sst_trial_slot(r.stage)]!=0x00000000 && return SpeciesNumericalRejection(r.stage,:sst_trial_strip,r.dt)
    r
end
_strip_attribute(h,r)=r
"The rejection `f()` throws as a SpeciesNumericalRejection, or nothing; any other exception propagates."
function _trial_rejection(f)
    try
        f()
    catch err
        err isa SpeciesNumericalRejection && return err
        rethrow()
    end
    nothing
end
"""The pair's trials of one stage in the immediate attempt (plan 4.6, version 1.5): both blocks' phase trials, then with SST
both blocks' SST trials, in the order of version 1; a kind's two checks are evaluated before either is reported (so when the C
block's trial of a kind fails, the strip's of that kind now runs too before the rejection is thrown), and the strip's rejection
(renamed `:phase_trial_strip`, `:sst_trial_strip`) is thrown in preference to the C block's, as the deferred decision names it
from the strip's slots."""
function _trials_pair!(c,w::Float32,stage::Int)
    x=c.strip
    rC=_trial_rejection(()->_check_stage!(c,w;retryable=true,stage))
    rS=_trial_rejection(()->_check_stage!(x,w;retryable=true,stage))
    rS===nothing || throw(SpeciesNumericalRejection(rS.stage,:phase_trial_strip,rS.dt))
    rC===nothing || throw(rC)
    if c.solver.sst
        rC=_trial_rejection(()->_sst_trial!(c,w;retryable=true,stage))
        rS=_trial_rejection(()->_sst_trial!(x,w;retryable=true,stage))
        rS===nothing || throw(SpeciesNumericalRejection(rS.stage,:sst_trial_strip,rS.dt))
        rC===nothing || throw(rC)
    end
    nothing
end

# ---- the conservation ledger over both blocks (version 1.5, plan section 4.8; 7 Oct 2026) ----------------------------
# The pair's ledger is the C block's one 21 x 3 ledger (c.ledger, its readback packet and _publish_ledger! unchanged). Each
# ledger point launches _k_ledger! twice on the one stream: the C block's launch writes its entries (`accumulate` 0: the
# single-block launch, bitwise), then the strip's launch adds the strip's own entries to them (`accumulate` 1) from the strip's
# flux, source and SST arrays with its own (NI, NJ) = (NIW, NS). So each entry is the sum over both blocks' boundary faces of
# its kind: entry 1 adds the strip's i = 1 faces, the base wall (no mass or species flux; k and omega diffuse to the wall
# values there, so its entries 9 and 13 are non-zero, as the foil wall's 11 and 15 are, and they are part of the published k
# and omega boundary); entry 2 adds the strip's outlet, i = NIW+1; entry 3 adds the strip's j = 1 faces, the lower interface,
# whose C-block twins (the lower leg's j = 1 entries) are their exact negatives after the S6 fallback's copy; entry 4 adds the
# strip's j = NS+1 faces, the upper interface, equal to the upper leg's j = 1 entries in entry 3. In _ledger_boundary
# (l1 - l2 + l3 - l4) the lower pairs cancel inside l3 and the upper pairs between l3 and l4, so the published boundary of the
# pair is that of its outer boundaries alone, to the Float64 rounding of the two sums. The ledger points are those of the
# single-block attempt, after all three of the stage's copies (plan 4.8 as amended on 6 Oct): a launch before a copy would sum
# the C block's own evaluation of the interface faces, which on CUDA differs from the strip's in the last bits (stage2/gpu).
function _ledger_strip!(c,stage,mask::UInt32)
    x=c.strip;ss=x.solver;b=x.turbulence
    Fik,Fjk,Fiw,Fjw,sk,sw,kfl,wpn=b===nothing ? (x.Fi,x.Fj,x.Fi,x.Fj,x.source,x.source,x.source,x.source) :
                                                (b.Fik,b.Fjk,b.Fiw,b.Fjw,b.sk,b.sw,b.kfloor,b.wpin)
    _k_ledger!(ss.backend,_LEDGER_WG)(c.ledger,Int32(stage),mask,Int32(1),ss.Fi_r,ss.Fj_r,x.Fi,x.Fj,Fik,Fjk,Fiw,Fjw,x.source,sk,sw,kfl,wpn,
        Int32(ss.NI),Int32(ss.NJ);ndrange=21*_LEDGER_WG)
    nothing
end
"""The pair's stage ledger (plan 4.8): the C block's launch of `_ledger_stage!`'s mask (`=`), then the strip's (`+=`)."""
function _ledger_stage_pair!(c,stage,reactive)
    mask=_LEDGER_CARRIER | (reactive ? _LEDGER_SOURCE : 0x00000000) |
         (c.turbulence!==nothing ? (_LEDGER_SST_FACES | _LEDGER_SST_SOURCES) : 0x00000000)
    _ledger_fused!(c,stage,mask)
    _ledger_strip!(c,stage,mask)
end
"""The pair's post-commit ledger (the k floor; the wall-function pin, refused with a strip, never set): C block, then strip."""
function _ledger_post_commit_pair!(c,stage)
    mask=_ledger_post_commit(c)
    _ledger_fused!(c,stage,mask)
    _ledger_strip!(c,stage,mask)
end

# ---- S7's instrument (plan section 9, S7 version 1; stage 2 part 3) --------------------------------------------------
# S7's first form (version 1, the ledger then refused with a strip) is a two-block flux balance summed on the host from the
# flux arrays after each stage; its ledger form (version 1.5) reads the pair's ledger and checks it against the same arrays.
# STRIP_STAGE_OBSERVER[] is nothing in production: the attempts then call and
# read nothing. A test sets it to `f(c, stage)`, called by both of the pair's attempts (the immediate _step_pair_once! and the
# deferred _enqueue_pair_deferred!) after each stage's pair projection, when the carrier, species and SST flux arrays and the
# source still hold that stage's values and the k-floor array its commit's corrections. The observer must only read. It is
# refused with a device executor (step_species_ka!): a captured graph would not replay it.
const STRIP_STAGE_OBSERVER=Ref{Any}(nothing)
function _strip_stage_observe(c,stage::Int)
    f=STRIP_STAGE_OBSERVER[]
    f===nothing || f(c,stage)
    nothing
end

# ---- the deferred attempt of a strip pair (plan sections 4.2 and 4.6; critique B 2.1, critique A 6.1) ------------------
# The twin of _enqueue_species_deferred! for a strip pair, reached from it (and so from the CUDA graph executor, which
# captures that function): the same launches as the pair's immediate attempt (_step_pair_once!), in the same order, with
# every status flag and clock cap reduced into the pair's one admission vector and the global step filled on the device. The
# C block's batch at each batch point writes its listed slots directly (_k_slot_batch!, unchanged), then the strip's batch
# reduces the strip's own banks into the same slots by atomic min/max (_k_slot_batch_accumulate!, below), so each slot holds
# the reduction over both blocks: the clock slots the pair's minimum, the flag slots the pair's maximum (min and max are exact
# in any order). The global step is then filled into both blocks' CFL fields from the one slot vector (_k_species_fill_dt! on
# each), so both blocks advance with the one dt the host decision (_deferred_decision, unchanged) admits. The S6 fallback's
# three copies (plan 4.3) are device launches inside the same pair helpers as in the immediate attempt (_stage_residual_pair!,
# _species_flux_pair!, _sst_residual_pair!), in the same places, so the captured graph replays them too.
function _residual_pair_deferred!(c,wallmode::Int32,base)
    x=c.strip
    _stage_residual_pair!(c,wallmode;slots=(_residual_slots(c,base),_residual_slots(x,base)))
    _check_state_deferred!(c,_status_slice(c,base+_S_STATE);coherent=false)
    _check_state_deferred!(x,_status_slice(x,base+_S_STATE);coherent=false)
end
function _enqueue_pair_deferred!(c,cfl::Float32;wallmode::Int32)
    x=c.strip;s,t=c.solver,c.tem;ss,ts=x.solver,x.tem
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)                    # flags: 0 = admitted
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)  # clocks: minimum of positive Float32 bit patterns
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    recon=c.reconstruction!==nothing
    specs=_batch_specs(c,reactive,viscous);sspecs=_batch_specs(x,reactive,viscous)
    base=_stage_base(1)
    _residual_pair_deferred!(c,wallmode,base)
    CMeshSolverKA.prepare_dt_ka!(s,cfl;slot=:batched)
    CMeshSolverKA.prepare_dt_ka!(ss,cfl;slot=:batched)
    _mixture_cap!(c,cfl;slot=_clock_slice(c,base+_S_MIX));_mixture_cap!(x,cfl;slot=_clock_slice(x,base+_S_MIX))
    if reactive
        _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE));_source_terms!(x;slot=_clock_slice(x,base+_S_SOURCE))
    end
    if viscous
        _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC));_viscous_cap!(x;slot=_clock_slice(x,base+_S_VISC))
    end
    if s.sst
        _sst_residual_pair!(c;slots=(_clock_slice(c,base+_S_SST_CAP),_clock_slice(x,base+_S_SST_CAP)),
            faces_slots=(_status_slice(c,base+_S_SST_FACES),_status_slice(x,base+_S_SST_FACES)))
    end
    _slot_batch!(c,specs.residual[1]);_slot_batch!(x,sspecs.residual[1])
    # the one global step in both blocks' CFL fields, from the slots both blocks' clocks were reduced into
    _k_species_fill_dt!(s.backend)(s.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,c.stage_margin,Int32(base),
        Int32(reactive),Int32(viscous),Int32(s.sst),ni,nj;ndrange=(s.NI,s.NJ))
    _k_species_fill_dt!(ss.backend)(ss.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,c.stage_margin,Int32(base),
        Int32(reactive),Int32(viscous),Int32(ss.sst),nis,njs;ndrange=(ss.NI,ss.NJ))
    copyto!(c.conserved.U0,c.conserved.U);copyto!(x.conserved.U0,x.conserved.U)
    if s.sst
        _sst_snapshot!(c);_sst_snapshot!(x)
    end
    LEDGER_ENABLED[] && fill!(c.ledger,0.0)
    for (stage,(w,b)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
        base=_stage_base(stage)
        if stage>1
            _residual_pair_deferred!(c,wallmode,base)
            _mixture_cap!(c,cfl;stage,slot=_clock_slice(c,base+_S_MIX));_mixture_cap!(x,cfl;stage,slot=_clock_slice(x,base+_S_MIX))
        end
        _species_flux_pair!(c)                        # both blocks' species fluxes, then the S6 fallback's species copy
        if stage>1
            if reactive
                _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE));_source_terms!(x;slot=_clock_slice(x,base+_S_SOURCE))
            end
            if viscous
                _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC));_viscous_cap!(x;slot=_clock_slice(x,base+_S_VISC))
            end
            if s.sst
                _sst_residual_pair!(c;slots=(_clock_slice(c,base+_S_SST_CAP),_clock_slice(x,base+_S_SST_CAP)),
                    faces_slots=(_status_slice(c,base+_S_SST_FACES),_status_slice(x,base+_S_SST_FACES)))
            end
            _slot_batch!(c,specs.residual[stage]);_slot_batch!(x,sspecs.residual[stage])
        end
        LEDGER_ENABLED[] && _ledger_stage_pair!(c,stage,reactive)   # after the stage's three S6 copies (plan 4.8)
        _check_stage!(c,w;retryable=true,stage,slot=_status_slice(c,base+_S_TRIAL))
        _check_stage!(x,w;retryable=true,stage,slot=_status_slice(x,base+_S_TRIAL))
        if s.sst
            _sst_trial!(c,w;retryable=true,stage,slot=_status_slice(c,base+_S_SST_TRIAL))
            _sst_trial!(x,w;retryable=true,stage,slot=_status_slice(x,base+_S_SST_TRIAL))
        end
        _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
            Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
        _k_species_commit!(ss.backend)(x.conserved.U,ss.rho,ss.ux,ss.uy,ts.alpha,x.conserved.Utrial,
            Float64(ts.rho_l_sat),Float64(ts.rho_v_sat),nis,njs,g;ndrange=(ss.NI,ss.NJ))
        if s.sst
            _sst_commit!(c);_sst_commit!(x)
            LEDGER_ENABLED[] && _ledger_post_commit_pair!(c,stage)
        end
        _project_pair!(c;batched=true)
        _strip_stage_observe(c,stage)
        _slot_batch!(c,specs.project[stage]);_slot_batch!(x,sspecs.project[stage])
        # the trial-flag attribution (plan 4.6, version 1.5): the strip's own trial statuses again, into their own free slots,
        # at the batch point that reduces them into the shared ones; read by the host only to name a trial rejection
        _slot_max!(_slot(c,base+_S_STRIP_TRIAL),_status_slice(x,base+_S_TRIAL))
        s.sst && _slot_max!(_slot(c,_strip_sst_trial_slot(stage)),_status_slice(x,base+_S_SST_TRIAL))
    end
    s.sst && _sst_prepare_pair!(c;slots=(_status_slice(c,_SLOT_POST+3),_status_slice(x,_SLOT_POST+3)))
    _check_state_deferred!(c,_status_slice(c,_SLOT_POST+4));_check_state_deferred!(x,_status_slice(x,_SLOT_POST+4))
    _slot_batch!(c,specs.post);_slot_batch!(x,sspecs.post)
    # the pair's one readback packet, the slot vector and the pair's ledger (the C block's packet, untouched: plan 4.8)
    LEDGER_ENABLED[] && _k_pack_readback!(s.backend)(c.readback.device,A,c.ledger;ndrange=_READBACK_WORDS)
    nothing
end

# ---- the accumulate admission (plan section 4.6; critique B 2.1, the lead's decision 4) --------------------------------
# The strip's batches reduce into the pair's one admission vector after the C block's: the twins of _k_slot_batch! and of the
# tiled finish _k_tile_finish! below are those kernels statement for statement (the same workgroup reduction ladder over the
# same banks and arrays) but for the final write, an atomic max (flag slots) or min (clock slots) into the slot the C block's
# batch has just written directly, instead of a direct write. min and max are exact and commute, so each slot ends as the
# reduction over both blocks whatever the order of the atomics. The C block's two kernels are not touched: their source is
# what the measured reduction harness reads and launches (tools/bench_species_reduction_tiles.jl takes _k_slot_batch! from
# this module's source, its tiled kernels' text is locked to the harness by tools/test_cmesh_species_reduction_driver.jl), so
# twins replace the plan's `accumulate::Int32` argument on them (stage 2 part 3, departure D1). The tiled partial kernel
# _k_tile_partial! is reused as it is, on the strip's own reduction workspace.
@kernel function _k_slot_batch_accumulate!(A,@Const(sbank),@Const(cbank),@Const(dtf),@Const(p),@Const(outlet),@Const(rstat),@Const(spec),
                                           ns::Int32,nc::Int32,ndtf::Int32,np::Int32,nout::Int32,nr::Int32)
    li=@index(Local,Linear); grp=@index(Group,Linear)
    buf=@localmem UInt32 (_BATCH_WG,)
    @inbounds begin
        kind=spec[2,grp]; slice=spec[3,grp]
        v=_batch_ismax(kind) ? 0x00000000 : 0xffffffff
        i=Int32(li)
        if kind==_BANK_KIND_STATUS
            off=(slice-Int32(1))*ns
            while i<=ns; v=max(v,sbank[off+i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_CLOCK
            off=(slice-Int32(1))*nc
            while i<=nc; v=min(v,_bank_cap_bits(cbank[off+i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_DTF
            while i<=ndtf; v=min(v,_bank_cap_bits(dtf[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_PRESSURE
            while i<=np; v=min(v,_bank_cap_bits(p[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_OUTLET
            while i<=nout; v=max(v,outlet[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        else
            while i<=nr; v=max(v,rstat[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        end
        buf[li]=v
    end
    @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=128 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+128]) : min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=64 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+64]) : min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=32 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+32]) : min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=16 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+16]) : min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=8 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+8]) : min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=4 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+4]) : min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=2 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+2]) : min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear)
    @inbounds if li==1
        id=spec[1,grp]; kind=spec[2,grp]
        if _batch_ismax(kind)
            KernelAbstractions.@atomic max(A[id],max(buf[1],buf[2]))
        else
            KernelAbstractions.@atomic min(A[ADMISSION_SLOTS+id],min(buf[1],buf[2]))
        end
    end
end
@kernel function _k_tile_finish_accumulate!(A,@Const(partial),@Const(spec),ntiles::Int32)
    li=@index(Local,Linear); grp=@index(Group,Linear)
    buf=@localmem UInt32 (_BATCH_WG,)
    @inbounds begin
        high=_batch_ismax(spec[2,grp]); v=high ? 0x00000000 : 0xffffffff; i=Int32(li)
        while i<=ntiles; v=high ? max(v,partial[i,grp]) : min(v,partial[i,grp]); i+=Int32(_BATCH_WG); end
        buf[li]=v
    end
    @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=128 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+128]) : min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=64 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+64]) : min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=32 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+32]) : min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=16 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+16]) : min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=8 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+8]) : min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=4 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+4]) : min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=2 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+2]) : min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear)
    @inbounds if li==1
        high=_batch_ismax(spec[2,grp]); id=spec[1,grp]
        if high
            KernelAbstractions.@atomic max(A[id],max(buf[1],buf[2]))
        else
            KernelAbstractions.@atomic min(A[ADMISSION_SLOTS+id],min(buf[1],buf[2]))
        end
    end
end
"""The strip's batch at a batch point (plan 4.6): its own banks, CFL field, pressure, outlet status and reconstruction status
reduced into the listed slots of the pair's admission vector by atomic max/min, after the C block's direct batch at the
same point (_enqueue_pair_deferred! launches them in that order on the one stream). With the reduction tile (the C block's
setting, the strip's own workspace) the tiled partial kernel on the strip's arrays, then the accumulating finish."""
function _slot_batch!(x::SpeciesStripKA,spec)
    s=x.solver; b=x.banks
    rstat=x.reconstruction===nothing ? x.pressure : x.reconstruction.status   # never read without a reconstruction
    if b.reduction===nothing
        _k_slot_batch_accumulate!(s.backend,_BATCH_WG)(x.admission,b.status,b.clock,s.dtf,x.pressure,s.outlet_status,rstat,spec.m,
            Int32(size(s.rho,1)*size(s.rho,2)),Int32(s.NI*s.NJ),Int32(length(s.dtf)),Int32(length(x.pressure)),
            Int32(length(s.outlet_status)),Int32(length(rstat));ndrange=Int(spec.n)*_BATCH_WG)
    else
        w=b.reduction
        _k_tile_partial!(s.backend,_BATCH_WG)(w.partial,b.status,b.clock,s.dtf,x.pressure,s.outlet_status,rstat,spec.m,
            Int32(length(s.rho)),Int32(s.NI*s.NJ),Int32(length(s.dtf)),Int32(length(x.pressure)),
            Int32(length(s.outlet_status)),Int32(length(rstat)),w.ntiles,w.tile;ndrange=Int(w.ntiles)*Int(spec.n)*_BATCH_WG)
        _k_tile_finish_accumulate!(s.backend,_BATCH_WG)(x.admission,w.partial,spec.m,w.ntiles;ndrange=Int(spec.n)*_BATCH_WG)
    end
    nothing
end

# ---- owner checks of a strip pair (the strip's share of _check_owner_host and _check_owner) -------------------------
function _check_strip_owner_host(c)
    x=c.strip;ss,ts=x.solver,x.tem;conf=c.configuration
    _strip_switch_refusals()
    length(conf)>=3 && conf[end]===_strip_tag(ss) && _scalars_match(ss,conf[end-2]) && _scalars_match(ts,conf[end-1]) &&
        _bindings_match(ss,x.bindings[1]) && _bindings_match(ts,x.bindings[2]) ||
        throw(ArgumentError("the strip's configuration or bound arrays changed; initialize/restore explicitly"))
    all(a isa AbstractArray{Float64,3} && size(a)==(ss.NI,ss.NJ,4) && KA.get_backend(a)==ss.backend for a in values(x.conserved)) ||
        throw(ArgumentError("the strip's conserved workspace shape/type/backend changed"))
    f=x.interface
    all(size(a)==(c.solver.NIW,c.solver.g,4,2) && eltype(a)==Float64 for a in (f.U_legs,f.U_strip)) ||
        throw(ArgumentError("the strip pair's interface buffers changed shape"))
    # the deferred admission (plan 4.6): the strip's batches reduce into the pair's one slot vector, never a copy of it
    x.admission===c.admission && size(x.banks.status)==(size(ss.rho)...,length(_BANK_STATUS_IDS)) &&
        size(x.banks.clock)==(ss.NI,ss.NJ,length(_BANK_CLOCK_IDS)) && _reduction_tile(x)==_reduction_tile(c) ||
        throw(ArgumentError("the strip's admission vector or banks changed (they must be the pair's vector and the strip's own banks, with the C block's reduction tile)"))
    _admit(ss,ts,x.wallmode;strip_carrier=true)
    nothing
end
function _strip_geometry_status!(c)
    x=c.strip;ss=x.solver
    fill!(x.status,0.0f0)
    geometry=_strip_geometry(ss)
    n=max(maximum(length,geometry),length(ss.outer_face_kind))
    n<=length(x.status) || throw(ArgumentError("the strip's status scratch is smaller than the geometry it must compare"))
    _k_species_geometry!(ss.backend)(x.status,geometry,x.geometry,ss.outer_face_kind,x.boundary_kinds,Int32(n);ndrange=n)
    maximum(x.status)==0 || throw(ArgumentError("the strip's geometry/boundary arrays changed in place"))
    nothing
end

# ---- loads on the base (plan section 4.1, "the i-face wall family") --------------------------------------------------
"""Force coefficients `(Cl, Cd)` of the strip's base (its column-1 faces, `i = 1`, rows `1..NJ`), in the C block's
convention (`loads_ka`): the pressure `-p S_i1` and, with viscosity and a no-slip wall, the Stokes traction with `d1 =
0.5 vol[1,j]/|S_i1|`, from the strip's column-1 host mirrors. The pair's loads are `loads_ka(c.solver; ...) .+
loads_strip_ka(c.strip; ...)`. `mixture_pressure` (the strip's species pressure by default) or `nothing` for the
pedestal pressure `rho cs2 T (+ pexc)`, as `loads_ka`."""
loads_strip_ka(x::SpeciesStripKA;mixture_pressure=x.pressure,wallmode::Int32=x.wallmode)=
    loads_strip_ka(x.solver,x.Six1_h,x.Siy1_h,x.voli1_h;mixture_pressure,wallmode)
function loads_strip_ka(s::CMeshSolverKA.KSol,Six1_h,Siy1_h,voli1_h;mixture_pressure=nothing,wallmode::Int32=Int32(2))
    s.outer_mode==Int32(2) || throw(ArgumentError("loads_strip_ka integrates a strip carrier's base (outer_mode 2); use loads_ka for the C block"))
    length(Six1_h)==length(Siy1_h)==length(voli1_h)==s.NJ || throw(ArgumentError("loads_strip_ka needs the strip's column-1 mirrors (length NJ)"))
    mixture_pressure===nothing || CMeshSolverKA.CMeshPressureOutlet.check_pressure_array(s,mixture_pressure)
    g=s.g;r=g+1:g+s.NJ
    KA.synchronize(s.backend)
    rw=Array(view(s.rho,g+1,r));Tw=Array(view(s.T,g+1,r));uw=Array(view(s.ux,g+1,r));vw=Array(view(s.uy,g+1,r))
    pxw=s.cav>=Int32(1) ? Array(view(s.pexc,g+1,r)) : nothing
    pmw=mixture_pressure===nothing ? nothing : Array(view(mixture_pressure,g+1,r))
    Fx=0.0;Fy=0.0
    for j in 1:s.NJ
        pw=pmw===nothing ? Float64(rw[j])*Float64(CMeshSolverKA.CS2)*Float64(Tw[j])+(pxw===nothing ? 0.0 : Float64(pxw[j])) : Float64(pmw[j])
        Fx-=pw*Float64(Six1_h[j]);Fy-=pw*Float64(Siy1_h[j])
        if s.nu_lam>0.0f0 && (pmw===nothing || wallmode!=Int32(1))
            Sf=sqrt(Float64(Six1_h[j])^2+Float64(Siy1_h[j])^2)
            d1=0.5*Float64(voli1_h[j])/Sf+1e-30
            muw=Float64(rw[j])*Float64(s.nu_lam)*Sf/d1
            if pmw===nothing
                Fx+=muw*Float64(uw[j]);Fy+=muw*Float64(vw[j])
            else
                tx,ty=CMeshSolverKA._wall_stokes_traction(Float64(uw[j]),Float64(vw[j]),Float64(Six1_h[j])/Sf,Float64(Siy1_h[j])/Sf,muw)
                Fx+=tx;Fy+=ty
            end
        end
    end
    U2=Float64(s.uxi)^2+Float64(s.uyi)^2
    q=0.5*Float64(s.rho_inf)*U2
    (( Fy*Float64(s.ca)-Fx*Float64(s.sa))/q,(Fx*Float64(s.ca)+Fy*Float64(s.sa))/q)
end

# ---- the line-implicit step through the strip (stage 4, plan section 5; 7 Oct 2026) ------------------------------------
# The line-implicit physical-time step (cmesh_species_imex.jl) makes each SSPRK3 stage linearly implicit along the j lines:
# foil lines from the wall to the outer boundary and, without a strip, one wake line per wake station through the cut. With a
# strip bound the wake line of station st runs through the strip (plan section 5 item 1; map_imex.md 1.4): down the upper
# leg's column NIW+NIF+st from j = NJ to its row 1, across the upper interface, down the strip's column st from j_s = NS to 1,
# across the lower interface, up the lower leg's column NIW+1-st from its row 1 to NJ: M = 2 NJ + NS cells. Both interfaces
# are j faces of the strip (its rows j = 1 and j = NS+1), so each lies inside a line and nothing thin is left explicit.
# - The foil lines are solved by the existing sweep kernels (_k_blk_line_factor!, _k_blk_line_substitute!), launched here with
#   ndrange NIF (the foil lines only; the default path's launches, NIF + NIW, are untouched), and the wake lines by this twin
#   (_k_strip_wake_line_*!, ndrange NIW). The existing sweep kernels, _blk_row and their thirteen launch sites are unchanged.
# - One linearisation per interface face (item 2): the strip's launch of _k_blk_face_jac! linearises its j = 1 face with the
#   lower leg's committed row-1 U (U_nb) on the left and its j = NS+1 face with the upper leg's on the right; the C block's
#   launch is the existing kernel, whose leg-column j = 1 Jacobians (the within-block cut, now the other leg) nothing reads.
# - The signs (item 3; critique A section 5, derived from _blk_row and _k_blk_face_corr!), with JL_b, JR_b the strip's
#   bottom-face Jacobians (w.r.t. the lower-leg cell and strip cell 1) and JL_t, JR_t its top-face Jacobians (strip cell NS,
#   upper-leg cell):
#       strip (k, 1):       B = -JR_b + JL(k, 2)      As = -JL_b (to the lower-leg cell)     dF =   JL_b D_L + JR_b D_s1
#       lower leg (iL, 1):  B = +JL_b + JL(iL, 2)     As = +JR_b (to strip cell 1)           dF = -(JL_b D_L + JR_b D_s1)
#       strip (k, NS):      B = -JR(k, NS) + JL_t     An = +JR_t (to the upper-leg cell)     dF =   JL_t D_sNS + JR_t D_U
#       upper leg (iU, 1):  B = -JR_t + JL(iU, 2)     As = -JL_t (to strip cell NS)          dF = +(JL_t D_sNS + JR_t D_U)
#   The strip's bottom row keeps the cut owner's form (_blk_row's i <= NIW branch), its top row couples north, the lower leg
#   takes the cut partner's form pointing at the strip's bottom face, the upper leg the owner's form pointing at its top face;
#   relative to the strip's faces the dF signs are lower -1, upper +1 (not a transcription of _k_blk_face_corr!'s sg).
# - The four "top is an interface" gates (item 4; critique A 3): _blk_row's j < NJ (lifted on the strip in _blk_row_strip);
#   _k_blk_face_corr!'s j == NJ+1, JL x only (on the strip _k_strip_face_corr!'s interface form: both Jacobians, both D);
#   _k_blk_sst_coeffs!'s j < NJ (lifted by its top_interface argument, 1 on the strip's launch only); the scalar SST line map's
#   j < NJ ? lamN : 0 (lifted on the strip in the scalar twin _k_strip_line_solve!).
# - The S6 fallback's copies (item 7, amended 6 Oct 2026): the pair's attempt takes its explicit residual through the pair
#   helpers that end in them (_stage_residual_pair!, _species_flux_pair!, _sst_residual_pair!), so every reader of the interface
#   entries (the right sides _k_line_rhs!, the SST transport, line coefficients and right sides, the clocks) reads the strip's
#   one value of each interface face; and _k_strip_face_corr! gives each interface face one implicit correction (the strip's
#   linearisation; the C block's leg entries its exact negative at the lower interface and its bits at the upper), so the
#   corrected flux of every interface face is single-valued and the increments telescope across both interfaces.

# The strip block's share of the line-implicit workspace (plan section 5 item 6), sized by the strip's carrier; nothing on a cache
# without a strip. step_species_imex_ka!'s ownership test takes _imex_strip_owner.
function _imex_strip_workspace(c)
    c.strip===nothing && return nothing
    ss=c.strip.solver
    (line=_block_workspace(ss),scratch=fill!(similar(ss.dtf),0.0f0),source_scratch=fill!(similar(ss.dtf),0.0f0),
     onset=fill!(similar(ss.dtf,ss.NI,ss.NJ),0.0f0))
end
# The strip block's layout check of build_species_imex (plan section 5 item 6: the layout check per block; the C block's is
# build_species_imex's own, NI = 2 NIW + NIF). The twin's line map (_strip_line_cell) runs the strip's column st as wake
# station st of the C block's legs, and the strip's launch of _k_blk_face_jac! treats every strip column as a cut-owner column
# (i <= NIW_s): the strip needs NI_s = NIW (the C block's wake stations), NIW_s = NI_s, NIF_s = 0, and at least one row.
# _admit_strip checks the same at the pair's build; this is the line map's own precondition, checked where the map is set up.
function _imex_strip_layout(c)
    s=c.solver;ss=c.strip.solver
    ss.NI==s.NIW && ss.NIW==ss.NI && ss.NIF==0 && ss.NJ>=1 || throw(ArgumentError(
        "the line-implicit step needs the strip block's layout NI = NIW = $(s.NIW) (the C block's wake stations), NIF = 0 and NJ >= 1 (got NI $(ss.NI), NIW $(ss.NIW), NIF $(ss.NIF), NJ $(ss.NJ))"))
    nothing
end
function _imex_strip_owner(c,w)
    c.strip===nothing && return w.strip===nothing
    ss=c.strip.solver;ws=w.strip
    ws!==nothing && size(ws.scratch)==size(ss.dtf) && typeof(ws.scratch)==typeof(ss.dtf) &&
        size(ws.line.X)==(ss.NI,ss.NJ,4) && size(ws.onset)==(ss.NI,ss.NJ)
end

@inline _blk_x4(X,i,j)=(X[i,j,1],X[i,j,2],X[i,j,3],X[i,j,4])
# Cell m of the wake line of station st through the strip (plan section 5 item 1): (blk, i, j, dir), blk 0 the C block and 1 the
# strip, dir +1 where the line runs towards +j of its block and -1 against it.
@inline function _strip_line_cell(st,m,NJ,NS,NIW,NIF)
    m<=NJ && return (Int32(0),NIW+NIF+st,NJ+Int32(1)-m,Int32(-1))    # the upper leg, from j = NJ down to its row 1
    m<=NJ+NS && return (Int32(1),st,NJ+NS+Int32(1)-m,Int32(-1))      # the strip column, from j_s = NS down to 1
    (Int32(0),NIW+Int32(1)-st,m-NJ-NS,Int32(1))                     # the lower leg, from its row 1 up to NJ
end
# The flux-Jacobian part of a wake-line cell's block row (critique A's table above): the diagonal block and the couplings to the
# cell below (As) and above (An) in its own block's j direction, across an interface the other block's cell.
@inline function _blk_row_strip(JLC,JRC,JLS,JRS,blk,i,j,NJ,NS,NIW,NIF)
    As=_b4zero(); An=_b4zero()
    if blk==Int32(1)
        # the strip: the owner of every face of its column; its bottom row the cut owner's form (As to the lower leg's row-1
        # cell), its top row coupled north through the upper interface (the j < NJ gate of _blk_row lifted)
        B=_b4neg(_b4load(JRS,i,j)); As=_b4neg(_b4load(JLS,i,j))
        B=_b4add(B,_b4load(JLS,i,j+1)); An=_b4load(JRS,i,j+1)
    elseif j>1
        # a leg cell above its row 1: _blk_row's interior form
        B=_b4neg(_b4load(JRC,i,j)); As=_b4neg(_b4load(JLC,i,j))
        B=_b4add(B,_b4load(JLC,i,j+1)); j<NJ && (An=_b4load(JRC,i,j+1))
    elseif i<=NIW
        # the lower leg's row 1 (station NIW+1-i): the cut partner's form on the strip's bottom face, As to strip cell 1
        st=NIW+Int32(1)-i
        B=_b4load(JLS,st,1); As=_b4load(JRS,st,1)
        B=_b4add(B,_b4load(JLC,i,2)); j<NJ && (An=_b4load(JRC,i,2))
    else
        # the upper leg's row 1 (station i-NIW-NIF): the cut owner's form on the strip's top face, As to strip cell NS
        st=i-NIW-NIF
        B=_b4neg(_b4load(JRS,st,NS+Int32(1))); As=_b4neg(_b4load(JLS,st,NS+Int32(1)))
        B=_b4add(B,_b4load(JLC,i,2)); j<NJ && (An=_b4load(JRC,i,2))
    end
    (B,As,An)
end

# The wake-line twin of _k_blk_line_solve! / _k_blk_line_factor! / _k_blk_line_substitute! (plan section 5 item 1): one thread
# per station, the block Thomas algorithm along the three-segment line in place in the two blocks' X (rhs in, increment out),
# with the arithmetic of the existing sweeps (the same products, sums and inverse; d0 = V/dtf of the cell's own block; GSC, GSS
# the two blocks' implicit source Jacobians, or nothing for both). The factor kernel stores, per cell, the modified
# super-diagonal (CP), the inverse modified diagonal (BI) and the sub-diagonal in line order (AP) for the substitutions.
@kernel function _k_strip_wake_line_solve!(XC,CPC,@Const(JLC),@Const(JRC),@Const(dtfC),@Const(volC),GSC,
                                           XS,CPS,@Const(JLS),@Const(JRS),@Const(dtfS),@Const(volS),GSS,
                                           NJ::Int32,NS::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    st=Int32(lg)
    @inbounds if st<=NIW
        M=Int32(2)*NJ+NS
        cprev=_b4zero(); xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            blk,i,j,dir=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            B,As,An=_blk_row_strip(JLC,JRC,JLS,JRS,blk,i,j,NJ,NS,NIW,NIF)
            B=_b4add(B,_b4eye(sb ? Float64(volS[i,j])/Float64(dtfS[i,j]) : Float64(volC[i,j])/Float64(dtfC[i,j])))
            if GSC!==nothing
                g1=sb ? GSS[i,j,1] : GSC[i,j,1]; g2=sb ? GSS[i,j,2] : GSC[i,j,2]
                B=Base.setindex(B,B[_b4(4,1)]-g1,_b4(4,1)); B=Base.setindex(B,B[_b4(4,4)]-g2,_b4(4,4))
            end
            Ap=dir==Int32(1) ? As : An; Cn=dir==Int32(1) ? An : As
            Bp=m==Int32(1) ? B : _b4sub(B,_b4mul(Ap,cprev))
            Bi=_b4inv(Bp)
            cp=_b4mul(Bi,Cn)
            CP=sb ? CPS : CPC; X=sb ? XS : XC
            for k in 1:16; CP[i,j,k]=cp[k]; end
            rhs=_blk_x4(X,i,j)
            if m>Int32(1)
                a=_b4vec(Ap,xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(Bi,rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            cprev=cp; xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            blk,i,j,_=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            CP=sb ? CPS : CPC; X=sb ? XS : XC
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end
@kernel function _k_strip_wake_line_factor!(XC,CPC,BIC,APC,@Const(JLC),@Const(JRC),@Const(dtfC),@Const(volC),GSC,
                                            XS,CPS,BIS,APS,@Const(JLS),@Const(JRS),@Const(dtfS),@Const(volS),GSS,
                                            NJ::Int32,NS::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    st=Int32(lg)
    @inbounds if st<=NIW
        M=Int32(2)*NJ+NS
        cprev=_b4zero(); xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            blk,i,j,dir=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            B,As,An=_blk_row_strip(JLC,JRC,JLS,JRS,blk,i,j,NJ,NS,NIW,NIF)
            B=_b4add(B,_b4eye(sb ? Float64(volS[i,j])/Float64(dtfS[i,j]) : Float64(volC[i,j])/Float64(dtfC[i,j])))
            if GSC!==nothing
                g1=sb ? GSS[i,j,1] : GSC[i,j,1]; g2=sb ? GSS[i,j,2] : GSC[i,j,2]
                B=Base.setindex(B,B[_b4(4,1)]-g1,_b4(4,1)); B=Base.setindex(B,B[_b4(4,4)]-g2,_b4(4,4))
            end
            Ap=dir==Int32(1) ? As : An; Cn=dir==Int32(1) ? An : As
            Bp=m==Int32(1) ? B : _b4sub(B,_b4mul(Ap,cprev))
            Bi=_b4inv(Bp)
            cp=_b4mul(Bi,Cn)
            CP=sb ? CPS : CPC; BI=sb ? BIS : BIC; AP=sb ? APS : APC; X=sb ? XS : XC
            for k in 1:16; CP[i,j,k]=cp[k]; BI[i,j,k]=Bi[k]; AP[i,j,k]=Ap[k]; end
            rhs=_blk_x4(X,i,j)
            if m>Int32(1)
                a=_b4vec(Ap,xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(Bi,rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            cprev=cp; xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            blk,i,j,_=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            CP=sb ? CPS : CPC; X=sb ? XS : XC
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end
@kernel function _k_strip_wake_line_substitute!(XC,@Const(CPC),@Const(BIC),@Const(APC),XS,@Const(CPS),@Const(BIS),@Const(APS),
                                                NJ::Int32,NS::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    st=Int32(lg)
    @inbounds if st<=NIW
        M=Int32(2)*NJ+NS
        xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            blk,i,j,_=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            X=sb ? XS : XC; BI=sb ? BIS : BIC; AP=sb ? APS : APC
            rhs=_blk_x4(X,i,j)
            if m>Int32(1)
                a=_b4vec(_b4load(AP,i,j),xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(_b4load(BI,i,j),rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            blk,i,j,_=_strip_line_cell(st,m,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            CP=sb ? CPS : CPC; X=sb ? XS : XC
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end

# The strip's face corrections and the legs' interface entries (plan section 5 items 2-4; critique A's dF column): the strip's
# interior j faces dF = JL D_south + JR D_north as _k_blk_face_corr! forms them; its j = 1 face (the lower interface) JL_b D_L +
# JR_b D_s1 with D_L the lower leg's row-1 increment; its j = NS+1 face (the upper interface) JL_t D_sNS + JR_t D_U with D_U the
# upper leg's (both Jacobians and both D: the outer face's form, JL x only, is not the strip's top row's); and, after the C
# block's own _k_blk_face_corr! (which wrote the leg columns' j = 1 entries across the within-block cut), the leg entries of
# the interface faces from the strip's one correction: the lower leg's its exact negative (the two face vectors are exact
# negatives), the upper leg's the same bits. One thread per strip face, each writing its own entries.
@kernel function _k_strip_face_corr!(dFS,dFC,@Const(XS),@Const(XC),@Const(JLS),@Const(JRS),NS::Int32,NIW::Int32,NIF::Int32)
    st,j=@index(Global,NTuple)
    @inbounds if st<=NIW && j<=NS+1
        s32=Int32(st);j32=Int32(j)
        if j32==Int32(1)
            iL=NIW+Int32(1)-s32
            a=_b4vec(_b4load(JLS,s32,1),_blk_x4(XC,iL,1)); b=_b4vec(_b4load(JRS,s32,1),_blk_x4(XS,s32,1))
            f=(a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
            for k in 1:4; dFS[s32,1,k]=f[k]; dFC[iL,1,k]=-f[k]; end
        elseif j32==NS+Int32(1)
            iU=NIW+NIF+s32
            a=_b4vec(_b4load(JLS,s32,j32),_blk_x4(XS,s32,NS)); b=_b4vec(_b4load(JRS,s32,j32),_blk_x4(XC,iU,1))
            f=(a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
            for k in 1:4; dFS[s32,j32,k]=f[k]; dFC[iU,1,k]=f[k]; end
        else
            a=_b4vec(_b4load(JLS,s32,j32),_blk_x4(XS,s32,j32-Int32(1))); b=_b4vec(_b4load(JRS,s32,j32),_blk_x4(XS,s32,j32))
            for k in 1:4; dFS[s32,j32,k]=a[k]+b[k]; end
        end
    end
end

# The scalar twin of the SST line solve's wake lines (plan section 5 item 4): position m of station st's line with its couplings
# (blk, i, j, a, c), a to position m-1 and c to m+1 (both positive lambdas entered with a minus sign), as _line_cell gives them,
# but through the strip: the strip's top row couples north to the upper leg's row-1 cell (its lamN, the j < NJ gate lifted).
@inline function _strip_scalar_cell(st,m,lamSC,lamNC,lamSS,lamNS,NJ,NS,NIW,NIF)
    if m<=NJ
        i=NIW+NIF+st;j=NJ+Int32(1)-m
        return (Int32(0),i,j,j<NJ ? lamNC[i,j] : 0.0f0,lamSC[i,j])     # previous = north, next = south (strip row NS at j = 1)
    elseif m<=NJ+NS
        j=NJ+NS+Int32(1)-m
        return (Int32(1),st,j,lamNS[st,j],lamSS[st,j])                 # previous = north (the upper leg at j = NS), next = south
    end
    i=NIW+Int32(1)-st;j=m-NJ-NS
    (Int32(0),i,j,lamSC[i,j],j<NJ ? lamNC[i,j] : 0.0f0)                # previous = south (strip row 1 at j = 1), next = north
end
# The Thomas algorithm of _k_line_solve! along the wake line of each station through the strip, for ncomp components of the two
# blocks' X (rhs in, increment out); cpC, cpS the two blocks' Float64 scratch of the modified super-diagonal (stored negative).
@kernel function _k_strip_line_solve!(XC,cpC,@Const(lamSC),@Const(lamNC),@Const(d0C),XS,cpS,@Const(lamSS),@Const(lamNS),@Const(d0S),
                                      ncomp::Int32,NJ::Int32,NS::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    st=Int32(lg)
    @inbounds if st<=NIW
        M=Int32(2)*NJ+NS
        cprev=0.0
        for m in Int32(1):M
            blk,i,j,a,c=_strip_scalar_cell(st,m,lamSC,lamNC,lamSS,lamNS,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            b=sb ? Float64(d0S[i,j])+Float64(lamSS[i,j])+Float64(lamNS[i,j]) :
                   Float64(d0C[i,j])+Float64(lamSC[i,j])+Float64(lamNC[i,j])
            den=b+Float64(a)*cprev   # c' is stored negative: den = b - A_m c'_{m-1} with A_m = -a
            cprev=-Float64(c)/den
            if sb
                cpS[i,j]=cprev
            else
                cpC[i,j]=cprev
            end
            X=sb ? XS : XC
            if m==Int32(1)
                for k in Int32(1):ncomp; X[i,j,k]=X[i,j,k]/den; end
            else
                pb,ip_,jp_,_,_=_strip_scalar_cell(st,m-Int32(1),lamSC,lamNC,lamSS,lamNS,NJ,NS,NIW,NIF)
                Xp=pb==Int32(1) ? XS : XC
                for k in Int32(1):ncomp; X[i,j,k]=(X[i,j,k]+Float64(a)*Xp[ip_,jp_,k])/den; end
            end
        end
        for m in M-Int32(1):Int32(-1):Int32(1)
            blk,i,j,_,_=_strip_scalar_cell(st,m,lamSC,lamNC,lamSS,lamNS,NJ,NS,NIW,NIF)
            nb,inx,jnx,_,_=_strip_scalar_cell(st,m+Int32(1),lamSC,lamNC,lamSS,lamNS,NJ,NS,NIW,NIF)
            sb=blk==Int32(1)
            X=sb ? XS : XC; Xn=nb==Int32(1) ? XS : XC; cpm=sb ? cpS[i,j] : cpC[i,j]
            for k in Int32(1):ncomp; X[i,j,k]-=cpm*Xn[inx,jnx,k]; end
        end
    end
end

# The pair's block line solve (rhs in L.X and Ls.X, increment out): the foil lines by the existing sweeps (ndrange NIF), the
# wake lines through the strip by the twin (ndrange NIW); factorise at stage 1 and substitute at stages 2-3 (W3a's reuse: a
# substitution is the factorising sweep's own arithmetic). `GS`, `GSs`: both blocks' implicit source Jacobians, or neither.
function _blk_line_solve_pair!(c,L,Ls;GS=nothing,GSs=nothing,factor::Bool=true)
    s=c.solver;ss=c.strip.solver
    ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);ns=Int32(ss.NJ)
    (GS===nothing)==(GSs===nothing) || throw(ArgumentError("the pair's implicit source Jacobians are given for both blocks or for neither"))
    if factor
        _k_blk_line_factor!(s.backend)(L.X,L.CP,L.BI,L.AP,L.JL,L.JR,s.dtf,s.vol,GS,ni,nj,niw,nif;ndrange=s.NIF)
        _k_strip_wake_line_factor!(s.backend)(L.X,L.CP,L.BI,L.AP,L.JL,L.JR,s.dtf,s.vol,GS,
            Ls.X,Ls.CP,Ls.BI,Ls.AP,Ls.JL,Ls.JR,ss.dtf,ss.vol,GSs,nj,ns,niw,nif;ndrange=s.NIW)
    else
        _k_blk_line_substitute!(s.backend)(L.X,L.CP,L.BI,L.AP,ni,nj,niw,nif;ndrange=s.NIF)
        _k_strip_wake_line_substitute!(s.backend)(L.X,L.CP,L.BI,L.AP,Ls.X,Ls.CP,Ls.BI,Ls.AP,nj,ns,niw,nif;ndrange=s.NIW)
    end
    nothing
end
"""The pair's conservative block line solve (the twin of _blk_line_solve_conservative!; plan section 5 items 1-3): the right
sides in `L.X` (the C block) and `Ls.X` (the strip) in, the increments out, rewritten as flux differences, `D = dt/V (rhs -
(dF north - dF south))`, in each block; `L.R`, `Ls.R` keep the right sides and `L.dF`, `Ls.dF` the face corrections, every
interface face's correction single-valued (the strip's one linearisation; the lower leg's entry its exact negative, the upper
leg's its bits), so the pair's increments telescope across both interfaces whatever the accuracy of the solve."""
function _blk_line_solve_conservative_pair!(c,L,Ls;GS=nothing,GSs=nothing,factor::Bool=true)
    s=c.solver;ss=c.strip.solver
    ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);nis,ns=Int32(ss.NI),Int32(ss.NJ)
    copyto!(L.R,L.X);copyto!(Ls.R,Ls.X)
    _blk_line_solve_pair!(c,L,Ls;GS,GSs,factor)
    _k_blk_face_corr!(s.backend)(L.dF,L.X,L.JL,L.JR,ni,nj,niw,nif;ndrange=(s.NI,s.NJ+1))
    _k_strip_face_corr!(ss.backend)(Ls.dF,L.dF,Ls.X,L.X,Ls.JL,Ls.JR,ns,niw,nif;ndrange=(ss.NI,ss.NJ+1))
    _k_blk_flux_increment!(s.backend)(L.X,L.R,L.dF,s.dtf,s.vol,GS,ni,nj;ndrange=(s.NI,s.NJ))
    _k_blk_flux_increment!(ss.backend)(Ls.X,Ls.R,Ls.dF,ss.dtf,ss.vol,GSs,nis,ns;ndrange=(ss.NI,ss.NJ))
    nothing
end
# The pair's SST lines: k and omega each on their own (one component per launch), the foil lines by the existing scalar sweep
# (ndrange NIF), the wake lines through the strip by the scalar twin (ndrange NIW).
function _line_solve_sst_pair!(c,w)
    s=c.solver;ss=c.strip.solver;L=w.line;Ls=w.strip.line
    ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);ns=Int32(ss.NJ)
    _k_line_solve!(s.backend)(@view(L.DK[:,:,1:1]),L.cp,L.lamSk,L.lamNk,L.d0k,Int32(1),ni,nj,niw,nif;ndrange=s.NIF)
    _k_strip_line_solve!(s.backend)(@view(L.DK[:,:,1:1]),L.cp,L.lamSk,L.lamNk,L.d0k,@view(Ls.DK[:,:,1:1]),Ls.cp,Ls.lamSk,Ls.lamNk,Ls.d0k,
        Int32(1),nj,ns,niw,nif;ndrange=s.NIW)
    _k_line_solve!(s.backend)(@view(L.DK[:,:,2:2]),L.cp,L.lamSw,L.lamNw,L.d0w,Int32(1),ni,nj,niw,nif;ndrange=s.NIF)
    _k_strip_line_solve!(s.backend)(@view(L.DK[:,:,2:2]),L.cp,L.lamSw,L.lamNw,L.d0w,@view(Ls.DK[:,:,2:2]),Ls.cp,Ls.lamSw,Ls.lamNw,Ls.d0w,
        Int32(1),nj,ns,niw,nif;ndrange=s.NIW)
    nothing
end
# The strip block's SST line coefficients: _blk_sst_coeffs!'s launch with top_interface = 1 (the strip's top row couples north
# through the upper interface) and the strip's context (its interface conductances).
function _blk_sst_coeffs_strip!(x::SpeciesStripKA,L,wallmode)
    s=x.solver;a=x.turbulence
    _k_blk_sst_coeffs!(s.backend)(L.lamSk,L.lamNk,L.lamSw,L.lamNw,L.d0k,L.d0w,s.dtf,s.Fj_r,a.rate,s.rho,s.nut,
        a.sigk,a.sigw,s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.nu_lam,wallmode,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),
        Int32(s.g),_wall_function(s),Int32(1),_strip_ctx(x);ndrange=(s.NI,s.NJ))
    nothing
end

# One block's i-direction clocks of the line-implicit step (_imex_caps!'s mixture face, source and viscous launches, written out
# for a block of either kind with its context: the strip's base wall in the viscous clock, plan section 5 item 5) into that
# block's bank slices; the pair's SST transport and SST clocks follow in _imex_caps_pair!.
function _imex_clocks_block!(q,source_scratch,cfl::Float32,base,reactive::Bool,viscous::Bool,implicit_source::Bool)
    s=q.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    a2=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2,1.0f-6))
    r=q.reconstruction
    states=r===nothing ? nothing : r.states
    _k_imex_face_dt!(s.backend)(_clock_slice(q,base+_S_MIX),states,s.rho,s.ux,s.uy,s.am,q.pressure,s.Fi_r,s.vol,
        s.Six,s.Siy,cfl,minf2,ni,nj,g;ndrange=(s.NI,s.NJ))
    if reactive
        if implicit_source
            _source_terms!(q;slot=source_scratch)
            _k_imex_source_gate!(s.backend)(_clock_slice(q,base+_S_SOURCE),source_scratch,ni,nj;ndrange=(s.NI,s.NJ))
        else
            _source_terms!(q;slot=_clock_slice(q,base+_S_SOURCE))
        end
    end
    viscous && _k_imex_viscous_dt!(s.backend)(_clock_slice(q,base+_S_VISC),s.rho,s.nut,s.vol,s.xc,s.yc,
        s.Six,s.Siy,s.Sjx,s.Sjy,s.nu_lam,Int32(s.sst),s.fullvisc,ni,nj,g,_strip_ctx(q);ndrange=(s.NI,s.NJ))
    nothing
end
function _imex_sst_cap_block!(q,base)
    s=q.solver;a=q.turbulence;r=q.reconstruction
    _k_imex_sst_cap!(s.backend)(_clock_slice(q,base+_S_SST_CAP),s.Fi_r,a.rk,a.rw,
        r===nothing ? nothing : r.sst_states,r===nothing ? nothing : r.sst_diffusion_terms,
        s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.vol,s.nu_lam,Int32(s.NI),Int32(s.NJ),Int32(s.g),q.sst_climit,
        _strip_ctx(q);ndrange=(s.NI,s.NJ))
    nothing
end
"""The pair's clocks of one stage of the line-implicit step (the twin of _imex_caps!; plan section 5 items 5 and 7): both
blocks' i-direction mixture, source and viscous clocks into their own bank slices (the same slots: the strip's batch reduces
them into the pair's), then the pair's SST transport (`_sst_residual_pair!`: both blocks' SST fluxes, the S6 fallback's SST copy,
both SST residuals, which the SST line right-hand sides read; their explicit caps go to the unused scratch) and both blocks'
i-direction SST clocks."""
function _imex_caps_pair!(c,w,cfl::Float32,base,reactive::Bool,viscous::Bool)
    x=c.strip
    _imex_clocks_block!(c,w.source_scratch,cfl,base,reactive,viscous,w.implicit_source)
    _imex_clocks_block!(x,w.strip.source_scratch,cfl,base,reactive,viscous,w.implicit_source)
    if c.solver.sst
        _sst_residual_pair!(c;slots=(w.scratch,w.strip.scratch),
            faces_slots=(_status_slice(c,base+_S_SST_FACES),_status_slice(x,base+_S_SST_FACES)))
        _imex_sst_cap_block!(c,base);_imex_sst_cap_block!(x,base)
    end
    nothing
end
"""The stage-1 evaluation and frozen operator of a strip pair's line-implicit attempt (the first part of
_enqueue_species_imex!'s structure, for the pair): the slot vector reset; the pair's stage residual (ending in the carrier copy)
and both state checks; both CFL fields floatmax (the base solver's dt field does not bind this step); the pair's clocks
(`_imex_caps_pair!`, the SST copy inside); the stage-1 batches of both blocks; the one global dt in both CFL fields; then, on the
stage-1 state, both blocks' face Jacobians (the strip's interface faces linearised once, by its launch), the implicit source
blocks when selected, and both blocks' SST line coefficients (the strip's top row coupled north). Returns both blocks' batch
specs. Called by _enqueue_pair_imex!; no host read."""
function _pair_imex_prepare!(c,w,cfl::Float32;wallmode::Int32)
    x=c.strip;s,t=c.solver,c.tem;ss=x.solver
    ni,nj=Int32(s.NI),Int32(s.NJ);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    L=w.line;Ls=w.strip.line;A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    recon=c.reconstruction!==nothing
    specs=_batch_specs(c,reactive,viscous);sspecs=_batch_specs(x,reactive,viscous)
    base=_stage_base(1)
    _residual_pair_deferred!(c,wallmode,base)
    fill!(s.dtf,floatmax(Float32));fill!(ss.dtf,floatmax(Float32))
    _imex_caps_pair!(c,w,cfl,base,reactive,viscous)
    _slot_batch!(c,specs.residual[1]);_slot_batch!(x,sspecs.residual[1])
    # -1: no stage-1 clock margin on this integrator (refused by _admit_imex), as _enqueue_species_imex!
    _k_species_fill_dt!(s.backend)(s.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,-1.0f0,Int32(base),
        Int32(reactive),Int32(viscous),Int32(s.sst),ni,nj;ndrange=(s.NI,s.NJ))
    _k_species_fill_dt!(ss.backend)(ss.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,-1.0f0,Int32(base),
        Int32(reactive),Int32(viscous),Int32(ss.sst),nis,njs;ndrange=(ss.NI,ss.NJ))
    _blk_jacobians!(c,L,wallmode);_blk_jacobians!(x,Ls,wallmode)
    if w.implicit_source
        _blk_source_jac!(c,L);_blk_source_jac!(x,Ls)
    end
    if s.sst
        _blk_sst_coeffs!(c,L,wallmode);_blk_sst_coeffs_strip!(x,Ls,wallmode)
    end
    (specs,sspecs)
end
# Both blocks' line right-hand sides of a stage (the stage's flux divergence minus source, Float64), read after the stage's
# carrier and species copies (plan section 5 item 7).
function _pair_line_rhs!(c,w,reactive::Bool)
    x=c.strip;s,ss=c.solver,x.solver;L=w.line;Ls=w.strip.line
    _k_line_rhs!(s.backend)(L.X,s.Fi_r,s.Fj_r,s.Fi_mx,s.Fj_mx,s.Fi_my,s.Fj_my,c.Fi,c.Fj,c.source,Int32(reactive),
        Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    _k_line_rhs!(ss.backend)(Ls.X,ss.Fi_r,ss.Fj_r,ss.Fi_mx,ss.Fj_mx,ss.Fi_my,ss.Fj_my,x.Fi,x.Fj,x.source,Int32(reactive),
        Int32(ss.NI),Int32(ss.NJ);ndrange=(ss.NI,ss.NJ))
    nothing
end
"""One speculative line-implicit attempt of a strip pair (the twin of _enqueue_species_imex!, reached from it; plan section 5):
the stage-1 evaluation and operator (`_pair_imex_prepare!`), then per stage the pair's stage residual (stages 2-3) and both
blocks' species fluxes, each ending in its S6 copy; the pair's clocks again at stages 2-3; both blocks' right sides; the pair's
conservative block line solve through the strip (the foil lines by the existing sweeps, the wake lines by the twin, one
correction per interface face); both blocks' trials, their flags reduced at once into the pair's slots (the CPU backend throws
where the GPU propagates NaN), the onset flags when reactive; the SST lines of both blocks (the scalar twin through the strip)
and their trials; both commits, the SST commits, the pair's projection (C, strip, exchange) and the stage observer; both
blocks' project batches. After the stages the pair's SST preparation and both blocks' post checks and batches. The same
decision and wrapper (step_species_imex_ka!) admit it; the trial rejections of a pair are not attributed to a block here."""
function _enqueue_pair_imex!(c,w,cfl::Float32;wallmode::Int32)
    x=c.strip;s,t=c.solver,c.tem;ss,ts=x.solver,x.tem
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g);nis,njs=Int32(ss.NI),Int32(ss.NJ)
    L=w.line;ws=w.strip;Ls=ws.line
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    specs,sspecs=_pair_imex_prepare!(c,w,cfl;wallmode)
    copyto!(c.conserved.U0,c.conserved.U);copyto!(x.conserved.U0,x.conserved.U)
    if s.sst
        _sst_snapshot!(c);_sst_snapshot!(x)
    end
    for (stage,(wt,_)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
        base=_stage_base(stage)
        stage>1 && _residual_pair_deferred!(c,wallmode,base)
        _species_flux_pair!(c)                        # both blocks' species fluxes, then the S6 fallback's species copy
        if stage>1
            _imex_caps_pair!(c,w,cfl,base,reactive,viscous)
            _slot_batch!(c,specs.residual[stage]);_slot_batch!(x,sspecs.residual[stage])
        end
        _pair_line_rhs!(c,w,reactive)                 # after the stage's three copies
        _blk_line_solve_conservative_pair!(c,L,Ls;GS=w.implicit_source ? L.GS : nothing,GSs=w.implicit_source ? Ls.GS : nothing,
            factor=stage==1)
        _k_imex_trial!(s.backend)(c.conserved.Utrial,_status_slice(c,base+_S_TRIAL),c.conserved.U,c.conserved.U0,
            L.X,wt,Float64.(s.outlet_eos),Int32(c.eos_single),Q_FLUSH[],ni,nj;ndrange=(s.NI,s.NJ))
        _k_imex_trial!(ss.backend)(x.conserved.Utrial,_status_slice(x,base+_S_TRIAL),x.conserved.U,x.conserved.U0,
            Ls.X,wt,Float64.(ss.outlet_eos),Int32(x.eos_single),Q_FLUSH[],nis,njs;ndrange=(ss.NI,ss.NJ))
        _slot_max!(_slot(c,base+_S_TRIAL),_status_slice(c,base+_S_TRIAL))
        _slot_max!(_slot(c,base+_S_TRIAL),_status_slice(x,base+_S_TRIAL))
        if reactive
            _k_imex_onset!(s.backend)(w.onset,c.conserved.Utrial,c.conserved.U0,Float64.(s.outlet_eos),Int32(c.eos_single),
                ONSET_FRACTION[],ni,nj;ndrange=(s.NI,s.NJ))
            _k_imex_onset!(ss.backend)(ws.onset,x.conserved.Utrial,x.conserved.U0,Float64.(ss.outlet_eos),Int32(x.eos_single),
                ONSET_FRACTION[],nis,njs;ndrange=(ss.NI,ss.NJ))
            _slot_max!(_slot(c,base+_S_IMEX_ONSET),w.onset);_slot_max!(_slot(c,base+_S_IMEX_ONSET),ws.onset)
        end
        if s.sst
            a,b=c.turbulence,x.turbulence
            _k_sst_line_rhs!(s.backend)(L.DK,a.dck,a.dcw,ni,nj;ndrange=(s.NI,s.NJ))
            _k_sst_line_rhs!(ss.backend)(Ls.DK,b.dck,b.dcw,nis,njs;ndrange=(ss.NI,ss.NJ))
            _line_solve_sst_pair!(c,w)
            _k_sst_trial_line!(s.backend)(a.ktrial,a.wtrial,_status_slice(c,base+_S_SST_TRIAL),a.rk,a.rw,a.k0,a.w0,L.DK,
                c.conserved.Utrial,s.rho,wt,ni,nj,g;ndrange=(s.NI,s.NJ))
            _k_sst_trial_line!(ss.backend)(b.ktrial,b.wtrial,_status_slice(x,base+_S_SST_TRIAL),b.rk,b.rw,b.k0,b.w0,Ls.DK,
                x.conserved.Utrial,ss.rho,wt,nis,njs,g;ndrange=(ss.NI,ss.NJ))
            _slot_max!(_slot(c,base+_S_SST_TRIAL),_status_slice(c,base+_S_SST_TRIAL))
            _slot_max!(_slot(c,base+_S_SST_TRIAL),_status_slice(x,base+_S_SST_TRIAL))
        end
        _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
            Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
        _k_species_commit!(ss.backend)(x.conserved.U,ss.rho,ss.ux,ss.uy,ts.alpha,x.conserved.Utrial,
            Float64(ts.rho_l_sat),Float64(ts.rho_v_sat),nis,njs,g;ndrange=(ss.NI,ss.NJ))
        if s.sst
            _sst_commit!(c);_sst_commit!(x)
        end
        _project_pair!(c;batched=true)
        _strip_stage_observe(c,stage)
        _slot_batch!(c,specs.project[stage]);_slot_batch!(x,sspecs.project[stage])
    end
    s.sst && _sst_prepare_pair!(c;slots=(_status_slice(c,_SLOT_POST+3),_status_slice(x,_SLOT_POST+3)))
    _check_state_deferred!(c,_status_slice(c,_SLOT_POST+4));_check_state_deferred!(x,_status_slice(x,_SLOT_POST+4))
    _slot_batch!(c,specs.post);_slot_batch!(x,sspecs.post)
    nothing
end
# Diagnostic on retry exhaustion of a strip pair (stderr only; the state is already restored): the strip block's cells whose phase
# trial the last attempt refused, per stage, with their restored base state (the C block's are _imex_exhaustion_report's).
function _imex_exhaustion_report_strip(c)
    x=c.strip;ss=x.solver
    try
        U=Array(x.conserved.U);P=Array(x.pressure);g=ss.g
        for stage in 1:3
            st=Array(_status_slice(x,_stage_base(stage)+_S_TRIAL))
            bad=findall(>(0),@view st[1:ss.NI,1:ss.NJ])
            isempty(bad) && continue
            println(stderr,"LINE-IMPLICIT EXHAUSTION: stage $stage phase trial refused in $(length(bad)) cells of the strip block; first cells (base state):")
            for I in bad[1:min(end,8)]
                i,j=Tuple(I)
                println(stderr,"   strip (",i,",",j,") r=",U[i,j,1]," q=",U[i,j,4]," u=",U[i,j,2]/U[i,j,1]," v=",U[i,j,3]/U[i,j,1],
                        " p=",P[i+g,j+g])
            end
        end
    catch err
        println(stderr,"LINE-IMPLICIT EXHAUSTION: the strip block's report failed: ",sprint(showerror,err))
    end
    nothing
end
