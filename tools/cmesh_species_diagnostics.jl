# Host observations for the opt-in SpeciesKA movie path. Include after the
# existing solver/species modules; these functions do not advance or project them.
const CMESH_SPECIES_VOLUME_RTOL = 64.0 * eps(Float32)

function _cmesh_species_phase_volumes(r, q, p, pv, A, rl, rv, psm, lq=0.0)
    r, q, p, pv, A, rl, rv, psm, lq = Float64.((r, q, p, pv, A, rl, rv, psm, lq))
    all(isfinite, (r, q, p, pv, A, rl, rv, psm, lq)) &&
        r > 0 && 0 <= q <= r && pv > 0 && A > 0 && rl > rv > 0 && psm >= 0 && lq >= 0 &&
        p > psm/2 || throw(ArgumentError("invalid conservative phase-volume diagnostic state"))
    # Invert the guard of the actual cached pressure, including boundary values.
    # A second host EOS evaluation would be a different rounded pressure field.
    # lq (2026-09-24): the liquid specific volume carries the TIER-3 dissolved-gas compliance
    # exactly as _rhomix_vl's denominator does at alpha=0 (aef = lq); lq=0 is bitwise the
    # previous arithmetic (the added term is (raw-pv)*0.0/raw = 0.0 and x - 0.0 == x).
    raw = p - psm*psm/(4p)
    vl = (1 - (raw-pv)/(rl*A) - (raw-pv)*lq/raw)/rl
    vv = pv/(rv*raw)
    isfinite(raw) && raw > 0 && isfinite(vl) && vl > 0 && isfinite(vv) && vv > 0 ||
        throw(ArgumentError("nonpositive raw pressure or phase specific volume"))
    liquid = (r-q)*vl
    vapor = q*vv
    total = liquid + vapor
    # Rebuild (22 Sep 2026): this throw used to apply the interior tolerance CMESH_SPECIES_VOLUME_RTOL
    # (64 eps(Float32)) to EVERY cell, ghosts included. Measured on saved states, the conserved interior closes
    # to 4e-8 at density ratio 100 and 1000 alike, while the wall GHOST cells -- boundary copies whose
    # (rho, alpha-derived q, pressure) triple is not an EOS root -- sit at 2e-6 and scale with the ratio; at
    # ratio 1000 they crossed 7.6e-6 and ended arm R10 at frame 19 for no physical reason. The closure is now a
    # REPORTED quantity (phase_volume_sum, and the driver's interior/ghost columns); only an invalid state throws.
    isfinite(total) && total > 0 && abs(total-1) <= 1e-2 ||
        throw(ArgumentError("represented phase volumes do not close to one: sum=$total"))
    # Normalize represented positive phase volumes, rather than clip raw vapor
    # volume. The unnormalized value and closure error remain observable.
    beta = vapor/total
    isfinite(beta) && 0 <= beta <= 1 || throw(ArgumentError("invalid current vapor fraction"))
    (; raw_pressure=raw, beta, raw_beta=vapor, phase_volume_sum=total)
end

"""Return full-ghost host `pressure, beta, rho, q` and phase-volume diagnostics.

`pressure` is an exact copy of the physical pressure cache used by the mixture
flux and its boundary conditions. The caller must first fill that cache with the
existing pressure-BC routine after construction. No fallback to the carrier
pedestal or fresh EOS projection is performed here.

Interior `rho` and `q` are the authoritative Float64 packed conserved state.
Ghost `rho` is the stored Float32 value promoted to Float64; ghost `q` is derived
in Float64 from that stored rho/reference-alpha and the material constants.
`beta` is current vapor volume divided by the sum of represented
phase volumes; it is not the reference-alpha label. `raw_beta=q*v_v` and
`phase_volume_sum` retain the unnormalized value and the actual closure defect.
The sum must agree with one to 64 Float32 eps relative to max(1,abs(sum));
neither invalid states nor beta are silently clipped. All observations are
read-only; a failed cache must be restored before calling this function.
"""
function cmesh_species_fields(c)
    c.valid || throw(ArgumentError("cannot diagnose an invalid species cache"))
    s, t = c.solver, c.tem
    t.eos == 1 && t.wallis == 1 && t.vaploc == 1 && t.lq >= 0 && t.ncg == 0 ||
        throw(ArgumentError("phase-volume diagnostics require the admitted local-pressure species EOS"))
    KernelAbstractions.synchronize(s.backend)
    stored_rho, pressure, alpha, conserved = map(Array, (s.rho, c.pressure, t.alpha, c.conserved.U))
    ni, nj, g = s.NI, s.NJ, s.g
    shape = (ni+2g, nj+2g)
    ni > 0 && nj > 0 && g >= 0 && size(stored_rho) == size(pressure) == size(alpha) == shape &&
        conserved isa Array{Float64,3} && size(conserved) == (ni,nj,4) && all(isfinite,conserved) ||
        throw(ArgumentError("invalid species diagnostic array shapes or conserved state"))
    rho = Float64.(stored_rho)
    q = Matrix{Float64}(undef, shape)
    beta, raw_pressure, raw_beta, phase_volume_sum = (similar(q) for _ in 1:4)
    pv, A, rl, rv, psm, lq = Float64.((t.p_v, t.a2l, t.rho_l_sat, t.rho_v_sat, t.psm, t.lq))
    for j in axes(rho,2), i in axes(rho,1)
        if g < i <= ni+g && g < j <= nj+g
            rho[i,j] = conserved[i-g,j-g,1]
            q[i,j] = conserved[i-g,j-g,4]
        else
            a = Float64(alpha[i,j])
            isfinite(a) && 0 <= a <= 1 || throw(ArgumentError("invalid ghost reference-alpha at ($i,$j)"))
            q[i,j] = rho[i,j]*a*rv/((1-a)*rl+a*rv)
        end
        state = _cmesh_species_phase_volumes(rho[i,j],q[i,j],pressure[i,j],pv,A,rl,rv,psm,lq)
        beta[i,j] = state.beta
        raw_pressure[i,j] = state.raw_pressure
        raw_beta[i,j] = state.raw_beta
        phase_volume_sum[i,j] = state.phase_volume_sum
    end
    (; pressure, beta, rho, q, raw_pressure, raw_beta, phase_volume_sum)
end

"Volume-weighted interior mass/species inventories; pass the same frame's fields to avoid recopying."
function cmesh_species_inventory(c; fields=nothing)
    c.valid || throw(ArgumentError("cannot diagnose an invalid species cache"))
    s = c.solver
    f = fields === nothing ? cmesh_species_fields(c) : fields
    vol = Array(s.vol)
    size(f.rho) == size(f.q) == (s.NI+2s.g,s.NJ+2s.g) && size(vol) == (s.NI,s.NJ) ||
        throw(ArgumentError("invalid species inventory array shapes"))
    mass = 0.0; species = 0.0
    for j in 1:s.NJ, i in 1:s.NI
        r = Float64(f.rho[i+s.g,j+s.g]); q = Float64(f.q[i+s.g,j+s.g]); v = Float64(vol[i,j])
        all(isfinite,(r,q,v)) && r > 0 && 0 <= q <= r && v > 0 ||
            throw(ArgumentError("invalid species inventory cell ($i,$j)"))
        mass += r*v
        species += q*v
    end
    isfinite(mass) && isfinite(species) || throw(ArgumentError("nonfinite species inventory"))
    (; mass, species)
end

# The strip block of a strip pair (strip trailing-edge topology, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 4.7; stage 3,
# the driver's field and inventory outputs over both blocks). The two functions above read a cache through `valid`,
# `solver`, `tem`, `pressure` and `conserved` only; the strip block is offered to them as such a view (the pair's validity, the
# strip's carrier, TEM, published pressure and conserved state), so its fields and inventory are the same arithmetic on the
# second block and the two functions themselves are unchanged. A pair's totals are the two blocks' sums (the driver adds them).
function _cmesh_species_strip_view(c)
    hasproperty(c, :strip) && c.strip !== nothing ||
        throw(ArgumentError("the strip block's diagnostics need a strip pair (build_species_ka(...; strip=(solver=..., tem=...))); this cache has no strip block"))
    x = c.strip
    (; valid=c.valid, solver=x.solver, tem=x.tem, pressure=x.pressure, conserved=x.conserved)
end

"""The strip block's host fields of a strip pair: `cmesh_species_fields` on the strip's carrier, TEM, published pressure
and conserved state (every ghost included: the interface rows hold the neighbour block's published values, the base-wall
side the mirror). Refused for a cache without a strip and for an invalid pair."""
cmesh_species_strip_fields(c) = cmesh_species_fields(_cmesh_species_strip_view(c))

"""The strip block's interior mass and species inventories of a strip pair (`cmesh_species_inventory` on the strip block;
pass the same frame's `cmesh_species_strip_fields` to avoid recopying). The pair's inventory is the sum of this and the C
block's `cmesh_species_inventory(c)`."""
cmesh_species_strip_inventory(c; fields=nothing) = cmesh_species_inventory(_cmesh_species_strip_view(c); fields)

"""Closure against explicitly accumulated accepted-step ledgers over the same interval.

Boundary amounts are positive inward; source is positive vapor production.
Accumulate `c.last_mass_boundary`, `c.last_species_boundary` and
`c.last_species_source` once after each successful step. They are already RK/dt
weighted amounts, not rates. This helper applies no admission tolerance and
does not accumulate them or infer conservation from a small residual.
"""
function cmesh_species_closure(initial, current; mass_boundary, species_boundary, species_source)
    mi, qi, m, q, mb, qb, qs = Float64.((initial.mass,initial.species,current.mass,current.species,
                                       mass_boundary,species_boundary,species_source))
    all(isfinite,(mi,qi,m,q,mb,qb,qs)) && mi > 0 && m > 0 && 0 <= qi <= mi && 0 <= q <= m ||
        throw(ArgumentError("invalid species closure inputs"))
    mass_change = m-mi; species_change = q-qi
    mass_residual = mass_change-mb; species_residual = species_change-qb-qs
    isfinite(mass_residual) && isfinite(species_residual) ||
        throw(ArgumentError("nonfinite species closure residual"))
    (; mass=m, species=q, mass_change, species_change, mass_boundary=mb,
       species_boundary=qb, species_source=qs,
       mass_residual, species_residual)
end

# Disk restart for this restricted species path. Store host arrays and plain
# metadata, never device objects, object identities or the cache's work buffers.
using Serialization
const CMESH_SPECIES_RESTART_FORMAT = "CMeshSpeciesRestart/2"
_cmesh_restart_names(sst) = sst ?
    (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:pressure,:dtf,:tk,:tom,:nut,:sst_rk,:sst_rw) :
    (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:pressure,:dtf)

# U is stored once, in species.conserved. These are the remaining published
# Float32 fields; never reconstruct their exact saved bits from U on restore.
function _cmesh_restart_state(c)
    s,t=c.solver,c.tem
    base=(rho=Array(s.rho),ux=Array(s.ux),uy=Array(s.uy),T=Array(s.T),
          pexc=Array(s.pexc),am=Array(s.am),alpha=Array(t.alpha),
          pressure=Array(c.pressure),dtf=Array(s.dtf))
    s.sst ? (;base...,tk=Array(s.tk),tom=Array(s.tom),nut=Array(s.nut),
             sst_rk=Array(c.turbulence.rk),sst_rw=Array(c.turbulence.rw)) : base
end

# Match the core's published rollback order explicitly. The old position-eight
# q matrix is now the complete packed Float64 U, with no redundant q payload.
# A strip pair's /3 file (strip topology, plan section 4.7): the C block's arrays, then the strip's in the same order (its U
# from the checkpoint's :strip entry), then the pair's two U_nb buffers: exactly CMeshCavSpecies._rollback_arrays(c)'s
# order. A /2 file (no :strip entry) gives the tuple as before.
function _cmesh_restart_rollback_payload(d,sst)
    a=d.state
    base=(a.rho,a.ux,a.uy,a.T,a.pexc,a.am,a.alpha,d.species.conserved,a.pressure,a.dtf)
    main=sst ? (base...,a.tk,a.tom,a.nut,a.sst_rk,a.sst_rw) : base
    haskey(d,:strip) || return main
    b=d.strip.state
    sbase=(b.rho,b.ux,b.uy,b.T,b.pexc,b.am,b.alpha,d.species.strip.conserved,b.pressure,b.dtf)
    strip=sst ? (sbase...,b.tk,b.tom,b.nut,b.sst_rk,b.sst_rw) : sbase
    (main...,strip...,d.strip.interface.U_legs,d.strip.interface.U_strip)
end

# The strip topology's restart format (plan section 4.7): /2's six entries and a seventh, `strip = (state, interface)`: the
# strip block's published fields under the same names (_cmesh_restart_state on the strip block) and the pair's two U_nb
# buffers. Written by a strip pair only; a cache without a strip writes /2, byte for byte. Mutually refused, never converted.
const CMESH_SPECIES_RESTART_FORMAT_STRIP = "CMeshSpeciesRestart/3"
_cmesh_restart_strip_entry(c)=(state=_cmesh_restart_state(c.strip),
    interface=(U_legs=Array(c.strip.interface.U_legs),U_strip=Array(c.strip.interface.U_strip)))
_cmesh_restart_strip_carriers(c)=c.strip===nothing ? nothing : (solver=c.strip.solver,tem=c.strip.tem)

# One block's published Float32 fields and its authoritative U (shapes, finiteness, phase bounds) and, with SST, its
# conserved SST state bound to its own wall-distance hash (`sp` the block's carrier).
function _cmesh_restart_validate_fields(state,U,turbulence,sp,bad,label)
    names=_cmesh_restart_names(sp.sst)
    state isa NamedTuple && keys(state)==names || bad("published field inventory$label")
    ghost=(sp.NI+2sp.g,sp.NJ+2sp.g);inner=(sp.NI,sp.NJ)
    for name in names
        a=getproperty(state,name)
        shape=name in (:dtf,:sst_rk,:sst_rw) ? inner : ghost
        a isa Matrix{Float32} && size(a)==shape && all(isfinite,a) || bad("shape/type/nonfinite $name$label")
    end
    U isa Array{Float64,3} && size(U)==(sp.NI,sp.NJ,4) && all(isfinite,U) ||
        bad("packed Float64 conserved shape/type/nonfinite state$label")
    for j in 1:sp.NJ, i in 1:sp.NI
        U[i,j,1]>0.0 && 0.0<=U[i,j,4]<=U[i,j,1] || bad("conserved density/species bounds$label")
    end
    if sp.sst
        a=turbulence
        a isa NamedTuple && keys(a)==(:format,:geometry_sha256,:rk,:rw) &&
            a.format=="CMeshSpeciesSST/1" && a.geometry_sha256==CMeshCavSpecies._sst_geometry_hash(sp) &&
            a.rk isa Matrix{Float32} && a.rw isa Matrix{Float32} &&
            isequal(a.rk,state.sst_rk) && isequal(a.rw,state.sst_rw) || bad("SST state/geometry$label")
    else
        turbulence===nothing || bad("unexpected SST state$label")
    end
    nothing
end

# The strip pair's published interface (plan sections 4.1 and 4.7): the saved state of a strip pair is a pair projection's
# (every step and the build end in the exchange), so each interface ghost row of either block holds the neighbour's interior
# row bit for bit in the eleven exchanged fields, the strip's base-wall corner blocks the mirrored station's leg row with
# the velocity reflected, and the two U_nb buffers the neighbours' saved U. A file that fails any of them is refused.
function _cmesh_restart_validate_interface(d,s,ss,bad)
    NIW,NIF,NS,g=s.NIW,s.NIF,ss.NJ,s.g
    f=d.strip.interface
    f isa NamedTuple && keys(f)==(:U_legs,:U_strip) &&
        all(a isa Array{Float64,4} && size(a)==(NIW,g,4,2) && all(isfinite,a) for a in (f.U_legs,f.U_strip)) ||
        bad("the strip pair's interface buffers (U_legs, U_strip: Float64 (NIW, g, 4, 2))")
    UC,US=d.species.conserved,d.species.strip.conserved
    all(isequal(f.U_legs[st,k,q,1],US[st,k,q]) && isequal(f.U_legs[st,k,q,2],US[st,NS+1-k,q]) &&
        isequal(f.U_strip[st,k,q,1],UC[NIW+1-st,k,q]) && isequal(f.U_strip[st,k,q,2],UC[NIW+NIF+st,k,q])
        for st in 1:NIW,k in 1:g,q in 1:4) || bad("the strip pair's U_nb is not the neighbours' saved U")
    A,B=d.state,d.strip.state
    names=s.sst ? (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:pressure,:tk,:tom,:nut) : (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:pressure)
    for name in names
        a=getproperty(A,name);b=getproperty(B,name)
        all(isequal(a[NIW+1-st+g,g+1-k],b[st+g,g+k]) && isequal(a[NIW+NIF+st+g,g+1-k],b[st+g,NS+g+1-k]) &&
            isequal(b[st+g,g+1-k],a[NIW+1-st+g,g+k]) && isequal(b[st+g,NS+g+k],a[NIW+NIF+st+g,g+k])
            for st in 1:NIW,k in 1:g) || bad("the strip pair's interface ghost rows of $name are not the neighbours' saved interior")
    end
    for name in (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:pressure)
        a=getproperty(A,name);b=getproperty(B,name);sg=name in (:ux,:uy) ? -1.0f0 : 1.0f0
        all(isequal(b[g+1-st,g+1-k],sg*a[NIW+1-st+g,g+k]) && isequal(b[g+1-st,NS+g+k],sg*a[NIW+NIF+st+g,g+k])
            for st in 1:g,k in 1:g) || bad("the strip's base-wall corner ghosts of $name are not the mirrored stations' leg cells")
    end
    nothing
end

# `strip`: nothing (a cache without a strip: the /2 format, every check as before) or the strip pair's strip carrier and
# TEM `(solver=ss, tem=ts)`: the /3 format, the pair's configuration and geometry hash, the strip block's fields and conserved
# state, its CFL field holding the applied global step, and the published interface (_cmesh_restart_validate_interface).
# (10 Oct 2026) `reboud_form`, `reboud_prod` (13 Oct), `source_model` and `ss` are the cache's (CMeshCavSpecies._configuration's keywords, defaults the
# tuples of every existing positional caller): the Reboud form, the phase-change source model and its Schnerr-Sauer parameters.
function _cmesh_restart_validate(d,s,t,wallmode,reconstruction::Bool=false,kinetic::Bool=false,eos_single::Bool=false,
                                 source_beta::Bool=false,reboud_n::Float32=0.0f0,trip::NTuple{2,Float32}=(0.0f0,0.0f0),
                                 sst_climit::Float32=-1.0f0,stage_margin::Float32=-1.0f0;strip=nothing,
                                 reboud_form::Int32=Int32(1),reboud_prod::Int32=Int32(0),source_model::Int32=Int32(0),
                                 ss::NTuple{4,Float64}=(0.0,0.0,0.0,0.0),recon_retreat::Int32=Int32(0))
    bad(message)=throw(ArgumentError("invalid species restart: "*message))
    format=d isa NamedTuple && hasproperty(d,:format) ? d.format : nothing
    if strip===nothing
        format==CMESH_SPECIES_RESTART_FORMAT_STRIP &&
            bad("a $CMESH_SPECIES_RESTART_FORMAT_STRIP file (strip topology, CMeshSpeciesStripTopologyV1) restores only into a strip pair: pass strip=(solver=..., tem=...)")
        d isa NamedTuple && keys(d)==(:format,:state,:species,:step,:retry,:metadata) &&
            d.format==CMESH_SPECIES_RESTART_FORMAT || bad("format; old formats are not converted")
    else
        format==CMESH_SPECIES_RESTART_FORMAT &&
            bad("a strip pair (CMeshSpeciesStripTopologyV1) restores only a $CMESH_SPECIES_RESTART_FORMAT_STRIP file; this one is $CMESH_SPECIES_RESTART_FORMAT (written by a cache without a strip)")
        d isa NamedTuple && keys(d)==(:format,:state,:species,:step,:retry,:metadata,:strip) &&
            d.format==CMESH_SPECIES_RESTART_FORMAT_STRIP && d.strip isa NamedTuple && keys(d.strip)==(:state,:interface) ||
            bad("format; old formats are not converted")
    end
    cp,m,r=d.species,d.step,d.retry
    if strip===nothing
        cp isa NamedTuple && keys(cp)==(:format,:configuration,:geometry_sha256,:wallmode,:conserved,:steps,:elapsed,:turbulence) &&
            cp.format=="CMeshSpeciesKA/2" && cp.wallmode===wallmode || bad("species schema or wall mode; old formats are not converted")
    else
        cp isa NamedTuple && keys(cp)==CMeshCavSpecies._STRIP_CHECKPOINT_KEYS &&
            cp.format==CMeshCavSpecies.STRIP_CHECKPOINT_FORMAT && cp.wallmode===wallmode &&
            cp.strip isa NamedTuple && keys(cp.strip)==(:conserved,:turbulence) || bad("species schema or wall mode; old formats are not converted")
    end
    # (3 Oct 2026) the stage clock margin is bound like every other option; a V1 limited-correction file is named in the refusal
    configuration=CMeshCavSpecies._configuration(s,t,reconstruction,kinetic,eos_single,source_beta,reboud_n,trip,sst_climit,stage_margin;
                                                 reboud_form,reboud_prod,source_model,ss,recon_retreat)
    strip===nothing || (configuration=CMeshCavSpecies._pair_configuration(configuration,strip.solver,strip.tem))
    cp.configuration==configuration ||
        bad(CMeshCavSpecies._configuration_mismatch("configuration mismatch",cp.configuration,configuration))
    cp.geometry_sha256==(strip===nothing ? CMeshCavSpecies._geometry_hash(s) : CMeshCavSpecies._geometry_hash_pair(s,strip.solver)) ||
        bad("geometry mismatch")
    _cmesh_restart_validate_fields(d.state,cp.conserved,cp.turbulence,s,bad,"")
    strip===nothing || _cmesh_restart_validate_fields(d.strip.state,cp.strip.conserved,cp.strip.turbulence,strip.solver,bad," (the strip block)")
    m isa NamedTuple && keys(m)==(:steps,:elapsed,:dt,:mass,:species,:source,:sst) || bad("step schema")
    m.steps isa Int && 0<=m.steps<typemax(Int) && m.elapsed isa Float64 &&
        isfinite(m.elapsed) && m.elapsed>=0 && (m.steps==0)==(m.elapsed==0) &&
        cp.steps===m.steps && isequal(cp.elapsed,m.elapsed) || bad("physical clock")
    m.dt isa Float32 && isfinite(m.dt) && m.dt>=0 && (m.steps==0 || m.dt>0) &&
        all(x->isequal(x,m.dt),d.state.dtf) || bad("applied global timestep")
    if strip!==nothing
        # the strip's CFL field holds the one global step of the pair (plan 4.6), and its interface is the pair's published one
        all(x->isequal(x,m.dt),d.strip.state.dtf) || bad("applied global timestep (the strip block)")
        _cmesh_restart_validate_interface(d,s,strip.solver,bad)
    end
    # (rebuild, 22 Sep 2026) the ledger scalars are NaN when the per-step ledger is off: unmeasured, not invalid.
    all(x->x isa Float64 && !isinf(x),(m.mass,m.species,m.source)) || bad("phase/mass ledgers")
    (s.sst ? m.sst isa NTuple{4,Float64} && all(x->!isinf(x),m.sst) : m.sst===nothing) || bad("SST ledgers")
    r isa NamedTuple && keys(r)==(:last_retries,:last_rejection) &&
        r.last_retries isa Int && 0<=r.last_retries<=8 || bad("retry metadata")
    if r.last_rejection!==nothing
        e=r.last_rejection
        e isa NamedTuple && keys(e)==(:stage,:reason,:dt) && e.stage isa Int && 1<=e.stage<=3 &&
            e.reason isa Symbol && e.dt isa Float32 && isfinite(e.dt) && e.dt>0 || bad("last rejection")
    end
    nothing
end

"""Exclusively create a disk checkpoint of an admitted, complete species state.

The host NamedTuple preserves all 10/15 published arrays: packed Float64 U and
the remaining 9/14 Float32 primitive/pressure/dt/SST arrays. U is stored once in
the species /2 checkpoint; q is its fourth component, not a separate payload.
It preserves the physical clock, last accepted-step ledgers and retry metadata. It excludes RK,
residual and backup work buffers. `metadata` is opaque application provenance;
it does not restore movie histories, output ownership or scored-segment clocks.
No existing path is overwritten. A failed serialization leaves its partial file
for inspection. Save only between complete steps, without concurrent mutation.
A strip pair (strip topology, plan section 4.7) writes CMeshSpeciesRestart/3: the
same six entries (its species checkpoint is CMeshSpeciesKA/3) and `strip = (state,
interface)`, the strip block's published fields and the pair's two U_nb buffers;
every other cache writes CMeshSpeciesRestart/2, byte for byte as before.
"""
function write_cmesh_species_restart(path::AbstractString,c;metadata=nothing)
    cp=CMeshCavSpecies.species_checkpoint(c)
    # (rebuild, 22 Sep 2026) the second full cmesh_species_fields pass per frame is gone; the validator below
    # already rejects a non-finite pressure cache, and the driver has just run the diagnostics on this state.
    state=_cmesh_restart_state(c)
    e=c.last_rejection
    retry=(last_retries=c.last_retries,last_rejection=e===nothing ? nothing :
           (stage=e.stage,reason=e.reason,dt=e.dt))
    d=c.strip===nothing ? (format=CMESH_SPECIES_RESTART_FORMAT,state=state,species=cp,
       step=CMeshCavSpecies._step_metadata(c),retry=retry,metadata=metadata) :
      (format=CMESH_SPECIES_RESTART_FORMAT_STRIP,state=state,species=cp,
       step=CMeshCavSpecies._step_metadata(c),retry=retry,metadata=metadata,strip=_cmesh_restart_strip_entry(c))
    _cmesh_restart_validate(d,c.solver,c.tem,c.wallmode,c.reconstruction!==nothing,c.kinetic!==nothing,c.eos_single,c.source_beta,c.reboud_n,c.trip,c.sst_climit,c.stage_margin;
        strip=_cmesh_restart_strip_carriers(c),reboud_form=c.reboud_form,reboud_prod=c.reboud_prod,source_model=c.source_model,ss=c.ss,
        recon_retreat=CMeshCavSpecies._recon_retreat(c))
    io=Base.Filesystem.open(path,Base.JL_O_WRONLY|Base.JL_O_CREAT|Base.JL_O_EXCL,0o600)
    try
        serialize(io,d)
    finally
        close(io)
    end
    path
end

"""Restore into a fresh, identically configured solver/TEM pair; return `(cache,metadata)`.

Construct geometry, material, boundary aliases and a valid initial EOS state as
usual first. This helper checks configuration/geometry before copying state and
uses the existing cache restore/admission. Direct pressure is restored exactly,
not recomputed through a different arithmetic path. Expected restore failures
put the destination's published flow/TEM arrays and last timestep back. It does
not promise recovery from allocation/device failure during rollback, or support
mid-stage restart. The caller starts a fresh scored/output segment explicitly.
The reconstruction and kinetic-face choices must match the stored method tags.
Old /1 formats are rejected before destination writes, without conversion. Its
face states, half masses, U0, Utrial and geometry workspace are rebuilt; only U
is authoritative.
`strip=(solver=ss, tem=ts)` restores a strip pair's CMeshSpeciesRestart/3 file into
fresh strip carriers prepared like the C block's (plan section 4.7): the strip
block's saved fields go to the strip carrier and TEM, the pair is built from the
/3 checkpoint, the projected pressure of both blocks is compared at every ghost the
block-aware donor maps (interface ghosts and the strip's base-wall corners
included), and the saved rollback payload of both blocks and U_nb is restored.
A /2 file is refused with a strip and a /3 file without one, by name.
The Reboud treatment (`nut_density`, `reboud_n`, `reboud_production`) and the phase-change source
(`source_model`, `ss_n0`, `ss_dnuc`) are those of build_species_ka (10 and 13 Oct 2026) and must
match the file's: an archived file (no Reboud tag) restores with `nut_density=:none`, and the
refusal says so.
"""
function restore_cmesh_species_restart(s,t,path::AbstractString;wallmode::Int32=Int32(2),reconstruction::Bool=false,kinetic::Bool=false,
                                       eos_precision::Symbol=CMeshCavSpecies.DEFAULT_EOS_PRECISION[],
                                       source_weight::Symbol=CMeshCavSpecies.DEFAULT_SOURCE_WEIGHT[],
                                       nut_density::Symbol=CMeshCavSpecies.DEFAULT_NUT_DENSITY[],reboud_n::Real=CMeshCavSpecies.DEFAULT_REBOUD_N[],
                                       reboud_production::Bool=false,
                                       source_model::Symbol=CMeshCavSpecies.DEFAULT_SOURCE_MODEL[],ss_n0=nothing,ss_dnuc=nothing,
                                       trip_x::Real=0.0,trip_width::Real=CMeshCavSpecies.DEFAULT_TRIP_WIDTH,
                                       reduction_tile::Int=0,sst_correction_limit::Real=-1.0,stage_clock_margin::Real=-1.0,
                                       recon_retreat::Union{Nothing,Symbol}=nothing,strip=nothing)
    CMeshCavSpecies._validate_reduction_tile(reduction_tile)
    eos_precision in (:double,:single) || throw(ArgumentError("eos_precision must be :double or :single"))
    source_weight in (:alpha,:beta) || throw(ArgumentError("source_weight must be :alpha or :beta"))
    rn,rform,rprod=CMeshCavSpecies._nut_density_setting(nut_density,reboud_n,reboud_production)
    sm,ssp=CMeshCavSpecies._source_model_setting(source_model,ss_n0,ss_dnuc,source_weight)
    strip===nothing || (strip isa NamedTuple && keys(strip)==(:solver,:tem)) ||
        throw(ArgumentError("strip must be (solver=<the strip's KSol>, tem=<its TEMState>)"))
    # the types here, before the validator's pair configuration reads the strip carrier's fields (_admit_strip's text)
    strip===nothing || (strip.solver isa CMeshSolverKA.KSol && strip.tem isa CMeshCavTEM.TEMState) ||
        throw(ArgumentError("strip=(solver=..., tem=...) must hold the strip's KSol and TEMState"))
    trip=CMeshCavSpecies._trip_setting(s,trip_x,trip_width)
    climit=CMeshCavSpecies._sst_climit_setting(s,reconstruction,sst_correction_limit)
    smargin=CMeshCavSpecies._stage_margin_setting(stage_clock_margin)
    retreat=CMeshCavSpecies._recon_retreat_setting(reconstruction,CMeshCavSpecies._recon_retreat_choice(reconstruction,recon_retreat))
    d=open(path,"r") do io
        value=deserialize(io)
        eof(io) || throw(ArgumentError("species restart has trailing bytes"))
        value
    end
    _cmesh_restart_validate(d,s,t,wallmode,reconstruction,kinetic,eos_precision===:single,source_weight===:beta,rn,trip,climit,smargin;strip,
                            reboud_form=rform,reboud_prod=rprod,source_model=sm,ss=ssp,recon_retreat=retreat)
    CMeshCavSpecies._admit(s,t,wallmode)
    names=s.sst ? (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:dtf,:tk,:tom,:nut) :
                  (:rho,:ux,:uy,:T,:pexc,:am,:alpha,:dtf)
    targets=map(name->name===:alpha ? t.alpha : getproperty(s,name),names)
    before=map(copy,targets);old_dt=s.last_global_dt
    # a strip pair: the strip block's fields go to the strip carrier and TEM (rolled back with the C block's on failure)
    stargets=strip===nothing ? () : map(name->name===:alpha ? strip.tem.alpha : getproperty(strip.solver,name),names)
    sbefore=map(copy,stargets);old_sdt=strip===nothing ? 0.0f0 : strip.solver.last_global_dt
    KernelAbstractions.synchronize(s.backend)
    try
        for (name,target) in zip(names,targets)
            copyto!(target,getproperty(d.state,name))
        end
        if strip!==nothing
            for (name,target) in zip(names,stargets)
                copyto!(target,getproperty(d.strip.state,name))
            end
        end
        c=CMeshCavSpecies.build_species_ka(s,t;checkpoint=d.species,wallmode=wallmode,reconstruction=reconstruction,kinetic=kinetic,
                                           eos_precision=eos_precision,source_weight=source_weight,nut_density=nut_density,reboud_n=reboud_n,
                                           reboud_production=reboud_production,source_model=source_model,ss_n0=ss_n0,ss_dnuc=ss_dnuc,
                                           trip_x=trip_x,trip_width=trip_width,reduction_tile=reduction_tile,
                                           sst_correction_limit=sst_correction_limit,stage_clock_margin=stage_clock_margin,
                                           recon_retreat=recon_retreat,strip=strip)
        payload=_cmesh_restart_rollback_payload(d,s.sst)
        length(c.backup)==length(payload) &&
            all(size(target)==size(saved) && eltype(target)==eltype(saved)
                for (target,saved) in zip(c.backup,payload)) ||
            throw(ArgumentError("species restart rollback schema changed"))
        # The restored cache projected its pressure at build from the saved conserved state (rebuild,
        # 22 Sep 2026). Compare it with the saved field before the saved rollback payload overwrites it.
        KernelAbstractions.synchronize(s.backend)
        projected=Array(c.pressure);kinds=Array(s.outer_face_kind)
        pressure_rtol=64.0*eps(Float32) # pressure scale; not pedestal or stiffness
        if strip===nothing
        for j in axes(projected,2),i in axes(projected,1)
            ii,jj=CMeshSolverKA._mixture_conserved_donor(i,j,kinds,s.NI,s.NJ,s.NIW,s.NIF,s.g)
            ii==0 && continue # characteristic/unmapped ghosts retain boundary semantics
            saved_p=Float64(d.state.pressure[i,j]);expected_p=Float64(projected[i,j])
            isfinite(saved_p) && saved_p>0 && isfinite(expected_p) && expected_p>0 &&
                abs(saved_p-expected_p)<=pressure_rtol*max(saved_p,expected_p) ||
                throw(ArgumentError("invalid species restart: saved direct pressure disagrees with conserved EOS at ($i,$j)"))
        end
        else
            # both blocks, through the block-aware donor (plan 4.1 item 2, 4.7): the interface ghosts (src > 0, the
            # neighbour's projection, written by the exchange) and the strip's base-wall corners are compared too
            x=c.strip
            _cmesh_restart_pressure_walk(projected,d.state.pressure,kinds,s,CMeshCavSpecies._strip_ctx(c),pressure_rtol,"")
            _cmesh_restart_pressure_walk(Array(x.pressure),d.strip.state.pressure,Array(x.solver.outer_face_kind),x.solver,
                CMeshCavSpecies._strip_ctx(x),pressure_rtol," in the strip block")
        end
        for (target,saved) in zip(c.backup,payload)
            copyto!(target,saved)
        end
        CMeshCavSpecies._restore_step!(c,d.step)
        c.last_retries=d.retry.last_retries
        e=d.retry.last_rejection
        c.last_rejection=e===nothing ? nothing : CMeshCavSpecies.SpeciesNumericalRejection(e.stage,e.reason,e.dt)
        CMeshCavSpecies._check_owner(c);CMeshCavSpecies._check_state!(c)
        strip===nothing || CMeshCavSpecies._check_state!(c.strip)
        cmesh_species_fields(c)
        (;cache=c,metadata=d.metadata)
    catch error
        try
            for (target,saved) in zip(targets,before)
                copyto!(target,saved)
            end
            s.last_global_dt=old_dt
            if strip!==nothing
                for (target,saved) in zip(stargets,sbefore)
                    copyto!(target,saved)
                end
                strip.solver.last_global_dt=old_sdt
            end
            KernelAbstractions.synchronize(s.backend)
        catch rollback_error
            throw(CompositeException(Any[error,rollback_error]))
        end
        rethrow()
    end
end

# The restore's pressure walk of one block of a strip pair (plan 4.7): every ghost the block-aware donor maps (an interior
# cell of the block, an interface ghost through the other block, a base-wall corner through the mirrored station) holds the
# saved pressure to the pressure scale; unmapped ghosts (the outlet's characteristic columns) keep their boundary semantics.
function _cmesh_restart_pressure_walk(projected,saved,kinds,sp,ctx,rtol,label)
    for j in axes(projected,2),i in axes(projected,1)
        ii,jj,src=CMeshSolverKA._donor3(i,j,kinds,sp.NI,sp.NJ,sp.NIW,sp.NIF,sp.g,ctx)
        ii==0 && continue # characteristic/unmapped ghosts retain boundary semantics
        saved_p=Float64(saved[i,j]);expected_p=Float64(projected[i,j])
        isfinite(saved_p) && saved_p>0 && isfinite(expected_p) && expected_p>0 &&
            abs(saved_p-expected_p)<=rtol*max(saved_p,expected_p) ||
            throw(ArgumentError("invalid species restart: saved direct pressure disagrees with conserved EOS at ($i,$j)$label"))
    end
    nothing
end
