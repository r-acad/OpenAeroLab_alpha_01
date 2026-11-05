# The run state: per level, two population buffers (post-collision at t and at t + 1), two velocity buffers and
# two density buffers, in the mesh's oct layout. DESIGN.md sections 3 and 5. Populations are shifted (f - w_i).
using Base.Threads

mutable struct LevelState{T<:AbstractFloat}
    f::Array{T,3}            # (8, Q, nocts) buffer A
    g::Array{T,3}            # buffer B
    u::Array{T,3}            # (8, 3, nocts) velocity A
    v::Array{T,3}            # velocity B
    rho::Matrix{T}           # (8, nocts) density A
    rhob::Matrix{T}          # density B
    nu::T                    # molecular viscosity, lattice units of this level
    omega::T
    real_octs::Vector{Int32} # the octs the step iterates
    chunks::Vector{UnitRange{Int}}   # thread partition of real_octs
    tau_w::Vector{T}         # wall-model friction velocity per cut cell (phase 4)
    # transfer tables (coupling.jl): the distinct donor slots of this level's ghosts (at level - 1) and the distinct
    # source slots of its restricted slots (at level + 1), with the stencil tables re-indexed into them, and the
    # moment buffers (rho, u, Pi_neq: 10 rows) filled once per transfer
    donor_cells::Vector{Int32}
    ghost_dix::Matrix{Int32}         # (64, nghost): index into donor_cells, 0 = unused node
    donor_m::Matrix{T}               # (10, ndonor)
    rsrc_cells::Vector{Int32}
    restrict_six::Matrix{Int32}      # (64, nrestrict): index into rsrc_cells
    rsrc_m::Matrix{T}                # (10, nsource)
    # the cut links of each cut cell spread over the directions, for the step's O(1) lookup (the CSR is the file form)
    cut_qdir::Matrix{Float32}        # (Q, ncut): q of the cut link in direction i, 0.5 where none
    cut_okdir::Matrix{Bool}          # (Q, ncut): the Bouzidi partner is a plain fluid slot
end

"The cut-link CSR of a level spread over the directions: (Q, ncut) tables of q and of the partner flag."
function cut_direction_tables(Lv::OctLevel)
    nc = length(Lv.cut_slot)
    qdir = fill(0.5f0, Q, nc); okdir = fill(false, Q, nc)
    for ci in 1:nc, li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
        qdir[Int(Lv.cut_dir[li]), ci] = Lv.cut_q[li]; okdir[Int(Lv.cut_dir[li]), ci] = Lv.cut_partner[li]
    end
    qdir, okdir
end

"Distinct slot ids of a (64, n) stencil table and the table re-indexed into them (0 stays 0)."
function index_table(tab::Matrix{Int32})
    idx = Dict{Int32,Int32}(); cells = Int32[]; ix = zeros(Int32, size(tab))
    for j in 1:size(tab, 2), n in 1:size(tab, 1)
        s = tab[n, j]; s == 0 && continue
        ix[n, j] = get!(idx, s) do
            push!(cells, s); Int32(length(cells))
        end
    end
    cells, ix
end

mutable struct State{T<:AbstractFloat}
    mesh::OctMesh
    cfg::CaseConfig
    scaling::LatticeScaling
    levels::Vector{LevelState{T}}
    step::Int                # coarse steps completed
    u_inlet::T               # current inlet speed, lattice units (the same on every level)
    u_target::T
    inlet_dir::NTuple{3,T}   # unit vector of the inlet velocity (alpha/beta are applied to the geometry, so +x)
    forces::Vector{Float64}  # last total force (lattice units of the finest level, converted at output): Fx, Fy, Fz
    tri_force::Matrix{Float64}      # (3, ntriangles) per-triangle force, last collection
    thread_tri::Vector{Matrix{Float64}}  # per-thread accumulators
    mass0::Float64
end

function chunk_ranges_min(n::Int, nt::Int, minsize::Int)
    nt = max(1, min(nt, cld(n, minsize)))
    chunk_ranges(n, nt)
end

"""Contiguous oct ranges with roughly equal fluid work rather than equal oct counts.
An oct near the body may have few fluid children or several costly cut links;
equal-count ranges left some C212 threads with over three times the fluid work.
The special-cell increment is only a scheduling weight, not a numerical change.
"""
function weighted_oct_chunks(Lv::OctLevel, real_octs::Vector{Int32}, nt::Int; minsize::Int=256)
    n = length(real_octs)
    nt = max(1, min(nt, cld(n, minsize)))
    nt == 1 && return UnitRange{Int}[1:n]
    function weight(oi)
        o = Int(real_octs[oi]); w = 0
        @inbounds for k in 1:8
            if Lv.kind[k, o] == K_FLUID
                w += 1 + (Lv.special[k, o] != 0)
            end
        end
        w
    end
    total = sum(weight(oi) for oi in 1:n)
    total == 0 && return chunk_ranges(n, nt)
    ranges = UnitRange{Int}[]
    start = 1; cumulative = 0; part = 1
    for oi in 1:n
        cumulative += weight(oi)
        if part < nt && oi - start + 1 >= minsize && n - oi >= (nt - part) * minsize && cumulative * nt >= part * total
            push!(ranges, start:oi)
            start = oi + 1
            part += 1
        end
    end
    push!(ranges, start:n)
    ranges
end

"Allocate the state at rest (rho = 1, u = 0: shifted populations are zero) with first touch by the owning thread."
function State(mesh::OctMesh, cfg::CaseConfig, ::Type{T}) where {T<:AbstractFloat}
    sc = lattice_scaling(cfg)
    nt = nthreads()
    levels = LevelState{T}[]
    for (l, Lv) in enumerate(mesh.levels)
        real_octs = Int32[o for o in 1:Lv.nocts if Lv.real[o]]
        chunks = weighted_oct_chunks(Lv, real_octs, nt)                  # balance fluid work, retain oct locality
        f = Array{T,3}(undef, 8, Q, Lv.nocts); g = similar(f)
        u = Array{T,3}(undef, 8, 3, Lv.nocts); v = similar(u)
        rho = Matrix{T}(undef, 8, Lv.nocts); rhob = similar(rho)
        # first touch: real octs by their thread, the rest (ghost, boundary) by thread 1
        @threads :static for t in 1:length(chunks)
            @inbounds for oi in chunks[t]
                o = real_octs[oi]
                for i in 1:Q, k in 1:8; f[k, i, o] = zero(T); g[k, i, o] = zero(T); end
                for a in 1:3, k in 1:8; u[k, a, o] = zero(T); v[k, a, o] = zero(T); end
                for k in 1:8; rho[k, o] = one(T); rhob[k, o] = one(T); end
            end
        end
        for o in 1:Lv.nocts
            Lv.real[o] && continue
            @inbounds for i in 1:Q, k in 1:8; f[k, i, o] = zero(T); g[k, i, o] = zero(T); end
            @inbounds for a in 1:3, k in 1:8; u[k, a, o] = zero(T); v[k, a, o] = zero(T); end
            @inbounds for k in 1:8; rho[k, o] = one(T); rhob[k, o] = one(T); end
        end
        nu = T(sc.nu_lattice[l])
        donor_cells, ghost_dix = index_table(Lv.ghost_donor)
        rsrc_cells, restrict_six = index_table(Lv.restrict_src)
        qdir, okdir = cut_direction_tables(Lv)
        push!(levels, LevelState{T}(f, g, u, v, rho, rhob, nu, omega_of(nu), real_octs, chunks, zeros(T, length(Lv.cut_slot)),
                                    donor_cells, ghost_dix, Matrix{T}(undef, 10, length(donor_cells)), rsrc_cells, restrict_six, Matrix{T}(undef, 10, length(rsrc_cells)),
                                    qdir, okdir))
    end
    ntri = max(length(mesh.triangles), 1)
    State{T}(mesh, cfg, sc, levels, 0, zero(T), T(sc.u_lattice), (one(T), zero(T), zero(T)), zeros(3), zeros(3, ntri), [zeros(3, ntri) for _ in 1:nt], 0.0)
end

"""
Set a uniform or user-given initial field: `init(p) -> (rho, ux, uy, uz)` in lattice units (p in metres). With
`nonequilibrium` (the default) the populations carry the Chapman-Enskog non-equilibrium of the field,
Pi_neq = -rho cs^2 tau (grad u + grad u^T) from central differences of `init` over one cell of each level, stored
as the post-collision state. An equilibrium start builds that part up through the over-relaxation oscillation
(1 - omega about -0.95, a hundred steps) at every level's own cadence, and the levels' errors meet out of phase at
the interfaces: a vortex started at equilibrium radiated a pressure pulse of a quarter of its deficit from a
2-level box, nothing from one level. A uniform field has no gradient and is unaffected.
"""
function initialise!(st::State{T}, init; nonequilibrium::Bool=true) where T
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]; omega = S.omega; tau = one(T) / omega; h = Lv.dx / 2
        ranges = chunk_ranges_min(Lv.nocts, nthreads(), 256)
        @threads :static for t in 1:length(ranges)
          for o in ranges[t], k in 1:8
            kd = Lv.kind[k, o]
            (kd == K_FLUID || kd == K_GHOST || kd == K_COVERED) || continue
            p = slot_centre(Lv, slot_id(k, o))
            rho, ux, uy, uz = init(p)
            rho = T(rho); ux = T(ux); uy = T(uy); uz = T(uz); usq = ux * ux + uy * uy + uz * uz
            if nonequilibrium
                # d_a u_b over one cell: velocity difference between the faces of the cell along a
                d = ntuple(a -> begin
                        e = ntuple(b -> b == a ? h : 0.0, 3)
                        up = init((p[1] + e[1], p[2] + e[2], p[3] + e[3])); um = init((p[1] - e[1], p[2] - e[2], p[3] - e[3]))
                        (T(up[2] - um[2]), T(up[3] - um[3]), T(up[4] - um[4]))
                    end, 3)
                c = -rho * tau / T(3)
                pn = (c * T(2) * d[1][1], c * T(2) * d[2][2], c * T(2) * d[3][3], c * (d[1][2] + d[2][1]), c * (d[1][3] + d[3][1]), c * (d[2][3] + d[3][2]))
                fs = rebuild(rho, ux, uy, uz, pn, omega)
            else
                fs = ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q))
            end
            @inbounds for i in 1:Q
                S.f[k, i, o] = fs[i]; S.g[k, i, o] = fs[i]
            end
            @inbounds S.u[k, 1, o] = ux; S.u[k, 2, o] = uy; S.u[k, 3, o] = uz
            @inbounds S.v[k, 1, o] = ux; S.v[k, 2, o] = uy; S.v[k, 3, o] = uz
            @inbounds S.rho[k, o] = rho; S.rhob[k, o] = rho
          end
        end
    end
    nlevels(st.mesh) > 1 && couple_initial!(st)
    st.mass0 = total_mass(st)
    st
end

"Uniform equilibrium start for production cases. The generic initializer samples a user field and its
gradient seven times per active slot; a constant freestream needs neither coordinates nor gradients."
function initialise_uniform!(st::State{T}, rho0::Real, ux0::Real, uy0::Real, uz0::Real) where T
    rho = T(rho0); ux = T(ux0); uy = T(uy0); uz = T(uz0)
    usq = ux * ux + uy * uy + uz * uz
    fs = ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q))
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]
        ranges = chunk_ranges_min(Lv.nocts, nthreads(), 256)
        @threads :static for t in 1:length(ranges)
            @inbounds for o in ranges[t], k in 1:8
                kd = Lv.kind[k, o]
                (kd == K_FLUID || kd == K_GHOST || kd == K_COVERED) || continue
                for i in 1:Q
                    S.f[k, i, o] = fs[i]; S.g[k, i, o] = fs[i]
                end
                S.u[k, 1, o] = ux; S.u[k, 2, o] = uy; S.u[k, 3, o] = uz
                S.v[k, 1, o] = ux; S.v[k, 2, o] = uy; S.v[k, 3, o] = uz
                S.rho[k, o] = rho; S.rhob[k, o] = rho
            end
        end
    end
    nlevels(st.mesh) > 1 && couple_initial!(st)
    st.mass0 = total_mass(st)
    st
end

"Total mass over the fluid leaves of all levels, weighted by cell volume (finest-cell units), in Float64."
function total_mass(st::State)
    m = 0.0; L = nlevels(st.mesh)
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]; w = 8.0^(L - l)
        for o in S.real_octs, k in 1:8
            Lv.kind[k, o] == K_FLUID && (m += w * Float64(S.rho[k, o]))
        end
    end
    m
end

"Total momentum (finest-cell units) over the fluid leaves."
function total_momentum(st::State)
    p = zeros(3); L = nlevels(st.mesh)
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]; w = 8.0^(L - l)
        for o in S.real_octs, k in 1:8
            Lv.kind[k, o] == K_FLUID || continue
            r = Float64(S.rho[k, o])
            for a in 1:3; p[a] += w * r * Float64(S.u[k, a, o]); end
        end
    end
    p
end
