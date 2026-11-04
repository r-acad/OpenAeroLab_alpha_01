# The fused pull-stream and collide step of one level over its real octs (DESIGN.md section 5, step 2).
# For every fluid slot: gather the 19 populations from the neighbour slots by the pull rule; directions flagged in
# the slot's `special` mask are supplied by a domain-face condition (the source is a boundary slot) or by the
# Bouzidi rule (the opposite link is cut); then the subgrid viscosity from the previous step's velocity, the
# collision, and the write of the post-collision populations, velocity and density into the B buffers.
# Momentum exchange of the cut links is accumulated per thread when `forces` is on.

"Gradient of the stored velocity (buffer A) by central differences over the 6 face neighbours; a missing or solid neighbour contributes the slot's own velocity (one-sided)."
@inline function velocity_gradient(Lv::OctLevel, u::Array{T,3}, k::Int, o::Int) where T
    @inline function at(dx, dy, dz)
        ni, nk = neighbour_slot(k, dx, dy, dz)
        no = Lv.onbr[ni, o]
        if no == 0 || Lv.kind[nk, no] == K_BOUNDARY
            return (u[k, 1, o], u[k, 2, o], u[k, 3, o])
        end
        (u[nk, 1, no], u[nk, 2, no], u[nk, 3, no])
    end
    xp = at(1, 0, 0); xm = at(-1, 0, 0); yp = at(0, 1, 0); ym = at(0, -1, 0); zp = at(0, 0, 1); zm = at(0, 0, -1)
    h = T(0.5)
    (h * (xp[1] - xm[1]), h * (xp[2] - xm[2]), h * (xp[3] - xm[3]),
     h * (yp[1] - ym[1]), h * (yp[2] - ym[2]), h * (yp[3] - ym[3]),
     h * (zp[1] - zm[1]), h * (zp[2] - zm[2]), h * (zp[3] - zm[3]))
end

"The open-face forms: equilibrium at the face state plus the bounce-back of the own non-equilibrium (regularised-inlet form)."
@inline function face_dirichlet(cond::Symbol, j::Int, fo::NTuple{Q,T}, rho::T, ux::T, uy::T, uz::T, uin::NTuple{3,T}) where T
    i = OPP[j]
    usq = ux * ux + uy * uy + uz * uz
    if cond == :inlet
        vx, vy, vz = uin; vsq = vx * vx + vy * vy + vz * vz
        return feq_shifted(j, rho, vx, vy, vz, vsq) + (fo[i] - feq_shifted(i, rho, ux, uy, uz, usq))
    elseif cond == :outlet
        return feq_shifted(j, one(T), ux, uy, uz, usq) + (fo[i] - feq_shifted(i, rho, ux, uy, uz, usq))
    else   # :freestream
        vx, vy, vz = uin; vsq = vx * vx + vy * vy + vz * vz
        return feq_shifted(j, one(T), vx, vy, vz, vsq) + (fo[i] - feq_shifted(i, rho, ux, uy, uz, usq))
    end
end

"""
Population j arriving at slot (k, o) from a domain-face boundary slot whose `mask` names the faces it lies beyond,
from the cell's own post-collision populations (all local): inlet, outlet, freestream, slip, symmetry, noslip,
ground. An edge or corner slot touches several faces (DESIGN.md A7): a wall among them bounces back; else an open
face among them imposes the freestream equilibrium (the outlet's own-velocity form fed back on itself at the
outlet's edges and grew a reversed jet there); else, slip and symmetry only, the reflection in every touched axis.
`fo` are the cell's own shifted post-collision populations of the previous step.
"""
@inline function face_population(mask::UInt8, faces::NTuple{6,Symbol}, j::Int, fo::NTuple{Q,T}, rho::T, ux::T, uy::T, uz::T, uin::NTuple{3,T}) where T
    i = OPP[j]
    if count_ones(mask) == 1
        f = trailing_zeros(mask) + 1
        cond = @inbounds faces[f]
        if cond == :noslip || cond == :ground
            return fo[i]                                           # bounce-back
        elseif cond == :slip || cond == :symmetry
            m = f <= 2 ? MIRROR_X[j] : f <= 4 ? MIRROR_Y[j] : MIRROR_Z[j]
            return fo[m]                                           # specular reflection across the face plane
        else
            return face_dirichlet(cond, j, fo, rho, ux, uy, uz, uin)
        end
    end
    wall = false; open = false
    for f in 1:6
        (mask >> (f - 1)) & 0x01 == 0x00 && continue
        c = @inbounds faces[f]
        (c == :noslip || c == :ground) && (wall = true)
        (c == :inlet || c == :outlet || c == :freestream) && (open = true)
    end
    wall && return fo[i]
    open && return face_dirichlet(:freestream, j, fo, rho, ux, uy, uz, uin)
    m = j
    (mask & 0x03) != 0x00 && (m = MIRROR_X[m]); (mask & 0x0c) != 0x00 && (m = MIRROR_Y[m]); (mask & 0x30) != 0x00 && (m = MIRROR_Z[m])
    fo[m]
end

"""
One step of level `l`: read buffers A (f, u, rho), write buffers B (g, v, rhob). `uin` is the inlet velocity
(lattice units) of this step; `collect` turns the per-triangle force accumulation on.
"""
function step_level!(st::State{T}, l::Int; collect::Bool=false, scale::Float64=1.0) where T
    Lv = st.mesh.levels[l]; S = st.levels[l]; cfg = st.cfg
    fin = S.f; fout = S.g; uin_a = S.u; vout = S.v; rho_a = S.rho; rhob = S.rhob
    onbr = Lv.onbr; kind = Lv.kind; special = Lv.special; cutidx = Lv.cutidx; bface = Lv.face
    cut_off = Lv.cut_off; cut_dir = Lv.cut_dir; cut_tri = Lv.cut_tri
    qdir = S.cut_qdir; okdir = S.cut_okdir                      # per cut cell, q and the partner flag per direction
    faces = cfg.mesh.faces
    nu = S.nu; omega0 = S.omega
    sgs = cfg.numerics.sgs; cw = T(cfg.numerics.c_wale); cs = T(cfg.numerics.c_smagorinsky); cv = T(2.5 * cfg.numerics.c_smagorinsky^2)
    sigma = cfg.numerics.collision == :hrr ? T(cfg.numerics.hrr_sigma) : one(T)
    uinlet = (st.u_inlet * st.inlet_dir[1], st.u_inlet * st.inlet_dir[2], st.u_inlet * st.inlet_dir[3])
    usq_inlet = uinlet[1] * uinlet[1] + uinlet[2] * uinlet[2] + uinlet[3] * uinlet[3]
    # a uniform body force given in finest-level lattice units scales with dt_l^2 / dx_l = 2^(L - l) on level l
    Fs = 2.0^(nlevels(st.mesh) - l)
    F = (T(cfg.numerics.body_force[1] * Fs), T(cfg.numerics.body_force[2] * Fs), T(cfg.numerics.body_force[3] * Fs))
    # the outlet sponge (DESIGN.md 7.2, A7): cells within `sponge_thickness` reference lengths of the outlet face blend
    # toward the freestream equilibrium with a quadratic ramp of the configured strength
    sponge_on = cfg.sponge_strength > 0 && cfg.sponge_thickness > 0 && faces[2] == :outlet
    sp_len = cfg.sponge_thickness * cfg.reference_length; x_s = cfg.mesh.domain[1] - sp_len; s_max = T(cfg.sponge_strength)
    origin = Lv.origin; dxl = Lv.dx
    real_octs = S.real_octs; chunks = S.chunks
    wall = cfg.walls.scheme
    tri_acc = st.thread_tri
    @threads :static for t in 1:length(chunks)
        acc = tri_acc[t]
        @inbounds for oi in chunks[t]
            o = Int(real_octs[oi])
            for k in 1:8
                kind[k, o] == K_FLUID || continue
                sp = special[k, o]
                if sp == 0
                    # (the closures carry their own @inbounds: the block's does not reach into them, and a run
                    # without --check-bounds=no would pay a bounds check per gathered population)
                    f = ntuple(j -> (@inbounds fin[SRC_CHILD[k, j], j, onbr[SRC_OFF[k, j], o]]), Val(Q))
                else
                    fo = ntuple(j -> (@inbounds fin[k, j, o]), Val(Q))        # own post-collision populations
                    rho0 = rho_a[k, o]; ux0 = uin_a[k, 1, o]; uy0 = uin_a[k, 2, o]; uz0 = uin_a[k, 3, o]
                    ci = cutidx[k, o]
                    f = ntuple(Val(Q)) do j
                        if (sp >> (j - 1)) & UInt32(1) == 0
                            @inbounds fin[SRC_CHILD[k, j], j, onbr[SRC_OFF[k, j], o]]
                        else
                            so = onbr[SRC_OFF[k, j], o]
                            if so != 0 && kind[SRC_CHILD[k, j], so] == K_BOUNDARY
                                face_population(bface[SRC_CHILD[k, j], so], faces, j, fo, rho0, ux0, uy0, uz0, uinlet)
                            else
                                # the link opposite to j is cut: Bouzidi from the own populations and the partner
                                # (a direction is special for a boundary source or a cut link, so ci != 0 here)
                                i = OPP[j]
                                q = T(qdir[i, ci]); ok = okdir[i, ci]
                                if wall == :bounce_back || !ok
                                    fo[i]
                                elseif q < T(0.5)
                                    pi_, pk_ = neighbour_slot(k, -CX[i], -CY[i], -CZ[i])
                                    po = onbr[pi_, o]
                                    T(2) * q * fo[i] + (one(T) - T(2) * q) * fin[pk_, i, po]
                                else
                                    inv2q = one(T) / (T(2) * q)
                                    inv2q * fo[i] + (one(T) - inv2q) * fo[j]
                                end
                            end
                        end
                    end
                    if collect && ci != 0
                        # momentum exchange on the cut links: (f_i* + f_j(t+1)) c_i per link, i toward the wall
                        for li in cut_off[ci]:cut_off[ci + 1]-1
                            i = Int(cut_dir[li]); j = OPP[i]
                            m = scale * Float64(fo[i] + f[j]); tr = Int(cut_tri[li])
                            if tr > 0
                                acc[1, tr] += m * CX[i]; acc[2, tr] += m * CY[i]; acc[3, tr] += m * CZ[i]
                            else
                                acc[1, 1] += m * CX[i]; acc[2, 1] += m * CY[i]; acc[3, 1] += m * CZ[i]
                            end
                        end
                    end
                end
                # subgrid viscosity and HRR moment from the previous step's velocity
                omega = omega0; pfd = (zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
                if sgs != :none || sigma != one(T)
                    g = velocity_gradient(Lv, uin_a, k, o)
                    nut = sgs == :wale ? nu_wale(g, cw) : sgs == :smagorinsky ? nu_smagorinsky(g, cs) : sgs == :vreman ? nu_vreman(g, cv) : zero(T)
                    omega = omega_of(nu + nut)
                    sigma != one(T) && (pfd = pi_fd_from_gradient(g, rho_a[k, o], one(T) / omega))
                end
                fo2, rho, ux, uy, uz = collide_rr(f, omega, sigma, pfd, F)
                # the sponge blend takes a fresh binding: a reassigned tuple captured by the closure would be boxed
                xs = sponge_on ? (Float64(origin[1, o]) + slot_offset(k)[1] + 0.5) * dxl : -Inf
                fo3 = if xs > x_s
                    s = s_max * T(((xs - x_s) / sp_len)^2)
                    ntuple(i -> (one(T) - s) * fo2[i] + s * feq_shifted(i, one(T), uinlet[1], uinlet[2], uinlet[3], usq_inlet), Val(Q))
                else
                    fo2
                end
                for j in 1:Q; fout[k, j, o] = fo3[j]; end
                vout[k, 1, o] = ux; vout[k, 2, o] = uy; vout[k, 3, o] = uz; rhob[k, o] = rho
            end
        end
    end
    nothing
end

"Swap the A and B buffers of a level after its step."
function swap_buffers!(S::LevelState)
    S.f, S.g = S.g, S.f; S.u, S.v = S.v, S.u; S.rho, S.rhob = S.rhob, S.rho
    S
end

"Reduce the per-thread triangle accumulators into st.tri_force and st.forces (lattice units of the level stepped), then clear them."
function reduce_forces!(st::State)
    fill!(st.tri_force, 0.0)
    for acc in st.thread_tri
        st.tri_force .+= acc; fill!(acc, 0.0)
    end
    st.forces .= vec(sum(st.tri_force; dims=2))
    st
end
