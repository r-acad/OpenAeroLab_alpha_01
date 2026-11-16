# The fused pull-stream and collide step of one level over its real octs (DESIGN.md section 5, step 2).
# For every fluid slot: gather the 19 populations from the neighbour slots by the pull rule; directions flagged in
# the slot's `special` mask are supplied by a domain-face condition (the source is a boundary slot) or by the
# Bouzidi rule (the opposite link is cut); then the subgrid viscosity from the previous step's velocity, the
# collision, and the write of the post-collision populations, velocity and density into the B buffers.
# Momentum exchange of the cut links is accumulated per thread when `forces` is on.
#
# Since 14 Nov 2026 the plain fluid cells of an oct are stepped together, eight lanes of a SIMD vector (kernel.jl);
# cells with a special direction are stepped one by one with the same arithmetic.

"Gradient of the stored velocity (buffer A) by central differences over the 6 face neighbours; a missing or solid neighbour contributes the slot's own velocity (one-sided)."
@inline velocity_gradient(Lv::OctLevel, u::Array{T,3}, k::Int, o::Int) where T = velocity_gradient(Lv.onbr, Lv.kind, u, k, o)
@inline function velocity_gradient(onbr::Matrix{Int32}, kind::Matrix{UInt8}, u::Array{T,3}, k::Int, o::Int) where T
    @inline function at(dx, dy, dz)
        ni, nk = neighbour_slot(k, dx, dy, dz)
        no = onbr[ni, o]
        if no == 0 || kind[nk, no] == K_BOUNDARY
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

"The viscosity of a cell: the wall model's viscosity where the cell has one (`numin` > 0), else the molecular plus subgrid one (the same selection for a scalar and for lanes)."
@inline _cell_viscosity(total, numin) = _select(numin > zero(numin), numin + zero(total), total + zero(numin))

"""
The collision of the eight cells of an oct from their arriving populations: the subgrid viscosity and the HRR
moment from the previous step's velocity (central differences over the face neighbours, as `velocity_gradient`),
then `collide_q19`. `numin` is the wall model's viscosity per lane (zero where none), which replaces the molecular
plus subgrid one in those lanes when the level has a wall model (`walled`). Returns the lanes and (drho, rho, u);
no positivity safeguard.
"""
@inline function collide_lanes(fv::NTuple{Q,Vec{8,T}}, uprev::Vector{T}, rhoprev::Vector{T}, nb::NTuple{27,Int}, o::Int, nu::T, omega0::T,
                               sgs_code::Int, cw::T, cs::T, cv::T, sigma::T, F::NTuple{3,T}, forced::Bool, numin::Vec{8,T}, walled::Bool) where T
    z = zero(Vec{8,T})
    if sgs_code != 0 || sigma != one(T)
        g = gradient_lanes(uprev, nb)
        nut = sgs_code == 1 ? nu_wale(g, cw) : sgs_code == 2 ? nu_smagorinsky(g, cs) : sgs_code == 3 ? nu_vreman(g, cv) : z
        omega = omega_of(walled ? _cell_viscosity(nu + nut, numin) : nu + nut)
        pfd = sigma != one(T) ? pi_fd_from_gradient(g, (@inbounds vload(Vec{8,T}, rhoprev, 1 + 8 * (o - 1))), one(T) / omega) : (z, z, z, z, z, z)
        return collide_q19(fv, omega, sigma, pfd, F, forced)
    end
    collide_q19(fv, walled ? omega_of(_cell_viscosity(nu + z, numin)) : Vec{8,T}(omega0), sigma, (z, z, z, z, z, z), F, forced)
end

"The arrays and settings of one level's step that the one-by-one gather of a cell reads."
struct StepContext{T}
    fin::Array{T,3}
    uprev::Array{T,3}
    rhoprev::Matrix{T}
    onbr::Matrix{Int32}
    kind::Matrix{UInt8}
    special::Matrix{UInt32}
    cutidx::Matrix{Int32}
    bface::Matrix{UInt8}
    cut_off::Vector{Int32}
    cut_dir::Vector{UInt8}
    cut_tri::Vector{Int32}
    qdir::Matrix{Float32}            # per cut cell, q of the cut link per direction
    okdir::Matrix{Bool}              # and whether the Bouzidi partner is a plain fluid slot
    faces::NTuple{6,Symbol}
    uinlet::NTuple{3,T}
    collect::Bool
    scale::Float64
    wm::Bool                         # the wall model (walls.jl)
    pressure_gradient::Bool
    nu::T
    wm_n::Matrix{T}
    wm_y::Vector{T}
    wm_ref::Vector{Int32}
    wm_yref::Vector{T}
    wm_d::Matrix{T}
    wm_us::Matrix{T}
    wm_eps::T
    utau::Vector{T}
    sgs_code::Int                    # the subgrid model (the wall cell keeps at least its viscosity)
    cw::T
    cs::T
    cv::T
end

"""
Gradient of the stored density by central differences over the six face neighbours; a missing neighbour, or one
that carries no fluid state (solid, boundary, unused), contributes the slot's own density (one-sided).
"""
@inline function density_gradient(onbr::Matrix{Int32}, kind::Matrix{UInt8}, rho::Matrix{T}, k::Int, o::Int) where T
    own = @inbounds rho[k, o]
    @inline function at(dx, dy, dz)
        ni, nk = neighbour_slot(k, dx, dy, dz)
        no = @inbounds onbr[ni, o]
        no == 0 && return own
        kd = @inbounds kind[nk, no]
        (kd == K_FLUID || kd == K_GHOST || kd == K_COVERED) ? (@inbounds rho[nk, no]) : own
    end
    h = T(0.5)
    h * (at(1, 0, 0) - at(-1, 0, 0)), h * (at(0, 1, 0) - at(0, -1, 0)), h * (at(0, 0, 1) - at(0, 0, -1))
end

"""
The wall model of cut cell `ci` (slot k of oct o; walls.jl) from the previous step's velocity and density: the slip
velocity of the wall and the viscosity of the cell's collision, both averaged in time; zeros where the cell has no
model. The friction velocity is kept as the next step's starting value.
"""
@inline function wall_slip(c::StepContext{T}, ci::Int, k::Int, o::Int) where T
    z = zero(T)
    (c.wm && ci != 0) || return z, z, z, z
    @inbounds begin
        nx = c.wm_n[1, ci]; ny = c.wm_n[2, ci]; nz = c.wm_n[3, ci]
        (nx == z && ny == z && nz == z) && return z, z, z, z
        r = Int(c.wm_ref[ci])
        rk = r == 0 ? k : slot_k(r); ro = r == 0 ? o : slot_oct(r)
        ux = c.uprev[rk, 1, ro]; uy = c.uprev[rk, 2, ro]; uz = c.uprev[rk, 3, ro]
        un = ux * nx + uy * ny + uz * nz
        tx = ux - un * nx; ty = uy - un * ny; tz = uz - un * nz
        speed = sqrt(tx * tx + ty * ty + tz * tz)
        moving = speed > T(1e-12)
        e = c.wm_eps
        # the adverse part of the pressure gradient along the sampled flow, (1/rho) dp/ds with p = rho / 3, averaged in time
        P = z
        if c.pressure_gradient
            gx, gy, gz = density_gradient(c.onbr, c.kind, c.rhoprev, rk, ro)
            now = moving ? max(z, (tx * gx + ty * gy + tz * gz) / (T(3) * c.rhoprev[rk, ro] * speed)) : z
            P = c.wm_us[5, ci] + e * (now - c.wm_us[5, ci])
            c.wm_us[5, ci] = P
        end
        # the cell's own tangential velocity and its molecular plus subgrid viscosity
        vx = c.uprev[k, 1, o]; vy = c.uprev[k, 2, o]; vz = c.uprev[k, 3, o]
        vn = vx * nx + vy * ny + vz * nz
        cx = vx - vn * nx; cy = vy - vn * ny; cz = vz - vn * nz
        own = sqrt(cx * cx + cy * cy + cz * cz)
        g = c.sgs_code == 0 ? (z, z, z, z, z, z, z, z, z) : velocity_gradient(c.onbr, c.kind, c.uprev, k, o)
        nusgs = c.nu + (c.sgs_code == 1 ? nu_wale(g, c.cw) : c.sgs_code == 2 ? nu_smagorinsky(g, c.cs) : c.sgs_code == 3 ? nu_vreman(g, c.cv) : z)
        ut, ulaw, nunow = moving ? wall_model(speed, c.wm_yref[ci], c.nu, nusgs, c.wm_y[ci], P, c.utau[ci]) : (z, z, nusgs)
        c.utau[ci] = ut
        nubar = c.wm_us[4, ci]
        nubar = nubar > z ? nubar + e * (nunow - nubar) : nunow
        s = own > T(1e-12) ? min(own, wall_slip_speed(ulaw, ut, c.wm_y[ci], nubar)) / own : z
        wx = c.wm_us[1, ci] + e * (s * cx - c.wm_us[1, ci]); wy = c.wm_us[2, ci] + e * (s * cy - c.wm_us[2, ci]); wz = c.wm_us[3, ci] + e * (s * cz - c.wm_us[3, ci])
        c.wm_us[1, ci] = wx; c.wm_us[2, ci] = wy; c.wm_us[3, ci] = wz; c.wm_us[4, ci] = nubar
        # the wall velocity is kept normal to the cell's link sum: the wall term then adds no mass (walls.jl)
        dx = c.wm_d[1, ci]; dy = c.wm_d[2, ci]; dz = c.wm_d[3, ci]; dd = dx * dx + dy * dy + dz * dz
        a = dd > z ? (wx * dx + wy * dy + wz * dz) / dd : z
        return wx - a * dx, wy - a * dy, wz - a * dz, nubar
    end
end

"""
The 19 populations arriving at fluid slot (k, o) by the pull rule, and the wall model's viscosity of the cell
(zero where none). Directions flagged in the slot's `special` mask are supplied by a domain-face condition (the
source is a boundary slot) or by the Bouzidi rule (the opposite link is cut), with the wall term of the slip
velocity where the wall model is on. The momentum exchange of the cut links is added to `acc` when collecting.
"""
@inline function arriving(c::StepContext{T}, k::Int, o::Int, acc::Matrix{Float64}) where T
    fin = c.fin; onbr = c.onbr
    sp = @inbounds c.special[k, o]
    # (the closures carry their own @inbounds: a run without --check-bounds=no would pay a bounds check per
    # gathered population)
    sp == 0 && return ntuple(j -> (@inbounds fin[SRC_CHILD[k, j], j, onbr[SRC_OFF[k, j], o]]), Val(Q)), zero(T)
    fo = ntuple(j -> (@inbounds fin[k, j, o]), Val(Q))        # own post-collision populations
    rho0, ux0, uy0, uz0 = @inbounds (c.rhoprev[k, o], c.uprev[k, 1, o], c.uprev[k, 2, o], c.uprev[k, 3, o])
    ci = Int(@inbounds c.cutidx[k, o])
    wx, wy, wz, numin = wall_slip(c, ci, k, o)
    moving = !(wx == zero(T) && wy == zero(T) && wz == zero(T))
    f = ntuple(Val(Q)) do j
        @inbounds if (sp >> (j - 1)) & UInt32(1) == 0
            fin[SRC_CHILD[k, j], j, onbr[SRC_OFF[k, j], o]]
        else
            so = onbr[SRC_OFF[k, j], o]
            if so != 0 && c.kind[SRC_CHILD[k, j], so] == K_BOUNDARY
                face_population(c.bface[SRC_CHILD[k, j], so], c.faces, j, fo, rho0, ux0, uy0, uz0, c.uinlet)
            else
                # the link opposite to j is cut: Bouzidi from the own populations and the partner
                # (a direction is special for a boundary source or a cut link, so ci != 0 here);
                # a wall moving at u_w returns -6 w_i rho (c_i . u_w) with the reflected population
                i = OPP[j]
                q = T(c.qdir[i, ci]); ok = c.okdir[i, ci]
                wt = moving ? T(6) * T(WT[i]) * rho0 * (T(CX[i]) * wx + T(CY[i]) * wy + T(CZ[i]) * wz) : zero(T)
                if !ok
                    fo[i] - wt
                elseif q < T(0.5)
                    pi_, pk_ = neighbour_slot(k, -CX[i], -CY[i], -CZ[i])
                    po = onbr[pi_, o]
                    T(2) * q * fo[i] + (one(T) - T(2) * q) * fin[pk_, i, po] - wt
                else
                    inv2q = one(T) / (T(2) * q)
                    inv2q * fo[i] + (one(T) - inv2q) * fo[j] - inv2q * wt
                end
            end
        end
    end
    @inbounds if c.collect && ci != 0
        # momentum exchange on the cut links: (f_i* + f_j(t+1)) c_i per link, i toward the wall
        for li in c.cut_off[ci]:c.cut_off[ci + 1]-1
            i = Int(c.cut_dir[li]); j = OPP[i]
            mom = c.scale * Float64(fo[i] + f[j]); tr = Int(c.cut_tri[li])
            col = tr > 0 ? tr : 1
            acc[1, col] += mom * CX[i]; acc[2, col] += mom * CY[i]; acc[3, col] += mom * CZ[i]
        end
    end
    f, numin
end

"The arriving populations of an oct with the lanes `vm` replaced by the scratch lanes, and the wall-model viscosity of those lanes (zero elsewhere)."
@inline function patch_lanes(fv::NTuple{Q,Vec{8,T}}, scr::Vector{T}, vm::UInt8) where T
    vm == 0x00 && return fv, zero(Vec{8,T})
    pm = lane_mask(vm)
    ntuple(j -> vifelse(pm, (@inbounds vload(Vec{8,T}, scr, 8 * (j - 1) + 1)), fv[j]), Val(Q)), vifelse(pm, (@inbounds vload(Vec{8,T}, scr, 8 * Q + 1)), zero(Vec{8,T}))
end

"""
One step of level `l`: read buffers A (f, u, rho), write buffers B (g, v, rhob). `collect` turns the per-triangle
force accumulation on. The cells of an oct are collided together, eight lanes of a vector: the plain fluid cells
pull their populations as vectors; a cell with a special direction (a cut link) has its arriving populations
gathered one by one (`arriving`) and joins the oct's collision. Cells beside a domain face, and every cell of an
oct in which the positivity safeguard is needed, are stepped one by one with the same arithmetic.
"""
function step_level!(st::State{T}, l::Int; collect::Bool=false, scale::Float64=1.0) where T
    Lv = st.mesh.levels[l]; S = st.levels[l]; cfg = st.cfg
    fin = S.f; fout = S.g; uin_a = S.u; vout = S.v; rho_a = S.rho; rhob = S.rhob
    finv = vec(fin); foutv = vec(fout); uinv = vec(uin_a); voutv = vec(vout); rhoav = vec(rho_a); rhobv = vec(rhob)
    onbr = Lv.onbr
    nu = S.nu; omega0 = S.omega
    sgs = cfg.numerics.sgs; cw = T(cfg.numerics.c_wale); cs = T(cfg.numerics.c_smagorinsky); cv = T(2.5 * cfg.numerics.c_smagorinsky^2)
    sgs_code = sgs == :wale ? 1 : sgs == :smagorinsky ? 2 : sgs == :vreman ? 3 : 0
    sigma = T(cfg.numerics.hrr_sigma)
    need_gradient = sgs_code != 0 || sigma != one(T)
    uinlet = (st.u_inlet * st.inlet_dir[1], st.u_inlet * st.inlet_dir[2], st.u_inlet * st.inlet_dir[3])
    usq_inlet = uinlet[1] * uinlet[1] + uinlet[2] * uinlet[2] + uinlet[3] * uinlet[3]
    # a uniform body force given in finest-level lattice units scales with dt_l^2 / dx_l = 2^(L - l) on level l
    Fs = 2.0^(nlevels(st.mesh) - l)
    F = (T(cfg.numerics.body_force[1] * Fs), T(cfg.numerics.body_force[2] * Fs), T(cfg.numerics.body_force[3] * Fs))
    forced = !(F[1] == zero(T) && F[2] == zero(T) && F[3] == zero(T))
    # absorbing layers (state.jl, `sponge_field`): the blend toward the freestream equilibrium of this step
    sponge = S.sponge; spv = vec(sponge); osponge = S.osponge; sponge_on = !isempty(sponge)
    feq_inf = ntuple(i -> feq_shifted(i, one(T), uinlet[1], uinlet[2], uinlet[3], usq_inlet), Val(Q))
    real_octs = S.real_octs; chunks = S.chunks; fmask = S.fmask; smask = S.smask; vmask = S.vmask; scratch = S.scratch
    walled = S.wm_on
    ctx = StepContext{T}(fin, uin_a, rho_a, onbr, Lv.kind, Lv.special, Lv.cutidx, Lv.face, Lv.cut_off, Lv.cut_dir, Lv.cut_tri, S.cut_qdir, S.cut_okdir,
                         cfg.mesh.faces, uinlet, collect, scale,
                         walled, cfg.walls.pressure_gradient, nu, S.wm_n, S.wm_y, S.wm_ref, S.wm_yref, S.wm_d, S.wm_us, T(1 / max(1, cfg.walls.filter_steps)), S.tau_w,
                         sgs_code, cw, cs, cv)
    tri_acc = st.thread_tri
    vector_on = VECTOR_STEP[]
    # whole octs of the large levels are written with streaming stores (no read of the destination line first);
    # the lanes are 32 bytes apart from the start of the buffer, which must itself be aligned
    streaming = STREAMING_STORES[] && length(foutv) >= STREAMING_MIN[] && UInt(pointer(foutv)) % 32 == 0
    @threads :static for t in 1:length(chunks)
        acc = tri_acc[t]; scr = scratch[t]
        @inbounds for oi in chunks[t]
            o = Int(real_octs[oi])
            fm = fmask[oi]; vm = vector_on ? vmask[oi] : 0x00
            vlanes = vector_on ? (fm | vm) : 0x00
            vector_done = false
            if vlanes != 0x00
                nb = neighbour_octs(onbr, o)
                if vm != 0x00
                    for k in 1:8
                        (vm >> (k - 1)) & 0x01 == 0x01 || continue
                        fk, nk = arriving(ctx, k, o, acc)
                        for j in 1:Q; scr[k + 8 * (j - 1)] = fk[j]; end
                        scr[k + 8 * Q] = nk
                    end
                end
                fv, numin = patch_lanes(pull_populations(finv, nb), scr, vm)
                outv, drv, rhov, uxv, uyv, uzv = collide_lanes(fv, uinv, rhoav, nb, o, nu, omega0, sgs_code, cw, cs, cv, sigma, F, forced, numin, walled)
                m = lane_mask(vlanes)
                if !any(below_floor(outv) & m)
                    ib = 152 * (o - 1); i8 = 1 + 8 * (o - 1); i24 = 1 + 24 * (o - 1)
                    if sponge_on && osponge[oi]
                        sv = vload(Vec{8,T}, spv, i8); keepv = one(T) - sv
                        if vlanes == 0xff
                            for j in 1:Q; vstore(keepv * outv[j] + sv * feq_inf[j], foutv, ib + 1 + 8 * (j - 1)); end
                        else
                            for j in 1:Q; vstore(keepv * outv[j] + sv * feq_inf[j], foutv, ib + 1 + 8 * (j - 1), m); end
                        end
                    elseif vlanes == 0xff && streaming
                        for j in 1:Q; vstorent(outv[j], foutv, ib + 1 + 8 * (j - 1)); end
                    elseif vlanes == 0xff
                        for j in 1:Q; vstore(outv[j], foutv, ib + 1 + 8 * (j - 1)); end
                    else
                        for j in 1:Q; vstore(outv[j], foutv, ib + 1 + 8 * (j - 1), m); end
                    end
                    if vlanes == 0xff
                        vstore(uxv, voutv, i24); vstore(uyv, voutv, i24 + 8); vstore(uzv, voutv, i24 + 16); vstore(rhov, rhobv, i8)
                    else
                        vstore(uxv, voutv, i24, m); vstore(uyv, voutv, i24 + 8, m); vstore(uzv, voutv, i24 + 16, m); vstore(rhov, rhobv, i8, m)
                    end
                    vector_done = true
                end
            end
            # the cells stepped one by one: those beside a domain face, and every fluid cell of an oct that needs
            # the safeguard (its special lanes are already gathered in the scratch, with their forces collected)
            lanes = vector_done ? (smask[oi] & ~vm) : (smask[oi] | fm)
            lanes == 0x00 && continue
            for k in 1:8
                (lanes >> (k - 1)) & 0x01 == 0x01 || continue
                gathered = (vm >> (k - 1)) & 0x01 == 0x01
                f, nk = gathered ? (ntuple(j -> (@inbounds scr[k + 8 * (j - 1)]), Val(Q)), scr[k + 8 * Q]) : arriving(ctx, k, o, acc)
                # subgrid viscosity and HRR moment from the previous step's velocity
                gradient = need_gradient ? velocity_gradient(Lv, uin_a, k, o) : (zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
                nut = sgs_code == 1 ? nu_wale(gradient, cw) : sgs_code == 2 ? nu_smagorinsky(gradient, cs) : sgs_code == 3 ? nu_vreman(gradient, cv) : zero(T)
                omega = walled ? omega_of(_cell_viscosity(nu + nut, nk)) : need_gradient ? omega_of(nu + nut) : omega0
                pfd = sigma != one(T) ? pi_fd_from_gradient(gradient, rho_a[k, o], one(T) / omega) : (zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
                fo2, rho, ux, uy, uz = collide_rr(f, omega, sigma, pfd, F)
                s = sponge_on ? sponge[k, o] : zero(T)
                if s != zero(T)
                    for j in 1:Q; fout[k, j, o] = (one(T) - s) * fo2[j] + s * feq_inf[j]; end
                else
                    for j in 1:Q; fout[k, j, o] = fo2[j]; end
                end
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
