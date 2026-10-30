# =====================================================================================================================
# bc.jl -- face boundary conditions (plan section 9). Every boundary face has exactly one condition, held per face in a
# BCSet (its kind and its values u_b, v_b, p_b), so a verification case can set any face to any value (the MMS sets them
# from the exact solution). Stage P0b, 14 Oct 2026.
#
# THE KINDS AND WHAT EACH ONE MEANS IN THE EQUATIONS OF P0b (psi = 0, laminar):
#   velocity Dirichlet: BC_NOSLIP, BC_MOVING_WALL, BC_INLET, BC_FARFIELD -- u_b given. Momentum: the two-point viscous
#       term mu |S|^2/(S.d_b) (u_P - u_b) implicit, its non-orthogonal part deferred; convection upwind (an inflow face is
#       an implicit neighbour of value u_b; an outflow face carries u_P implicitly and the deferred correction to u_b).
#       Pressure: the flux u_b . S is fixed (a Neumann face of the pressure equation). The plan's foil and base (no slip;
#       zero phase flux) and the inlet (velocity U along the channel, pure liquid) are this kind.
#   BC_SLIP (tunnel walls): zero normal flux and zero tangential stress. Momentum: mu |S|^2/(S.d_b) (u_P - u_t) with u_t the
#       tangential projection of u_P, deferred, so at convergence only the normal stress acts. Pressure: Neumann.
#   BC_PRESSURE_OUTLET: p_b given (the plan's p_out = p_v + sigma q); zero-gradient velocity (no viscous flux); backflow
#       enters at the cell's velocity (deferred). Pressure: a Dirichlet face of the pressure equation, its flux from the
#       momentum interpolation of the owner cell (rhie_chow_flux!).
# Pressure at a face without a pressure value (the Green-Gauss gradient): p_P + grad(p)_P . d_b, the least-squares
# gradient taken over the internal faces and the pressure-Dirichlet faces.
# =====================================================================================================================

"The condition kinds."
@enum BCKind::UInt8 BC_NOSLIP = 1 BC_SLIP = 2 BC_INLET = 3 BC_PRESSURE_OUTLET = 4 BC_FARFIELD = 5 BC_MOVING_WALL = 6

"Velocity-Dirichlet kinds (u_b given, flux fixed)."
is_velocity_bc(k::UInt8) = k == UInt8(BC_NOSLIP) || k == UInt8(BC_MOVING_WALL) || k == UInt8(BC_INLET) ||
                           k == UInt8(BC_FARFIELD)
"Pressure-Dirichlet kinds (p_b given)."
is_pressure_bc(k::UInt8) = k == UInt8(BC_PRESSURE_OUTLET)

"""
    default_bc_kind(tag::UInt8) -> BCKind

Plan section 9's condition for a C-mesh boundary tag (box sides have no default: each verification case names its own).
"""
function default_bc_kind(tag::UInt8)
    tag == BT_FOIL && return BC_NOSLIP
    tag == BT_BASE && return BC_NOSLIP
    (tag == BT_BAFFLE_FRONT || tag == BT_BAFFLE_BACK) && return BC_NOSLIP     # an interior wall's two sides (baffle.jl)
    tag == BT_INLET && return BC_INLET
    (tag == BT_LOWER || tag == BT_UPPER) && return BC_SLIP
    tag == BT_FAR && return BC_FARFIELD
    (tag == BT_OUTLET || tag == BT_SOUTLET) && return BC_PRESSURE_OUTLET
    throw(ArgumentError("boundary tag $tag has no default condition (box sides are set by each case)"))
end

"""
    BCSet

Per boundary face, on the backend: `kind` (UInt8 of BCKind), `ub, vb` (velocity values), `pb` (pressure value).
`npdir` counts the pressure-Dirichlet faces (none: the pressure is fixed by a reference cell, see assemble_pressure.jl).
"""
struct BCSet{VU<:AbstractVector{UInt8},VF<:AbstractVector{Float64}}
    kind::VU; ub::VF; vb::VF; pb::VF
    npdir::Int
end

"""
    bcset(mesh; kinds, values=Dict(), backend=CPU()) -> BCSet

`kinds`: boundary tag => BCKind for every tag the mesh has (a missing tag is refused); `values`: tag => (u, v, p) or
tag => f(x, y) -> (u, v, p) at the face centroid (default (0, 0, 0)).
"""
function bcset(m::PBMesh; kinds::AbstractDict, values::AbstractDict=Dict{UInt8,Any}(), backend=CPU())
    bt = _host(m.btag); xf = _host(m.bxf); yf = _host(m.byf)
    k = zeros(UInt8, m.nb); ub = zeros(m.nb); vb = zeros(m.nb); pb = zeros(m.nb)
    for b in 1:m.nb
        haskey(kinds, bt[b]) || throw(ArgumentError("bcset: boundary tag $(bt[b]) has no condition"))
        k[b] = UInt8(kinds[bt[b]])
        v = get(values, bt[b], (0.0, 0.0, 0.0))
        uvp = v isa Function ? v(xf[b], yf[b]) : v
        ub[b], vb[b], pb[b] = Float64.(uvp)
    end
    BCSet(_dev(backend, k), _dev(backend, ub), _dev(backend, vb), _dev(backend, pb), count(is_pressure_bc, k))
end

"""
    bcset_faces(mesh; rule, backend=CPU()) -> BCSet

Face by face (W1, 21 Oct 2026: a side that is partly symmetry and partly wall, as TMR's flat plate): `rule(tag, x, y)`
returns `(kind::BCKind, (u, v, p))` for the boundary face of tag `tag` with centroid (x, y). Host setup, once.
"""
function bcset_faces(m::PBMesh; rule, backend=CPU())
    bt = _host(m.btag); xf = _host(m.bxf); yf = _host(m.byf)
    k = zeros(UInt8, m.nb); ub = zeros(m.nb); vb = zeros(m.nb); pb = zeros(m.nb)
    for b in 1:m.nb
        kd, uvp = rule(bt[b], xf[b], yf[b])
        k[b] = UInt8(kd::BCKind); ub[b], vb[b], pb[b] = Float64.(uvp)
    end
    BCSet(_dev(backend, k), _dev(backend, ub), _dev(backend, vb), _dev(backend, pb), count(is_pressure_bc, k))
end

"""
    channel_direction(mesh) -> (ex, ey)

The channel's axis on a C-mesh, measured from the mesh: the length-weighted mean tangent of the lower tunnel wall's faces,
oriented downstream (+x). No incidence convention is assumed.
"""
function channel_direction(m::PBMesh)
    q = _host(m.slip_lower); isempty(q) && throw(ArgumentError("channel_direction: the mesh has no lower tunnel wall"))
    Sx = _host(m.bSx)[q]; Sy = _host(m.bSy)[q]
    tx = sum(-Sy); ty = sum(Sx)                   # the faces' tangents (S rotated by +90 degrees), length-weighted
    tx < 0 && (tx = -tx; ty = -ty)
    h = hypot(tx, ty); (tx/h, ty/h)
end

"""
    channel_bcset(mesh; U=1.0, p_out=0.0, backend=CPU()) -> BCSet

Plan section 9 on a C-mesh: foil and base no slip, inlet velocity U along the channel (`channel_direction`), tunnel walls
slip, outlets at p_out.
"""
function channel_bcset(m::PBMesh; U::Real=1.0, p_out::Real=0.0, backend=CPU())
    m.family === :cmesh || throw(ArgumentError("channel_bcset needs a C-mesh"))
    ex, ey = channel_direction(m)
    tags = unique(_host(m.btag))
    kinds = Dict(t => default_bc_kind(t) for t in tags)
    vals = Dict{UInt8,Any}(BT_INLET => (U*ex, U*ey, 0.0), BT_OUTLET => (0.0, 0.0, p_out), BT_SOUTLET => (0.0, 0.0, p_out))
    bcset(m; kinds, values=vals, backend)
end

"Set every face's values from an MMS field at time t (u_b, v_b, p_b = the exact solution at the face centroid)."
function set_bc_from_mms!(bc::BCSet, m::PBMesh, f, t::Real; backend=CPU())
    _launch!(_k_bc_mms!, backend, m.nb, bc.ub, bc.vb, bc.pb, m.bxf, m.byf, f, Float64(t))
    pb_sync(backend)
    bc
end
@kernel function _k_bc_mms!(ub, vb, pb, bxf, byf, f, t::Float64)
    b = @index(Global)
    @inbounds begin
        u, v, p = mms_exact(f, bxf[b], byf[b], t)
        ub[b] = u; vb[b] = v; pb[b] = p
    end
end

"""
    apply_bc!(sys, mesh, state, bcs, config)

The boundary conditions enter the assemblers face by face (assemble_momentum!, assemble_pressure!, rhie_chow_flux!),
each reading the BCSet; there is no separate pass. Calling this is an error, so nothing relies on it silently.
"""
apply_bc!(sys, mesh, state, bcs, config) =
    throw(ArgumentError("apply_bc!: the boundary conditions are applied inside the assemblers (see bc.jl); there is no separate pass"))
