# =====================================================================================================================
# freeair.jl -- work item W2 of plan section 32 (Amendment 12, 21 Oct 2026): free air and the NACA 0012 case. OPT-IN: nothing
# here is reached by the tunnel case (PB_DOMAIN=tunnel, the default), and no default of any other file changes.
#
# THE DOMAIN. Version 1's generator, read-only, builds its rectangular channel (the app's free-air meshes: 60 c high, from
# -29.5 c to +30.5 c, the outer domain at 0 degrees so one mesh serves a whole polar; knobs PB_FARFIELD_*). `freeair_blocks`
# re-tags every outer j face of the C block from the channel's kinds (inlet, lower and upper walls) to OUTER_FARFIELD, so
# `build_faces` gives them BT_FAR; the node arrays are not touched (their hashes, and version 1's MESH COORDINATE RECEIPT, are
# the generator's). The downstream ends of the C block (and the strip of a blunt edge) stay the outlets, BT_OUTLET/BT_SOUTLET.
#
# THE CONDITIONS (`freeair_bcset`, bc.jl's kinds, nothing new in the assemblers):
#   foil (and base)        BC_NOSLIP, u_b = 0;
#   far field (BT_FAR)     BC_FARFIELD, velocity Dirichlet at the freestream U (cos alpha, sin alpha), on the inflow and on
#                          the lateral faces alike (bc.jl: an outflow velocity face convects its own u_b, deferred);
#   outlets                BC_PRESSURE_OUTLET at p_inf (zero-gradient velocity, backflow at the cell's velocity).
#   The incidence is the freestream's (PB_ALPHA); the mesh is not rotated.
# THE POINT-VORTEX CORRECTION (optional; Thomas and Salas 1986, AIAA J. 24(7) 1074-1080, incompressible form, beta = 1): the
#   far-field velocity is the freestream plus the velocity of a point vortex at the quarter chord (x_v, y_v) = (0.25, 0) whose
#   circulation, clockwise positive, is the current lift's by Kutta-Joukowski, Gamma = C_L U c / 2:
#       u = U cos(alpha) + Gamma (y - y_v)/(2 pi r^2),    v = U sin(alpha) - Gamma (x - x_v)/(2 pi r^2);
#   the outlet pressure is the same potential flow's, by Bernoulli on the vortex's velocity u' alone,
#       p_b = p_inf - rho U.u' - rho |u'|^2/2    (Gamma = 0 gives p_inf exactly, so the uncorrected far field is bitwise);
#   the circulation is lagged (updated after every PB_VORTEX_EVERY committed steps from the loads of the committed state) and
#   relaxed, Gamma <- Gamma + omega (C_L U c/2 - Gamma). The field u' is divergence-free and its flux through a closed contour
#   around the vortex is zero; the outlet carries whatever the far field's fixed fluxes leave (the mass balance is measured,
#   `boundary_fluxes`). THE STEADY STOP WITH THE VORTEX (review of W2, R4; 22 Oct 2026): a committed state is settled when its
#   UNRELAXED circulation defect |C_L U c/2 - Gamma| (Gamma the far field it was computed under) is at most vortex_tol; the
#   run stops when the state is steady AND settled, and then leaves the far field as it is (no update after the last state),
#   so the final state is steady under its own far field whatever PB_VORTEX_EVERY and PB_VORTEX_RELAX are (before: the
#   relaxed increment was tested, and an update after every step, PB_VORTEX_EVERY=1, could never stop).
# THE LOADS (`airfoil_loads!`), from the solver's own boundary terms on every wall face b (bc.jl, assemble_momentum.jl):
#   pressure      f_p = p_b S_b, p_b the face pressure the Green-Gauss gradient of the momentum equation uses (`_k_bface_p!`:
#                 the owner's value extrapolated with its least-squares pressure gradient; S_b points out of the fluid);
#   viscous       f_v = -(bfx_b - bdg_b u_P, bfy_b - bdg_b v_P): minus the force the momentum's OWN boundary kernel applies
#                 to the fluid cell through the face (`momentum_bface_terms!`, the kernel every assembly launches: laminar
#                 `_k_mom_bface!`, mu |S_b|^2/(S_b.d_b) (u_b - u_P) + nonorth mu k_b.(grad u)_P, limited as the momentum's when
#                 0 < nonorth < 1; SST `_k_mom_bface_t!` with the momentum's face viscosity w.mub, mu_eff = mu at a no-slip
#                 wall since mu_t = 0 there, and the TRANSPOSE term mu_eff (grad u)_P^T . S_b, which the momentum applies on
#                 every velocity face under SST; it vanishes at a wall in the continuum, not in the discrete cell gradient).
#                 The kernel is given a ZERO face flux (LoadsWork.fb0), so the loads are its viscous terms: at an impermeable
#                 wall every committed state has F_b = u_b . S_b = 0 and the force is the momentum's boundary term exactly; a
#                 state set by init_state! carries F_b = u_P . S_b there, and a permeable test face would add a momentum flux,
#                 which is not a load (the suite's linear-field case, W2 (e)).
#                 The integration of 22 Oct 2026 (review of W2, R1): before it the loads re-derived the laminar terms with
#                 cfg.mu and no transpose term, which under SST was not the momentum's boundary term. The same terms give
#                 `wall_shear`'s :assembled C_f (assemble_komega.jl), so the loads, the wall-shear instrument and the momentum
#                 are one definition; testset INT checks the loads against the momentum's assembled boundary force with SST.
#   Their sums give the force (body axes), its wind-axis drag (along the freestream) and lift (normal to it), and the moment
#   about (x_ref, y_ref) = the quarter chord; C_M is NOSE-UP POSITIVE, C_M = -M_z/(q c^2). The gradients and boundary terms
#   are computed into the loads' own work arrays (`LoadsWork`), never the step's, so a march with the loads called after every
#   step is bitwise the march without them (tools/test_pbcav.jl checks it, laminar and SST). FreeStream.rho must equal
#   PBConfig.rho_l (q and the Bernoulli outlet use the one, the momentum the other): refused otherwise (review of W2, R8).
# TURBULENCE (the W1+W2 integration, 22 Oct 2026; review of W2, R2): the free-air run takes PB_TURBULENCE and the SST knobs
#   (`freeair_drive!`: turbulence_settings), and with SST the steady stop also needs the k and omega steady residuals at or
#   below turb_steady_tol (PB_TURB_STEADY_TOL); the driver prints the configuration that ran (PB FREEAIR CONFIG).
# THE MESH AUDIT (`mesh_audit`): a HOST DIAGNOSTIC (as `convection_gain` and `oddeven_operators`), never in a step: the
#   non-orthogonality, skewness, aspect ratio, volume ratio, wall-normal and surface growth, corner angles and the outlet faces'
#   angle, with the definitions of the mesh-quality audit (docs/_records/2026-10-15_mesh_quality/mesh_quality_lib_20261015.jl,
#   main checkout), from version 2's own geometry (true centroids).
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

# ---- the domain -------------------------------------------------------------------------------------------------------
"""
    freeair_blocks(b::CBlocks) -> CBlocks

The same node arrays with every outer j face of the C block tagged OUTER_FARFIELD (BT_FAR after `build_faces`); the label
gains "+freeair". Works on a generated or a subdivided mesh (the kinds of a subdivision are repeated, so the order of the two
operations does not matter).
"""
function freeair_blocks(b::CBlocks)
    CBlocks(string(b.label, "+freeair"), b.NI, b.NJ, b.NIW, b.NIF, b.X, b.Y, fill(OUTER_FARFIELD, length(b.kind)),
            b.NIs, b.NS, b.Xs, b.Ys, b.level, b.hashes)
end

"""
    freeair_mesh(call::CMeshCall; level=0, backend=CPU(), label="pbcav", remap=nothing) -> (gen, blocks, mesh)

Version 1's generator mesh for `call` (a channel of the free-air extents at the outer angle, foil not rotated), the W4 remap
of its columns when `remap` is a RemapSpec (mesh_remap.jl, on the host's node arrays; `nothing`, the default, is the
generator's mesh bit for bit), `level` bilinear subdivisions, the far-field tags, and the faces BUILT ON THE CPU and moved to
`backend` (`to_backend`; review of W3, F6, 22 Oct 2026: building them with the geometry kernels on a device would let FMA
contraction move the metrics by ulps and change the receipts; moved, they are the CPU's bit for bit, as the driver's). On
CPU() the mesh is the CPU build itself. The remapped mesh follows the same rule (the merge of W4/W4b, 22 Oct 2026): the
remap and the faces are both host work, so a remapped mesh on a device is the CPU's remapped mesh bit for bit.
"""
function freeair_mesh(call::CMeshCall; level::Integer=0, backend=CPU(), label::AbstractString="pbcav", remap=nothing)
    gen = generate_v1(call)
    b0 = cblocks(gen; label)
    remap === nothing || (b0 = remap_columns(b0, remap).blocks)
    b = freeair_blocks(subdivide(b0, level))
    (gen, b, to_backend(backend, build_faces(b; backend=CPU())))
end

"""The NACA 0012 section of W2 (sharp trailing edge): apps/airfoil_web/validation/references/naca0012_tmr/naca0012_tmr_sharp.dat
of the main checkout (untracked there), copied byte for byte under a new name, because tools/naca0012_tmr_sharp.dat is version
1's NA0 section (tools/naca0012_tmr_section.jl, 401 points per surface) and stays untouched. This file is the original NACA
0012 equation extended to x = 1.008930411365, where it closes, and scaled to unit chord, 1,201 points. THE SAME CURVE AS
VERSION 1'S FILE (review of W2, R5; checked again in the integration, testset INT): TMR's modified-coefficient formula IS that
extension scaled to unit chord, both files' points lie on it to about 1e-11, and both have t/c 0.118972; the "t/c 0.11894" of
this file's header line is a wrong number in a header. The choice between the files is a choice of point count only."""
const NACA0012_TMR_FOIL = "tools/naca0012_tmr_sharp_app1201.dat"
"Its SHA-256 (the source file's, recorded when it was copied, 21 Oct 2026)."
const NACA0012_TMR_SHA256 = "fb333bf10a5e6528857659922f6d6075e8604a007d7d0086f7799049bc956cb0"

"""
    freeair_call(; foil=NACA0012_TMR_FOIL, n_foil, n_normal, dn1, growth=1.20, smooth=80, n_wake, le_ds=0.0, te=nothing,
                 H=60.0, xmin=-29.5, xmax=30.5, angle=0.0) -> CMeshCall

The generator call of a free-air mesh (the app's extents by default; a sharp edge: `te = nothing`).
"""
freeair_call(; foil::AbstractString=NACA0012_TMR_FOIL, n_foil::Integer, n_normal::Integer, dn1::Real, growth::Real=1.20,
             smooth::Integer=80, n_wake::Integer, le_ds::Real=0.0, te=nothing, H::Real=60.0, xmin::Real=-29.5,
             xmax::Real=30.5, angle::Real=0.0) =
    CMeshCall(; foil=String(foil), n_foil=Int(n_foil), n_normal=Int(n_normal), dn1=Float64(dn1), growth=Float64(growth),
              smooth=Int(smooth), n_wake=Int(n_wake), le_ds=Float64(le_ds), te, H=Float64(H), xin=Float64(xmin),
              xout=Float64(xmax), alpha=Float64(angle))

# ---- the freestream and the far-field conditions ---------------------------------------------------------------------
"""
    FreeStream(; U=1.0, alpha=0.0, p_inf=0.0, rho=1.0, chord=1.0, xref=0.25, yref=0.0)

The freestream (speed U at `alpha` degrees to the foil's x axis), the reference pressure and density, the chord, and the
reference point of the moment and of the point vortex (the quarter chord of a unit chord from x = 0; one point serves both,
review of W2 R8 c). `rho` must equal the configuration's rho_l (`airfoil_loads!` and `freeair_run!` refuse otherwise: q and
the Bernoulli outlet use this one, the momentum rho_l).
"""
Base.@kwdef struct FreeStream
    U::Float64 = 1.0; alpha::Float64 = 0.0; p_inf::Float64 = 0.0; rho::Float64 = 1.0; chord::Float64 = 1.0
    xref::Float64 = 0.25; yref::Float64 = 0.0
end
"The freestream velocity components."
uinf(fs::FreeStream) = (fs.U*cosd(fs.alpha), fs.U*sind(fs.alpha))
"The dynamic pressure rho U^2 / 2."
qinf(fs::FreeStream) = 0.5*fs.rho*fs.U^2

"""
    point_vortex_velocity(x, y, G, xv, yv) -> (u', v')

The velocity of a point vortex of circulation G, CLOCKWISE positive (a lifting foil in a flow from left to right has G > 0),
at (xv, yv): u' = G (y - yv)/(2 pi r^2), v' = -G (x - xv)/(2 pi r^2). Zero at G = 0 and r > 0. At r = 0 it is NaN (0/0 or
Inf*0, any G), which `set_farfield!` refuses (a far-field or outlet face centroid on the vortex; review of W2, R8 b).
"""
@inline function point_vortex_velocity(x, y, G, xv, yv)
    dx = x - xv; dy = y - yv; r2 = dx*dx + dy*dy
    c = G/(2pi*r2)
    (c*dy, -c*dx)
end

@kernel function _k_farfield_values!(ub, vb, pb, bkind, bxf, byf, Ux::Float64, Uy::Float64, pinf::Float64, rho::Float64,
                                     G::Float64, xv::Float64, yv::Float64)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]
        if k == UInt8(BC_FARFIELD)
            du, dv = point_vortex_velocity(bxf[b], byf[b], G, xv, yv)
            ub[b] = Ux + du; vb[b] = Uy + dv
        elseif k == UInt8(BC_PRESSURE_OUTLET)
            du, dv = point_vortex_velocity(bxf[b], byf[b], G, xv, yv)
            pb[b] = pinf - rho*(Ux*du + Uy*dv) - 0.5*rho*(du*du + dv*dv)
        end
    end
end

"""
    set_farfield!(bc, mesh, fs, G; backend=CPU()) -> bc

The far-field faces' velocity (freestream plus the point vortex of circulation G at fs.xref, fs.yref) and the outlets'
pressure (p_inf plus the vortex's Bernoulli term), in place on the backend; G = 0 is the uncorrected far field, bitwise.
Refuses (ArgumentError) a non-finite G, and a far-field or outlet face centroid at the vortex (r = 0, where the values are
NaN; three device reductions, no host loop).
"""
function set_farfield!(bc::BCSet, m::PBMesh, fs::FreeStream, G::Real; backend=CPU())
    isfinite(G) || throw(ArgumentError("set_farfield!: the circulation must be finite (got $G)"))
    Ux, Uy = uinf(fs)
    _launch!(_k_farfield_values!, backend, m.nb, bc.ub, bc.vb, bc.pb, bc.kind, m.bxf, m.byf, Ux, Uy, fs.p_inf, fs.rho,
             Float64(G), fs.xref, fs.yref)
    pb_sync(backend)
    all(x -> isfinite(pb_maxabs(x; backend)), (bc.ub, bc.vb, bc.pb)) || throw(ArgumentError("set_farfield!: a far-field or " *
        "outlet face centroid lies on the point vortex at ($(fs.xref), $(fs.yref)) (r = 0, where its velocity is undefined), " *
        "or the freestream is not finite"))
    bc
end

"""
    freeair_bcset(mesh, fs::FreeStream; G=0.0, backend=CPU()) -> BCSet

The free-air conditions (header): foil and base no slip, BT_FAR far field (velocity Dirichlet), outlets pressure at p_inf;
with G != 0 the point-vortex values. Refuses a mesh with tunnel tags (inlet or slip walls): build it with `freeair_blocks`.
"""
function freeair_bcset(m::PBMesh, fs::FreeStream; G::Real=0.0, backend=CPU())
    m.family === :cmesh || throw(ArgumentError("freeair_bcset needs a C-mesh"))
    tags = unique(_host(m.btag))
    bad = intersect(tags, (BT_INLET, BT_LOWER, BT_UPPER))
    isempty(bad) || throw(ArgumentError("freeair_bcset: the mesh has tunnel tags $(Int.(bad)) (inlet / slip walls); a free-air " *
                                        "mesh re-tags its outer faces far field (freeair_blocks)"))
    BT_FAR in tags || throw(ArgumentError("freeair_bcset: the mesh has no far-field faces"))
    kinds = Dict(t => (t == BT_FOIL || t == BT_BASE || t == BT_BAFFLE_FRONT || t == BT_BAFFLE_BACK ? BC_NOSLIP :
                       t == BT_FAR ? BC_FARFIELD :
                       (t == BT_OUTLET || t == BT_SOUTLET) ? BC_PRESSURE_OUTLET :
                       throw(ArgumentError("freeair_bcset: boundary tag $t has no free-air condition"))) for t in tags)
    bc = bcset(m; kinds, backend)
    set_farfield!(bc, m, fs, G; backend)
end

"""
    PointVortex(; relax=0.5, every=10, G=0.0)

The point-vortex correction's state: the circulation `G` now in the far field, its relaxation factor and update period (in
committed steps), and the history of its updates (step, G).
"""
Base.@kwdef mutable struct PointVortex
    relax::Float64 = 0.5
    every::Int = 10
    G::Float64 = 0.0
    hist::Vector{Tuple{Int,Float64}} = Tuple{Int,Float64}[]
end

"""
    vortex_update!(pv, CL, fs, bc, mesh, nstep; backend=CPU()) -> Bool

After committed step `nstep`: when it is a multiple of pv.every, G <- G + relax (C_L U c / 2 - G), the far field reset
(`set_farfield!`), and the update recorded; returns whether it updated.
"""
function vortex_update!(pv::PointVortex, CL::Real, fs::FreeStream, bc::BCSet, m::PBMesh, nstep::Integer; backend=CPU())
    nstep % pv.every == 0 || return false
    target = 0.5*CL*fs.U*fs.chord
    pv.G += pv.relax*(target - pv.G)
    set_farfield!(bc, m, fs, pv.G; backend)
    push!(pv.hist, (Int(nstep), pv.G))
    true
end

# ---- the lifting potential start (W6, 22 Oct 2026; records air/w6_start/) ---------------------------------------------
"""
    vortex_projection_start!(st, mesh, bc, cfg, fs, G; work) -> NamedTuple

The projection start (`project_start!`) from the freestream PLUS the velocity of a point vortex of circulation G at fs.xref,
fs.yref (clockwise positive, `point_vortex_velocity`; the quarter chord, inside the foil), with the far-field faces carrying the
same vortex (`set_farfield!`, so bc is left at G) and, under SST, k and omega at the inflow values (`init_turbulence!`). The
vortex field is divergence-free and irrotational outside its centre, so the projected flow is the potential flow of circulation
G about the foil. G = 0 is the plain start of `freeair_drive!` (the same calls in the same order, bit for bit).
"""
function vortex_projection_start!(st::PBState, m::PBMesh, bc::BCSet, cfg, fs::FreeStream, G::Real; work, prestart=nothing)
    be = cfg.backend; Ux, Uy = uinf(fs); Gf = Float64(G)
    set_farfield!(bc, m, fs, Gf; backend=be)
    init_state!(st, m, (x, y) -> (d = point_vortex_velocity(x, y, Gf, fs.xref, fs.yref); (Ux + d[1], Uy + d[2], fs.p_inf)); backend=be)
    cfg.turbulence === :sst && init_turbulence!(st, m, bc, cfg; work)
    prestart === nothing || prestart(st)          # a diagnostic hook (the parity tool's one-ulp draws, W6b); none by default
    project_start!(st, m, bc, cfg; work)
end

"""
    kutta_circulation(mesh, bc, cfg, fs; work, nk=4) -> (G, jump0, jumpG)

THE KUTTA CIRCULATION OF THE LIFTING POTENTIAL START (W6): the projection is linear in G, u(G) = u0 + G u1, so two projection
starts (G = 0 and G = 1, `vortex_projection_start!`, into a scratch state) give u0 and u1, and G is the least-squares zero of the
velocity jump across the first `nk` cut faces behind the sharp trailing edge (C cells (NIW+1-n, 1) and (NIW+NIF+n, 1), n = 1..nk):
the flow then leaves the trailing edge smoothly instead of wrapping round it, as the circulation-free start does at incidence.
Returns G and the largest jump at G = 0 and at G (U units). A sharp-edged C-mesh only (refused with a base: the strip's cut is
not built here). bc is left at G = 1; the caller sets the far field it wants. A host read of 4 nk velocities.
"""
function kutta_circulation(m::PBMesh, bc::BCSet, cfg, fs::FreeStream; work, nk::Integer=4, prestart=nothing)
    m.family === :cmesh || throw(ArgumentError("kutta_circulation needs a C-mesh"))
    isempty(_host(m.base)) || throw(ArgumentError("kutta_circulation: a sharp trailing edge only (this mesh has a base)"))
    isempty(_host(m.baffle)) || throw(ArgumentError("kutta_circulation: a mesh without a baffle only (a baffle such as a Gurney " *
        "plate moves the point the flow leaves from; the Kutta jump across the cut behind the trailing edge is not its condition)"))
    (1 <= nk <= m.NIW) || throw(ArgumentError("kutta_circulation: 1 <= nk <= NIW required (got $nk)"))
    cell(i, j) = (i - 1)*m.NJ + j
    pr = [(cell(m.NIW + m.NIF + n, 1), cell(m.NIW + 1 - n, 1)) for n in 1:nk]   # upper and lower cell of the n-th cut face
    s = allocate_state(m; backend=cfg.backend)
    take() = (u = _host(lev(s.ux, 1)); v = _host(lev(s.uy, 1)); ([u[a] - u[b] for (a, b) in pr], [v[a] - v[b] for (a, b) in pr]))
    vortex_projection_start!(s, m, bc, cfg, fs, 0.0; work, prestart); du0, dv0 = take()
    vortex_projection_start!(s, m, bc, cfg, fs, 1.0; work, prestart); du1, dv1 = take(); du1 .-= du0; dv1 .-= dv0
    G = -(sum(du0 .* du1) + sum(dv0 .* dv1))/(sum(abs2, du1) + sum(abs2, dv1))
    jump(g) = maximum(hypot.(du0 .+ g .* du1, dv0 .+ g .* dv1))
    (G, jump(0.0), jump(G))
end

# ---- the loads --------------------------------------------------------------------------------------------------------
"""
    LoadsWork

The loads' own work arrays (never the step's): velocity and pressure gradients, the boundary face velocities and pressures,
the momentum's boundary terms of every boundary face (bdg, bfx, bfy: `momentum_bface_terms!`, given the zero face flux fb0,
never written), and per wall face (the foil list, then the base list) its pressure and viscous force on the body and their
moments.
"""
struct LoadsWork{V<:AbstractVector{Float64},VI<:AbstractVector{Int32}}
    gux::V; guy::V; gvx::V; gvy::V; ubf::V; vbf::V; gpx::V; gpy::V; ggx::V; ggy::V; pbf::V
    walls::VI; nwall::Int
    fpx::V; fpy::V; fvx::V; fvy::V; mp::V; mv::V
    farfield::VI; outlets::VI; fscr::V
    bdg::V; bfx::V; bfy::V; fb0::V
end
"""
    LoadsWork(mesh; backend=CPU(), walls=nothing)

The loads' work arrays; `walls` (host vector of boundary-face indices) defaults to the foil list, the base list and the baffle
list (a C-mesh; the baffle list, both sides of every plate, is empty without a baffle, so the default is then the one before);
a box case names its own wall faces (the suite's Couette check).
"""
function LoadsWork(m::PBMesh; backend=CPU(), walls=nothing)
    z(n) = KernelAbstractions.zeros(backend, Float64, max(n, 1))
    wl = walls === nothing ? vcat(_host(m.foil), _host(m.base), _host(m.baffle)) : Int32.(collect(walls))
    walls = _dev(backend, wl); nw = length(wl)
    LoadsWork(z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nb), z(m.nb), z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nb),
              walls, nw, z(nw), z(nw), z(nw), z(nw), z(nw), z(nw), m.farfield, m.outlets, z(m.nb), z(m.nb), z(m.nb), z(m.nb),
              z(m.nb))
end

"""The wall faces' loads on the body (header): the pressure force p_b S_b and the viscous force, minus the momentum's own
boundary force on the fluid (bdg, bfx, bfy of `momentum_bface_terms!` at the same velocity: F_fluid = (bfx - bdg u_P,
bfy - bdg v_P)), and their moments about (xr, yr)."""
@kernel function _k_wall_loads!(fpx, fpy, fvx, fvy, mp, mv, walls, pbf, u, v, bdg, bfx, bfy, bown, bSx, bSy, bxf, byf,
                                xr::Float64, yr::Float64)
    q = @index(Global)
    @inbounds begin
        b = walls[q]; P = bown[b]
        px = pbf[b]*bSx[b]; py = pbf[b]*bSy[b]
        vx = -(bfx[b] - bdg[b]*u[P]); vy = -(bfy[b] - bdg[b]*v[P])
        rx = bxf[b] - xr; ry = byf[b] - yr
        fpx[q] = px; fpy[q] = py; fvx[q] = vx; fvy[q] = vy
        mp[q] = rx*py - ry*px; mv[q] = rx*vy - ry*vx
    end
end

@kernel function _k_gather!(out, x, idx)
    i = @index(Global)
    @inbounds out[i] = x[idx[i]]
end

"""
    loads_gradients!(lw, st, mesh, bc, w, cfg)

The velocity's least-squares gradients (the step's velocity stencil and weights) and the pressure's least-squares and
Green-Gauss gradients with the boundary face pressures, of the state's level 1, into the loads' arrays.
"""
function loads_gradients!(lw::LoadsWork, st::PBState, m::PBMesh, bc, w, cfg)
    be = cfg.backend; u = lev(st.ux, 1); v = lev(st.uy, 1); G = w.lsq
    _launch!(_k_bvel!, be, m.nb, lw.ubf, lw.vbf, u, v, bc.kind, bc.ub, bc.vb, m.bown, m.bSx, m.bSy, m.bSmag)
    for (g1, g2, phi, phib) in ((lw.gux, lw.guy, u, lw.ubf), (lw.gvx, lw.gvy, v, lw.vbf))
        _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, phib, G.u11, G.u12, G.u22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), Int32(0), Int32(0), Int32(0))
    end
    pb_sync(be)
    pressure_gradients!(lw.gpx, lw.gpy, lw.ggx, lw.ggy, lw.pbf, lev(st.p, 1), w, m, bc, cfg)
    nothing
end

_lsum(x, n, be) = n == 0 ? 0.0 : pb_sum(view(x, 1:n); backend=be)

"""
    airfoil_loads!(lw, st, mesh, bc, w, cfg, fs) -> NamedTuple

The loads of the state's level 1 (header): CL, CD (with its pressure part CDp and friction part CDf), CLp, CLf, CM (nose-up
positive about fs.xref, fs.yref), the body-axis coefficients Cx, Cy, and the dimensional force (Fx, Fy) and moment Mz
(counter-clockwise positive). The viscous part is minus the momentum's own boundary force (`momentum_bface_terms!`; under SST
with the momentum's face viscosity w.mub and the transpose term). Refuses fs.rho != cfg.rho_l. Device sums (deterministic,
two-pass), a few scalars read back.
"""
function airfoil_loads!(lw::LoadsWork, st::PBState, m::PBMesh, bc, w, cfg, fs::FreeStream)
    be = cfg.backend
    fs.rho == cfg.rho_l || throw(ArgumentError("airfoil_loads!: FreeStream.rho = $(fs.rho) differs from PBConfig.rho_l = " *
        "$(cfg.rho_l) (q and the outlet's Bernoulli pressure use the one, the momentum the other; review of W2, R8)"))
    loads_gradients!(lw, st, m, bc, w, cfg)
    u = lev(st.ux, 1); v = lev(st.uy, 1)
    momentum_bface_terms!(lw.bdg, lw.bfx, lw.bfy, m, bc, cfg, u, v, lw.fb0, lw.ubf, lw.vbf, lw.gux, lw.guy, lw.gvx,
                          lw.gvy; mub=cfg.turbulence === :sst ? w.mub : nothing)      # the viscous terms (header: zero flux)
    if lw.nwall > 0
        _launch!(_k_wall_loads!, be, lw.nwall, lw.fpx, lw.fpy, lw.fvx, lw.fvy, lw.mp, lw.mv, lw.walls, lw.pbf, u, v, lw.bdg,
                 lw.bfx, lw.bfy, m.bown, m.bSx, m.bSy, m.bxf, m.byf, fs.xref, fs.yref)
        pb_sync(be)
    end
    n = lw.nwall
    Fpx = _lsum(lw.fpx, n, be); Fpy = _lsum(lw.fpy, n, be); Fvx = _lsum(lw.fvx, n, be); Fvy = _lsum(lw.fvy, n, be)
    Mp = _lsum(lw.mp, n, be); Mv = _lsum(lw.mv, n, be)
    qc = qinf(fs)*fs.chord; ca = cosd(fs.alpha); sa = sind(fs.alpha)
    drag(fx, fy) = (fx*ca + fy*sa)/qc
    lift(fx, fy) = (-fx*sa + fy*ca)/qc
    (CL=lift(Fpx + Fvx, Fpy + Fvy), CD=drag(Fpx + Fvx, Fpy + Fvy), CLp=lift(Fpx, Fpy), CLf=lift(Fvx, Fvy),
     CDp=drag(Fpx, Fpy), CDf=drag(Fvx, Fvy), CM=-(Mp + Mv)/(qc*fs.chord), Cx=(Fpx + Fvx)/qc, Cy=(Fpy + Fvy)/qc,
     Fx=Fpx + Fvx, Fy=Fpy + Fvy, Mz=Mp + Mv)
end

"""
    surface_distribution(lw, mesh, fs) -> NamedTuple of host vectors

After `airfoil_loads!`: per wall face (the foil from the lower trailing edge round the leading edge to the upper trailing
edge, then the base, then the baffle faces when there are any, tagged BT_BAFFLE_FRONT or BT_BAFFLE_BACK: `baffle_table` in
baffle.jl gives them with their plate, side and row): the face centroid (x, y), the face length ds, C_p = (p_b - p_inf)/q, the viscous force on the body per
unit length over q (cfx, cfy), and C_f, its component along the face from its first node to its second (the generator's wrap
order: forward along the lower surface, aft along the upper), so C_f is positive where the flow runs that way. A one-time
host read, for the output files.
"""
function surface_distribution(lw::LoadsWork, m::PBMesh, fs::FreeStream)
    n = lw.nwall; wq = _host(lw.walls)
    xf = _host(m.bxf)[wq]; yf = _host(m.byf)[wq]; Sx = _host(m.bSx)[wq]; Sy = _host(m.bSy)[wq]; ds = hypot.(Sx, Sy)
    q = qinf(fs)
    pb = _host(lw.pbf)[wq]
    fvx = _host(lw.fvx)[1:n]; fvy = _host(lw.fvy)[1:n]
    tx = -Sy ./ ds; ty = Sx ./ ds                           # the face's direction from node a to node b (S rotated +90 deg)
    cf = (fvx .* tx .+ fvy .* ty) ./ (q .* ds)
    (face=Int.(wq), tag=Int.(_host(m.btag)[wq]), x=xf, y=yf, ds=ds, Cp=(pb .- fs.p_inf) ./ q, cfx=fvx ./ (q .* ds),
     cfy=fvy ./ (q .* ds), Cf=cf)
end

"""
    boundary_fluxes(lw, st, mesh; backend=CPU()) -> (far, outlet, net)

The volumetric flux of the state's level 1 out of the domain through the far-field faces, through the outlets, and their
sum with the walls' (zero by construction): `net` is the discrete mass imbalance of the boundary, rho_l net per unit time.
"""
function boundary_fluxes(lw::LoadsWork, st::PBState, m::PBMesh; backend=CPU())
    Fb = lev(st.Fb, 1)
    nfar = length(lw.farfield); nout = length(lw.outlets)
    nfar > 0 && _launch!(_k_gather!, backend, nfar, lw.fscr, Fb, lw.farfield)
    pb_sync(backend)
    far = _lsum(lw.fscr, nfar, backend)
    nout > 0 && _launch!(_k_gather!, backend, nout, lw.fscr, Fb, lw.outlets)
    pb_sync(backend)
    out = _lsum(lw.fscr, nout, backend)
    (far=far, outlet=out, net=pb_sum(Fb; backend))
end

# ---- the run -------------------------------------------------------------------------------------------------------------
"""The columns of the load history (one row per committed step), written by `freeair_run!`. Gamma is the far field's
circulation the state was computed under; Gamma_defect = |C_L U c/2 - Gamma| (0 without the vortex); steady_k, steady_w the
k and omega steady residuals under SST (NaN laminar). The last three columns are the integration's (22 Oct 2026)."""
const LOADS_HISTORY_HEADER = "step,t,dt,outer,linear_its,steady,CL,CD,CDp,CDf,CM,Gamma,flux_far,flux_outlet,flux_net,mass_step_err_max,Gamma_defect,steady_k,steady_w"

"""
    freeair_run!(st, mesh, bc, cfg, fs; wall_cap, vortex=nothing, max_steps=typemax(Int), t_end=Inf, steady_tol=0.0,
                 vortex_tol=1e-6, turb_steady_tol=nothing, history=nothing, work=nothing, ledger=PBLedger(), on_step=nothing,
                 max_halvings=10) -> NamedTuple

One free-air run: `pbcav_run!` (one march, its outcomes and wall cap) with, after every committed step, the loads of the
committed state, the boundary fluxes, the load history (`history`, an IO: one CSV row per step, LOADS_HISTORY_HEADER), and,
with `vortex::PointVortex`, the circulation update (`vortex_update!`).
THE STEADY STOP (pbcav_march!'s, the momentum's steady residual at or below steady_tol) also needs (the march's `steady_ok`):
  * with a vortex, the state SETTLED: its unrelaxed circulation defect |C_L U c/2 - Gamma|, Gamma the far field the state was
    computed under, at most vortex_tol (U c units; review of W2, R4). Once the state is steady and settled the far field is
    not updated again, so the final state is steady under its own far field, at any PB_VORTEX_EVERY (1 included) and any
    relaxation factor;
  * with SST (cfg.turbulence = :sst), the k and omega steady residuals (`turbulence_steady_residual`) at most the registered
    turb_steady_tol: `pbcav_march!`'s own SST rule (re-review of W1b, R1; refused when it is not given, or when it lies at
    or below the k-omega line solve's goal floor, `_kw_goal_floor`: 0.1 turb_tol under turb_coupling = :outer, 0.1 min(turb_tol, turb_step_tol) under :step; corrected wording of W6, the review of the
    integration's F3).
Without a vortex and laminar the stop is pbcav_run!'s, bit for bit. Refuses fs.rho != cfg.rho_l. Returns pbcav_run!'s fields
and `loads` (the final loads), `fluxes`, `hist` (the per-step records), `vortex` and the loads' work (for
`surface_distribution`).
"""
function freeair_run!(st::PBState, m::PBMesh, bc::BCSet, cfg, fs::FreeStream; wall_cap::Real, vortex=nothing,
                      max_steps::Integer=typemax(Int), t_end::Real=Inf, steady_tol::Real=0.0, vortex_tol::Real=1e-6,
                      turb_steady_tol=nothing, history=nothing, work=nothing, ledger::PBLedger=PBLedger(), on_step=nothing,
                      max_halvings::Integer=10, ramp=nothing)
    fs.rho == cfg.rho_l || throw(ArgumentError("freeair_run!: FreeStream.rho = $(fs.rho) differs from PBConfig.rho_l = " *
        "$(cfg.rho_l) (review of W2, R8)"))
    ramp === nothing || check_start_ramp(ramp)
    ramping(t) = ramp !== nothing && t < ramp.T     # the ramp start (24 Oct 2026): no vortex update and no steady stop while on
    vortex === nothing || (vortex.every >= 1 && 0 < vortex.relax <= 1) ||
        throw(ArgumentError("freeair_run!: the point vortex needs every >= 1 and 0 < relax <= 1"))
    vortex_tol >= 0 || throw(ArgumentError("freeair_run!: vortex_tol must be >= 0"))
    cfg.turbulence === :sst && steady_tol > 0 && check_turb_steady_tol(cfg, turb_steady_tol)
    be = cfg.backend
    w = work === nothing ? pbwork(m, bc, cfg) : work
    lw = LoadsWork(m; backend=be)
    recs = NamedTuple[]; settled = Ref(true)
    history === nothing || println(history, LOADS_HISTORY_HEADER)
    function cb(s, rec, led)
        L = airfoil_loads!(lw, s, m, bc, w, cfg, fs); fl = boundary_fluxes(lw, s, m; backend=be)
        G0 = vortex === nothing ? 0.0 : vortex.G
        defect = vortex === nothing ? 0.0 : abs(0.5*L.CL*fs.U*fs.chord - G0)
        sk = get(rec, :steady_k, NaN); sw = get(rec, :steady_w, NaN)      # the march's (SST with a steady stop)
        settled[] = defect <= vortex_tol
        tsteady = !haskey(rec, :steady_k) || (sk <= turb_steady_tol && sw <= turb_steady_tol)
        stopping = steady_tol > 0 && rec.steady <= steady_tol && settled[] && tsteady && !ramping(s.t)
        # the far field is not moved under a state the march is about to stop on (R4: the final state is steady under it)
        vortex === nothing || stopping || ramping(s.t) || vortex_update!(vortex, L.CL, fs, bc, m, s.nstep; backend=be)
        r = (step=s.nstep, t=s.t, dt=s.dt_prev, outer=rec.outer, linear_its=sum(rec.fcg; init=0), steady=rec.steady,
             L..., Gamma=G0, flux_far=fl.far, flux_outlet=fl.outlet, flux_net=fl.net, mass_step_err_max=led.step_err_max,
             Gamma_defect=defect, steady_k=sk, steady_w=sw)
        push!(recs, r)
        history === nothing || (println(history, join((r.step, r.t, r.dt, r.outer, r.linear_its, r.steady, r.CL, r.CD, r.CDp,
                                                        r.CDf, r.CM, r.Gamma, r.flux_far, r.flux_outlet, r.flux_net,
                                                        r.mass_step_err_max, r.Gamma_defect, r.steady_k, r.steady_w), ","));
                                flush(history))
        on_step === nothing || on_step(s, r, led)
    end
    ok = ramp !== nothing ? (s -> !ramping(s.t) && (vortex === nothing || settled[])) : vortex === nothing ? nothing : (s -> settled[])
    # the ramp start: before every attempt the far field at the attempt's time (only where the factor changes: after the ramp it
    # is set once at the full freestream, with the vortex's G, and the vortex's own updates take over)
    rlast = Ref(NaN)
    pre = ramp === nothing ? nothing : function (s)
        rf = start_ramp_factor(s.t + s.dt, ramp)
        rf == rlast[] && return nothing
        set_farfield!(bc, m, scaled_freestream(fs, rf), vortex === nothing ? 0.0 : vortex.G; backend=be); rlast[] = rf
        nothing
    end
    r = pbcav_run!(st, m, bc, cfg; wall_cap, t_end, steady_tol, max_steps, work=w, ledger, on_step=cb, max_halvings,
                   steady_ok=ok, turb_steady_tol, pre_step=pre)
    L = airfoil_loads!(lw, st, m, bc, w, cfg, fs); fl = boundary_fluxes(lw, st, m; backend=be)
    merge(r, (loads=L, fluxes=fl, hist=recs, vortex=vortex, lw=lw))
end

# ---- the ramp start (24 Oct 2026, records air/baffle/; opt-in, PB_START=ramp) ------------------------------------------
"""THE RAMP START (PB_START=ramp, PB_START_RAMP=T,r0; opt-in, measured 24 Oct 2026 on the free-air C+H mesh with and without
a Gurney plate, records air/baffle/README.md). The plain projection start is the impulsively started flow: its potential flow
is singular at every sharp corner (the base corners of a finite trailing edge, a plate's tip, |u| ~ r^-1/2), and the first
coupled steps from it need dt cut by 2^8 to 2^9 (the Newton iterate diverges at larger dt), then, with a plate, by a further
factor of 30 while the starting vortex forms at the tip. Instead the freestream is ACCELERATED SMOOTHLY: the far field (and
the outlets' point-vortex Bernoulli term) carry U r(t), r(t) = r0 + (1 - r0)(1 - cos(pi t/T))/2 for t < T and 1 after
(zero slope at both ends), from the projected potential flow at r0 U (project_start! under that far field, its Bernoulli
pressure at speed r0 U). While t < T the point vortex is not updated and the run may not stop as steady; the loads are
reported against the full freestream's q throughout. The far field is the full freestream from T on, so the steady state the
march converges to is the plain start's problem."""
start_ramp_factor(t::Real, ramp) = t >= ramp.T ? 1.0 : ramp.r0 + (1 - ramp.r0)*0.5*(1 - cos(pi*t/ramp.T))
"A FreeStream at r times the speed of fs (every other field the same)."
scaled_freestream(fs::FreeStream, r::Real) = FreeStream(; U=fs.U*r, alpha=fs.alpha, p_inf=fs.p_inf, rho=fs.rho, chord=fs.chord,
                                                       xref=fs.xref, yref=fs.yref)
"Refuse a ramp that is not (T > 0, 0 < r0 <= 1)."
check_start_ramp(ramp) = (isfinite(ramp.T) && ramp.T > 0 && 0 < ramp.r0 <= 1) ||
    throw(ArgumentError("the ramp start needs T > 0 and 0 < r0 <= 1 (got T = $(ramp.T), r0 = $(ramp.r0))"))

"""
    ramp_start!(st, mesh, bc, cfg, fs, ramp; work) -> NamedTuple

The ramp start's initial state: the far field at r0 U (set_farfield!, G = 0), every level at the uniform flow r0 U, under SST
k and omega at the inflow values (init_turbulence!, the full freestream's), and project_start! with its Bernoulli pressure at
speed r0 U. Returns project_start!'s fields. freeair_run!(...; ramp) then carries the far field to U.
"""
function ramp_start!(st::PBState, m::PBMesh, bc::BCSet, cfg, fs::FreeStream, ramp; work)
    check_start_ramp(ramp)
    be = cfg.backend; f0 = scaled_freestream(fs, ramp.r0); Ux, Uy = uinf(f0)
    set_farfield!(bc, m, f0, 0.0; backend=be)
    init_state!(st, m, (x, y) -> (Ux, Uy, fs.p_inf); backend=be)
    cfg.turbulence === :sst && init_turbulence!(st, m, bc, cfg; work)
    project_start!(st, m, bc, cfg; work, bernoulli_U=f0.U)
end

"""
    freeair_exit_code(outcome) -> Int

The driver's exit code of a free-air run (review of W2, R3; the 14 Oct rule: a non-zero exit is a FAIL): 0 for :STEADY and
:REACHED (the steps or time asked for were made), 2 for :NOT_REACHED (the march was refused), 3 for :INCOMPLETE (the wall cap
was reached), 1 for anything else.
"""
freeair_exit_code(outcome::Symbol) = outcome in (:STEADY, :REACHED) ? 0 : outcome === :NOT_REACHED ? 2 :
                                     outcome === :INCOMPLETE ? 3 : 1

"""
    freeair_config(knobs; backend=CPU()) -> PBConfig

The configuration of the driver's free-air run: rho = U = c = 1, mu = 1/PB_RE, backward Euler at PB_DT on the coupled step,
nonorth 1, PB_N_OUTER_MAX, PBConfig's defaults otherwise, and THE TURBULENCE KNOBS (review of W2, R2; re-review of W1b, R1 d):
`turbulence_settings`, i.e. PB_TURBULENCE, PB_SST_TU_INF, PB_SST_MUT_RATIO, PB_WALL_OMEGA, PB_TURB_COUPLING, PB_TURB_TOL,
PB_TURB_STEP_TOL, PB_TURB_NONORTH and PB_TURB_OUTER_RAMP (their defaults are PBConfig's, so the laminar configuration is the
one before the integration field for field, and none of them is read on the laminar path), and since then W6's, W6b's and
PB_TURB_MUT_BASE (27 Oct 2026; default :extrapolated, PBConfig's).
"""
function freeair_config(k::AbstractDict; backend=CPU())
    check_speed_gate(k)        # W7's PROVISIONAL gate (the second review's R1), after PB_SPEED=auto applied its table
    cfg = PBConfig(; mu=1/k["PB_RE"], dt=k["PB_DT"], nonorth=1.0, n_outer_max=k["PB_N_OUTER_MAX"], backend, turbulence_settings(k)...,
                   speed_settings(k)...,
                   cpl_visc_nonorth=get(k, "PB_CPL_VISC_NONORTH", :jacobian), cpl_au=get(k, "PB_CPL_AU", 0.8),  # W6b (the review of W6, R2)
                   speed_dev=get(k, "PB_SPEED_DEV", false),
                   domain=:freeair)    # the PROVISIONAL gate's case class (the merge review of cf838408, M2; no arithmetic)
    check_speed_config(cfg)    # and on the configuration itself, its backend included (the third review's G1)
    cfg
end

"The receipt of the configuration a free-air run actually used (the driver prints it as PB FREEAIR CONFIG)."
function freeair_config_line(cfg, fs::FreeStream, k::AbstractDict)
    sst = cfg.turbulence === :sst
    kin, win = sst ? sst_freestream(cfg.U_ref, cfg.sst_tu, cfg.sst_mut_ratio, cfg.mu/cfg.rho_l) : (NaN, NaN)
    s = @sprintf("PB FREEAIR CONFIG: turbulence=%s Re=%.6g mu=%.6g rho=%.6g U=%.6g chord=%.6g alpha=%.6g dt=%.6g coupling=%s convection=%s nonorth=%.3g n_outer_max=%d mom_tol=%.3g theta_tol=%.3g steady_tol=%.3g vortex=%s vortex_relax=%.3g vortex_every=%d",
                 cfg.turbulence, 1/cfg.mu, cfg.mu, cfg.rho_l, fs.U, fs.chord, fs.alpha, cfg.dt, cfg.coupling, cfg.convection,
                 cfg.nonorth, cfg.n_outer_max, cfg.mom_tol, cfg.theta_tol, k["PB_STEADY_TOL"], k["PB_VORTEX"],
                 k["PB_VORTEX_RELAX"], k["PB_VORTEX_EVERY"])
    s *= " vortex_tol=1e-06"                         # freeair_drive!'s (freeair_run!'s default; the review of the integration, F6 a)
    s *= " cpl_visc_nonorth=" * string(cfg.cpl_visc_nonorth) * @sprintf(" cpl_au=%.4g", cfg.cpl_au)       # W6b
    # W7's speed settings, printed only when one differs from its default (so the receipt of a default run is the one before W7);
    # the CPU launch shape counts as one (the W7 review's finding 3: a balanced CPU run's receipt showed no trace of it)
    spd = (sync=:all, line_solver=:thomas, prec_precision=:f64, kw_precision=:f64, cpl_gs=:mgs, graphs=:off, dot=:chunk)
    (any(getproperty(cfg, f) !== v for (f, v) in pairs(spd)) || PB_CPU_LAUNCH[] !== :fixed) &&
        (s *= " speed=" * join(("$f:$(getproperty(cfg, f))" for f in keys(spd)), ",") * ",cpu_launch:$(PB_CPU_LAUNCH[])")
    # the development acknowledgement of the PROVISIONAL gate (the second W7 review's R1): any reading carries it
    (cfg.speed_dev || get(k, "PB_SPEED_DEV", false) === true) &&
        (s *= " speed_dev=1(W7 PROVISIONAL settings acknowledged: a development run, not a registered reading)")
    sst || return s
    s *= @sprintf(" sst_tu=%.6g sst_mut_ratio=%.6g k_in=%.6e omega_in=%.6e wall_omega=%s turb_coupling=%s turb_mut=%s turb_omega_nonorth=%s turb_omega_bound=%.3g turb_outer_ramp=%d turb_outer_tol=%.3g turb_nonorth=%.3g turb_nonorth_split=%s turb_tol=%.3g turb_step_tol=%.3g turb_goal_floor=%.3g turb_steady_tol=%.3g k_min=%.3g w_min=%.3g turb_lin_rtol=%.3g turb_sweeps_max=%d turb_step_maxit=%d",
                 cfg.sst_tu, cfg.sst_mut_ratio, kin, win, cfg.wall_omega, cfg.turb_coupling, cfg.turb_mut, cfg.turb_omega_nonorth, cfg.turb_omega_bound, cfg.turb_outer_ramp, cfg.turb_outer_tol,
                 cfg.turb_nonorth, cfg.turb_nonorth_split, cfg.turb_tol, cfg.turb_step_tol, _kw_goal_floor(cfg),
                 k["PB_TURB_STEADY_TOL"], cfg.k_min, cfg.w_min, cfg.turb_lin_rtol, cfg.turb_sweeps_max, cfg.turb_step_maxit)
    # turb_mut_base (27 Oct 2026, records air/mutbase/): appended only when not the default, so a default receipt is the one
    # before it, byte for byte
    cfg.turb_mut_base === :extrapolated ? s : s * " turb_mut_base=" * string(cfg.turb_mut_base)
end

"""
    write_surface_csv(path, sd)

The surface distribution (`surface_distribution`) as a CSV: face, tag, x, y, ds, Cp, Cf, cfx, cfy.
"""
function write_surface_csv(path::AbstractString, sd)
    open(path, "w") do io
        println(io, "face,tag,x,y,ds,Cp,Cf,cfx,cfy")
        for k in eachindex(sd.x)
            println(io, join((sd.face[k], sd.tag[k], sd.x[k], sd.y[k], sd.ds[k], sd.Cp[k], sd.Cf[k], sd.cfx[k], sd.cfy[k]), ","))
        end
    end
    path
end

"""
    floor_census(st, mesh, cfg; L=2) -> (nk, nw, w_at)

The cells of the state's level L sitting AT the floors (k = rho_l k_min, omega = rho_l w_min; W6, the review of the integration,
F1: the ledger counts floor events, this the cells held there in the final state): their numbers and, for omega, where the first
lies ("x,y" of its centroid, or "none"). A host diagnostic of the result line (a one-time read).
"""
function floor_census(st::PBState, m::PBMesh, cfg; L::Integer=2)
    rk = _host(lev(st.rk, L)); rw = _host(lev(st.rw, L)); xc = _host(m.xc); yc = _host(m.yc)
    ik = findall(x -> x <= cfg.rho_l*cfg.k_min, rk); iw = findall(x -> x <= cfg.rho_l*cfg.w_min, rw)
    (nk=length(ik), nw=length(iw), w_at=isempty(iw) ? "none" : @sprintf("%.6g,%.6g", xc[iw[1]], yc[iw[1]]))
end

"Whether `turb_omega_bound` can act under `cfg` (SST, a bound > 0, the k-omega correction split: the kernel's gate)."
omega_bound_on(cfg) = cfg.turbulence === :sst && cfg.turb_omega_bound > 0 && cfg.turb_nonorth == 1 && cfg.turb_nonorth_split

"""
    omega_bound_census(w, mesh, cfg; ncells=12) -> NamedTuple

WHERE `turb_omega_bound` ACTED in the LAST k-omega assembly of the work `w` (W6b; the review of W6b, F1: the bound's census, as
`floor_census` is the floors'): `active` (false, with n = 0, where the bound cannot act: `omega_bound_on`), `n` the cells it
cut, `ratio` the largest cut/Q (cut = `TurbWork.wcut`, the part of omega's net negative correction removed; Q =
`TurbWork.wbq`, the cell's steady budget), its cell `at_b`, `at_i`, `at_j` (block, and the mesh's ci, cj; 0 for none) and
`at` ("(i,j)", "(b:i,j)" on a mesh of more than one block, or "none"), and `cells` the first `ncells` cut cells by decreasing
cut/Q ("(i,j):cut/Q" joined by spaces; "none"). A host read of the work's arrays: the caller decides which assembly it reads
(after a committed step of a march with the SST steady stop, `turbulence_steady_residual`'s, i.e. the committed state's
steady assembly; `omega_bound_census!` for the committed state explicitly).
"""
function omega_bound_census(w, m::PBMesh, cfg; ncells::Integer=12)
    none = (active=omega_bound_on(cfg), n=0, ratio=0.0, at_b=0, at_i=0, at_j=0, at="none", cells="none")
    omega_bound_on(cfg) || return none
    cut = _host(w.turb.wcut); Q = _host(w.turb.wbq)
    a = findall(>(0.0), cut)
    isempty(a) && return none
    ci = _host(m.ci); cj = _host(m.cj); cb = _host(m.cblk)
    lab(c) = m.nblk > 1 ? "($(Int(cb[c])):$(ci[c]),$(cj[c]))" : "($(ci[c]),$(cj[c]))"
    rq = [cut[c]/Q[c] for c in a]; o = sortperm(rq; rev=true); c1 = a[o[1]]
    (active=true, n=length(a), ratio=rq[o[1]], at_b=Int(cb[c1]), at_i=Int(ci[c1]), at_j=Int(cj[c1]), at=lab(c1),
     cells=ncells < 1 ? "" : join([lab(a[o[q]]) * @sprintf(":%.3g", rq[o[q]]) for q in 1:min(ncells, length(o))], " "))
end

"""
    omega_bound_census!(w, mesh, st, bc, cfg; ncells=12) -> NamedTuple

The bound's census AT THE COMMITTED STATE (level 2; the review of W6b, F1 a): level 1 set to level 2 (what every step's
`restore_level!` does), `turb_update!` (mu_t and the k-omega terms re-formed there) and `assemble_komega!` without the time
term (the bound's budget excludes it, so the cut is the same with it), then `omega_bound_census`. It overwrites level 1 (with
level 2) and the work's turbulence arrays, nothing of level 2 or 3; the driver calls it after the run's loads, wall shear and
floor census are read. Returns `omega_bound_census`'s fields, or its inactive form where the bound cannot act.
"""
function omega_bound_census!(w, m::PBMesh, st::PBState, bc, cfg; ncells::Integer=12)
    omega_bound_on(cfg) || return omega_bound_census(w, m, cfg; ncells)
    restore_level!(st)
    turb_update!(w, m, st, bc, cfg) ||
        return (active=true, n=-1, ratio=NaN, at_b=0, at_i=0, at_j=0, at="terms_not_finite", cells="none")
    assemble_komega!(w, m, st, bc, cfg; time_term=false)
    omega_bound_census(w, m, cfg; ncells)
end

"The columns of the bound's per-step history (`omega_bound_history.csv`, freeair_drive! under SST with the bound on)."
const OMEGA_BOUND_HISTORY_HEADER = "step,t,bound_cells,max_cut_over_Q,at_block,at_i,at_j"

"""
    freeair_drive!(mesh, knobs) -> NamedTuple

The driver's free-air run (tools/pbcav_driver.jl, PB_DOMAIN=freeair, PB_STOP=run): the configuration of `freeair_config`
(W6: with PB_START=kutta the start is the lifting potential flow of `kutta_circulation` and `vortex_projection_start!`, and the
point vortex, if any, starts at G_K; PB_START=projection, the default, is the plain start bit for bit)
(laminar, or SST with PB_TURBULENCE=sst: review of W2, R2), rho = U = c = 1, the freestream at PB_ALPHA, the projection start
from the uniform freestream (under SST, k and omega the inflow values in every cell, `init_turbulence!`), PB_VORTEX, at most
PB_MAX_STEPS steps to PB_STEADY_TOL (and, under SST, PB_TURB_STEADY_TOL) under the wall cap PB_WALL_CAP; the load history
and the final surface distribution in PB_OUT_DIR (when set). Prints the configuration that ran (PB FREEAIR CONFIG), the start
and one summary line; the caller exits with `freeair_exit_code(r.outcome)` (R3).
THE BOUND'S CENSUS (W6b; the review of W6b, F1 a), under SST where `turb_omega_bound` can act (`omega_bound_on`): after every
committed step, whether the bound cut any cell in the step's last k-omega assembly (with the SST steady stop, the committed
state's steady assembly) and, if so, its cells and largest cut/Q (`omega_bound_census`), written to omega_bound_history.csv
in PB_OUT_DIR (OMEGA_BOUND_HISTORY_HEADER; loads_history.csv is unchanged); at the end the census AT THE COMMITTED STATE
(`omega_bound_census!`), in the summary line (omega_bound_steps, omega_bound_last_step, omega_bound_final_cells,
omega_bound_final_max_cut_over_Q, omega_bound_final_at) and in one line PB FREEAIR OMEGA BOUND with the cut cells. Laminar,
nothing new is printed; under SST with the bound unable to act the summary line says omega_bound=0 or omega_bound=unsplit(b).
"""
function freeair_drive!(m::PBMesh, k::AbstractDict; backend=CPU())
    fs = FreeStream(; alpha=k["PB_ALPHA"])
    bc = freeair_bcset(m, fs; backend)
    cfg = freeair_config(k; backend)
    sst = cfg.turbulence === :sst
    println(freeair_config_line(cfg, fs, k))
    st = allocate_state(m; backend); Ux, Uy = uinf(fs)
    G0 = 0.0; kut = ""; ramp = nothing
    if get(k, "PB_START", :projection) === :ramp         # the ramp start (24 Oct 2026; opt-in): the freestream accelerated
        ramp = k["PB_START_RAMP"]
        w = pbwork(m, bc, cfg)
        ps = ramp_start!(st, m, bc, cfg, fs, ramp; work=w)
        kut = @sprintf(", start ramp: the far field from %.4g U to U over t = %.4g c/U (r = r0 + (1 - r0)(1 - cos(pi t/T))/2)",
                       ramp.r0, ramp.T)
    elseif get(k, "PB_START", :projection) === :kutta    # W6: the lifting potential start (opt-in)
        w = pbwork(m, bc, cfg)
        G0, j0, jG = kutta_circulation(m, bc, cfg, fs; work=w)
        kut = @sprintf(", start kutta: G_K %.6f (potential C_L %.5f), trailing-edge velocity jump %.4e -> %.4e", G0,
                       2G0/(fs.U*fs.chord), j0, jG)
        ps = vortex_projection_start!(st, m, bc, cfg, fs, G0; work=w)
        # the far field the run starts from: with the point vortex its G is G_K (pv.G below), without it the uniform freestream
        k["PB_VORTEX"] === :point || set_farfield!(bc, m, fs, 0.0; backend)
    else                                                  # the plain projection start (before W6, bit for bit)
        init_state!(st, m, (x, y) -> (Ux, Uy, fs.p_inf); backend)
        w = pbwork(m, bc, cfg)
        sst && init_turbulence!(st, m, bc, cfg; work=w)
        ps = project_start!(st, m, bc, cfg; work=w)
    end
    @printf("PB FREEAIR START: projection FCG %d, rel %.2e, max |div F|/V %.3e -> %.3e%s\n", ps.fcg, ps.rel, ps.div_before,
            ps.div_after, kut)
    pv = k["PB_VORTEX"] === :point ? PointVortex(; relax=k["PB_VORTEX_RELAX"], every=k["PB_VORTEX_EVERY"], G=G0) : nothing
    dir = k["PB_OUT_DIR"]; isempty(dir) || mkpath(dir)
    hio = isempty(dir) ? nothing : open(joinpath(dir, "loads_history.csv"), "w")
    # W6b (the review of W6b, F1 a): where turb_omega_bound acted, per committed step (its own file beside the load history,
    # so loads_history.csv keeps its columns and every recorded proof on it), and the steps it acted in, for the result line
    bon = omega_bound_on(cfg)
    bio = (bon && !isempty(dir)) ? open(joinpath(dir, "omega_bound_history.csv"), "w") : nothing
    bio === nothing || println(bio, OMEGA_BOUND_HISTORY_HEADER)
    bsteps = Ref(0); blast = Ref(0)
    onb = !bon ? nothing : function (s, rec, led)
        n = 0; c = nothing
        if pb_maxabs(w.turb.wcut; backend) > 0
            c = omega_bound_census(w, m, cfg; ncells=0); n = c.n; bsteps[] += 1; blast[] = rec.step
        end
        bio === nothing || (println(bio, join((rec.step, rec.t, n, n > 0 ? c.ratio : 0.0, n > 0 ? c.at_b : 0,
                                               n > 0 ? c.at_i : 0, n > 0 ? c.at_j : 0), ",")); flush(bio))
    end
    r = try
        freeair_run!(st, m, bc, cfg, fs; wall_cap=k["PB_WALL_CAP"], vortex=pv, max_steps=k["PB_MAX_STEPS"],
                     steady_tol=k["PB_STEADY_TOL"], turb_steady_tol=k["PB_TURB_STEADY_TOL"], history=hio, work=w, on_step=onb, ramp)
    finally
        hio === nothing || close(hio)
        bio === nothing || close(bio)
    end
    isempty(dir) || write_surface_csv(joinpath(dir, "surface.csv"), surface_distribution(r.lw, m, fs))
    # the baffles (baffle.jl; only on a mesh that has one): their faces with plate, side and row, and each plate's share of the loads
    if !isempty(_host(m.baffle))
        sdb = surface_distribution(r.lw, m, fs)
        isempty(dir) || write_baffle_csv(joinpath(dir, "baffle.csv"), baffle_table(sdb, m))
        println(baffle_loads_line(sdb, m, fs))
    end
    L = r.loads; h = r.hist
    tail = sst ? begin
        ws = wall_shear(st, m, bc, cfg; L=2, form=:assembled, work=w); yp = sort(ws.yplus)
        fc = floor_census(st, m, cfg)
        @sprintf(" turbulence=sst steady_k=%.3e steady_w=%.3e mut_ratio_max=%.4g yplus_median=%.4g yplus_max=%.4g floor_k=%d floor_w=%d turb_retries=%d",
                 isempty(h) ? NaN : h[end].steady_k, isempty(h) ? NaN : h[end].steady_w,
                 pb_maxabs(lev(st.mut, 2); backend)/cfg.mu, isempty(yp) ? NaN : yp[cld(length(yp), 2)],
                 isempty(yp) ? NaN : yp[end], r.ledger.turb_floor_k_cells, r.ledger.turb_floor_w_cells, r.ledger.turb_retries) *
        @sprintf(" floor_cells_final_k=%d floor_cells_final_w=%d floor_w_at=%s", fc.nk, fc.nw, fc.w_at)
    end : " turbulence=laminar"
    # the bound's census at the committed state (the review of W6b, F1 a), read after everything above (it overwrites level 1
    # with level 2 and the work's turbulence arrays: omega_bound_census!)
    bcs = sst ? omega_bound_census!(w, m, st, bc, cfg) : nothing
    if sst
        tail *= bon ? @sprintf(" omega_bound=%.3g omega_bound_steps=%d omega_bound_last_step=%d omega_bound_final_cells=%d omega_bound_final_max_cut_over_Q=%.3e omega_bound_final_at=%s",
                               cfg.turb_omega_bound, bsteps[], blast[], bcs.n, bcs.ratio, bcs.at) :
                      (cfg.turb_omega_bound > 0 ? @sprintf(" omega_bound=unsplit(%.3g)", cfg.turb_omega_bound) : " omega_bound=0")
    end
    @printf("PB FREEAIR RESULT: outcome=%s t=%.6g steps=%d wall=%.1f CL=%.6f CD=%.6f CDp=%.6f CDf=%.6f CM=%.6f Gamma=%.6f Gamma_defect=%.3e flux_net=%.3e mass_step_err_max=%.3e rejections=%s%s\n",
            r.outcome, r.t, r.nstep, r.wall, L.CL, L.CD, L.CDp, L.CDf, L.CM, pv === nothing ? 0.0 : pv.G,
            isempty(h) ? NaN : h[end].Gamma_defect, r.fluxes.net, r.ledger.step_err_max, string(r.tally), tail)
    (sst && bon) && @printf("PB FREEAIR OMEGA BOUND: at the committed state (steady assembly, bound %.3g): %d cells cut, max cut/Q %.3e at %s; cells %s; acted in %d of %d committed steps (last %d)\n",
                            cfg.turb_omega_bound, bcs.n, bcs.ratio, bcs.at, bcs.cells, bsteps[], r.nstep, blast[])
    isempty(r.refusal) || println("PB FREEAIR REFUSAL: ", r.refusal)
    r
end

# ---- the mesh audit (a host diagnostic) -------------------------------------------------------------------------------
"""
    mesh_audit(mesh::PBMesh; band=40, outlet_cells=3) -> NamedTuple

Cell quality of a C-mesh from version 2's geometry (true centroids), the mesh-quality audit's definitions (main checkout,
docs/_records/2026-10-15_mesh_quality/mesh_quality_lib_20261015.jl): `theta` (max over internal faces, the angle between S and
d; counts above 60, 65, 70 degrees), `skew` (|x_f - x_f'|/|d|), `AR` (longest over shortest edge per cell; max, median, and the
max within `outlet_cells` cells of the outlets), `AR_principal` (extent along i over extent along j, or the inverse), `VR`
(adjacent volume ratio, max over internal faces), `rn` (ratio of consecutive cells' extents along j in a C column; max over
j <= band and over the rest), `rt_foil` (ratio of consecutive foil face lengths; the surface spacing ratio), `rt_cut` (the
same along the wake cut), `angle_min` / `angle_max` (corner angles, degrees), `theta_b_outlet` (max angle between S_b and
x_f - x_P on the outlet faces), with the worst cell's (block, i, j) where it helps, and the counts.
"""
function mesh_audit(m::PBMesh; band::Integer=40, outlet_cells::Integer=3)
    m.family === :cmesh || throw(ArgumentError("mesh_audit is written for C-meshes"))
    H(x) = _host(x)
    xn = H(m.xn); yn = H(m.yn); cn = H(m.cnode); vol = H(m.vol); xc = H(m.xc); yc = H(m.yc)
    th = H(m.theta); own = H(m.own); nei = H(m.nei); ftag = H(m.ftag); dx = H(m.dx); dy = H(m.dy); skx = H(m.skx); sky = H(m.sky)
    cblk = H(m.cblk); ci = H(m.ci); cj = H(m.cj); NI, NJ = m.NI, m.NJ
    nc = m.nc
    ar = zeros(nc); arp = zeros(nc); amin = fill(Inf, nc); amax = zeros(nc); Lj = zeros(nc)
    for c in 1:nc
        p = [(xn[cn[k, c]], yn[cn[k, c]]) for k in 1:4]
        L = [hypot(p[mod1(k+1, 4)][1] - p[k][1], p[mod1(k+1, 4)][2] - p[k][2]) for k in 1:4]
        ar[c] = maximum(L)/minimum(L)
        # principal extents: along i between the i-face midpoints (sides 4 = n4-n1 and 2 = n2-n3), along j between j faces
        li = hypot((p[2][1] + p[3][1] - p[1][1] - p[4][1])/2, (p[2][2] + p[3][2] - p[1][2] - p[4][2])/2)
        lj = hypot((p[4][1] + p[3][1] - p[1][1] - p[2][1])/2, (p[4][2] + p[3][2] - p[1][2] - p[2][2])/2)
        arp[c] = max(li/lj, lj/li); Lj[c] = lj
        for k in 1:4
            a = p[mod1(k - 1, 4)]; o = p[k]; b = p[mod1(k + 1, 4)]
            e1 = (a[1] - o[1], a[2] - o[2]); e2 = (b[1] - o[1], b[2] - o[2])
            ang = acosd(clamp((e1[1]*e2[1] + e1[2]*e2[2])/(hypot(e1...)*hypot(e2...)), -1.0, 1.0))
            amin[c] = min(amin[c], ang); amax[c] = max(amax[c], ang)
        end
    end
    sk = [hypot(skx[f], sky[f])/hypot(dx[f], dy[f]) for f in 1:m.nf]
    vr = [max(vol[own[f]]/vol[nei[f]], vol[nei[f]]/vol[own[f]]) for f in 1:m.nf]
    # wall-normal growth along each C column (cells (i, j) and (i, j + 1))
    rn_band = 1.0; rn_out = 1.0; rn_band_at = (0, 0); rn_out_at = (0, 0)
    for i in 1:NI, j in 1:NJ-1
        a = (i - 1)*NJ + j; b = a + 1
        r = max(Lj[b]/Lj[a], Lj[a]/Lj[b])
        if j + 1 <= band
            r > rn_band && (rn_band = r; rn_band_at = (i, j))
        else
            r > rn_out && (rn_out = r; rn_out_at = (i, j))
        end
    end
    # surface and cut spacing ratios
    bown = H(m.bown); bS = hypot.(H(m.bSx), H(m.bSy))
    fl = H(m.foil); dsf = bS[fl]
    rtf = length(dsf) > 1 ? maximum(max(dsf[k+1]/dsf[k], dsf[k]/dsf[k+1]) for k in 1:length(dsf)-1) : 1.0
    cut = H(m.cut); dsc = H(m.Smag)[cut]
    rtc = length(dsc) > 1 ? maximum(max(dsc[k+1]/dsc[k], dsc[k]/dsc[k+1]) for k in 1:length(dsc)-1) : 1.0
    # outlet faces' angle
    ol = H(m.outlets); bxf = H(m.bxf); byf = H(m.byf); bSx = H(m.bSx); bSy = H(m.bSy)
    thb = isempty(ol) ? 0.0 : maximum(acosd(clamp((bSx[b]*(bxf[b] - xc[bown[b]]) + bSy[b]*(byf[b] - yc[bown[b]]))/
                                             (bS[b]*hypot(bxf[b] - xc[bown[b]], byf[b] - yc[bown[b]])), -1.0, 1.0)) for b in ol)
    outc = [cblk[c] == 0x00 ? (ci[c] <= outlet_cells || ci[c] > NI - outlet_cells) : (ci[c] > m.NIs - outlet_cells) for c in 1:nc]
    loc(c) = (blk=Int(cblk[c]), i=Int(ci[c]), j=Int(cj[c]), x=xc[c], y=yc[c])
    ft = argmax(th); fs_ = argmax(sk); fv = argmax(vr)
    ca = argmax(ar); cam = argmin(amin)
    (cells=nc, faces=m.nf, hmax=sqrt(maximum(vol)), hmin=sqrt(minimum(vol)),
     theta_max=maximum(th), theta_gt60=count(>(60), th), theta_gt65=count(>(65), th), theta_gt70=count(>(70), th),
     theta_at=loc(own[ft]), skew_max=maximum(sk), skew_at=loc(own[fs_]),
     AR_max=maximum(ar), AR_median=sort(ar)[cld(nc, 2)], AR_at=loc(ca), AR_outlet_max=maximum(ar[outc]),
     AR_gt1000=count(>(1000), ar), AR_principal_max=maximum(arp),
     VR_max=maximum(vr), VR_at=loc(own[fv]), VR_cut_max=isempty(cut) ? 1.0 : maximum(vr[cut]),
     rn_band_max=rn_band, rn_band_at=rn_band_at, rn_out_max=rn_out, rn_out_at=rn_out_at,
     rt_foil_max=rtf, rt_cut_max=rtc, ds_foil_min=minimum(dsf), ds_foil_max=maximum(dsf),
     angle_min=minimum(amin), angle_min_at=loc(cam), angle_max=maximum(amax), theta_b_outlet_max=thb,
     dn1_wall=minimum(2*abs(hypot(bxf[b] - xc[bown[b]], byf[b] - yc[bown[b]])) for b in fl))
end

"""
    mesh_audit_gates(a) -> Vector of (gate, value, limit, pass)

The audit against version 1's mesh-quality option's G2 limits (docs/_records/2026-10-15_mesh_quality/DESIGN.md section 7, as
registered there for version 1's channel presets; quoted here as reference limits, not registered for this family): G2a
theta <= 65; G2c AR within 3 cells of the outlet <= 1,000 (a cavitating-preset gate); G2d VR <= 2; G2e r_n <= 1.15 in the
band, <= 1.2 elsewhere; G2f r_t on the foil and the cut <= 1.2; G2g skew <= 4 (version 2's skew/d, the audit's stand-in for
skew_OF); G2h outlet theta_b <= 15.
"""
mesh_audit_gates(a) = [("G2a theta_max", a.theta_max, 65.0, a.theta_max <= 65), ("G2c AR_outlet3", a.AR_outlet_max, 1000.0,
                         a.AR_outlet_max <= 1000), ("G2d VR_max", a.VR_max, 2.0, a.VR_max <= 2),
                        ("G2e rn_band", a.rn_band_max, 1.15, a.rn_band_max <= 1.15), ("G2e rn_out", a.rn_out_max, 1.2, a.rn_out_max <= 1.2),
                        ("G2f rt_foil", a.rt_foil_max, 1.2, a.rt_foil_max <= 1.2), ("G2f rt_cut", a.rt_cut_max, 1.2, a.rt_cut_max <= 1.2),
                        ("G2g skew", a.skew_max, 4.0, a.skew_max <= 4), ("G2h theta_b_outlet", a.theta_b_outlet_max, 15.0,
                         a.theta_b_outlet_max <= 15)]

"One line of the mesh audit (the driver prints it for a free-air mesh)."
function mesh_audit_line(a)
    g = mesh_audit_gates(a)
    @sprintf("PB MESH AUDIT: cells=%d hmax=%.4g theta_max=%.2f gt65=%d skew_max=%.3f AR_max=%.0f AR_median=%.2f AR_outlet3=%.0f VR_max=%.3f rn_band=%.4f rn_out=%.4f rt_foil=%.4f rt_cut=%.4f angle_min=%.2f theta_b_outlet=%.2f dn1_wall=%.3e G2=%s",
             a.cells, a.hmax, a.theta_max, a.theta_gt65, a.skew_max, a.AR_max, a.AR_median, a.AR_outlet_max, a.VR_max,
             a.rn_band_max, a.rn_out_max, a.rt_foil_max, a.rt_cut_max, a.angle_min, a.theta_b_outlet_max, a.dn1_wall,
             join([string(n, ":", p ? "pass" : "FAIL") for (n, _, _, p) in g], ","))
end
