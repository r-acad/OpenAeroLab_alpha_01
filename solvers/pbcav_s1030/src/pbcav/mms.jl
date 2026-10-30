# =====================================================================================================================
# mms.jl -- the manufactured-solution forcing hook (plan 10.1), present from the first commit. An MMS field supplies the
# exact solution and the source terms that make it exact; the assemblers add V_P * mms_source at the true centroid to each
# equation's right side (momentum) and to the volume constraint (mass) when a field is attached. P0b: u, v, p (psi = 0,
# constant rho and mu); P1/P5a: alpha_v with psi != 0 and a manufactured source; P2: k, omega.
# The fields are isbits structs and the sources plain functions, so the same call runs inside a kernel on any backend.
# =====================================================================================================================

"A manufactured solution: `mms_exact(field, x, y, t)` and `mms_source(field, eq, x, y, t)`."
abstract type MMSField end

"No manufactured solution (production): every source is zero."
struct NoMMS <: MMSField end

"Equation codes for the kernels (mms_source's `eq` as an integer)."
const EQ_MASS = Int32(1); const EQ_XMOM = Int32(2); const EQ_YMOM = Int32(3)
_eqcode(eq::Symbol) = eq === :mass ? EQ_MASS : eq === :xmom ? EQ_XMOM : eq === :ymom ? EQ_YMOM :
    throw(ArgumentError("mms_source: equation $eq is not built before its stage (P0b: :mass, :xmom, :ymom)"))

"""
    MMSTrig(; U0, V0, A, kx, ky, x0, y0, B, kpx, kpy, et, om, rho, mu)

A smooth divergence-free field for the incompressible equations with constant rho and mu (plan 10.1):
  u = U0 + A g(t) sin X cos Y,   v = V0 - A g(t) (kx/ky) cos X sin Y,   X = kx (x - x0), Y = ky (y - y0),
  p = B g(t) cos(kpx x) cos(kpy y),   g(t) = 1 + et sin(om t)   (et = 0: steady).
The momentum sources are rho (du/dt + u.grad u) + grad p - mu lap u (exact derivatives below) minus the constants (cx, cy)
(default 0: bitwise the plain sources, x - 0.0 === x); the mass source is zero. The constants exist for the doubly periodic
box (`mms_zero_mean`, plan 25.2): there the discrete momentum budget is sum_P V_P s_P, so a steady discrete state exists only
if that sum vanishes, and the sampled sources' sum is a quadrature error, not zero.
"""
Base.@kwdef struct MMSTrig <: MMSField
    U0::Float64 = 0.3; V0::Float64 = 0.2; A::Float64 = 0.5
    kx::Float64 = 2pi; ky::Float64 = 2pi; x0::Float64 = 0.1; y0::Float64 = -0.05
    B::Float64 = 0.25; kpx::Float64 = pi; kpy::Float64 = pi
    et::Float64 = 0.0; om::Float64 = 2pi
    rho::Float64 = 1.0; mu::Float64 = 0.01
    cx::Float64 = 0.0; cy::Float64 = 0.0
end

"""
    mms_zero_mean(f::MMSTrig, m::PBMesh; t=0.0) -> MMSTrig

The field `f` with (cx, cy) set to the volume-weighted means of its momentum sources over the cells of `m` at time t (cx, cy
of `f` itself ignored), so that sum_P V_P s_P vanishes to round-off (plan 25.2: the doubly periodic box). The exact field then
solves the forced equations up to that constant body force, which is the quadrature error of the cell-centre rule (O(h^2) on a
smooth mesh), so the order of accuracy is unaffected.
"""
function mms_zero_mean(f::MMSTrig, m::PBMesh; t::Real=0.0)
    g = MMSTrig(; U0=f.U0, V0=f.V0, A=f.A, kx=f.kx, ky=f.ky, x0=f.x0, y0=f.y0, B=f.B, kpx=f.kpx, kpy=f.kpy, et=f.et, om=f.om,
                rho=f.rho, mu=f.mu)
    V = _host(m.vol); xc = _host(m.xc); yc = _host(m.yc)
    sx = sum(V[c]*mms_source(g, EQ_XMOM, xc[c], yc[c], t) for c in eachindex(V))
    sy = sum(V[c]*mms_source(g, EQ_YMOM, xc[c], yc[c], t) for c in eachindex(V))
    MMSTrig(; U0=f.U0, V0=f.V0, A=f.A, kx=f.kx, ky=f.ky, x0=f.x0, y0=f.y0, B=f.B, kpx=f.kpx, kpy=f.kpy, et=f.et, om=f.om,
            rho=f.rho, mu=f.mu, cx=sx/sum(V), cy=sy/sum(V))
end

"""
    MMSPoiseuille(; G, mu, y0=0.0, H=1.0)

Plane Poiseuille flow driven by a uniform body force G along x between walls at y0 and y0 + H (plan 10.2, 25.6): the exact
solution u = G/(2 mu) (y - y0)(y0 + H - y), v = 0, p = 0 (uniform); the source is G in the x-momentum equation only (the
body force), zero in the mass and y-momentum equations. Steady.
"""
Base.@kwdef struct MMSPoiseuille <: MMSField
    G::Float64; mu::Float64; y0::Float64 = 0.0; H::Float64 = 1.0
end
@inline mms_exact(f::MMSPoiseuille, x, y, t) = (f.G/(2f.mu)*(y - f.y0)*(f.y0 + f.H - y), 0.0, 0.0)
@inline mms_source(f::MMSPoiseuille, eq::Int32, x, y, t) = eq == EQ_XMOM ? f.G : 0.0

@inline _mms_g(f::MMSTrig, t) = (1 + f.et*sin(f.om*t), f.et*f.om*cos(f.om*t))

"Cells per velocity wavelength on the coarsest level of a mesh's MMS ladder, registered (plan 10.1, amended 14 Oct 2026)."
const MMS_CELLS_PER_WAVELENGTH = 12

"""
    mms_for_mesh(m::PBMesh; ppw=MMS_CELLS_PER_WAVELENGTH, kw...) -> MMSTrig

The MMS field of plan 10.1 sized for a mesh's domain (review F6, 14 Oct 2026): MMSTrig's defaults (kx = ky = 2 pi,
kpx = kpy = pi) suit a unit box, but on N60R's 11.8 x 5 c channel its far-field cells would carry a few cells per
wavelength. Here the velocity wavelength is lambda = ppw h_max, h_max the largest sqrt(V) of the mesh (the coarsest level
of the ladder is the one to pass), so EVERY cell has at least ppw cells per wavelength; the pressure wavenumbers are half
the velocity's (as the defaults), and the phase origin sits 0.1 lambda from the node box's centre (no symmetry with the
domain). Other fields (U0, V0, A, B, et, om, rho, mu) are MMSTrig's defaults unless given. The cost: near a foil, whose
cells are far smaller than h_max, the field is close to linear over many cells; the norms with and without the special
cells (10.1) show that region separately.
"""
function mms_for_mesh(m::PBMesh; ppw::Real=MMS_CELLS_PER_WAVELENGTH, kw...)
    ppw >= 4 || throw(ArgumentError("mms_for_mesh: ppw must be >= 4 (got $ppw)"))
    hmax = sqrt(maximum(_host(m.vol))); lam = ppw*hmax; k = 2pi/lam
    xn = _host(m.xn); yn = _host(m.yn)
    xm = (minimum(xn) + maximum(xn))/2; ym = (minimum(yn) + maximum(yn))/2
    MMSTrig(; kx=k, ky=k, kpx=k/2, kpy=k/2, x0=xm + 0.1lam, y0=ym - 0.05lam, kw...)
end

"""
    mms_density_neutral(field) -> Bool

True when the manufactured field keeps the density uniform (rho = rho_l: no mass source, constant rho), the condition SST
needs (W1b, review F3; `check_turbulence_config`). FAIL-CLOSED: false for any field that does not declare it, so a field of
stage P1 or P5a (psi != 0, a phase source) refuses SST until its stage decides.
"""
mms_density_neutral(f::MMSField) = false
mms_density_neutral(f::NoMMS) = true
mms_density_neutral(f::MMSTrig) = true            # mass source zero, constant rho (above)
mms_density_neutral(f::MMSPoiseuille) = true      # mass source zero

"The exact (u, v, p) of a field at (x, y, t)."
mms_exact(f::NoMMS, x, y, t) = (0.0, 0.0, 0.0)
@inline function mms_exact(f::MMSTrig, x, y, t)
    g, _ = _mms_g(f, t); X = f.kx*(x - f.x0); Y = f.ky*(y - f.y0)
    (f.U0 + f.A*g*sin(X)*cos(Y), f.V0 - f.A*g*(f.kx/f.ky)*cos(X)*sin(Y), f.B*g*cos(f.kpx*x)*cos(f.kpy*y))
end

"The source of equation `eq` (:mass, :xmom, :ymom; later :vapour, :k, :omega) at (x, y, t)."
mms_source(f::MMSField, eq::Symbol, x, y, t) = mms_source(f, _eqcode(eq), x, y, t)
mms_source(f::NoMMS, eq::Int32, x, y, t) = 0.0
@inline function mms_source(f::MMSTrig, eq::Int32, x, y, t)
    eq == EQ_MASS && return 0.0
    g, gt = _mms_g(f, t); A = f.A; kx = f.kx; ky = f.ky
    X = f.kx*(x - f.x0); Y = f.ky*(y - f.y0); sX = sin(X); cX = cos(X); sY = sin(Y); cY = cos(Y)
    u = f.U0 + A*g*sX*cY; v = f.V0 - A*g*(kx/ky)*cX*sY
    if eq == EQ_XMOM
        ut = A*gt*sX*cY; ux = A*g*kx*cX*cY; uy = -A*g*ky*sX*sY; lap = -A*g*(kx^2 + ky^2)*sX*cY
        px = -f.B*g*f.kpx*sin(f.kpx*x)*cos(f.kpy*y)
        return f.rho*(ut + u*ux + v*uy) + px - f.mu*lap - f.cx
    else
        vt = -A*gt*(kx/ky)*cX*sY; vx = A*g*(kx^2/ky)*sX*sY; vy = -A*g*kx*cX*cY; lap = A*g*(kx/ky)*(kx^2 + ky^2)*cX*sY
        py = -f.B*g*f.kpy*cos(f.kpx*x)*sin(f.kpy*y)
        return f.rho*(vt + u*vx + v*vy) + py - f.mu*lap - f.cy
    end
end
