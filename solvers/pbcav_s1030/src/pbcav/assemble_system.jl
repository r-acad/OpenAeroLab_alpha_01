# =====================================================================================================================
# assemble_system.jl -- the face-based linear system every assembler fills (plan 4.5, 5.1). Stage P0b.
#
# A cell-centred finite-volume matrix on a PBMesh is a diagonal (one per cell) and one coefficient pair per internal face:
# `upper[f]`, the owner's row coefficient on the neighbour, and `lower[f]`, the neighbour's row coefficient on the owner
# (equal for a symmetric operator). Boundary conditions enter the diagonal and the right side. The product A x is a
# deterministic gather over each cell's four faces (mesh.cface): no atomics, the same sum order on every backend
# (plan 5.5). Krylov methods, the line solves and the multigrid all take this form.
# =====================================================================================================================

"""
    FaceSystem

`diag` (nc), `upper` and `lower` (nf), `rhs` (nc), on the mesh's backend.
"""
struct FaceSystem{V<:AbstractVector{<:AbstractFloat}}   # (Float64; W7 also makes Float32 copies of the coupled preconditioner's)
    diag::V; upper::V; lower::V; rhs::V
end

"A zeroed FaceSystem for `mesh` on `backend`."
function FaceSystem(m::PBMesh; backend=CPU())
    z(n) = KernelAbstractions.zeros(backend, Float64, n)
    FaceSystem(z(m.nc), z(m.nf), z(m.nf), z(m.nc))
end

# No @Const (review F6, 14 Oct): the header's rule below now holds for every kernel in this file, and y === x would have
# been undefined behaviour under an @Const x.
@kernel function _k_facesys_mul!(y, x, dg, up, lo, cface, own, nei, nf::Int32)
    c = @index(Global)
    @inbounds begin
        acc = dg[c]*x[c]
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                acc += up[code]*x[nei[code]]
            elseif code < 0
                acc += lo[-code]*x[own[-code]]
            end
        end
        y[c] = acc
    end
end

"""
    facesys_mul!(y, A::FaceSystem, x, mesh; backend=CPU())

y = A x by the four-face gather (deterministic, no atomics).
"""
function facesys_mul!(y, A::FaceSystem, x, m::PBMesh; backend=CPU())
    _launch!(_k_facesys_mul!, backend, m.nc, y, x, A.diag, A.upper, A.lower, m.cface, m.own, m.nei, Int32(m.nf))
    pb_sync(backend)
    y
end

# ---- discretisation primitives shared by the assemblers (plan 4.1) ---------------------------------------------------
# Every per-cell sum below is a gather over the cell's four sides (mesh.cface): code f in 1..nf, the cell owns internal
# face f (outward normal +S, neighbour nei[f], neighbour position x_N + t); code -f, the cell is f's neighbour (outward
# normal -S, the other cell own[f] at x_own - t); code nf + b, boundary face b (outward normal S_b).
# GRADIENTS. Least squares (weights 1/|d|^2) for the deferred corrections (convection, non-orthogonality), for the
# pressure at faces that carry no pressure value, and for the skewness corrections below; Green-Gauss on the face
# pressures for the momentum's pressure force, so force and flux use the same face pressures (plan 4.1). The least-squares
# stencil of a cell is its internal faces plus its boundary faces that carry a value of the field: every boundary face for
# the velocity (the condition's face value), only the pressure-Dirichlet faces for the pressure, none for a field without
# boundary values (the momentum interpolation's H, the start's velocity: the "internal" stencil). Its 2x2 normal matrix is
# inverted once per mesh and condition set.
# SKEWNESS (review F1; plan 4.1 as amended on 14 Oct 2026). Linear interpolation with the owner weight fw is exact at x_f',
# where the line from x_P to x_N crosses the face, not at the face centroid x_f. On a mesh whose metric jumps (every coarse
# grid line of a bilinear subdivision; the real C-meshes) |x_f - x_f'|/|d| stays O(1), and the Green-Gauss gradient built
# from x_f' values is not exact for a linear field (worst cell 1.61 |grad p| on N60R). Every face VALUE is therefore
#   phi_f = fw phi_P + (1 - fw) phi_N + 1/2 (grad phi_P + grad phi_N)^LSQ . (x_f - x_f'),
# exact for a linear field: the face pressure of the Green-Gauss gradient (and of the correction's) and the start's face
# velocity (state.jl); the momentum interpolation's H carries the same correction in the form that keeps it consistent with
# the coefficients (assemble_momentum.jl). Coefficients (A, a, a^t, V) and face gradients keep the plain weight. PBConfig.skew = false restores the plain values (the negative examples of the suite).
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).

"""
    LSQGeom

The inverse least-squares normal matrices (g11, g12, g22 per cell) of the velocity stencil (`u*`), of the pressure
stencil (`p*`) and of the internal stencil (`i*`: internal faces only), all with the weights 1/|d|^2; and (`c*`, R2 of
15 Oct 2026, plan 26) the velocity stencil with UNIT weights, the gradient of the convection increment under
`PBConfig.conv_gradient = :unit` (assemble_momentum.jl).
"""
struct LSQGeom{V<:AbstractVector{Float64}}
    u11::V; u12::V; u22::V
    p11::V; p12::V; p22::V
    i11::V; i12::V; i22::V
    c11::V; c12::V; c22::V
end

"The stencil modes of the least-squares kernels: every boundary face, the pressure-Dirichlet faces, none."
const LSQ_VEL = Int32(0); const LSQ_PRES = Int32(1); const LSQ_INT = Int32(2)
"""The least-squares weight of a stencil row d: 1/|d|^2 (wunit = 0, every gradient of the solver except the one below), or 1
(wunit = 1: the convection increment's gradient under conv_gradient = :unit, plan 26). With 1/|d|^2 each row enters as a unit
direction, so in a thin cell (aspect ratio ~4000 on N60R's outlet rows) the short, tangentially offset j-neighbours set the
long direction's component too, and the linear-upwind increment r . g over the long cell can amplify a neighbour
perturbation ~460 times (probe r2 A); with unit weights the rows weigh by |d|^2 and the gain stays <= 0.87 on N60R. Both are
exact for a linear field."""
@inline _lsq_w(ex, ey, wunit::Int32) = wunit == Int32(1) ? 1.0 : 1/(ex*ex + ey*ey)
@inline _lsq_use(pmode::Int32, k::UInt8) = pmode == LSQ_VEL || (pmode == LSQ_PRES && is_pressure_bc(k))

@kernel function _k_lsq_geom!(g11, g12, g22, det_out, cface, dx, dy, bdx, bdy, bkind, nf::Int32, pmode::Int32, wunit::Int32)
    c = @index(Global)
    @inbounds begin
        a11 = 0.0; a12 = 0.0; a22 = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf
                if _lsq_use(pmode, bkind[b])
                    ex = bdx[b]; ey = bdy[b]; w = _lsq_w(ex, ey, wunit)
                    a11 += w*ex*ex; a12 += w*ex*ey; a22 += w*ey*ey
                end
            else
                f = abs(code); ex = dx[f]; ey = dy[f]; w = _lsq_w(ex, ey, wunit)
                a11 += w*ex*ex; a12 += w*ex*ey; a22 += w*ey*ey
            end
        end
        dt = a11*a22 - a12*a12
        det_out[c] = dt/(a11*a22 + 1e-300)
        g11[c] = a22/dt; g12[c] = -a12/dt; g22[c] = a11/dt
    end
end

"""
    lsq_geom(mesh, bc::BCSet; backend=CPU()) -> LSQGeom

Refuses a cell whose stencil is (near) collinear (relative determinant below 1e-8). `lsq_geom_internal(mesh)` gives the
internal stencil alone, which needs no condition set (the start's face velocity).
"""
function lsq_geom(m::PBMesh, bc; backend=CPU())
    z() = KernelAbstractions.zeros(backend, Float64, m.nc)
    G = LSQGeom(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z()); dt = z()
    for (pm, a, b, c) in ((LSQ_VEL, G.u11, G.u12, G.u22), (LSQ_PRES, G.p11, G.p12, G.p22), (LSQ_INT, G.i11, G.i12, G.i22))
        _lsq_geom_mode!(a, b, c, dt, m, bc.kind, pm, backend)
    end
    _lsq_geom_mode!(G.c11, G.c12, G.c22, dt, m, bc.kind, LSQ_VEL, backend; wunit=Int32(1))
    G
end
function _lsq_geom_mode!(a, b, c, dt, m::PBMesh, kind, pm::Int32, backend; wunit::Int32=Int32(0))
    _launch!(_k_lsq_geom!, backend, m.nc, a, b, c, dt, m.cface, m.dx, m.dy, m.bdx, m.bdy, kind, Int32(m.nf), pm, wunit)
    pb_sync(backend)
    bad = count(v -> !(v > 1e-8), _host(dt))
    lab = pm == LSQ_VEL ? "velocity" : pm == LSQ_PRES ? "pressure" : "internal"
    bad == 0 || throw(ArgumentError("lsq_geom: $bad cells have a collinear least-squares stencil ($lab)"))
    nothing
end
function lsq_geom_internal(m::PBMesh; backend=CPU())
    z() = KernelAbstractions.zeros(backend, Float64, m.nc)
    a, b, c, dt = z(), z(), z(), z()
    _lsq_geom_mode!(a, b, c, dt, m, KernelAbstractions.zeros(backend, UInt8, max(m.nb, 1)), LSQ_INT, backend)
    (a, b, c)
end

"Least-squares gradient of a scalar phi: boundary values phib (only where pmode admits the face; vzero = 1 uses 0 there)."
@kernel function _k_lsq_grad!(gx, gy, phi, phib, g11, g12, g22, cface, own, nei, dx, dy, bdx, bdy, bkind, nf::Int32,
                              pmode::Int32, vzero::Int32, wunit::Int32)
    c = @index(Global)
    @inbounds begin
        sx = 0.0; sy = 0.0; pc = phi[c]
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf
                if _lsq_use(pmode, bkind[b])
                    ex = bdx[b]; ey = bdy[b]; w = _lsq_w(ex, ey, wunit)
                    dv = (vzero == Int32(1) ? 0.0 : phib[b]) - pc
                    sx += w*ex*dv; sy += w*ey*dv
                end
            elseif code > 0
                f = code; ex = dx[f]; ey = dy[f]; w = _lsq_w(ex, ey, wunit); dv = phi[nei[f]] - pc
                sx += w*ex*dv; sy += w*ey*dv
            else
                f = -code; ex = -dx[f]; ey = -dy[f]; w = _lsq_w(ex, ey, wunit); dv = phi[own[f]] - pc
                sx += w*ex*dv; sy += w*ey*dv
            end
        end
        gx[c] = g11[c]*sx + g12[c]*sy; gy[c] = g12[c]*sx + g22[c]*sy
    end
end

"""Face pressure on every boundary face: the condition's value on a pressure-Dirichlet face (0 with vzero = 1, for a
correction), else the owner's value extrapolated with its least-squares gradient, p_P + grad p . d_b."""
@kernel function _k_bface_p!(pbf, p, gx, gy, pb, bkind, bown, bdx, bdy, vzero::Int32)
    b = @index(Global)
    @inbounds begin
        P = bown[b]
        if is_pressure_bc(bkind[b])
            pbf[b] = vzero == Int32(1) ? 0.0 : pb[b]
        else
            pbf[b] = p[P] + gx[P]*bdx[b] + gy[P]*bdy[b]
        end
    end
end

"""The skewness-corrected value at internal face f of a cell field phi with least-squares gradient (gx, gy) (header):
fw phi_P + (1 - fw) phi_N + skew/2 (grad phi_P + grad phi_N) . (x_f - x_f'); skew = 0 gives the plain linear value."""
@inline function face_value(phi, gx, gy, f, own, nei, fw, skx, sky, skew::Float64)
    P = own[f]; N = nei[f]; w = fw[f]
    w*phi[P] + (1 - w)*phi[N] + skew*0.5*((gx[P] + gx[N])*skx[f] + (gy[P] + gy[N])*sky[f])
end

"""Green-Gauss gradient on the face pressures: the skewness-corrected face value (face_value, with the pressure's
least-squares gradient lx, ly) inside, pbf on boundary faces."""
@kernel function _k_gg_grad!(gx, gy, p, lx, ly, pbf, vol, cface, own, nei, fw, skx, sky, Sx, Sy, bSx, bSy, nf::Int32,
                             skew::Float64)
    c = @index(Global)
    @inbounds begin
        sx = 0.0; sy = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                b = code - nf; sx += pbf[b]*bSx[b]; sy += pbf[b]*bSy[b]
            elseif code > 0
                f = code; pf = face_value(p, lx, ly, f, own, nei, fw, skx, sky, skew); sx += pf*Sx[f]; sy += pf*Sy[f]
            else
                f = -code; pf = face_value(p, lx, ly, f, own, nei, fw, skx, sky, skew); sx -= pf*Sx[f]; sy -= pf*Sy[f]
            end
        end
        gx[c] = sx/vol[c]; gy[c] = sy/vol[c]
    end
end

"Face values of a cell field on every internal face (face_value): the start's face flux and the suite use it."
@kernel function _k_face_values!(out, phi, gx, gy, own, nei, fw, skx, sky, skew::Float64)
    f = @index(Global)
    @inbounds out[f] = face_value(phi, gx, gy, f, own, nei, fw, skx, sky, skew)
end

"""
    convection_gain(mesh, bc; weights=:unit) -> Matrix{Float64} (2 x nf)

A host diagnostic of the linear-upwind increment's geometry (R2, plan 26; probe r2 A). For internal face f upwinded from
its owner (row 1) or its neighbour (row 2), the increment r . g_U (r = x_f - x_U, g_U the velocity stencil's
least-squares gradient with weights 1/|d|^2, `weights = :inverse_distance`, or 1, `:unit`) is a linear function
sum_k s_k (u_k - u_U) of the differences to U's internal neighbours; its GAIN sum_k |s_k| bounds how far a perturbation of
the neighbours moves the face value (1/2 for one-dimensional linear upwind on a uniform mesh). Boundary rows enter the
normal matrix only. On N60R L0 the inverse-distance gains reach 458 (the outlet rows' thin skewed cells) and the unit-weight
gains stay below 0.87.
"""
function convection_gain(m::PBMesh, bc; weights::Symbol=:unit)
    weights in (:unit, :inverse_distance) || throw(ArgumentError("weights must be :unit or :inverse_distance"))
    wu = Int32(weights === :unit ? 1 : 0)
    own = _host(m.own); nei = _host(m.nei); cf = _host(m.cface); dx = _host(m.dx); dy = _host(m.dy)
    bdx = _host(m.bdx); bdy = _host(m.bdy); xf = _host(m.xf); yf = _host(m.yf); xc = _host(m.xc); yc = _host(m.yc)
    ftx = _host(m.ftx); fty = _host(m.fty); bk = _host(bc.kind); nf = m.nf
    function gain(c, rx, ry)
        a11 = 0.0; a12 = 0.0; a22 = 0.0
        for k in 1:4
            code = cf[k, c]
            if code > nf
                b = code - nf
                _lsq_use(LSQ_VEL, bk[b]) || continue
                ex = bdx[b]; ey = bdy[b]
            else
                ex = dx[abs(code)]; ey = dy[abs(code)]
            end
            w = _lsq_w(ex, ey, wu); a11 += w*ex*ex; a12 += w*ex*ey; a22 += w*ey*ey
        end
        det = a11*a22 - a12*a12; g = 0.0
        for k in 1:4
            code = cf[k, c]; (0 < abs(code) <= nf) || continue
            sg = code > 0 ? 1.0 : -1.0; ex = sg*dx[abs(code)]; ey = sg*dy[abs(code)]; w = _lsq_w(ex, ey, wu)
            gx = (a22*w*ex - a12*w*ey)/det; gy = (-a12*w*ex + a11*w*ey)/det
            g += abs(rx*gx + ry*gy)
        end
        g
    end
    G = zeros(2, nf)
    for f in 1:nf
        P = own[f]; N = nei[f]
        G[1, f] = gain(P, xf[f] - xc[P], yf[f] - yc[P])
        G[2, f] = gain(N, xf[f] - (xc[N] + ftx[f]), yf[f] - (yc[N] + fty[f]))
    end
    G
end
