# =====================================================================================================================
# state.jl -- the structure-of-arrays state (plan 3.1, 4.7, 6.1). Stage P0b.
#
# THE VARIABLES (plan 3.1): the mixture density rho, the vapour partial density q = alpha_v rho_v, the velocity (ux, uy),
# the pressure p, and rho k, rho omega; the VOLUMETRIC face flux F = u_f . S on internal faces and Fb on boundary faces
# (the mass flux is rho_f F, rho_f the phase law's face density: rho_l in the incompressible single-phase limit of P0b).
# THREE TIME LEVELS: each field has its current iterate (level 1, n+1), n (level 2) and n-1 (level 3), so backward Euler
# and variable-step BDF2 read the same arrays (plan 4.2), and a rejected step restores level 1 from level 2 bitwise.
# THE SPAN INDEX k IS RESERVED (NK = 1): a field is (ncell, NK, NTIME); every kernel reads the k = 1 slice. A quasi-3-D or
# eddy-resolving extension (the shedding plan) adds span layers without changing a signature.
# Every array lives on the mesh's backend; nothing here loops over cells on the host.
# =====================================================================================================================

"Time levels held for each field: 1 the current iterate (n+1), 2 level n, 3 level n-1."
const NTIME = 3

"""
    PBState

Fields, each `n x NK x NTIME`: `rho`, `q`, `ux`, `uy`, `rk`, `rw` (rho k, rho omega), `p`, `mut` (the SST eddy viscosity
mu_t, dynamic; W1, 21 Oct 2026) (cells), `F` (internal faces), `Fb` (boundary faces). Scalars: physical time `t` (of
level 2, the last committed state), step `dt`, previous step `dt_prev`, step count `nstep`.
THE TURBULENCE FIELDS (W1, plan 32): rk and rw are transported only under PBConfig.turbulence = :sst (assemble_komega.jl);
mut is level 1's mu_t, recomputed from the iterate at every outer iteration (turb_update!) and committed and restored with
the other fields. On the laminar path (the default) all three stay zero and nothing reads them.
"""
mutable struct PBState{A3<:AbstractArray{Float64,3}}
    NK::Int
    rho::A3; q::A3; ux::A3; uy::A3; rk::A3; rw::A3
    p::A3
    F::A3; Fb::A3
    mut::A3
    t::Float64; dt::Float64; dt_prev::Float64; nstep::Int
end

"The fields of a PBState that carry time levels (the ones a commit shifts and a rejection restores)."
const STATE_FIELDS = (:rho, :q, :ux, :uy, :rk, :rw, :p, :F, :Fb, :mut)
"The fields before W1 (21 Oct 2026); the laminar-bitwise proof hashes these (mut is new and stays zero on the laminar path)."
const STATE_FIELDS_P0B = (:rho, :q, :ux, :uy, :rk, :rw, :p, :F, :Fb)

"""
    allocate_state(mesh::PBMesh; backend=CPU(), NK=1) -> PBState

A zeroed state on `backend` with the layout above. Only NK = 1 is admitted until a span direction exists.
"""
function allocate_state(m::PBMesh; backend=CPU(), NK::Integer=1)
    NK == 1 || throw(ArgumentError("NK = 1 only (the span index is reserved; got $NK)"))
    z(n) = KernelAbstractions.zeros(backend, Float64, n, NK, NTIME)
    PBState(Int(NK), z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nc), z(m.nf), z(m.nb), z(m.nc), 0.0, 0.0, 0.0, 0)
end

"The k = 1 slice of a field at time level L (1 current, 2 n, 3 n-1): a contiguous view, valid on any backend."
lev(A::AbstractArray{Float64,3}, L::Integer) = view(A, :, 1, L)

"Copy time level `src` of every field into level `dst` (bitwise)."
function copy_level!(st::PBState, dst::Integer, src::Integer)
    for f in STATE_FIELDS
        A = getfield(st, f); copyto!(lev(A, dst), lev(A, src))
    end
    st
end

"Commit the current iterate: level 3 <- level 2 <- level 1; time and step counters advance."
function commit_level!(st::PBState)
    copy_level!(st, 3, 2); copy_level!(st, 2, 1)
    st.t += st.dt; st.dt_prev = st.dt; st.nstep += 1
    st
end

"Restore the current iterate from level n (a rejected step): bitwise."
restore_level!(st::PBState) = copy_level!(st, 1, 2)

"""
    init_state!(st, mesh, f; rho_l=1.0, backend=CPU(), skew=true) -> st

Every time level set from `f(x, y) -> (u, v, p)` at the true centroids, rho = rho_l, q = 0, and the face fluxes from the
cell velocities' face values (F = u_f . S; boundary faces from the owner's value). The face value is skewness corrected
with the internal least-squares gradient (assemble_system.jl, plan 4.1 as amended 14 Oct; skew = false: the plain linear
value), so a linear velocity field gives its exact face flux. The start of a verification case; a run's start (the wetted
panel flow plus one projection, plan 4.7) is `build_pbcav_state`.
"""
function init_state!(st::PBState, m::PBMesh, f; rho_l::Real=1.0, backend=CPU(), skew::Bool=true)
    xc = _host(m.xc); yc = _host(m.yc)
    uvp = [f(xc[c], yc[c]) for c in 1:m.nc]
    u = _dev(backend, [t[1] for t in uvp]); v = _dev(backend, [t[2] for t in uvp]); p = _dev(backend, [t[3] for t in uvp])
    g11, g12, g22 = lsq_geom_internal(m; backend)
    z() = KernelAbstractions.zeros(backend, Float64, m.nc)
    gux, guy, gvx, gvy = z(), z(), z(), z()
    kz = KernelAbstractions.zeros(backend, UInt8, max(m.nb, 1)); bz = KernelAbstractions.zeros(backend, Float64, max(m.nb, 1))
    for (g1, g2, phi) in ((gux, guy, u), (gvx, gvy, v))
        _launch!(_k_lsq_grad!, backend, m.nc, g1, g2, phi, bz, g11, g12, g22, m.cface, m.own, m.nei, m.dx, m.dy, m.bdx, m.bdy,
                 kz, Int32(m.nf), LSQ_INT, Int32(0), Int32(0))
    end
    pb_sync(backend)
    sk = skew ? 1.0 : 0.0
    for L in 1:NTIME
        copyto!(lev(st.ux, L), u); copyto!(lev(st.uy, L), v); copyto!(lev(st.p, L), p)
        fill!(lev(st.rho, L), Float64(rho_l)); fill!(lev(st.q, L), 0.0)
        _launch!(_k_face_flux_interp!, backend, m.nf, lev(st.F, L), u, v, gux, guy, gvx, gvy, m.fw, m.skx, m.sky, m.own,
                 m.nei, m.Sx, m.Sy, sk)
        _launch!(_k_bface_flux_owner!, backend, m.nb, lev(st.Fb, L), u, v, m.bown, m.bSx, m.bSy)
    end
    pb_sync(backend)
    st.t = 0.0; st.nstep = 0
    st
end

@kernel function _k_face_flux_interp!(F, u, v, gux, guy, gvx, gvy, fw, skx, sky, own, nei, Sx, Sy, skew::Float64)
    f = @index(Global)
    @inbounds F[f] = face_value(u, gux, guy, f, own, nei, fw, skx, sky, skew)*Sx[f] +
                     face_value(v, gvx, gvy, f, own, nei, fw, skx, sky, skew)*Sy[f]
end
@kernel function _k_bface_flux_owner!(Fb, u, v, bown, bSx, bSy)
    b = @index(Global)
    @inbounds Fb[b] = u[bown[b]]*bSx[b] + v[bown[b]]*bSy[b]
end

"""
    build_pbcav_state(mesh, config) -> PBState

The initial state of a driver run: the wetted panel start plus one incompressible projection, or a restart (plan 4.7,
Start). Stage P3 (the first driver run, the NACA 0012 wetted check); the verification cases start from `init_state!`.
"""
build_pbcav_state(mesh, config) = _nyi(:build_pbcav_state, "stage P3, the first driver run")
