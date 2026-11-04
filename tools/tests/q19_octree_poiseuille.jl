# Gate G1.3 of the CPU octree D3Q19 solver: Poiseuille flow between two STL plates (closed slabs) driven by a
# body force, plates tilted by 0, 15, 30 and 45 degrees about the flow axis with random sub-cell offsets, three
# resolutions: profile convergence order, wall position, and the momentum-exchange friction against the exact
# wall shear. One level, periodic in x and y, slip at the outer z faces. CPU, a minute or two.
#   julia -t 8 --project=src/octree_q19 tools/tests/q19_octree_poiseuille.jl
using Test, Printf, LinearAlgebra, Random
using OctreeQ19
const G = OctreeQ19
const ONLY = get(ENV, "Q19_TEST_ONLY", "")
want(name) = isempty(ONLY) || occursin(ONLY, name)

"A closed box (12 triangles) from lo to hi."
function box_tris(lo, hi)
    v = [(x, y, z) for z in (lo[3], hi[3]) for y in (lo[2], hi[2]) for x in (lo[1], hi[1])]   # index = 1 + ix + 2iy + 4iz
    faces = ((1, 3, 7, 5), (2, 6, 8, 4), (1, 5, 6, 2), (3, 4, 8, 7), (1, 2, 4, 3), (5, 7, 8, 6))   # outward: -x +x -y +y -z +z
    tri = G.Tri[]
    for f in faces
        push!(tri, (v[f[1]], v[f[2]], v[f[3]])); push!(tri, (v[f[1]], v[f[3]], v[f[4]]))
    end
    tri
end

"""
Channel of half-width H (lattice units) between two slabs, tilted by `angle` degrees about the x axis (the flow
axis), the channel centre at a random sub-cell position. Periodic in x; the y and z faces are slip, and the slabs
are thick enough that every cell outside the channel is solid (so the only fluid is the channel and the global
force balance is exact). A tilted channel meets the slip faces at an angle, where the parabola is incompatible
with the slip condition (du/dn is not zero there), so the domain is tall enough (2n in z) that the band leaves it
only far out in y, and the profile is measured in the central quarter of y. Returns the mesh, config, state and
the channel geometry (centre, unit normal).
"""
function channel(n::Int, H::Float64, angle_deg::Float64, nu::Float64, Fx::Float64; precision=Float64, seed=1)
    Random.seed!(seed)
    dx = 1.0
    a = deg2rad(angle_deg); nrm = (0.0, -sin(a), cos(a))        # the channel normal, in the y-z plane
    Lx = 2 * (n ÷ 4); Ly = 2n; Lz = 2n             # the periodic x length must be an even number of cells
    c = (0.0, Ly / 2 + (rand() - 0.5), Lz / 2 + (rand() - 0.5))   # channel centre (random sub-cell offset)
    thick = 4.0 * n                                               # slabs reach far beyond the domain
    rot(p) = (p[1], cos(a) * p[2] - sin(a) * p[3], sin(a) * p[2] + cos(a) * p[3])
    function slab(sign)
        tri = G.Tri[]
        Wh = 4n
        for t in box_tris((-Lx, -Wh, sign > 0 ? H : -H - thick), (2Lx, Wh, sign > 0 ? H + thick : -H))
            push!(tri, Tuple((rot(p) .+ c) for p in t))
        end
        tri
    end
    tri = vcat(slab(+1), slab(-1))
    spec = G.MeshSpec(domain=(Float64(Lx), Float64(Ly), Float64(Lz)), periodic=(true, false, false), levels=1, dx_finest=dx, faces=(:periodic, :periodic, :slip, :slip, :slip, :slip))
    mesh = G.generate(tri, spec)
    cfg = G.test_case_config(spec; nu_lattice=nu, precision=precision, body_force=(Fx, 0.0, 0.0))
    st = G.State(mesh, cfg, precision)
    G.initialise!(st, p -> (1.0, 0.0, 0.0, 0.0))
    mesh, cfg, st, c, nrm
end

"Run to a steady state (the force-driven channel relaxes on H^2/nu); returns the max relative change of the last interval."
function run_steady!(st, nsteps; collect=true)
    for s in 1:nsteps
        G.advance!(st; collect=(collect && s == nsteps))
    end
end

"Sample the velocity profile in the central quarter of y: every fluid cell's signed distance from the centre plane and u_x."
function profile(st, c, nrm)
    Lv = st.mesh.levels[1]; S = st.levels[1]; Ly = st.mesh.domain[2]
    d = Float64[]; u = Float64[]
    for o in S.real_octs, k in 1:8
        Lv.kind[k, o] == G.K_FLUID || continue
        p = G.slot_centre(Lv, G.slot_id(k, o))
        abs(p[2] - Ly / 2) < Ly / 8 || continue
        s = (p[2] - c[2]) * nrm[2] + (p[3] - c[3]) * nrm[3]
        push!(d, s); push!(u, Float64(S.u[k, 1, o]))
    end
    d, u
end

want("G1.3") && @testset "G1.3 Poiseuille between tilted STL plates" begin
    nu = 0.1; Fx = 1e-5                              # tau 0.8
    results = Dict{Tuple{Float64,Int},Any}()
    sizes = (18, 36, 72)
    for angle in (0.0, 15.0, 30.0, 45.0), n in sizes
        H = n / 6                                   # channel half-width in cells: 3, 6, 12
        mesh, cfg, st, c, nrm = channel(n, H, angle, nu, Fx; seed=round(Int, angle) + n)
        @test mesh.counts["reports"]["leak_cells"] == 0
        nfluid = mesh.counts["fluid_total"]
        umax = Fx * H^2 / (2nu)
        steps = round(Int, 5 * H^2 / nu)            # five diffusion times
        run_steady!(st, steps)
        d, u = profile(st, c, nrm)
        # least-squares fit of u = a (H_eff^2 - d^2) gives the effective half-width (the wall position)
        A = hcat(ones(length(d)), d .^ 2); coef = A \ u
        H_eff = sqrt(-coef[1] / coef[2])
        wall_err = H_eff - H                         # in cells
        prof_err = maximum(abs.(u .- Fx / (2nu) .* (H^2 .- d .^ 2))) / umax
        # friction: in the steady state the body force over the whole fluid balances the wall force on the
        # fluid; the momentum exchange gives the force on the walls (lattice units, force per cell x cells)
        Fx_body = Fx * nfluid
        friction_err = abs(abs(st.forces[1]) - Fx_body) / Fx_body
        results[(angle, n)] = (prof_err, wall_err, friction_err)
        @printf("angle %4.1f n %3d H %5.1f: fluid %d, profile error %.3e, wall position error %+.4f cells, friction error %.3e (force %+.4e, body %+.4e; %d steps)\n",
                angle, n, H, nfluid, prof_err, wall_err, friction_err, st.forces[1], Fx_body, steps)
        @test isfinite(prof_err) && prof_err < 0.2
    end
    for angle in (0.0, 15.0, 30.0, 45.0)
        e = [results[(angle, n)][1] for n in sizes]
        order = log(e[1] / e[3]) / log(4)
        @printf("angle %4.1f: profile order %.2f (errors %s)\n", angle, order, string(round.(e; sigdigits=3)))
        @test order >= 1.8
        @test abs(results[(angle, sizes[end])][2]) < 0.05
        @test results[(angle, sizes[end])][3] < 0.01
    end
end
println("q19_octree_poiseuille: done")
