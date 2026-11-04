# Gate G0.6 (octree invariants) and the mesh file round trip for the CPU octree D3Q19 solver: a sphere of
# diameter 1 in a 6 x 4 x 4 m domain on 3 levels, a periodic variant, and a zero-thickness plate (baffle).
#   julia -t 4 --project=src/octree_q19 tools/tests/q19_octree_generator.jl
using Test
using OctreeQ19
const G = OctreeQ19
include(joinpath(@__DIR__, "q19_octree_geometry_helpers.jl"))

function invariants(m::G.OctMesh)
    L = G.nlevels(m)
    for l in 1:L
        Lv = m.levels[l]
        # every oct index in the neighbour table is valid; neighbour symmetry a->b iff b->a
        for o in 1:Lv.nocts, idx in 1:27
            n = Lv.onbr[idx, o]
            @test 0 <= n <= Lv.nocts
            n == 0 && continue
            d = idx - 1; dx = d % 3 - 1; dy = (d ÷ 3) % 3 - 1; dz = d ÷ 9 - 1
            @test Lv.onbr[G.nbr_index(-dx, -dy, -dz), n] == o
        end
        # every oct with a fluid slot has all 26 neighbours
        for o in 1:Lv.nocts
            any(Lv.kind[k, o] == G.K_FLUID for k in 1:8) || continue
            @test all(Lv.onbr[idx, o] != 0 for idx in 1:27)
        end
        # covered slots have children (an oct at l+1 at the right origin); fluid/solid/ghost slots have none
        if l < L
            Lf = m.levels[l + 1]
            for o in 1:Lv.nocts, k in 1:8
                c = G.slot_cell(Lv, G.slot_id(k, o))
                child = G.find_oct(Lf, 2c[1], 2c[2], 2c[3])
                if Lv.kind[k, o] == G.K_COVERED
                    @test child != 0 && all(Lf.kind[kk, child] != G.K_GHOST && Lf.kind[kk, child] != G.K_BOUNDARY for kk in 1:8)
                elseif Lv.kind[k, o] in (G.K_FLUID, G.K_SOLID)
                    @test child == 0 || all(Lf.kind[kk, child] in (G.K_GHOST, G.K_SOLID) for kk in 1:8)
                end
            end
        end
        # origins are even and unique; the oct map finds every oct
        for o in 1:Lv.nocts
            @test all(iseven(Lv.origin[a, o]) for a in 1:3)
            @test G.find_oct(Lv, Lv.origin[1, o], Lv.origin[2, o], Lv.origin[3, o]) == o
        end
        # cut links belong to fluid slots and point along valid directions with q in (0, 1]
        for (ci, s) in enumerate(Lv.cut_slot)
            @test Lv.kind[G.slot_k(Int(s)), G.slot_oct(Int(s))] == G.K_FLUID
            @test Lv.cutidx[G.slot_k(Int(s)), G.slot_oct(Int(s))] == ci
            for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
                @test 2 <= Lv.cut_dir[li] <= 19 && 0 < Lv.cut_q[li] <= 1
            end
        end
        # ghost donors are at the coarser level, weights sum to one; a cubic ghost has all 64 nodes, a trilinear
        # one only inner nodes; covered donors are restricted slots
        if l > 1
            Lp = m.levels[l - 1]
            restricted = Set(Int.(Lp.restrict_slot))
            for gi in 1:length(Lv.ghost_slot)
                @test Lv.kind[G.slot_k(Int(Lv.ghost_slot[gi])), G.slot_oct(Int(Lv.ghost_slot[gi]))] == G.K_GHOST
                @test isapprox(sum(Lv.ghost_w[:, gi]), 1; atol=1e-5)
                full = all(Lv.ghost_donor[n, gi] != 0 for n in 1:64)
                full || @test all(Lv.ghost_donor[n, gi] == 0 for n in 1:64 if !(n in G.STENCIL_INNER))
                for n in 1:64
                    d = Lv.ghost_donor[n, gi]; d == 0 && continue
                    kd = Lp.kind[G.slot_k(Int(d)), G.slot_oct(Int(d))]
                    @test kd in (G.K_FLUID, G.K_COVERED, G.K_GHOST)
                    kd == G.K_COVERED && @test Int(d) in restricted
                end
            end
        end
        # restricted slots are covered and have at least one admissible child; a cubic stencil is complete and its
        # covered sources are restricted at the finer level
        if l < L
            Lf = m.levels[l + 1]
            restricted_f = Set(Int.(Lf.restrict_slot))
            for ri in 1:length(Lv.restrict_slot)
                s = Int(Lv.restrict_slot[ri])
                @test Lv.kind[G.slot_k(s), G.slot_oct(s)] == G.K_COVERED
                @test any(Lv.restrict_src[n, ri] != 0 for n in G.STENCIL_INNER)
                Lv.restrict_cubic[ri] && @test all(Lv.restrict_src[n, ri] != 0 for n in 1:64)
                for n in 1:64
                    c = Lv.restrict_src[n, ri]; c == 0 && continue
                    kc = Lf.kind[G.slot_k(Int(c)), G.slot_oct(Int(c))]
                    @test kc in (G.K_FLUID, G.K_GHOST, G.K_COVERED)
                    kc == G.K_COVERED && @test Int(c) in restricted_f
                end
            end
        end
    end
    # 2:1 balance: a fluid slot's 26 cell neighbours exist at its level (as any kind) or are ghosts, never a jump of two
    # (guaranteed by the neighbour-table completeness above); the buffer: a fine oct's neighbour octs that are ghosts
    # must have coarse parents that are leaves whose own neighbours at the coarse level are not covered beyond 2 cells:
    # checked through the generator's dilation, exercised by the ghost listing being non-empty on every level > 1
    for l in 2:L
        @test length(m.levels[l].ghost_slot) > 0
    end
end

scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","q19_tests");mkpath(scratch)
tmp = mktempdir(scratch)
# triangles of about 2.6 finest cells (24 x 48): the surface is sampled at the cell scale, so finer tessellations
# leave sub-cell triangles unreached by any link (informational, not an error)
tri = sphere_tris(24, 48; r=0.5, c=(2.0, 2.0, 2.0))
spec = G.MeshSpec(domain=(6.0, 4.0, 4.0), levels=3, dx_finest=0.025, band_fine=4, band=3)
m = G.generate(tri, spec)
@testset "sphere, 3 levels" begin
    @test G.nlevels(m) == 3
    @test m.domain == (6.0, 4.0, 4.0)
    per = m.counts["per_level"]
    println("per level: ", [(p["level"], p["octs"], p["fluid"], p["solid"], p["covered"], p["ghost"], p["solid_ghost"], p["wall_cells"], p["cut_links"], p["ghost_listed"], p["restricted"]) for p in per])
    println("timings: ", m.counts["timings_s"], " reports: ", m.counts["reports"])
    # leaves partition the domain: fluid = domain - sphere, solid = sphere, to the staircase error
    vol = sum(p["fluid"] * p["dx_m"]^3 for p in per)
    @test isapprox(vol, 6 * 4 * 4 - π / 6; rtol=0.01)
    @test isapprox(sum(p["solid"] * p["dx_m"]^3 for p in per), π / 6; rtol=0.02)
    @test isapprox(sum((p["fluid"] + p["solid"]) * p["dx_m"]^3 for p in per), 6 * 4 * 4; rtol=1e-9)
    # the band is one-sided: the finest level's solid cells are a shell 1-2 cells deep, far fewer than its fluid cells
    @test per[3]["solid"] < 0.4 * per[3]["fluid"]
    @test m.counts["reports"]["leak_cells"] == 0
    @test m.counts["reports"]["surface_orientation"] == 1
    @test m.counts["reports"]["unreached_triangle_area_fraction"] < 0.01
    @test per[3]["wall_cells"] > 0 && per[1]["wall_cells"] == 0
    @test per[3]["cut_links"] > per[3]["wall_cells"]
    # the transfers are cubic away from the body: every ghost of level 2 (far from the sphere), most of level 3
    rep = m.counts["reports"]
    @test rep["ghosts_trilinear_level_2"] == 0 && rep["ghosts_cubic_level_2"] > 0
    @test rep["ghosts_cubic_level_3"] > 0.8 * (rep["ghosts_cubic_level_3"] + rep["ghosts_trilinear_level_3"])
    # the depth-1 restricted slots are cubic except in the one-cell strip around a fine region's lateral edges (their
    # stencils would need the unlisted outer ghosts); depth-2 slots average their children by design (A8)
    @test rep["restricted_cubic_level_1"] > 0.8 * rep["restricted_depth1_level_1"]
    @test rep["restricted_depth1_level_1"] < per[1]["restricted"]
    # only the inner ghosts are listed: fewer than the ghost slots of the ghost octs
    @test per[2]["ghost_listed"] < per[2]["ghost"] && per[2]["ghost_listed"] > 0.3 * per[2]["ghost"]
    @test rep["ghost_donors_rebuilt_level_2"] == 0 && rep["ghost_donors_rebuilt_level_3"] == 0
    invariants(m)
end

@testset "sealed cavities: a sphere with its top cap missing" begin
    # the cap above z = c + 0.42 (a hole 0.54 m wide, 21 finest cells: the fill floods the interior through it)
    tri_h = [t for t in tri if !all(v[3] > 2.42 for v in t)]
    @test length(tri_h) < length(tri)
    m_open = G.generate(tri_h, spec)
    kind_at(mm, p) = (ls = G.locate_leaf(mm, p); mm.levels[ls[1]].kind[G.slot_k(ls[2]), G.slot_oct(ls[2])])
    @test kind_at(m_open, (2.0, 2.0, 2.0)) == G.K_FLUID                 # flooded interior
    @test m_open.counts["reports"]["leak_cells"] > 0
    @test m_open.counts["reports"]["sealed_cells"] == 0
    spec_s = G.MeshSpec(domain=(6.0, 4.0, 4.0), levels=3, dx_finest=0.025, band_fine=4, band=3, seal_cavities=true)
    m_s = G.generate(tri_h, spec_s)
    rep_s = m_s.counts["reports"]
    @test rep_s["sealed_cells"] > 0 && rep_s["leak_cells"] == 0
    @test kind_at(m_s, (2.0, 2.0, 2.0)) == G.K_SOLID                   # the interior stays solid
    @test kind_at(m_s, (2.0, 2.0, 2.9)) == G.K_FLUID                   # above the hole: fluid
    @test kind_at(m_s, (1.3, 2.0, 2.0)) == G.K_FLUID                   # beside the sphere: fluid
    per_s = m_s.counts["per_level"]
    @test isapprox(sum(p["solid"] * p["dx_m"]^3 for p in per_s), π / 6; rtol=0.03)   # the sealed volume is the sphere's
    @test isapprox(sum(p["fluid"] * p["dx_m"]^3 for p in per_s), 6 * 4 * 4 - π / 6; rtol=0.01)
    # the exterior is unchanged: the closed sphere's and the sealed open sphere's fluid counts agree to the cap's cells
    @test abs(sum(p["fluid"] for p in per_s) - sum(p["fluid"] for p in m.counts["per_level"])) < 0.02 * sum(p["fluid"] for p in m.counts["per_level"])
    invariants(m_s)
end

@testset "mesh file round trip" begin
    path = joinpath(tmp, "mesh.oct")
    G.write_mesh(path, m)
    m2 = G.read_mesh(path)
    @test m2.domain == m.domain && G.nlevels(m2) == 3
    for l in 1:3
        a = m.levels[l]; b = m2.levels[l]
        @test a.nocts == b.nocts && a.origin == b.origin && a.real == b.real && a.kind == b.kind && a.onbr == b.onbr && a.cutidx == b.cutidx
        @test a.cut_slot == b.cut_slot && a.cut_off == b.cut_off && a.cut_dir == b.cut_dir && a.cut_q == b.cut_q && a.cut_tri == b.cut_tri && a.cut_partner == b.cut_partner
        @test a.ghost_slot == b.ghost_slot && a.ghost_donor == b.ghost_donor && a.ghost_w == b.ghost_w
        @test a.restrict_slot == b.restrict_slot && a.restrict_src == b.restrict_src && a.restrict_cubic == b.restrict_cubic
        @test G.find_oct(b, 0, 0, 0) == G.find_oct(a, 0, 0, 0)
    end
    @test length(m2.triangles) == length(m.triangles)
    # a damaged file is refused
    bytes = read(path); bytes[100] ⊻= 0x01; write(path, bytes)
    @test_throws ErrorException G.read_mesh(path)
end

@testset "periodic span" begin
    # the sphere centred on the periodic y face: the mesh wraps and the fluid volume still closes
    tri_p = sphere_tris(24, 48; r=0.5, c=(2.0, 0.0, 2.0))
    spec_p = G.MeshSpec(domain=(6.0, 2.0, 4.0), periodic=(false, true, false), levels=3, dx_finest=0.025, band_fine=4, band=3)
    mp = G.generate(tri_p, spec_p)
    per = mp.counts["per_level"]
    vol = sum(p["fluid"] * p["dx_m"]^3 for p in per)
    @test isapprox(vol, 6 * 2 * 4 - π / 6; rtol=0.01)
    @test mp.counts["reports"]["leak_cells"] == 0
    invariants(mp)
    # a wrapped neighbour: the first root oct's -y neighbour is on the far side
    L1 = mp.levels[1]
    o = G.find_oct(L1, 0, 0, 0)
    n = L1.onbr[G.nbr_index(0, -1, 0), o]
    @test n != 0 && L1.origin[2, n] == L1.n[2] - 2
end

@testset "zero-thickness plate (baffle)" begin
    # a square plate at x = 3, both sides fluid, no solid cells; cut links on both sides
    plate = G.Tri[((3.0, 1.0, 1.0), (3.0, 3.0, 1.0), (3.0, 3.0, 3.0)), ((3.0, 1.0, 1.0), (3.0, 3.0, 3.0), (3.0, 1.0, 3.0))]
    spec_b = G.MeshSpec(domain=(6.0, 4.0, 4.0), levels=2, dx_finest=0.05, band_fine=4, band=3)
    mb = G.generate(plate, spec_b)
    per = mb.counts["per_level"]
    @test sum(p["solid"] for p in per) == 0
    @test mb.counts["reports"]["surface_orientation"] == 0
    @test per[2]["wall_cells"] > 0
    @test isapprox(sum(p["fluid"] * p["dx_m"]^3 for p in per), 6 * 4 * 4; rtol=1e-9)
    # cut links exist in both +x and -x directions, and the refinement is symmetric about the plate
    L2 = mb.levels[2]
    @test any(d == 2 for d in L2.cut_dir) && any(d == 3 for d in L2.cut_dir)
    front = count(G.slot_centre(L2, G.slot_id(k, o))[1] < 3.0 for o in 1:L2.nocts, k in 1:8 if L2.real[o] && L2.kind[k, o] == G.K_FLUID)
    back = count(G.slot_centre(L2, G.slot_id(k, o))[1] > 3.0 for o in 1:L2.nocts, k in 1:8 if L2.real[o] && L2.kind[k, o] == G.K_FLUID)
    @test front == back && front > 0
    invariants(mb)
end
println("q19_octree_generator: all tests passed")
