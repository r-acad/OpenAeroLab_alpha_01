# Geometry queries of the CPU octree D3Q19 solver (part of gate G0.6): STL reading, BVH nearest distance against
# the analytic sphere, segment crossings (the q of a cut link), ray-parity inside test, ASCII STL. CPU, seconds.
#   julia --project=src/octree_q19 tools/tests/q19_octree_geometry.jl
using Test
using OctreeQ19
const G = OctreeQ19

# a unit sphere STL built here (icosphere-like latitude/longitude tessellation), binary and ASCII
function sphere_tris(n_lat::Int, n_lon::Int; r=0.5, c=(0.0, 0.0, 0.0))
    P(θ, φ) = (c[1] + r * sin(θ) * cos(φ), c[2] + r * sin(θ) * sin(φ), c[3] + r * cos(θ))
    tri = G.Tri[]
    for i in 0:n_lat-1, j in 0:n_lon-1
        θ0 = π * i / n_lat; θ1 = π * (i + 1) / n_lat; φ0 = 2π * j / n_lon; φ1 = 2π * (j + 1) / n_lon
        a = P(θ0, φ0); b = P(θ1, φ0); cc = P(θ1, φ1); d = P(θ0, φ1)
        i < n_lat - 1 && push!(tri, (a, b, cc))   # b == cc at the south pole
        i > 0 && push!(tri, (a, cc, d))           # a == d at the north pole
    end
    tri
end
function write_binary_stl(path, tri)
    open(path, "w") do io
        write(io, zeros(UInt8, 80)); write(io, UInt32(length(tri)))
        for t in tri
            n, _ = G.tri_normal_area(t)
            for v in (n, t[1], t[2], t[3]); write(io, Float32(v[1]), Float32(v[2]), Float32(v[3])); end
            write(io, UInt16(0))
        end
    end
end
function write_ascii_stl(path, tri)
    open(path, "w") do io
        println(io, "solid test")
        for t in tri
            println(io, " facet normal 0 0 0"); println(io, "  outer loop")
            for v in t; println(io, "   vertex $(v[1]) $(v[2]) $(v[3])"); end
            println(io, "  endloop"); println(io, " endfacet")
        end
        println(io, "endsolid test")
    end
end

scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","q19_tests");mkpath(scratch)
tmp = mktempdir(scratch)
tri0 = sphere_tris(64, 128)
write_binary_stl(joinpath(tmp, "s.stl"), tri0); write_ascii_stl(joinpath(tmp, "s_ascii.stl"), tri0)

@testset "STL reading" begin
    t1, d1 = G.read_stl(joinpath(tmp, "s.stl")); t2, d2 = G.read_stl(joinpath(tmp, "s_ascii.stl"))
    @test length(t1) == length(tri0) && d1 == 0
    @test length(t2) == length(tri0) && d2 == 0
    @test all(isapprox(t1[k][m][a], t2[k][m][a]; atol=1e-6) for k in 1:50, m in 1:3, a in 1:3)
    t3, _ = G.read_stl(joinpath(tmp, "s.stl"); offset=(1.0, 2.0, 3.0))
    @test t3[1][1][1] ≈ t1[1][1][1] + 1.0 atol=1e-6
    lo, hi = G.bounds(t1)
    @test all(isapprox.(lo, (-0.5, -0.5, -0.5); atol=1e-6)) && all(isapprox.(hi, (0.5, 0.5, 0.5); atol=1e-6))
end

tri, _ = G.read_stl(joinpath(tmp, "s.stl"))
bvh = G.build_bvh(tri)
st = G.new_stack()

@testset "nearest distance against the analytic sphere" begin
    # the tessellation error of a 64x128 sphere is below 1e-3 r
    for p in ((0.9, 0.0, 0.0), (0.0, 0.7, 0.1), (0.1, 0.1, 0.1), (0.0, 0.0, 0.0), (0.49, 0.0, 0.0), (0.3, -0.2, 0.6))
        d, ti, q = G.nearest(bvh, p, st)
        @test isapprox(d, abs(sqrt(sum(abs2, p)) - 0.5); atol=1e-3)
        @test 1 <= ti <= length(tri)
        @test isapprox(sqrt(sum(abs2, q)), 0.5; atol=1e-3)
    end
    d, ti, _ = G.nearest(bvh, (3.0, 0.0, 0.0), st; bound=1.0)
    @test ti == 0 && d == 1.0
    # brute force agreement on random points
    for _ in 1:200
        p = (randn(), randn(), randn()) .* 0.6
        d = G.nearest_distance(bvh, p, st)
        db = minimum(sqrt(G.vnorm2(G.vsub(p, G.tri_closest(p, t[1], t[2], t[3])))) for t in tri)
        @test d ≈ db atol=1e-12
    end
end

@testset "segment crossing (cut-link q)" begin
    # from inside radius 0.4 along +x to radius 0.6: the crossing is at q = (0.5 - 0.4) / 0.2 = 0.5
    q, ti = G.first_crossing(bvh, (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), st)
    @test isapprox(q, 0.5; atol=2e-3) && ti > 0
    q, ti = G.first_crossing(bvh, (0.6, 0.0, 0.0), (0.4, 0.0, 0.0), st)
    @test isapprox(q, 0.5; atol=2e-3)
    q, ti = G.first_crossing(bvh, (0.6, 0.0, 0.0), (0.8, 0.0, 0.0), st)
    @test q == 1.0 && ti == 0
    # a segment through the whole sphere: the first crossing is the near side
    q, ti = G.first_crossing(bvh, (-1.0, 0.05, 0.02), (1.0, 0.05, 0.02), st)
    @test isapprox(q, 0.25; atol=2e-3)
    # q is in (0, 1]
    for _ in 1:200
        a = (randn(), randn(), randn()) .* 0.5; b = (randn(), randn(), randn()) .* 0.5
        q, ti = G.first_crossing(bvh, a, b, st)
        ra = sqrt(sum(abs2, a)); rb = sqrt(sum(abs2, b))
        if (ra < 0.49 && rb > 0.51) || (ra > 0.51 && rb < 0.49)
            @test 0 < q <= 1 && ti > 0
        end
    end
end

@testset "ray parity inside test" begin
    @test G.is_inside(bvh, (0.0, 0.0, 0.0), st)
    @test G.is_inside(bvh, (0.3, 0.3, 0.1), st)
    @test !G.is_inside(bvh, (0.6, 0.0, 0.0), st)
    @test !G.is_inside(bvh, (0.0, 0.0, 0.7), st)
    ok = 0; tested = 0
    for _ in 1:5000
        p = (rand() - 0.5, rand() - 0.5, rand() - 0.5) .* 1.4
        r = sqrt(sum(abs2, p))
        abs(r - 0.5) < 2e-3 && continue      # inside the tessellation error band: no ground truth
        tested += 1
        ok += G.is_inside(bvh, p, st) == (r < 0.5)
    end
    @test ok == tested
end

@testset "triangle normal, area and bounds" begin
    n, a = G.tri_normal_area(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    @test n == (0.0, 0.0, 1.0) && a == 0.5
end
println("q19_octree_geometry: all tests passed")
