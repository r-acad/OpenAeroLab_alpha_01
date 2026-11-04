# shared sphere tessellation helpers of the geometry tests
function sphere_tris(n_lat::Int, n_lon::Int; r=0.5, c=(0.0, 0.0, 0.0))
    P(θ, φ) = (c[1] + r * sin(θ) * cos(φ), c[2] + r * sin(θ) * sin(φ), c[3] + r * cos(θ))
    tri = OctreeQ19.Tri[]
    for i in 0:n_lat-1, j in 0:n_lon-1
        θ0 = π * i / n_lat; θ1 = π * (i + 1) / n_lat; φ0 = 2π * j / n_lon; φ1 = 2π * (j + 1) / n_lon
        a = P(θ0, φ0); b = P(θ1, φ0); cc = P(θ1, φ1); d = P(θ0, φ1)
        i < n_lat - 1 && push!(tri, (a, b, cc))
        i > 0 && push!(tri, (a, cc, d))
    end
    tri
end
