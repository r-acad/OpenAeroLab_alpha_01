# julia --startup-file=no --project=src tools/tests/cmesh_marched_mesh.jl
# The opt-in marched airfoil mesh (src/cmesh/cmesh_marched.jl): a sharp section (one C block), a blunt section (C block
# plus strip block, C+H) and the wake first-row rule. Geometry checks only.
using Test
include(joinpath(@__DIR__,"..","..","src","cmesh","cmesh_gen.jl"))

"NACA 0012 thickness form; `cut` < 1 truncates the section there (a blunt base) and rescales it to unit chord."
function foil(;cut=1.0,n=40)
    yt(x)=5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    xs=[cut*.5*(1-cospi(k/n)) for k in 0:n];y=[x==1.0 ? 0.0 : yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end])./cut,vcat(-reverse(y),y[2:end])./cut)
end
area(X,Y,i,j)=0.5*((X[i+1,j+1]-X[i,j])*(Y[i,j+1]-Y[i+1,j])-(X[i,j+1]-X[i+1,j])*(Y[i+1,j+1]-Y[i,j]))
function convex(X,Y)
    for j in 1:size(X,2)-1,i in 1:size(X,1)-1
        px=(X[i,j],X[i+1,j],X[i+1,j+1],X[i,j+1]);py=(Y[i,j],Y[i+1,j],Y[i+1,j+1],Y[i,j+1])
        for k in 1:4
            a=k;b=k%4+1;c=(k+1)%4+1
            (px[b]-px[a])*(py[c]-py[b])-(py[b]-py[a])*(px[c]-px[b])>0||return false
        end
    end
    true
end
aspect(X,Y,i,j)=(a=hypot(X[i+1,j]-X[i,j],Y[i+1,j]-Y[i,j]);b=hypot(X[i,j+1]-X[i,j],Y[i,j+1]-Y[i,j]);max(a/b,b/a))
kw=(;n_foil=96,n_wake=24,n_normal=32,dn1=2e-4,alpha_deg=2.0,far=20.0)

@testset "sharp trailing edge: one C block" begin
    r=CMeshGen.marched_airfoil_mesh(foil();kw...);m=r.mesh
    @test m.strip===nothing
    @test (m.NI,m.NJ,m.NIW,m.NIF)==(144,32,24,96)&&size(m.X)==(145,33)
    @test convex(m.X,m.Y)&&r.info.area_min>0
    # the two wake rows are one line: node i of the lower row is node NI+2-i of the upper row
    @test all((m.X[i,1],m.Y[i,1])==(m.X[m.NI+2-i,1],m.Y[m.NI+2-i,1]) for i in 1:m.NIW+1)
    # one wall-normal law: every surface column has the wall spacing in its first cell
    h1=[hypot(m.X[i,2]-m.X[i,1],m.Y[i,2]-m.Y[i,1]) for i in m.NIW+1:m.NIW+m.NIF+1]
    @test maximum(abs.(h1.-kw.dn1))<1e-12
    # wake rule: the first-row aspect ratio at the outlet falls to about the cap, cells stay convex
    r2=CMeshGen.marched_airfoil_mesh(foil();kw...,wake_aspect=100.0).mesh
    @test convex(r2.X,r2.Y)
    @test aspect(m.X,m.Y,1,1)>20*aspect(r2.X,r2.Y,1,1)
    @test aspect(r2.X,r2.Y,1,1)<150
    # the surface nodes and the first surface rows do not depend on the wake rule
    w=m.NIW+1:m.NIW+m.NIF+1
    @test m.X[w,1]==r2.X[w,1]&&m.Y[w,1]==r2.Y[w,1]
end

@testset "blunt trailing edge: C block and strip (C+H)" begin
    r=CMeshGen.marched_airfoil_mesh(foil(cut=0.95);kw...,wake_aspect=100.0);m=r.mesh;s=m.strip
    @test s!==nothing&&s.NI==m.NIW&&s.NJ%4==0&&s.gap>0.01
    @test convex(m.X,m.Y)&&convex(s.X,s.Y)&&r.info.area_min>0
    # the strip's edge nodes are the C block's wake nodes, bit for bit
    @test all((s.X[k,1],s.Y[k,1])==(m.X[m.NIW+2-k,1],m.Y[m.NIW+2-k,1]) for k in 1:s.NI+1)
    @test all((s.X[k,end],s.Y[k,end])==(m.X[m.NIW+m.NIF+k,1],m.Y[m.NIW+m.NIF+k,1]) for k in 1:s.NI+1)
    # the base (column 1 of the strip) joins the two ends of the surface
    @test hypot(s.X[1,end]-s.X[1,1],s.Y[1,end]-s.Y[1,1])≈s.gap rtol=1e-12
    # cell height is continuous across the legs: strip edge cell against the C block's first row, every station
    for k in (1,s.NI÷2,s.NI)
        hs=hypot(s.X[k,2]-s.X[k,1],s.Y[k,2]-s.Y[k,1]);hc=hypot(m.X[m.NIW+2-k,2]-m.X[m.NIW+2-k,1],m.Y[m.NIW+2-k,2]-m.Y[m.NIW+2-k,1])
        @test isapprox(hs,hc;rtol=0.02)
    end
    # the strip widens downstream with the wake rule and keeps the base height without it
    @test hypot(s.X[end,end]-s.X[end,1],s.Y[end,end]-s.Y[end,1])>2s.gap
    s0=CMeshGen.marched_airfoil_mesh(foil(cut=0.95);kw...).mesh.strip
    @test hypot(s0.X[end,end]-s0.X[end,1],s0.Y[end,end]-s0.Y[end,1])≈s0.gap rtol=2e-3    # the legs are parallel; the base may be slightly inclined to them
end
