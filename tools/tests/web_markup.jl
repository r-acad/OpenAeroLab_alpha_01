using Test
const ROOT=normpath(joinpath(@__DIR__,"../.."))
@testset "Workbench HTML controls have unique IDs" begin
    for relative in ("apps/airfoil_web/static/index.html","apps/ludwig_web/static/index.html",
                     "apps/ludwig_web/static/aero_lab.html","apps/ludwig_web/static/solver_catalog.html")
        html=read(joinpath(ROOT,relative),String)
        ids=[m.captures[1] for m in eachmatch(r"\bid\s*=\s*\"([^\"]+)\"",html)]
        counts=Dict(id=>count(==(id),ids) for id in unique(ids))
        duplicates=sort!([id for (id,n) in counts if n>1])
        @testset "$relative" begin
            @test !isempty(ids)
            @test isempty(duplicates)
        end
    end
end

@testset "Low-speed C+H baseline control is wired into the form" begin
    html=read(joinpath(ROOT,"apps/airfoil_web/static/index.html"),String)
    script=read(joinpath(ROOT,"apps/airfoil_web/static/scripts/app.js"),String)
    @test occursin("id=\"lowspeed-ch-baseline\"",html)
    @test occursin("lowspeed_topology:\"ch_native\"",script)
    @test occursin("lowspeed_start:\"ramp\"",script)
    @test occursin("lowspeed_turb_mut_base:\"committed\"",script)
    @test occursin("id=\"lowspeed-steady-tol\"",html)
    @test occursin("id=\"lowspeed-turb-steady-tol\"",html)
end
