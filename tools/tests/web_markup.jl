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
