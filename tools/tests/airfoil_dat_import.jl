using Test
const ROOT=normpath(joinpath(@__DIR__,"../.."))
include(joinpath(ROOT,"apps/airfoil_web/src/backends/jx_backend.jl"))
const B=AirfoilJXBackend
foil=B.JXfoil.naca4_airfoil(2412;n=81)
rows=join((string(x," ",y) for (x,y) in zip(foil.x,foil.y)),"\n")
@testset "XFOIL .dat import" begin
    source=B.import_dat("My airfoil\n"*rows,"test.dat")["airfoil"]
    @test source["kind"]=="dat" && source["id"]=="test.dat"
    @test length(source["coordinates"]["x"])==81
    @test minimum(source["coordinates"]["x"])==0
    @test maximum(source["coordinates"]["x"])==1
    @test B.resolve_airfoil(source;offline=true).metadata["source_url"]===nothing
    @test B.resolve_airfoil(source;offline=true).airfoil.x==source["coordinates"]["x"]
    for text in (rows,"\ufeffMy foil\r\n"*replace(rows,"\n"=>"\r\n"),
        "# comment\n!comment\n"*rows,"My foil\n"*replace(rows,"e"=>"D"),
        join(reverse(split(rows,'\n')),"\n"))
        @test B.import_dat(text,"plain.DAT")["airfoil"]["coordinates"]["x"]≈source["coordinates"]["x"]
    end
    symmetric=B.JXfoil.naca4_airfoil(12;n=81)
    upper=collect(41:-1:1);lower=collect(41:81)
    counted="NACA0012\n41 41\n"*join((string(symmetric.x[i]," ",symmetric.y[i]) for i in upper),"\n")*"\n\n"*
        join((string(symmetric.x[i]," ",symmetric.y[i]) for i in lower),"\n")
    @test B.import_dat(counted,"two-surfaces.dat")["point_count"]==81
    scaled=join((string(3+2x," ",5+2y) for (x,y) in zip(foil.x,foil.y)),"\n")
    @test B.import_dat(scaled,"scaled.dat")["airfoil"]["coordinates"]["y"]≈source["coordinates"]["y"] atol=1e-14
    for text in ("", "Title\n1 abc\n"*rows, "Title\nNaN 0\n"*rows,
        "Title\nInf 0\n"*rows, "Title\n"*rows*"\nbad data", "\0"*rows,
        repeat("a",2*1024*1024+1),join(("1 $y" for y in 1:8),"\n"),
        join(("$x 0" for x in [1,.8,.6,.4,0,.4,.6,.8,1]),"\n"))
        @test_throws ArgumentError B.import_dat(text,"bad.dat")
    end
    @test_throws ArgumentError B.import_dat(rows,"bad.csv")
    @test_throws ArgumentError B.import_dat(rows,"bad\n0 10\nname.dat")
    @test_throws ArgumentError B.resolve_airfoil(merge(source,Dict("id"=>"bad\n0 10\nname.dat")))
    @test_throws ArgumentError B.resolve_airfoil(Dict("kind"=>"dat","id"=>"missing.dat"))
end

@testset "Imported contour is one simple polygon" begin
    repeated=rows*"\n"*rows
    crossed="1 0\n0.8 0.1\n0.2 -0.1\n0 0\n0.2 0.1\n0.8 -0.15\n0.9 -0.05\n1 0"
    @test_throws ArgumentError B.import_dat(repeated,"repeat.dat")
    @test_throws ArgumentError B.import_dat(crossed,"cross.dat")
    # Contact without a proper crossing, overlapping adjacent panels and a
    # repeated interior vertex must also be refused.
    @test_throws ArgumentError B._validate_dat_contour([1.,.8,.4,0.,.4,.8],[0.,.1,0.,0.,0.,-.1])
    @test_throws ArgumentError B._validate_dat_contour([1.,.8,.9,0.,.5],[0.,.1,.05,0.,-.1])
    @test_throws ArgumentError B._validate_dat_contour([1.,.8,0.,.8,.9],[0.,.1,0.,.1,-.1])
    # Both a finite TE and an explicit sharp closing TE are valid.
    for sharp in (false,true),count in (81,1001,24999)
        sample=B.JXfoil.naca4_airfoil(2412;n=count)
        sharp && (sample.x[end]=sample.x[1];sample.y[end]=sample.y[1])
        @test B._validate_dat_contour(sample.x,sample.y)===nothing
    end
    @test_throws ArgumentError B._validate_dat_contour(foil.x,foil.y;max_checks=1)
end
