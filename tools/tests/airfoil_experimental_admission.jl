using Test
const REPOSITORY=dirname(dirname(@__DIR__))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT"))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","experimental_admission_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
ENV["AIRFOIL_WEB_WARM_PREVIEW"]="0"
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb
function request(mesh)
    Dict{String,Any}("airfoil"=>Dict("kind"=>"naca4","id"=>"2412"),"backends"=>["transonic"],
        "flow"=>Dict("mach"=>.3,"reynolds"=>3e6,"alpha_deg"=>4.,"cfd_te_topology"=>"aplus",
            "cfd_mesh"=>mesh,"rans_method"=>"jfnk_native","cfd_implicit_globalization"=>"pseudo_transient",
            "cfd_implicit_basis"=>60,"cfd_implicit_linear_maxiter"=>120,"cfd_implicit_cfl"=>10.,"cfd_implicit_max_cfl"=>1e3))
end
@testset "Every local asset referenced by the actual page is served" begin
    html=read(joinpath(REPOSITORY,"apps","airfoil_web","static","index.html"),String)
    paths=unique([String(m.captures[1]) for m in eachmatch(r"(?:src|href)=\"(/static/[^\"]+)\"",html)])
    @test "/static/scripts/research_settings.js" in paths
    for path in paths
        response=A.app_handler(A.HTTP.Request("GET",path))
        @test response.status==200
        @test !isempty(response.body)
    end
end
@testset "HTTP intake preserves experimental implicit profile and budgets" begin
    for name in ("aplus_uniform_quality","aplus_uniform_fixed")
        accepted=A._validated_case(request(name))
        @test accepted["flow"]["cfd_mesh"]==name
        @test accepted["flow"]["cfd_te_topology"]=="aplus"
        @test accepted["flow"]["rans_method"]=="jfnk_native"
        cfg=Dict{String,Any}("steps"=>5000,"time_mode"=>"steady")
        A.CFDBackend.RANSAcceleration.configure_compressible!(cfg,accepted["flow"])
        @test cfg["implicit_basis"]==60&&cfg["implicit_linear_maxiter"]==120
        @test cfg["implicit_globalization"]=="pseudo_transient"
        automatic=request(name);delete!(automatic["flow"],"cfd_te_topology")
        single_choice=A._validated_case(automatic)
        @test single_choice["flow"]["cfd_te_topology"]=="aplus"
        @test !A.CFDBackend._auto_ch_selected(single_choice,"unused_for_explicit_profile.dat",single_choice["flow"])
        roundtrip=A.JSON3.read(A.JSON3.write(A._clean(single_choice)),Dict{String,Any})
        @test A._validated_case(roundtrip)["flow"]["cfd_mesh"]==name
    end
    @test_throws ArgumentError A._validated_case(request("unknown_profile"))
    # Existing studies keep their exact requested mesh family; no default is promoted here.
    for name in ("cavitation_aplus_farfield","aplus_nested_refined","legacy_sharp_cmesh")
        @test A._validated_case(request(name))["flow"]["cfd_mesh"]==name
    end
end
