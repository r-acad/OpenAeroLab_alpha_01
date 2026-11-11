using Test,UUIDs
const PUBLIC=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(PUBLIC,"tools","install","scratch_path.jl"))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(PUBLIC,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","windows_scratch_tests")
mkpath(SCRATCH)
@testset "Windows setup scratch keeps its real private directory" begin
    mktempdir(SCRATCH) do root
        private=joinpath(root,"private with spaces");other=joinpath(root,"other");mkpath(other)
        withenv("TEMP"=>"before-temp","TMP"=>"before-tmp","TMPDIR"=>"before-tmpdir") do
            result=OpenAeroLabScratch.prepare_windows_scratch!(;private,io=IOBuffer())
            @test result.canonical==joinpath(abspath(private),"TEMP_DELETEABLE","setup")
            @test isdir(result.canonical)&&Base.Filesystem.samefile(result.path,result.canonical)
            @test all(ENV[key]==result.path for key in ("TEMP","TMP","TMPDIR"))
            if Sys.iswindows()
                path=OpenAeroLabScratch.windows_short_path(result.canonical)
                @test path!==nothing&&Base.Filesystem.samefile(path,result.canonical)
            end
            fallback=OpenAeroLabScratch.choose_scratch_path(result.canonical;shortener=_ -> nothing)
            @test fallback.path==result.canonical&&!fallback.shortened
            invalid=OpenAeroLabScratch.choose_scratch_path(result.canonical;shortener=_ -> other)
            @test invalid.path==result.canonical&&!invalid.shortened
            absent=OpenAeroLabScratch.choose_scratch_path(result.canonical;shortener=_ -> joinpath(root,"absent"))
            @test absent.path==result.canonical&&!absent.shortened
        end
    end
end
