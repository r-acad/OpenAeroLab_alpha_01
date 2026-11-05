using Test, UUIDs
include(joinpath(@__DIR__, "..", "install", "setup_windows.jl"))

function test_vcredist()
    Sys.iswindows() || return
    root = abspath(get(ENV, "LUDWIG_PRIVATE_ROOT", joinpath(@__DIR__, "..", "..", "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
    scratch = joinpath(root, "TEMP_DELETEABLE", "windows_vcredist_tests", string(uuid4()))
    mkpath(scratch)
    @testset "Windows prerequisite admission" begin
        withenv("LUDWIG_VCREDIST_INSTALLER"=>"") do
            unused = joinpath(scratch, "already installed")
            @test install_visual_cpp(; offline=true, installed=()->true, private=unused) === nothing
            @test !ispath(unused)
            @test_throws ErrorException install_visual_cpp(; offline=true, installed=()->false, private=unused)
            @test !ispath(unused)
        end
        supplied = joinpath(scratch, "IT supplied installer.exe")
        write(supplied, "not a real executable")
        executed = Ref(false)
        withenv("LUDWIG_VCREDIST_INSTALLER"=>supplied) do
            @test_throws ErrorException install_visual_cpp(; offline=true, installed=()->false,
                private=scratch, verify=path->false, execute=command->(executed[]=true))
            @test !executed[]
            @test read(supplied,String)=="not a real executable"
            passed = Ref{Any}(nothing)
            install_visual_cpp(; offline=true, installed=()->false, private=scratch,
                verify=path->path==supplied, execute=command->(passed[]=command; (exitcode=0,)))
            @test passed[] isa Cmd
            @test passed[].exec==[supplied,"/install","/quiet","/norestart"]
            @test_throws ErrorException install_visual_cpp(; offline=true, installed=()->false,
                private=scratch, verify=path->true, execute=command->(exitcode=1603,))
            # Exercise the real Windows signature query on harmless unsigned data.
            @test !microsoft_signature_valid(supplied)
        end
        calls=Ref(0)
        withenv("LUDWIG_VCREDIST_INSTALLER"=>"") do
            install_visual_cpp(; installed=()->false, private=scratch,
                download=(url,path)->begin
                    calls[]+=1
                    calls[]==1 && error("HTTP status 503 temporarily unavailable")
                    write(path,"download fixture")
                end,
                verify=path->read(path,String)=="download fixture", execute=command->(exitcode=3010,))
            @test calls[]==2
            @test !isempty(readdir(joinpath(scratch,"runtime","install_logs")))
        end
    end
    println("Disposable prerequisite fixtures: $scratch")
end
test_vcredist()
