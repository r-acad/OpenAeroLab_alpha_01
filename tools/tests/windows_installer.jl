#!/usr/bin/env julia
# Tests the Windows-native bootstrap without changing machine policy, installing
# packages, or using a network connection. Optional full offline runtime test:
#   set LUDWIG_TEST_JULIA_ARCHIVE=C:\path\julia-1.13.2-win64.zip
#   julia --startup-file=no tools/tests/windows_installer.jl
using Test, UUIDs, SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const PRIVATE = abspath(get(ENV, "LUDWIG_PRIVATE_ROOT", joinpath(REPO, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH = joinpath(PRIVATE, "TEMP_DELETEABLE", "windows_installer_tests", string(uuid4()))
const EXPECTED = strip(read(joinpath(REPO, "tools", "install", "julia_version.txt"), String))

function fixture_repo(name)
    dest = joinpath(SCRATCH, name)
    install = joinpath(dest, "tools", "install")
    mkpath(install)
    cp(joinpath(REPO, "setup.cmd"), joinpath(dest, "setup.cmd"))
    for file in ("resolve_julia.cmd", "julia_version.txt", "runtimes.tsv", "install_julia.cmd", "promote_runtime.jl")
        source=joinpath(REPO,"tools","install",file)
        isfile(source) && cp(source,joinpath(install,file))
    end
    if get(ENV,"LUDWIG_TEST_CMD_EOL","")=="lf"
        for file in (joinpath(dest,"setup.cmd"),joinpath(install,"resolve_julia.cmd"),joinpath(install,"install_julia.cmd"))
            write(file,replace(read(file,String),"\r\n"=>"\n"))
        end
    end
    # Real Julia executes this stub, proving runtime discovery and validation,
    # while keeping actual package installation out of the isolated regression.
    write(joinpath(install,"setup_windows.jl"), """
    println("INSTALL_TEST_JULIA=", VERSION)
    println("INSTALL_TEST_ARGS=", join(ARGS, "|"))
    """)
    dest
end

function invoke_cmd(file, options; env=Pair{String,String}[])
    wrapper = joinpath(SCRATCH, "invoke_$(uuid4()).cmd")
    path = replace(file, "%"=>"%%")
    write(wrapper,"@echo off\r\nsetlocal DisableDelayedExpansion\r\ncall \"$path\" $(join(options, ' '))\r\nexit /b %ERRORLEVEL%\r\n")
    command=Cmd(Cmd([get(ENV,"COMSPEC",raw"C:\Windows\System32\cmd.exe"),"/d","/c",wrapper]);ignorestatus=true,windows_hide=true)
    common=["LUDWIG_WEB_NO_PAUSE"=>"1", "LUDWIG_JULIA"=>"", "AIRFOIL_JULIA"=>"",
        "LUDWIG_JULIA_ARCHIVE"=>"", "LUDWIG_PRIVATE_ROOT"=>joinpath(SCRATCH,"private runtime (spaces)"),
        "LOCALAPPDATA"=>joinpath(SCRATCH,"empty localappdata"),
        "PATH"=>joinpath(ENV["SystemRoot"],"System32")*";"*ENV["SystemRoot"]]
    io=IOBuffer()
    process=run(pipeline(addenv(command, common..., env...);stdout=io,stderr=io))
    output=String(take!(io))
    write(wrapper*".log",output)
    process.exitcode,output
end
invoke_setup(repo,options;kw...)=invoke_cmd(joinpath(repo,"setup.cmd"),options;kw...)

function main()
    if !Sys.iswindows()
        @info "Native Windows installer checks require Windows."
        return
    end
    mkpath(SCRATCH)
    repo=fixture_repo("OpenAeroLab_alpha_01-main (company download)")
    known=joinpath(Sys.BINDIR,"julia.exe")
    @testset "Native Windows setup" begin
        code,output=invoke_setup(repo,["--help"])
        @test code==0
        @test occursin("--install-julia",output)
        @test occursin("--offline",output)
        code,output=invoke_setup(repo,["--no-install-julia"])
        @test code!=0
        @test !occursin("INSTALL_TEST_JULIA=",output)
        code,output=invoke_setup(repo,["--offline"])
        @test code!=0
        @test !occursin("Downloading Julia",output)
        code,output=invoke_setup(repo,["--install-julia","--no-install-julia"])
        @test code!=0
        code,output=invoke_setup(repo,["--no-install-julia","--dry-run"];env=["LUDWIG_JULIA"=>known])
        @test code==0
        @test occursin("INSTALL_TEST_JULIA=$EXPECTED",output)
        @test occursin("INSTALL_TEST_ARGS=--dry-run",output)
        @test !isdir(joinpath(SCRATCH,"private runtime (spaces)","runtime"))
        code,output=invoke_setup(repo,["--install-julia"];env=["LUDWIG_JULIA"=>joinpath(SCRATCH,"missing julia.exe")])
        @test code!=0
        @test !occursin("Downloading Julia",output)
    end
    @testset "Offline runtime rejection preserves existing files" begin
        helper=joinpath(repo,"tools","install","install_julia.cmd")
        @test isfile(helper)
        private=joinpath(SCRATCH,"failed installation (keep original)")
        runtime=joinpath(private,"runtime","julia-$EXPECTED")
        mkpath(runtime)
        sentinel=joinpath(runtime,"preserve_unique_user_file.txt")
        write(sentinel,"unique existing runtime data")
        invalid=joinpath(SCRATCH,"proxy error response.zip")
        write(invalid,"<html>company proxy denied this request</html>")
        code,output=invoke_cmd(helper,["--offline"];env=["LUDWIG_PRIVATE_ROOT"=>private,"LUDWIG_JULIA_ARCHIVE"=>invalid])
        @test code!=0
        @test read(sentinel,String)=="unique existing runtime data"
        @test !isfile(joinpath(runtime,"julia-$EXPECTED","bin","julia.exe"))
        @test !occursin("Downloading Julia",output)
        code,output=invoke_cmd(helper,["--offline"];env=["LUDWIG_PRIVATE_ROOT"=>private,"LUDWIG_JULIA_ARCHIVE"=>joinpath(SCRATCH,"missing.zip")])
        @test code!=0
        @test isfile(sentinel)
        @test !occursin("Downloading Julia",output)
    end
    @testset "Concurrent installer lock" begin
        private=joinpath(SCRATCH,"locked installation")
        mkpath(joinpath(private,"runtime"))
        lockfile=joinpath(private,"runtime","julia-install.lock")
        ready=joinpath(SCRATCH,"lock_ready.txt")
        release=joinpath(SCRATCH,"lock_release.txt")
        holdjulia=joinpath(SCRATCH,"hold_lock.jl")
        write(holdjulia,"deadline=time()+15; while !isfile(ARGS[1]) && time()<deadline; sleep(0.05); end\n")
        holdercmd=joinpath(SCRATCH,"hold_lock.cmd")
        write(holdercmd,"@echo off\r\ncall :hold 9>\"$lockfile\"\r\nexit /b %ERRORLEVEL%\r\n:hold\r\necho ready>\"$ready\"\r\n\"$known\" --startup-file=no \"$holdjulia\" \"$release\"\r\nexit /b 0\r\n")
        command=Cmd(Cmd([get(ENV,"COMSPEC",raw"C:\Windows\System32\cmd.exe"),"/d","/c",holdercmd]);ignorestatus=true,windows_hide=true)
        open(joinpath(SCRATCH,"held_lock.log"),"w") do log
            holder=run(pipeline(command,stdout=log,stderr=log);wait=false)
            try
                deadline=time()+5
                while !isfile(ready) && time()<deadline; sleep(0.05); end
                @test isfile(ready)
                code,output=invoke_cmd(joinpath(repo,"tools","install","install_julia.cmd"),["--offline"];env=["LUDWIG_PRIVATE_ROOT"=>private])
                @test code!=0
                @test occursin("Another Julia installation",output)
                @test !isdir(joinpath(private,"TEMP_DELETEABLE"))
            finally
                write(release,"release")
                wait(holder)
            end
            @test holder.exitcode==0
        end
    end
    archive=get(ENV,"LUDWIG_TEST_JULIA_ARCHIVE","")
    if !isempty(archive)
        @testset "Pinned real Julia ZIP, local offline installation" begin
            record=only(filter(line->startswith(line,"windows-x86_64|"),readlines(joinpath(REPO,"tools","install","runtimes.tsv"))))
            expected=split(record,'|')[3]
            @test bytes2hex(open(sha256,archive)) == expected
            private=joinpath(SCRATCH,"company downloads (spaces)")
            runtime=joinpath(private,"runtime","julia-$EXPECTED")
            mkpath(runtime)
            write(joinpath(runtime,"preserve_original_partial_runtime.txt"),"original interrupted installation")
            helper=joinpath(repo,"tools","install","install_julia.cmd")
            code,output=invoke_cmd(helper,["--offline"];env=["LUDWIG_PRIVATE_ROOT"=>private,"LUDWIG_JULIA_ARCHIVE"=>abspath(archive)])
            code==0 || println(output)
            @test code==0
            managed=joinpath(runtime,"julia-$EXPECTED","bin","julia.exe")
            @test isfile(managed)
            @test strip(read(`$managed --startup-file=no --version`,String))=="julia version $EXPECTED"
            @test isfile(joinpath(runtime,"julia-$EXPECTED","LICENSE.md"))
            @test maximum(length(joinpath(dir,file)) for (dir,_,files) in walkdir(runtime) for file in files)>260
            preserved=[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(private,"runtime")) for file in files if file=="preserve_original_partial_runtime.txt"]
            @test length(preserved)==1
            @test read(only(preserved),String)=="original interrupted installation"
            timestamp=mtime(managed)
            code,output=invoke_cmd(helper,["--offline"];env=["LUDWIG_PRIVATE_ROOT"=>private])
            @test code==0
            @test mtime(managed)==timestamp
            code,output=invoke_setup(repo,["--offline","--no-install-julia"];env=["LUDWIG_PRIVATE_ROOT"=>private])
            @test code==0
            @test occursin("INSTALL_TEST_JULIA=$EXPECTED",output)
        end
    end
    println("Disposable test files: ",SCRATCH)
end
main()
