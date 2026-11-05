using Test, TOML

const REPOSITORY=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(REPOSITORY,"tools","install","setup.jl"))
const Setup=OpenAeroLabSetup
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(REPOSITORY,"..","02_PRIVATE_PROJECT_DEVELOPMENT"))),"TEMP_DELETEABLE","installer_tests")
mkpath(SCRATCH)

@testset "Package transfer retry policy" begin
    for message in ("HTTP/2 503", "Error status 429", "Connection reset by peer", "Operation timed out", "Could not resolve host")
        @test Setup.transient_download_error(ErrorException(message))
    end
    for message in ("certificate verification failed", "Error status 407: Proxy authentication required", "HTTP/2 403", "Permission denied", "Checksum mismatch", "Disk full", "could not resolve project dependency", "MethodError")
        @test !Setup.transient_download_error(ErrorException(message))
    end
    @test !Setup.transient_download_error(InterruptException())
    tries=Ref(0);delays=Int[];log=IOBuffer();console=IOBuffer()
    result=Setup.instantiate_with_retry(;log,console,sleepfn=d->push!(delays,d)) do
        tries[]+=1
        tries[]<3 && error("Connection reset by peer")
        :installed
    end
    @test result==:installed
    @test tries[]==3
    @test delays==[2,4]
    @test occursin("Cached downloads are retained",String(take!(console)))
    for (failure,offline) in ((ErrorException("Operation timed out"),true),
        (ErrorException("certificate verification failed"),false),
        (ErrorException("HTTP/2 407"),false), (InterruptException(),false))
        tries[]=0;empty!(delays)
        @test_throws typeof(failure) Setup.instantiate_with_retry(;offline,log=IOBuffer(),console=IOBuffer(),sleepfn=d->push!(delays,d)) do
            tries[]+=1;throw(failure)
        end
        @test tries[]==1
        @test isempty(delays)
    end
    tries[]=0
    @test_throws ErrorException Setup.instantiate_with_retry(;log=IOBuffer(),console=IOBuffer(),sleepfn=d->nothing) do
        tries[]+=1;error("Error status 503")
    end
    @test tries[]==3
end

@testset "Credentials and download tokens are redacted" begin
    text=Setup.safe_error(ErrorException("Could not download https://user:password@example.test/data?token=secret from http://proxy:credential@proxy.test:8080"))
    for secret in ("password", "credential", "token=secret")
        @test !occursin(secret,text)
    end
    @test occursin("example.test",text)
    @test occursin("proxy.test",text)
end

@testset "Durable progress, safe rerun, partial failure and offline install" begin
    mktempdir(SCRATCH) do fixture
        repo=joinpath(fixture,"public");private=joinpath(fixture,"private")
        selected=[(name="First project",path="first",cuda=false),(name="Second project",path="second",cuda=false)]
        for project in selected
            dir=joinpath(repo,project.path);mkpath(dir)
            write(joinpath(dir,"Project.toml"),"[deps]\n")
            write(joinpath(dir,"Manifest.toml"),"julia_version = \"1.13.2\"\nmanifest_format = \"2.0\"\n")
        end
        fingerprints=[Setup.fingerprint(p.path;root=repo) for p in selected]
        stages=String[];attempts=Ref(0)
        activate=(path,log)->push!(stages,"activate:"*basename(path))
        precompile=log->push!(stages,"precompile")
        verify=(p,log)->push!(stages,"verify:"*p.path)
        instantiate=log->begin
            attempts[]+=1
            attempts[]==1 && error("Operation timed out")
            push!(stages,"instantiate")
        end
        receipt=Setup.install_projects!(selected;root=repo,private,console=IOBuffer(),activate,instantiate,precompile,verify,sleepfn=d->nothing)
        successful=read(receipt)
        result=TOML.parsefile(receipt)
        @test result["status"]=="complete"
        @test result["verified"]===true
        @test length(result["projects"])==2
        @test isfile(result["log"])
        @test !occursin("TEMP_DELETEABLE",relpath(result["log"],private))
        @test attempts[]==3
        @test stages==["activate:first","instantiate","precompile","verify:first","activate:second","instantiate","precompile","verify:second"]
        @test !ispath(joinpath(private,"runtime","setup.lock"))
        attempts[]=0
        failing=log->begin
            attempts[]+=1
            attempts[]==2 && error("Error status 407 from https://user:secret@proxy.test")
        end
        @test_throws ErrorException Setup.install_projects!(selected;root=repo,private,console=IOBuffer(),activate,
            instantiate=failing,precompile,verify,sleepfn=d->nothing)
        @test read(receipt)==successful
        attempt=TOML.parsefile(joinpath(private,"runtime","installation_attempt.toml"))
        @test attempt["status"]=="failed"
        @test attempt["current_project"]=="second"
        @test attempt["stage"]=="download"
        @test length(attempt["projects"])==1
        @test !occursin("secret",attempt["error"])
        @test occursin("proxy authentication",read(attempt["log"],String))
        @test attempts[]==2
        @test !ispath(joinpath(private,"runtime","setup.lock"))
        empty!(stages)
        opts=Set(["--offline","--no-precompile","--no-verify"])
        Setup.install_projects!(selected;opts,root=repo,private,console=IOBuffer(),activate,
            instantiate=log->push!(stages,"instantiate"),precompile,verify)
        offline=TOML.parsefile(receipt)
        @test offline["status"]=="complete"
        @test offline["verified"]===false
        @test stages==["activate:first","instantiate","activate:second","instantiate"]
        @test [Setup.fingerprint(p.path;root=repo) for p in selected]==fingerprints
        @test length(readdir(joinpath(private,"runtime","install_logs")))==3
        @test all(isfile(joinpath(dir,"progress.toml")) for dir in readdir(joinpath(private,"runtime","install_logs");join=true))
        @test all(!endswith(file,".tmp") for (dir,_,files) in walkdir(private) for file in files)
    end
end
