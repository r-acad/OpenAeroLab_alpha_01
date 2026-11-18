using Test, TOML

const REPOSITORY=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(REPOSITORY,"tools","install","setup.jl"))
const Setup=OpenAeroLabSetup
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(REPOSITORY,"..","02_PRIVATE_PROJECT_DEVELOPMENT"))),"TEMP_DELETEABLE","installer_tests")
mkpath(SCRATCH)

@testset "Pinned manifests match their root package dependencies" begin
    for project in Setup.PROJECTS
        @test Setup.validate_manifest(project.path)===nothing
    end
    jx=TOML.parsefile(joinpath(REPOSITORY,"third_party","JXfoil","Manifest.toml"))
    @test VersionNumber(jx["julia_version"])==Setup.VERSION_RECOMMENDED
    @test "CairoMakie" in only(jx["deps"]["JXfoil"])["deps"]
    mktempdir(SCRATCH) do repo
        dir=joinpath(repo,"sample");mkpath(dir)
        write(joinpath(dir,"Project.toml"),"name = \"Example\"\nuuid = \"aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa\"\n[deps]\nLinearAlgebra = \"37e2e46d-f89d-539d-b4ee-838fcccc9c8e\"\n")
        file=joinpath(dir,"Manifest.toml")
        prefix="manifest_format = \"2.0\"\n[[deps.Example]]\nuuid = \"aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa\"\npath = \".\"\n"
        write(file,prefix*"deps = [\"LinearAlgebra\"]\n")
        @test Setup.validate_manifest("sample";root=repo)===nothing
        write(file,prefix*"deps = []\n")
        err=try Setup.validate_manifest("sample";root=repo);nothing catch e;e end
        @test err isa ArgumentError
        @test occursin("missing: LinearAlgebra",sprint(showerror,err))
        write(file,prefix*"deps = [\"LinearAlgebra\", \"Extra\"]\n")
        @test_throws ArgumentError Setup.validate_manifest("sample";root=repo)
        write(file,replace(prefix,"aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"=>"bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb")*"deps = [\"LinearAlgebra\"]\n")
        @test_throws ArgumentError Setup.validate_manifest("sample";root=repo)
        write(file,"manifest_format = \"2.0\"\n")
        @test Setup.validate_manifest("sample";root=repo)===nothing
    end
end

@testset "Any Julia from the minimum release on is accepted" begin
    @test Setup.VERSION_MINIMUM==v"1.13"
    @test Setup.julia_supported(VERSION)
    @test Setup.VERSION_RECOMMENDED>=Setup.VERSION_MINIMUM
    for v in (v"1.13.0", v"1.13.2", v"1.13.9", v"1.14.0-DEV.12", v"1.14.1", v"1.20.0", v"2.0.0")
        @test Setup.julia_supported(v)
    end
    for v in (v"1.12.6", v"1.10.10", v"1.9.4", v"0.7.0")
        @test !Setup.julia_supported(v)
    end
    @test Setup.minimum_label()=="1.13"
    @test Setup.same_minor(v"1.13.0",v"1.13.2") && !Setup.same_minor(v"1.13.2",v"1.14.0")
    # Every pinned environment accepts the minimum release in its own compat entry, if it has one.
    for project in Setup.PROJECTS
        compat=get(get(TOML.parsefile(joinpath(REPOSITORY,project.path,"Project.toml")),"compat",Dict()),"julia",nothing)
        compat===nothing && continue
        spec=Setup.Pkg.Types.semver_spec(compat)
        @test v"1.13.0" in spec && v"1.14.0" in spec
    end
end

@testset "Failure guidance follows the actual setup stage" begin
    for stage in ("metadata","precompile","verify","download")
        io=IOBuffer();Setup.failure_guidance(io,stage)
        guidance=String(take!(io))
        @test !occursin(r"(?i)\b(?:company|corporate|policy)\b",guidance)
        @test stage=="download" ? occursin("network",lowercase(guidance)) : occursin(stage,guidance)
    end
end

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
        verified=Ref(false)
        @test_throws ErrorException Setup.install_projects!(selected;root=repo,private,console=IOBuffer(),activate,
            instantiate=log->nothing,precompile=log->error("synthetic JXfoil compile failure"),
            verify=(p,log)->(verified[]=true))
        attempt=TOML.parsefile(joinpath(private,"runtime","installation_attempt.toml"))
        @test attempt["status"]=="failed"
        @test attempt["stage"]=="precompile"
        @test occursin("Setup failed in precompile",read(attempt["log"],String))
        @test !verified[]
        @test read(receipt)==successful
        empty!(stages)
        opts=Set(["--offline","--no-precompile","--no-verify"])
        Setup.install_projects!(selected;opts,root=repo,private,console=IOBuffer(),activate,
            instantiate=log->push!(stages,"instantiate"),precompile,verify)
        offline=TOML.parsefile(receipt)
        @test offline["status"]=="complete"
        @test offline["verified"]===false
        @test stages==["activate:first","instantiate","activate:second","instantiate"]
        @test [Setup.fingerprint(p.path;root=repo) for p in selected]==fingerprints
        @test length(readdir(joinpath(private,"runtime","install_logs")))==4
        @test all(isfile(joinpath(dir,"progress.toml")) for dir in readdir(joinpath(private,"runtime","install_logs");join=true))
        @test all(!endswith(file,".tmp") for (dir,_,files) in walkdir(private) for file in files)
    end
end

@testset "A manifest recorded with another Julia minor release is resolved again only after a failure" begin
    mktempdir(SCRATCH) do fixture
        repo=joinpath(fixture,"public");private=joinpath(fixture,"private")
        other="$(VERSION.major).$(VERSION.minor==0 ? 1 : VERSION.minor-1).0"
        selected=[(name="Older record",path="older",cuda=false),(name="Current record",path="current",cuda=false)]
        for (project,recorded) in zip(selected,(other,string(VERSION)))
            dir=joinpath(repo,project.path);mkpath(dir)
            write(joinpath(dir,"Project.toml"),"[deps]\n")
            write(joinpath(dir,"Manifest.toml"),"julia_version = \"$recorded\"\nmanifest_format = \"2.0\"\n")
        end
        @test Setup.manifest_julia("older";root=repo)==VersionNumber(other)
        @test Setup.manifest_julia("missing";root=repo)===nothing
        stages=String[];failures=Dict("older"=>1)
        current=Ref("")
        activate=(path,log)->(current[]=basename(path);push!(stages,"activate:"*current[]))
        instantiate=log->begin
            if get(failures,current[],0)>0
                failures[current[]]-=1
                error("expected package Example to be registered")   # not a transfer error: no retry
            end
            push!(stages,"instantiate")
        end
        resolve=log->push!(stages,"resolve:"*current[])
        console=IOBuffer()
        receipt=Setup.install_projects!(selected;opts=Set(["--no-precompile","--no-verify"]),root=repo,private,console,
            activate,instantiate,resolve,sleepfn=d->nothing)
        @test TOML.parsefile(receipt)["status"]=="complete"
        @test stages==["activate:older","resolve:older","instantiate","activate:current","instantiate"]
        @test occursin("resolving them again",String(take!(console)))
        # The same failure on a manifest recorded with this release is reported, not re-resolved.
        empty!(stages);failures["current"]=1
        @test_throws ErrorException Setup.install_projects!(selected;opts=Set(["--no-precompile","--no-verify"]),root=repo,
            private,console=IOBuffer(),activate,instantiate,resolve,sleepfn=d->nothing)
        @test !any(startswith(s,"resolve:current") for s in stages)
    end
end
