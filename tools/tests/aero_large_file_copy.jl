using Test, HTTP, JSON3, Dates, UUIDs, SHA

const PUBLIC_ROOT=normpath(joinpath(@__DIR__,"..",".."))
const PRIVATE_ROOT=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(PUBLIC_ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const TEMP_ROOT=joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","aero_file_copy_tests")
mkpath(TEMP_ROOT)
include(joinpath(PUBLIC_ROOT,"apps","ludwig_web","src","aero","outputs.jl"))

@testset "bounded Aero case and export copies" begin
    mktempdir(TEMP_ROOT) do root
        source=joinpath(root,"source.bin")
        payload=UInt8.(mod.(0:12032,251));write(source,payload)
        for chunk in (1,97,8192,32768)
            target=joinpath(root,"copy_$chunk.bin")
            @test _copy_regular_file!(source,target;buffer_bytes=chunk)==target
            @test read(target)==payload
        end
        empty_source=joinpath(root,"empty");write(empty_source,UInt8[])
        empty_target=joinpath(root,"empty_copy")
        @test _copy_regular_file!(empty_source,empty_target)==empty_target
        @test filesize(empty_target)==0
        existing=joinpath(root,"existing");write(existing,"preserve me")
        @test_throws ArgumentError _copy_regular_file!(source,existing)
        @test read(existing,String)=="preserve me"
        @test_throws ArgumentError _copy_regular_file!(source,source)
        @test_throws ArgumentError _copy_regular_file!(joinpath(root,"missing"),joinpath(root,"never_created"))
        @test !ispath(joinpath(root,"never_created"))
        @test_throws ArgumentError _copy_regular_file!(source,joinpath(root,"bad_buffer");buffer_bytes=0)
        @test !ispath(joinpath(root,"bad_buffer"))
        tree=joinpath(root,"tree");mkpath(joinpath(tree,"nested"))
        write(joinpath(tree,"mesh.oct"),payload);write(joinpath(tree,"nested","config.yaml"),"mesh: octree\n")
        copied=_copy_case_tree!(tree,joinpath(root,"copied"))
        @test read(joinpath(copied,"mesh.oct"))==payload
        @test read(joinpath(copied,"nested","config.yaml"),String)=="mesh: octree\n"
        export_path=joinpath(root,"export","mesh.oct")
        _copy_export_snapshot!(source,export_path)
        @test read(export_path)==payload
        write(source,reverse(payload));_copy_export_snapshot!(source,export_path)
        @test read(export_path)==reverse(payload)
        @test readdir(dirname(export_path))==["mesh.oct"]
    end
end

# Opt-in real >4 GiB verification uses bounded memory and removes only this
# test's generated directory. The source mesh remains untouched.
if haskey(ENV,"AERO_LARGE_COPY_SOURCE")
    source=abspath(ENV["AERO_LARGE_COPY_SOURCE"])
    @testset "actual multi-GiB mesh copy" begin
        @test isfile(source) && filesize(source)>4*1024^3
        mktempdir(TEMP_ROOT) do root
            target=joinpath(root,"large_mesh.oct")
            elapsed=@elapsed _copy_regular_file!(source,target)
            @test filesize(target)==filesize(source)
            original_hash=open(sha256,source);copied_hash=open(sha256,target)
            @test copied_hash==original_hash
            println("LARGE_COPY_OK bytes=",filesize(target)," seconds=",round(elapsed,digits=3)," sha256=",bytes2hex(copied_hash))
        end
    end
end
