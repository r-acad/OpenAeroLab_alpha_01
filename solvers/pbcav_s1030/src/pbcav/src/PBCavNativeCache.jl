module PBCavNativeCache

using PrecompileTools
using SHA
using TOML

const NATIVE_DIR=normpath(joinpath(@__DIR__,".."))
const CACHE_SCHEMA="PBCavNativeHost/1"
include(joinpath(NATIVE_DIR,"PBCav.jl"))
include("host_precompile_workload.jl")

"All native/generator Julia sources and the locked environment, including this wrapper."
function _bundle_signature()
    io=IOBuffer()
    for root in (NATIVE_DIR,normpath(joinpath(NATIVE_DIR,"..","cmesh")))
        for (dir,dirs,names) in walkdir(root)
            sort!(dirs)
            include_dependency(dir)
            for name in sort!(filter(n->endswith(n,".jl"),names))
                path=joinpath(dir,name)
                write(io,replace(relpath(path,dirname(NATIVE_DIR)),'\\'=>'/'))
                write(io,read(path))
            end
        end
    end
    for name in ("Project.toml","Manifest.toml","LocalPreferences.toml")
        path=joinpath(NATIVE_DIR,name)
        if isfile(path)
            include_dependency(path)
            write(io,name);write(io,read(path))
        end
    end
    bytes2hex(sha256(take!(io)))
end

# Captured together with the methods; never relabel an old module using files
# edited after loading. Julia also tracks ordinary include dependencies above.
const LOADED_BUNDLE_SHA256=_bundle_signature()
const LOADED_NATIVE_SHA256=PBCav.freeair_source_signature()

"Validate the exact loaded module and environment before a fresh worker uses it."
function validate_provenance!(;native_dir::AbstractString=NATIVE_DIR)
    expected=realpath(native_dir)
    realpath(NATIVE_DIR)==expected || throw(ArgumentError("Native host package was loaded from a different solver directory"))
    project=Base.active_project()
    project!==nothing && realpath(project)==realpath(joinpath(expected,"Project.toml")) ||
        throw(ArgumentError("Native host package is not running in the requested solver environment"))
    LOADED_NATIVE_SHA256==PBCav._freeair_disk_source_signature() ||
        throw(ArgumentError("Native solver source changed after loading; start a fresh checked worker"))
    LOADED_BUNDLE_SHA256==_bundle_signature() ||
        throw(ArgumentError("Native host package source or locked environment changed after loading; start a fresh checked worker"))
    (schema=CACHE_SCHEMA,native_sha256=LOADED_NATIVE_SHA256,bundle_sha256=LOADED_BUNDLE_SHA256,
        workload="cpu_sst_coupled_v1",julia_version=string(VERSION))
end

# No CUDA import, device query, context, device allocation or CUDA graph capture.
# Every state/work array is local and discarded after the host-only workload.
@compile_workload begin
    _cpu_sst_workload()
end

end
