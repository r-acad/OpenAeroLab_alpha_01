"Native Q19 wall samples: bounded headers and exact static-geometry identity, independent of STL tessellation."
const WALL_GEOMETRY_HASH_CACHE=Dict{Tuple{String,Int64,Float64},String}()

function _wall_geometry_header(path)
    open(path,"r") do io
        String(read(io,8))=="LBMWGEO\0" || throw(ArgumentError("Invalid native wall geometry"))
        version=Int(ltoh(read(io,UInt32)));stride=Int(ltoh(read(io,UInt32)));n=ltoh(read(io,UInt64))
        version==1 && stride==36 && 0<n<=typemax(Int)÷36 && 24+36Int(n)==filesize(path) ||
            throw(ArgumentError("Native wall geometry is incomplete or unsupported"))
        Dict{String,Any}("samples"=>Int(n),"format_version"=>version,"record_bytes"=>stride)
    end
end

function _wall_frame_header(path)
    open(path,"r") do io
        String(read(io,8))=="LBMWFLD\0" || throw(ArgumentError("Invalid native wall field"))
        version=Int(ltoh(read(io,UInt32)));mask=Int(ltoh(read(io,UInt32)));n=ltoh(read(io,UInt64))
        version==1 && mask==1 && 0<n<=typemax(Int)÷4 && 128+4Int(n)==filesize(path) ||
            throw(ArgumentError("Native wall field is incomplete or unsupported"))
        digest=String(read(io,64));occursin(r"^[0-9a-f]{64}$",digest) || throw(ArgumentError("Invalid wall geometry identity"))
        step=Int(ltoh(read(io,Int64)));time=read(io,Float64);density=read(io,Float64);speed=read(io,Float64);q=read(io,Float64)
        step>=0 && isfinite(time) && time>=0 && all(x->isfinite(x)&&x>0,(density,speed,q)) || throw(ArgumentError("Invalid native wall field references"))
        isapprox(q,.5density*speed^2;rtol=1e-8) || throw(ArgumentError("Native wall pressure reference is inconsistent"))
        Dict{String,Any}("samples"=>Int(n),"step"=>step,"time_s"=>time,"density_kg_m3"=>density,"speed_m_s"=>speed,
            "q_reference_pa"=>q,"geometry_sha256"=>digest,"geometry_file"=>"surface_wall_geometry_"*digest*".lbmwg",
            "geometry_frame"=>"domain","geometry_binding"=>"native_wall_samples","representation"=>"wall_samples",
            "field_units"=>Dict("pressure"=>"Pa","cp"=>"1"))
    end
end

function _wall_geometry_sha(path)
    info=stat(path);key=(abspath(path),Int64(info.size),Float64(info.mtime))
    get!(WALL_GEOMETRY_HASH_CACHE,key) do
        length(WALL_GEOMETRY_HASH_CACHE)>16 && empty!(WALL_GEOMETRY_HASH_CACHE)
        open(io->bytes2hex(sha256(io)),path,"r")
    end
end

function _wall_surface_metadata(path;geometry_path=nothing,verify=false)
    metadata=_wall_frame_header(path)
    geometry=geometry_path===nothing ? joinpath(dirname(path),metadata["geometry_file"]) : geometry_path
    isfile(geometry) && !islink(geometry) || throw(ArgumentError("Matching native wall geometry is missing"))
    _wall_geometry_header(geometry)["samples"]==metadata["samples"] || throw(ArgumentError("Native wall geometry/sample count mismatch"))
    verify && _wall_geometry_sha(geometry)!=metadata["geometry_sha256"] && throw(ArgumentError("Native wall geometry SHA-256 does not match the field"))
    sidecar=path*".toml"
    if isfile(sidecar)
        filesize(sidecar)<=65_536 || throw(ArgumentError("Native wall metadata is too large"))
        recorded=TOML.parsefile(sidecar)
        for (key,value) in (("geometry_sha256",metadata["geometry_sha256"]),("geometry_frame","domain"))
            haskey(recorded,key) && recorded[key]!=value && throw(ArgumentError("Native wall sidecar disagrees with its field"))
        end
        for key in ("source_checkpoint_sha256","config_sha256","stl_sha256","source_checkpoint","sampling_note")
            haskey(recorded,key) && (metadata[key]=recorded[key])
        end
    end
    metadata
end

function _wall_geometry_bounds(path)
    n=_wall_geometry_header(path)["samples"];lo=fill(Inf,3);hi=fill(-Inf,3)
    buffer=Vector{UInt8}(undef,36min(n,65_536))
    open(path,"r") do io
        seek(io,24);left=n
        while left>0
            count=min(left,length(buffer)÷36);bytes=@view buffer[1:36count];read!(io,bytes)
            floats=reinterpret(Float32,bytes)
            for i in 0:count-1,a in 1:3
                x=Float64(floats[9i+a]);isfinite(x) || throw(ArgumentError("Native wall geometry contains nonfinite coordinates"))
                lo[a]=min(lo[a],x);hi[a]=max(hi[a],x)
            end
            left-=count
        end
    end
    Dict("min"=>lo,"max"=>hi)
end
