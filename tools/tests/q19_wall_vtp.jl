# julia --project=src/octree_q19 tools/tests/q19_wall_vtp.jl
using Test,OctreeQ19,SHA,TOML
const G=OctreeQ19
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT"))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","q19_wall_vtp_tests");mkpath(SCRATCH)
const OUT=mktempdir(SCRATCH;cleanup=false)
function make_native(n;step=12,delta=0f0,dir=OUT)
    tmp=joinpath(dir,"geometry.tmp")
    open(tmp,"w") do io
        write(io,codeunits("LBMWGEO\0"),UInt32(1),UInt32(36),UInt64(n))
        for i in 1:n
            write(io,Float32[i,2i,3i],Float32[0,1,0],.25f0,UInt32(i==2 ? 1 : i),UInt8(2),UInt8(3),UInt16(i==3 ? 0 : 1))
        end
    end
    sha=bytes2hex(open(sha256,tmp));geom=joinpath(dir,"surface_wall_geometry_"*sha*".lbmwg");mv(tmp,geom;force=true)
    field=joinpath(dir,"surface_$(lpad(step,6,'0')).lbmw")
    open(field,"w") do io
        write(io,codeunits("LBMWFLD\0"),UInt32(1),UInt32(1),UInt64(n),codeunits(sha),Int64(step),Float64[step/10,1,10,50])
        for i in 1:n;write(io,i==3 ? NaN32 : Float32(i)+delta);end
    end
    (geom,field)
end
function appended(path)
    bytes=read(path);marker=Vector{UInt8}(codeunits("<AppendedData encoding=\"raw\">\n_"))
    start=findfirst(marker,bytes);start===nothing&&error("Missing appended marker")
    data=last(start)+1;xml=String(bytes[1:last(start)])
    offsets=[parse(Int,m.captures[1]) for m in eachmatch(r"offset=\"(\d+)\"",xml)]
    # Header emits PointData first, then Points, then Verts; offsets refer to the
    # actual binary order and do not depend on their XML order.
    function block(offset,T)
        i=data+offset;count=Int(only(reinterpret(UInt64,bytes[i:i+7])))
        @test count%sizeof(T)==0
        collect(reinterpret(T,bytes[i+8:i+7+count]))
    end
    (xml=xml,offsets=offsets,block=block,bytes=bytes,data=data)
end
@testset "Fine wall pressure as streaming ParaView PolyData" begin
    geom,frame=make_native(4)
    beforefield=bytes2hex(open(sha256,frame));beforegeom=bytes2hex(open(sha256,geom))
    vtp=G.write_wall_surface_vtp(frame)
    @test basename(vtp)=="surface_wall_samples_000012.vtp"
    @test bytes2hex(open(sha256,frame))==beforefield&&bytes2hex(open(sha256,geom))==beforegeom
    v=appended(vtp)
    @test occursin("type=\"PolyData\"",v.xml)&&occursin("NumberOfPoints=\"4\"",v.xml)&&occursin("NumberOfVerts=\"4\"",v.xml)
    @test length(v.offsets)==9
    pressures=v.block(v.offsets[1],Float32);cp=v.block(v.offsets[2],Float32);dx=v.block(v.offsets[3],Float32)
    parent=v.block(v.offsets[4],UInt32);level=v.block(v.offsets[5],UInt8);flags=v.block(v.offsets[6],UInt16)
    xyz=v.block(v.offsets[7],Float32);conn=v.block(v.offsets[8],Int32);offsets=v.block(v.offsets[9],Int32)
    @test pressures[1:2]==Float32[1,2] && isnan(pressures[3]) && pressures[4]==4
    @test cp[1:2]≈Float32[.02,.04] && isnan(cp[3])
    @test parent[1]==parent[2]==1 && pressures[1]!=pressures[2] # variation within one original facet
    @test dx==fill(.25f0,4)&&level==fill(UInt8(2),4)&&flags==UInt16[1,1,0,1]
    @test xyz==Float32[1,2,3,2,4,6,3,6,9,4,8,12]
    @test conn==Int32[0,1,2,3] && offsets==Int32[1,2,3,4]
    @test filesize(vtp)<39*4+4_000 && G.WALL_VTP_BYTES_PER_SAMPLE==39
    sidecar=TOML.parsefile(vtp*".toml")
    @test sidecar["geometry_sha256"]==beforegeom && sidecar["source_field_sha256"]==beforefield
    @test sidecar["samples"]==4 && sidecar["step"]==12 && sidecar["physical_time_s"]==1.2
    firsthash=bytes2hex(open(sha256,vtp));@test G.write_wall_surface_vtp(frame)==vtp
    @test bytes2hex(open(sha256,vtp))==firsthash
    _,frame2=make_native(4;step=24,delta=3f0);vtp2=G.write_wall_surface_vtp(frame2)
    pvd=read(joinpath(OUT,"surface_wall_samples.pvd"),String)
    @test occursin(basename(vtp),pvd)&&occursin(basename(vtp2),pvd)
    @test findfirst(basename(vtp),pvd)<findfirst(basename(vtp2),pvd)
    open(geom,"r+") do io;seek(io,24);write(io,0xff);end
    @test_throws ErrorException G.write_wall_surface_vtp(frame)
    @test bytes2hex(open(sha256,vtp))==firsthash
end
@testset "VTP chunk boundaries stay exact" begin
    dir=joinpath(OUT,"many");mkpath(dir);n=32_771
    _,frame=make_native(n;dir)
    path=G.write_wall_surface_vtp(frame);v=appended(path)
    @test filesize(path)<39n+4_000
    p=v.block(v.offsets[1],Float32);c=v.block(v.offsets[8],Int32);ends=v.block(v.offsets[9],Int32)
    @test p[32_768]==32768f0&&p[end]==32771f0
    @test c[32_768:32_770]==Int32[32767,32768,32769]
    @test ends[32_768:32_770]==Int32[32768,32769,32770]
end
println("ParaView VTP fixture directory: ",OUT)
