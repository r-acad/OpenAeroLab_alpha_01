# ParaView-compatible VTK XML PolyData, streamed from immutable wall samples.
# A vertex cell represents each measured cell/facet point; no normals, pressure
# or surface area are fabricated for missing/unsupported links.
const WALL_VTP_BYTES_PER_SAMPLE = 39
const WALL_VTP_ARRAYS = (
    ("Points", "Float32", 3, 12), ("Pressure", "Float32", 1, 4),
    ("Cp", "Float32", 1, 4), ("LocalCellSize", "Float32", 1, 4),
    ("OriginalFacetId", "UInt32", 1, 4), ("AMRLevel", "UInt8", 1, 1),
    ("SupportFlags", "UInt16", 1, 2), ("connectivity", "Int32", 1, 4),
    ("offsets", "Int32", 1, 4))

"Read and validate the lightweight native frame header and its exact geometry."
function _wall_vtp_sources(field::AbstractString)
    ENDIAN_BOM == 0x04030201 || error("Wall VTP currently requires a little-endian host")
    open(field,"r") do io
        filesize(field)>=WALL_FIELD_HEADER_BYTES || error("Wall pressure frame is truncated")
        String(read(io,8))==WALL_FIELD_MAGIC || error("Invalid wall pressure frame")
        read(io,UInt32)==1 && read(io,UInt32)==1 || error("Unsupported wall pressure frame")
        raw=read(io,UInt64);0<raw<=typemax(Int32)||error("Wall sample count exceeds VTP Int32 capacity")
        count=Int(raw);filesize(field)==WALL_FIELD_HEADER_BYTES+4count||error("Wall pressure frame is incomplete")
        hash=String(read(io,64));occursin(r"^[0-9a-f]{64}$",hash)||error("Invalid wall geometry identity")
        step=Int(read(io,Int64));time=read(io,Float64);rho=read(io,Float64);speed=read(io,Float64);q=read(io,Float64)
        step>=0 && all(isfinite,(time,rho,speed,q)) && time>=0 && rho>0 && speed>0 && q>0 || error("Invalid wall pressure references")
        isapprox(q,.5rho*speed^2;rtol=1e-8)||error("Inconsistent wall pressure reference")
        geometry=joinpath(dirname(field),"surface_wall_geometry_"*hash*".lbmwg")
        isfile(geometry) && filesize(geometry)==24+36count || error("Matching wall geometry is missing or incomplete")
        _wall_sha(geometry)==hash||error("Wall geometry hash does not match pressure frame")
        open(geometry,"r") do g
            String(read(g,8))==WALL_GEOMETRY_MAGIC && read(g,UInt32)==1 && read(g,UInt32)==WALL_GEOMETRY_RECORD_BYTES && read(g,UInt64)==count ||
                error("Wall geometry header differs from pressure frame")
        end
        (;count,geometry,geometry_sha256=hash,field_sha256=_wall_sha(field),step,time,q)
    end
end

"Copy one component of interleaved 36-byte wall records in bounded blocks."
function _wall_vtp_component!(out,geometry,count,offset,width)
    block=16_384;input=Vector{UInt8}(undef,36block);output=Vector{UInt8}(undef,width*block)
    open(geometry,"r") do io
        seek(io,24)
        for start in 1:block:count
            n=min(block,count-start+1);read!(io,@view input[1:36n])
            @inbounds for j in 0:n-1,k in 1:width
                output[j*width+k]=input[j*36+offset+k]
            end
            write(out,@view output[1:width*n])
        end
    end
    nothing
end

"Read pressure in bounded blocks and optionally convert it to Cp."
function _wall_vtp_pressure!(out,field,count,q;cp=false)
    block=32_768;values=Vector{Float32}(undef,block)
    open(field,"r") do io
        seek(io,WALL_FIELD_HEADER_BYTES)
        for start in 1:block:count
            n=min(block,count-start+1);read!(io,@view values[1:n])
            if cp
                @inbounds for i in 1:n;values[i]=Float32(values[i]/q);end
            end
            write(out,@view values[1:n])
        end
    end
    nothing
end
function _wall_vtp_vertices!(out,count;offsets=false)
    block=32_768;indices=Vector{Int32}(undef,block)
    for start in 1:block:count
        n=min(block,count-start+1)
        @inbounds for i in 1:n;indices[i]=Int32(start+i-(offsets ? 1 : 2));end
        write(out,@view indices[1:n])
    end
    nothing
end

"Update a ParaView collection for complete, naturally sorted fine-wall VTP frames."
function _wall_vtp_collection!(dir)
    files=sort(filter(name->occursin(r"^surface_wall_samples_[0-9]+\.vtp$",name),readdir(dir)))
    path=joinpath(dir,"surface_wall_samples.pvd");tmp=_wall_tmp(path)
    try
        open(tmp,"w") do io
            println(io,"<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n <Collection>")
            for name in files
                info=name*".toml";isfile(joinpath(dir,info))||continue
                metadata=TOML.parsefile(joinpath(dir,info));t=get(metadata,"physical_time_s",nothing)
                t isa Real && isfinite(t) && t>=0 || continue
                @printf(io,"  <DataSet timestep=\"%.17g\" group=\"\" part=\"0\" file=\"%s\"/>\n",t,name)
            end
            println(io," </Collection>\n</VTKFile>")
        end
        Base.Filesystem.rename(tmp,path)
    finally
        isfile(tmp)&&rm(tmp)
    end
    path
end

"""
Stream a ParaView PolyData point dataset from matching native wall geometry and pressure.
The file is uncompressed (39 bytes/sample plus a small header); peak conversion memory
is bounded independently of the number of samples. Pressure is local adjacent-cell
pressure, not an average over the original facet. Geometry and field stay unchanged.
"""
function write_wall_surface_vtp(field::AbstractString)
    source=_wall_vtp_sources(field);n=source.count;dir=dirname(abspath(field))
    name=@sprintf("surface_wall_samples_%06d.vtp",source.step);path=joinpath(dir,name);tmp=_wall_tmp(path)
    lengths=UInt64[UInt64(item[4])*UInt64(n) for item in WALL_VTP_ARRAYS]
    offsets=UInt64[];cursor=UInt64(0)
    for length in lengths;push!(offsets,cursor);cursor+=8+length;end
    try
        open(tmp,"w") do io
            println(io,"<?xml version=\"1.0\"?>")
            println(io,"<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">")
            println(io," <PolyData><Piece NumberOfPoints=\"$n\" NumberOfVerts=\"$n\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">")
            println(io,"  <PointData Scalars=\"Cp\">")
            for idx in 2:7
                label,kind,components,_=WALL_VTP_ARRAYS[idx]
                println(io,"   <DataArray type=\"$kind\" Name=\"$label\" NumberOfComponents=\"$components\" format=\"appended\" offset=\"$(offsets[idx])\"/>")
            end
            println(io,"  </PointData><CellData/>")
            println(io,"  <Points><DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$(offsets[1])\"/></Points>")
            println(io,"  <Verts><DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"$(offsets[8])\"/>")
            println(io,"   <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"$(offsets[9])\"/></Verts>")
            println(io," </Piece></PolyData>\n <AppendedData encoding=\"raw\">\n_")
            # The underscore marks offset zero. println above added one newline
            # after it, which must not become part of the appended stream.
            seek(io,position(io)-1)
            for i in eachindex(lengths)
                write(io,lengths[i])
                if i==1;_wall_vtp_component!(io,source.geometry,n,0,12)
                elseif i==2;_wall_vtp_pressure!(io,field,n,source.q)
                elseif i==3;_wall_vtp_pressure!(io,field,n,source.q;cp=true)
                elseif i==4;_wall_vtp_component!(io,source.geometry,n,24,4)
                elseif i==5;_wall_vtp_component!(io,source.geometry,n,28,4)
                elseif i==6;_wall_vtp_component!(io,source.geometry,n,32,1)
                elseif i==7;_wall_vtp_component!(io,source.geometry,n,34,2)
                elseif i==8;_wall_vtp_vertices!(io,n)
                else;_wall_vtp_vertices!(io,n;offsets=true);end
            end
            println(io,"\n </AppendedData>\n</VTKFile>")
        end
        _wall_sha(field)==source.field_sha256 || error("Wall pressure changed during VTP export")
        _wall_sha(source.geometry)==source.geometry_sha256 || error("Wall geometry changed during VTP export")
        metadata=Dict("format"=>"VTK XML PolyData","representation"=>"native_wall_cell_facet_samples",
            "samples"=>n,"step"=>source.step,"physical_time_s"=>source.time,"reference_dynamic_pressure_pa"=>source.q,
            "geometry_sha256"=>source.geometry_sha256,"source_field_sha256"=>source.field_sha256,
            "source_geometry"=>abspath(source.geometry),"source_field"=>abspath(field),
            "bytes_per_sample"=>WALL_VTP_BYTES_PER_SAMPLE,"pressure_unit"=>"Pa",
            "note"=>"Point data use adjacent fluid-cell pressure at wall-link intersections; no facet averaging or surface-area force quadrature.")
        if isfile(path)
            filesize(path)==filesize(tmp)&&_wall_sha(path)==_wall_sha(tmp)||error("Refusing to replace different existing VTP: $path")
        end
        if isfile(path*".toml")
            previous=TOML.parsefile(path*".toml")
            all(get(previous,key,nothing)==metadata[key] for key in ("geometry_sha256","source_field_sha256","samples","step")) ||
                error("Existing VTP provenance does not match its native field: $path.toml")
        else
            _wall_atomic_toml(path*".toml",metadata)
        end
        _wall_publish(tmp,path)
        _wall_vtp_collection!(dir)
        println("[WallSurface] ParaView point field: $n samples, $(filesize(path)) bytes; $path");flush(stdout)
        path
    finally
        isfile(tmp)&&rm(tmp)
    end
end
