#= Interactive axis-aligned cuts through actual saved voxel fields.

The solver writes inline base64/zlib VTU cell data. Reading these snapshots
does not run a solver, extrapolate an unavailable field, or mix AMR owners:
where exported cells overlap, the smallest containing cell supplies the pixel.
=#
import Base64
import Zlib_jll

const VOLUME_CUT_LOCK = ReentrantLock()
const VOLUME_CUT_CACHE = Ref{Any}(nothing)
const VOLUME_CUT_MAX_BYTES = 512_000_000
const VOLUME_CUT_MAX_ARRAY = 1_000_000_000

_vtk_attribute(text, name, default="") = begin
    m=match(Regex("\\b"*name*"\\s*=\\s*\"([^\"]*)\""),text)
    m===nothing ? default : String(m[1])
end

function _vtk_section(xml, name)
    m=match(Regex("<"*name*"\\b[^>]*>"),xml)
    m===nothing && throw(ArgumentError("Saved volume is missing its $name section"))
    start=m.offset+ncodeunits(m.match)
    closing=findnext("</"*name*">",xml,start)
    closing===nothing && throw(ArgumentError("Saved volume $name section is incomplete"))
    SubString(xml,start,prevind(xml,first(closing)))
end

function _vtk_arrays(section)
    arrays=Dict{String,Any}();cursor=firstindex(section)
    # Scan closing tags directly: a regex spanning a multi-megabyte base64
    # payload can hit PCRE's backtracking limit on otherwise valid snapshots.
    while (opening=findnext("<DataArray",section,cursor))!==nothing
        tagend=findnext('>',section,last(opening)+1)
        tagend===nothing && throw(ArgumentError("Saved volume array tag is incomplete"))
        start=nextind(section,tagend)
        closing=findnext("</DataArray>",section,start)
        closing===nothing && throw(ArgumentError("Saved volume array payload is incomplete"))
        attributes=SubString(section,last(opening)+1,prevind(section,tagend))
        arrays[_vtk_attribute(attributes,"Name")]=(attributes=attributes,
            text=SubString(section,start,prevind(section,first(closing))))
        cursor=nextind(section,last(closing))
    end
    arrays
end

function _vtk_inflate(source::Vector{UInt8}, count::Int)
    0<=count<=VOLUME_CUT_MAX_ARRAY || throw(ArgumentError("Saved volume array is too large for an interactive cut"))
    destination=Vector{UInt8}(undef,count)
    actual=Ref{Culong}(count)
    code=ccall((:uncompress,Zlib_jll.libz),Cint,
        (Ptr{UInt8},Ref{Culong},Ptr{UInt8},Culong),destination,actual,source,length(source))
    code==0 && actual[]==count || throw(ArgumentError("Saved volume compressed data is incomplete"))
    destination
end

function _vtk_binary(text, header_type, compressed)
    encoded=replace(String(text),r"\s"=>"")
    H=header_type=="UInt64" ? UInt64 : header_type=="UInt32" ? UInt32 :
        throw(ArgumentError("Unsupported saved volume header type"))
    width=sizeof(H)
    # WriteVTK encodes the header and the payload separately, including padding.
    # Decode enough complete base64 groups to inspect the compressed block count.
    prefixchars=4*cld(compressed ? 3width : width,3)
    length(encoded)>=prefixchars || throw(ArgumentError("Saved volume binary header is incomplete"))
    prefix=Base64.base64decode(encoded[1:prefixchars])
    if compressed
        length(prefix)>=3width || throw(ArgumentError("Saved volume compressed header is incomplete"))
        firstwords=reinterpret(H,prefix[1:3width])
        blocks=Int(firstwords[1]);blocksize=Int(firstwords[2]);lastsize=Int(firstwords[3])
        1<=blocks<=100_000 && 0<=blocksize<=VOLUME_CUT_MAX_ARRAY && 0<=lastsize<=blocksize ||
            throw(ArgumentError("Invalid saved volume compression header"))
        headerchars=4*cld((3+blocks)*width,3)
        length(encoded)>=headerchars || throw(ArgumentError("Saved volume compression sizes are incomplete"))
        header=Base64.base64decode(encoded[1:headerchars])
        sizes=Int.(reinterpret(H,header)[4:3+blocks])
        total=(blocks-1)*blocksize+(lastsize==0 ? blocksize : lastsize)
        total<=VOLUME_CUT_MAX_ARRAY || throw(ArgumentError("Saved volume array exceeds the interactive memory limit"))
        payload=Base64.base64decode(encoded[headerchars+1:end])
        sum(sizes)==length(payload) || throw(ArgumentError("Saved volume payload is incomplete"))
        output=Vector{UInt8}();sizehint!(output,total);offset=1
        for (i,n) in enumerate(sizes)
            n>0 || throw(ArgumentError("Invalid compressed block length"))
            count=i==blocks && lastsize!=0 ? lastsize : blocksize
            append!(output,_vtk_inflate(payload[offset:offset+n-1],count));offset+=n
        end
        output
    else
        header=Base64.base64decode(encoded[1:prefixchars])
        count=Int(reinterpret(H,header)[1])
        0<=count<=VOLUME_CUT_MAX_ARRAY || throw(ArgumentError("Saved volume array exceeds the interactive memory limit"))
        payload=Base64.base64decode(encoded[prefixchars+1:end])
        length(payload)==count || throw(ArgumentError("Saved volume payload is incomplete"))
        payload
    end
end

function _vtk_array(spec,header_type,compressed)
    types=Dict("Float32"=>Float32,"Float64"=>Float64,"Int32"=>Int32,"Int64"=>Int64,
               "UInt8"=>UInt8,"UInt32"=>UInt32,"UInt64"=>UInt64,"Int8"=>Int8,"Int16"=>Int16,"UInt16"=>UInt16)
    type=get(types,_vtk_attribute(spec.attributes,"type"),nothing)
    type===nothing && throw(ArgumentError("Unsupported saved volume array type"))
    format=_vtk_attribute(spec.attributes,"format","ascii")
    format=="ascii" && return parse.(type,split(spec.text))
    format=="binary" || throw(ArgumentError("Interactive cuts require inline VTU snapshots from the solver"))
    bytes=_vtk_binary(spec.text,header_type,compressed)
    length(bytes)%sizeof(type)==0 || throw(ArgumentError("Invalid saved volume array length"))
    copy(reinterpret(type,bytes))
end

function _read_volume_cut_grid(path)
    filesize(path)<=VOLUME_CUT_MAX_BYTES || throw(ArgumentError("Saved volume exceeds the 512 MB interactive reader limit; use recorded cuts or ParaView"))
    xml=read(path,String)
    root=match(r"<VTKFile\b([^>]*)>",xml)
    root===nothing && throw(ArgumentError("Saved volume snapshot is incomplete"))
    _vtk_attribute(root[1],"byte_order","LittleEndian")=="LittleEndian" && ENDIAN_BOM==0x04030201 ||
        throw(ArgumentError("Unsupported saved volume byte order"))
    compressor=_vtk_attribute(root[1],"compressor")
    compressor in ("","vtkZLibDataCompressor") || throw(ArgumentError("Unsupported saved volume compressor"))
    compressed=!isempty(compressor);header_type=_vtk_attribute(root[1],"header_type","UInt32")
    point_arrays=_vtk_arrays(_vtk_section(xml,"Points"))
    isempty(point_arrays) && throw(ArgumentError("Saved volume has no points"))
    points=_vtk_array(first(values(point_arrays)),header_type,compressed)
    length(points)%3==0 || throw(ArgumentError("Invalid volume point count"))
    coordinates=reshape(points,3,:)
    cells=_vtk_arrays(_vtk_section(xml,"Cells"))
    all(haskey(cells,k) for k in ("connectivity","offsets","types")) || throw(ArgumentError("Saved volume cell topology is incomplete"))
    connectivity=_vtk_array(cells["connectivity"],header_type,compressed)
    offsets=_vtk_array(cells["offsets"],header_type,compressed)
    kinds=_vtk_array(cells["types"],header_type,compressed)
    ncells=length(offsets)
    0<ncells<=8_000_000 && length(kinds)==ncells || throw(ArgumentError("Invalid or excessive volume cell count"))
    bounds=Matrix{Float32}(undef,6,ncells);previous=0
    for c in 1:ncells
        last=Int(offsets[c]);last-previous==8 && kinds[c] in (0x0b,0x0c) ||
            throw(ArgumentError("Interactive cuts support axis-aligned voxel/hexahedron cells only"))
        last<=length(connectivity) || throw(ArgumentError("Incomplete volume connectivity"))
        for axis in 1:3
            lo=Inf;hi=-Inf
            for j in previous+1:last
                index=Int(connectivity[j])+1
                1<=index<=size(coordinates,2) || throw(ArgumentError("Invalid volume point index"))
                v=Float64(coordinates[axis,index]);isfinite(v) || throw(ArgumentError("Nonfinite volume coordinates"))
                lo=min(lo,v);hi=max(hi,v)
            end
            hi>lo || throw(ArgumentError("Degenerate saved volume cell"))
            bounds[axis,c]=lo;bounds[axis+3,c]=hi
            # Reject general curved/hexahedral exports rather than invent a cut.
            tolerance=1e-5*(hi-lo)
            all(j->begin
                v=coordinates[axis,Int(connectivity[j])+1]
                abs(v-lo)<=tolerance || abs(v-hi)<=tolerance
            end,previous+1:last) || throw(ArgumentError("Saved volume cells are not axis-aligned"))
        end
        previous=last
    end
    fields=_vtk_arrays(_vtk_section(xml,"CellData"))
    Dict{String,Any}("bounds"=>bounds,"fields"=>fields,"header_type"=>header_type,
        "compressed"=>compressed,"decoded"=>Dict{String,Any}(),
        "min"=>[minimum(@view bounds[a,:]) for a in 1:3],
        "max"=>[maximum(@view bounds[a+3,:]) for a in 1:3])
end

function _volume_cell_array(grid,name)
    haskey(grid["fields"],name) || return nothing
    get!(grid["decoded"],name) do
        spec=grid["fields"][name]
        components=parse(Int,_vtk_attribute(spec.attributes,"NumberOfComponents","1"))
        data=_vtk_array(spec,grid["header_type"],grid["compressed"])
        length(data)==size(grid["bounds"],2)*components || throw(ArgumentError("Saved volume field length does not match the mesh"))
        reshape(data,components,:)
    end
end

function _volume_velocity_scale(case)
    _pond_case(case) && return 1.0 # Thermal FV volume fields are already SI.
    _q19_case(case) && return 1.0  # The CPU octree solver writes SI volume fields.
    flow=case["flow"]
    get(case,"lattice","d3q27")=="d3q39" ? Float64(flow["speed_of_sound"])/sqrt(2/3) : Float64(flow["velocity"])/.08
end

function _volume_sample_plane(grid,axis::String,position::Real,field::String,case;resolution::Int=384,step::Int=0)
    position=Float64(position)
    names=("x","y","z");normal=findfirst(==(axis),names)
    normal===nothing && throw(ArgumentError("Cut axis must be x, y or z"))
    1<=resolution<=768 || throw(ArgumentError("Cut resolution must be between 1 and 768"))
    lo=grid["min"];hi=grid["max"]
    tolerance=max(1e-9,1e-6*(hi[normal]-lo[normal]))
    isfinite(position) && lo[normal]-tolerance<=position<=hi[normal]+tolerance || throw(ArgumentError("Cut position is outside the saved volume"))
    # UI domain extents are Float64; exported voxel vertices are Float32.
    # Treat sub-roundoff boundary differences as the same boundary plane.
    position=clamp(position,Float64(lo[normal]),Float64(hi[normal]))
    a1,a2=normal==1 ? (2,3) : normal==2 ? (1,3) : (1,2)
    spacing=max((hi[a1]-lo[a1])/resolution,(hi[a2]-lo[a2])/resolution)
    n1=max(1,ceil(Int,(hi[a1]-lo[a1])/spacing));n2=max(1,ceil(Int,(hi[a2]-lo[a2])/spacing))
    source=field in ("velocity","u","v","w") ? "Velocity" :
           field=="density" ? "Density" : field=="pressure" ? "Pressure" :
           field=="vorticity" ? "Vorticity" : field=="mach" ? "Mach" : field=="temperature" ? "Temperature" :
           throw(ArgumentError("Unknown volume cut field"))
    array=_volume_cell_array(grid,source)
    array===nothing && return nothing
    expected=source in ("Velocity","Vorticity") ? 3 : 1
    size(array,1)==expected || throw(ArgumentError("Saved field component count is invalid"))
    scale=_volume_velocity_scale(case)
    factor=source in ("Velocity","Vorticity") ? scale : 1.0
    si=_pond_case(case) || _q19_case(case)
    units=source=="Velocity" ? "m/s" : source=="Vorticity" ? "1/s" :
        si && source=="Pressure" ? "Pa" :
        si && source=="Density" ? "kg/m³" :
        si && source=="Temperature" ? "K" : source=="Pressure" ? "lattice" : "1"
    obstacle=_volume_cell_array(grid,"Obstacle");level=_volume_cell_array(grid,"Level")
    selected=fill(NaN,n1,n2);cellsize=fill(Inf,n1,n2);owner=zeros(Int,n1,n2)
    b=grid["bounds"];component=field=="u" ? 1 : field=="v" ? 2 : field=="w" ? 3 : field=="vorticity" ? normal : 1
    for c in axes(b,2)
        cmin=Float64(b[normal,c]);cmax=Float64(b[normal+3,c])
        # Half-open cells avoid ownership flicker exactly on an internal face.
        cmin<=position && (position<cmax || position==hi[normal] && cmax==hi[normal]) || continue
        i0=max(1,ceil(Int,(b[a1,c]-lo[a1])/spacing+.5))
        i1=min(n1,ceil(Int,(b[a1+3,c]-lo[a1])/spacing+.5)-1)
        j0=max(1,ceil(Int,(b[a2,c]-lo[a2])/spacing+.5))
        j1=min(n2,ceil(Int,(b[a2+3,c]-lo[a2])/spacing+.5)-1)
        sizecell=Float64(b[4,c]-b[1,c])*Float64(b[5,c]-b[2,c])*Float64(b[6,c]-b[3,c])
        value=field=="velocity" ? sqrt(sum(abs2,@view array[:,c]))*factor : Float64(array[component,c])*factor
        solid=obstacle!==nothing && obstacle[1,c]!=0
        for j in j0:j1,i in i0:i1
            if sizecell<cellsize[i,j]
                cellsize[i,j]=sizecell;owner[i,j]=level===nothing ? 0 : Int(level[1,c])
                selected[i,j]=solid ? NaN : value
            end
        end
    end
    origin=Float64.(lo);origin[normal]=position
    Dict{String,Any}("ok"=>true,"source"=>"saved_volume","field"=>field,"source_field"=>source,
        "units"=>units,"n1"=>n1,"n2"=>n2,"axis1"=>names[a1],"axis2"=>names[a2],
        "normal_axis"=>axis,"normal_coord_m"=>position,"origin_m"=>origin,"spacing_m"=>spacing,
        "step"=>step,"sampling"=>"piecewise constant saved cell values; finest containing AMR cell wins",
        "values"=>Any[isfinite(v) ? v : nothing for v in vec(selected)],"owner_levels"=>vec(owner))
end

function volume_cut_json(id::AbstractString,req::HTTP.Request)
    job=_load_job(String(id));params=HTTP.queryparams(HTTP.URI(req.target))
    axis=String(get(params,"axis","x"));axis in ("x","y","z") || throw(ArgumentError("Cut axis must be x, y or z"))
    position=tryparse(Float64,String(get(params,"position","")))
    position!==nothing && isfinite(position) || throw(ArgumentError("Cut position must be a finite number in metres"))
    field=String(get(params,"field","velocity"))
    resolution=tryparse(Int,String(get(params,"resolution","384")))
    resolution!==nothing && 1<=resolution<=768 || throw(ArgumentError("Cut resolution must be between 1 and 768"))
    directory=joinpath(_case_dir(id,"run"),"RESULTS","VTK")
    snapshots=Tuple{Int,String}[]
    if isdir(directory)
        for filename in readdir(directory)
            m=match(r"^(?:flow|volume)_(\d+)\.vtu$",filename)
            # A new snapshot can be visible while WriteVTK is still filling it.
            # Keep using the last complete field until its successor is closed.
            m===nothing || (_artifact_complete(joinpath(directory,filename)) &&
                push!(snapshots,(parse(Int,m[1]),filename)))
        end
    end
    isempty(snapshots) && return _bad("No saved full-volume fields exist. Enable volume VTK outputs for a future run, or select a recorded cut.",404)
    sort!(snapshots);step,filename=last(snapshots)
    path=_case_file(id,"run","RESULTS/VTK/"*filename)
    path===nothing && return _bad("Saved volume snapshot is unavailable",404)
    filesize(path)<=VOLUME_CUT_MAX_BYTES || return _bad("Saved volume exceeds the 512 MB interactive reader limit; use recorded cuts or ParaView",413)
    lock(VOLUME_CUT_LOCK) do
        key=(path,stat(path).mtime,filesize(path));cached=VOLUME_CUT_CACHE[]
        grid=if cached!==nothing && cached.key==key
            cached.grid
        else
            loaded=_read_volume_cut_grid(path)
            VOLUME_CUT_CACHE[]=(key=key,grid=loaded);loaded
        end
        sampled=_volume_sample_plane(grid,axis,position,field,job["case"];resolution=resolution,step=step)
        sampled===nothing && return _bad("The saved volume does not contain the requested field",404)
        sampled["snapshot"]=filename
        _response(200,sampled)
    end
end
