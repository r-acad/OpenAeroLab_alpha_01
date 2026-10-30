module FullFieldArtifact

using SHA
using ..AirfoilWire

const COLUMNS = ("x", "y", "mach", "cp", "pressure", "density", "u", "v",
                 "temperature", "vorticity", "nut")
const HEADER_RESERVE = 8192

function field_layout(ni::Integer,nj::Integer;max_bytes::Integer=AirfoilWire.MAX_BYTES)
    ni > 1 && nj > 1 || throw(ArgumentError("invalid full field dimensions"))
    HEADER_RESERVE < max_bytes <= AirfoilWire.MAX_BYTES ||
        throw(ArgumentError("invalid full field chunk size limit"))
    count = Base.checked_mul(Int(ni),Int(nj))
    row_bytes = Base.checked_mul(Int(ni),length(COLUMNS)*9)
    rows_per_part = min(Int(nj),fld(Int(max_bytes)-HEADER_RESERVE,row_bytes))
    rows_per_part >= 1 || throw(ArgumentError(
        "one full field row exceeds the MessagePack artifact limit; reduce tangential mesh refinement"))
    (;count,rows_per_part,part_count=cld(Int(nj),rows_per_part))
end

function _field_rows(at::Function,ni::Int,j_start::Int,j_end::Int)
    count = ni*(j_end-j_start+1)
    arrays = Dict(name=>Vector{Float64}(undef,count) for name in COLUMNS)
    k=0
    for j in j_start:j_end, i in 1:ni
        row=at(i,j)
        length(row)==length(COLUMNS) || throw(ArgumentError(
            "full field callback must return $(length(COLUMNS)) values"))
        k+=1
        for n in eachindex(COLUMNS)
            value=Float64(row[n])
            isfinite(value) || throw(ArgumentError("nonfinite full field $(COLUMNS[n]) at cell ($i,$j)"))
            arrays[COLUMNS[n]][k]=value
        end
    end
    arrays
end

"""Write every final cell center and scalar in j-major, i-fastest order.

`at(i,j)` returns values in `COLUMNS` order. Small fields keep the historical
single MessagePack document. Larger fields publish a manifest and contiguous
row chunks, each below the 64 MiB wire limit, preserving every Float64 value.
"""
function write_full_field(at::Function, path::AbstractString,
                          ni::Integer, nj::Integer, mesh_path::AbstractString,
                          iteration::Integer;max_bytes::Integer=AirfoilWire.MAX_BYTES)
    ni > 1 && nj > 1 && iteration >= 1 ||
        throw(ArgumentError("invalid full field dimensions or final iteration"))
    layout=field_layout(ni,nj;max_bytes)
    count=layout.count
    isfile(mesh_path) || throw(ArgumentError("full field mesh artifact is missing"))
    mesh_hash = bytes2hex(sha256(read(mesh_path)))
    common=Dict{String,Any}("ni"=>Int(ni),"nj"=>Int(nj),
        "cell_order"=>"j-major, i-fastest","mesh_sha256"=>mesh_hash,"iteration"=>Int(iteration))
    total_bytes=0
    payload = if layout.part_count == 1
        arrays=_field_rows(at,Int(ni),1,Int(nj))
        # Keep the historical single-file payload unchanged below the cap.
        Dict{String,Any}(
        "kind"=>"structured_cells", "ni"=>Int(ni), "nj"=>Int(nj),
        "samples"=>arrays, "cell_order"=>"j-major, i-fastest",
        "mesh_sha256"=>mesh_hash, "iteration"=>Int(iteration))
    else
        parts=Any[]
        stem=splitext(basename(path))[1]
        for (index,j_start) in enumerate(1:layout.rows_per_part:Int(nj))
            j_end=min(j_start+layout.rows_per_part-1,Int(nj))
            part=merge(common,Dict{String,Any}("kind"=>"structured_cells_part",
                "nj"=>j_end-j_start+1,"full_nj"=>Int(nj),
                "j_start"=>j_start,"j_end"=>j_end,
                "samples"=>_field_rows(at,Int(ni),j_start,j_end)))
            bytes=AirfoilWire.encode(part,:msgpack)
            length(bytes)<=max_bytes || throw(ArgumentError("encoded full field chunk exceeds its size limit"))
            filename=stem*".part"*lpad(string(index),4,'0')*".msgpack"
            write(joinpath(dirname(path),filename),bytes)
            total_bytes+=length(bytes)
            push!(parts,Dict("path"=>filename,"j_start"=>j_start,"j_end"=>j_end,"bytes"=>length(bytes)))
        end
        merge(common,Dict{String,Any}("kind"=>"structured_cells_manifest",
            "columns"=>collect(COLUMNS),"parts"=>parts))
    end
    bytes = AirfoilWire.encode(payload, :msgpack)
    length(bytes) <= max_bytes ||
        throw(ArgumentError("encoded full field exceeds the 64 MiB MessagePack limit"))
    write(path, bytes)
    Dict{String,Any}("filename"=>basename(path), "cell_count"=>count,
                     "mesh_sha256"=>mesh_hash, "iteration"=>Int(iteration),
                     "bytes"=>total_bytes+length(bytes),"part_count"=>layout.part_count)
end

end # module FullFieldArtifact
