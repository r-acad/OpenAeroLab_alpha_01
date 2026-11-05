const MAX_STL_BYTES = 50_000_084
const Triangle = NTuple{3,NTuple{3,Float64}}

function _stl_triangles(bytes::Vector{UInt8})
    length(bytes) >= 84 || throw(ArgumentError("A binary STL needs an 84-byte header"))
    io = IOBuffer(bytes)
    seek(io, 80)
    count = Int(ltoh(read(io, UInt32)))
    1 <= count <= 1_000_000 || throw(ArgumentError("STL must contain 1 to 1,000,000 triangles"))
    length(bytes) == 84 + 50count || throw(ArgumentError("Binary STL length does not match its triangle count"))
    tris = Vector{Triangle}(undef, count)
    lo = [Inf, Inf, Inf]; hi = [-Inf, -Inf, -Inf]
    for i in 1:count
        skip(io, 12) # incoming normal is recomputed after transformation
        tri = ntuple(_ -> ntuple(_ -> Float64(read(io, Float32)), 3), 3)
        skip(io, 2)
        for v in tri, d in 1:3
            isfinite(v[d]) || throw(ArgumentError("STL has a nonfinite vertex"))
            lo[d] = min(lo[d], v[d]); hi[d] = max(hi[d], v[d])
        end
        tris[i] = tri
    end
    tris, (min=ntuple(i -> lo[i], 3), max=ntuple(i -> hi[i], 3))
end

function _write_stl(path::AbstractString, tris::Vector{Triangle})
    mkpath(dirname(path))
    open(path, "w") do io
        header = zeros(UInt8, 80)
        copyto!(header, 1, Vector{UInt8}(codeunits("Aero Lab transformed STL")), 1, 24)
        write(io, header)
        write(io, htol(UInt32(length(tris))))
        for tri in tris
            a, b, c = tri
            u = ntuple(i -> b[i] - a[i], 3); v = ntuple(i -> c[i] - a[i], 3)
            n = (u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1])
            mag = sqrt(sum(x*x for x in n))
            mag > 0 || throw(ArgumentError("STL contains a degenerate triangle"))
            for x in n; write(io, Float32(x/mag)); end
            for p in tri, x in p; write(io, Float32(x)); end
            write(io, UInt16(0))
        end
    end
    path
end

function _bounds(tris::Vector{Triangle})
    lo = [Inf, Inf, Inf]; hi = [-Inf, -Inf, -Inf]
    for tri in tris, p in tri, d in 1:3
        lo[d] = min(lo[d], p[d]); hi[d] = max(hi[d], p[d])
    end
    Dict("min" => lo, "max" => hi)
end

function upload_model(req::HTTP.Request)
    bytes = Vector{UInt8}(req.body)
    length(bytes) <= MAX_STL_BYTES || throw(ArgumentError("STL exceeds the 50 MB upload limit"))
    tris, bounds = _stl_triangles(bytes)
    tris, audit = _repair_winding(tris,bounds)
    model_id = string(uuid4())
    dir = joinpath(MODEL_ROOT, model_id)
    mkpath(dir)
    write(joinpath(dir, "original.stl"), bytes)
    _write_stl(joinpath(dir,"source.stl"),tris)
    meta = Dict{String,Any}("model_id" => model_id, "triangles" => length(tris),
                            "bounds" => Dict("min" => collect(bounds.min), "max" => collect(bounds.max)),
                            "geometry_audit" => audit)
    write(joinpath(dir, "metadata.json"), JSON3.write(meta))
    _response(201, merge(meta, Dict("ok" => true, "url" => "/api/aero/models/$model_id/file")))
end

"Repair local face winding and verify outward orientation for closed manifold components.
No backfaces are duplicated and no surface triangles are discarded. Open surfaces
have no uniquely inferable outside; their limitations are reported explicitly."
function _repair_winding(tris::Vector{Triangle},bounds)
    n=length(tris); span=maximum(bounds.max[i]-bounds.min[i] for i in 1:3)
    span>0 || throw(ArgumentError("Model has no spatial extent"))
    tolerance=max(span*1e-8,1e-14)
    center=ntuple(i->(bounds.min[i]+bounds.max[i])/2,3)
    vertices=Dict{NTuple{3,Int64},Int32}(); ids=Matrix{Int32}(undef,3,n)
    for face in 1:n, corner in 1:3
        point=tris[face][corner]
        key=ntuple(i->round(Int64,(point[i]-center[i])/tolerance),3)
        ids[corner,face]=get!(vertices,key,Int32(length(vertices)+1))
    end
    # Edge record: face, corner, direction, second face/corner/direction, count.
    edges=Dict{Tuple{Int32,Int32},Tuple{Int32,Int8,Int8,Int32,Int8,Int8,Int32}}()
    manifold=trues(n)
    for face in 1:n
        length(unique(ids[:,face]))==3 || throw(ArgumentError("Model contains a degenerate triangle; repair or remove zero-area faces before meshing"))
        for corner in 1:3
            a=ids[corner,face];b=ids[mod1(corner+1,3),face]; key=minmax(a,b); direction=Int8(a<b ? 1 : -1)
            if !haskey(edges,key)
                edges[key]=(Int32(face),Int8(corner),direction,Int32(0),Int8(0),Int8(0),Int32(1))
            else
                old=edges[key]
                edges[key]=old[7]==1 ? (old[1],old[2],old[3],Int32(face),Int8(corner),direction,Int32(2)) :
                    (old[1],old[2],old[3],old[4],old[5],old[6],old[7]+Int32(1))
                if old[7]>=2; manifold[face]=false;manifold[old[1]]=false;manifold[old[4]]=false; end
            end
        end
    end
    neighbors=zeros(Int32,3,n); opposite=falses(3,n);boundary_edges=0;nonmanifold_edges=0
    for (_,edge) in edges
        if edge[7]==1
            boundary_edges+=1;manifold[edge[1]]=false
        elseif edge[7]==2
            a,slot_a,dir_a,b,slot_b,dir_b,_=edge
            neighbors[slot_a,a]=b;neighbors[slot_b,b]=a
            opposite[slot_a,a]=opposite[slot_b,b]=dir_a==dir_b
        else
            nonmanifold_edges+=1
        end
    end
    flip=fill(Int8(-1),n);components=0;closed_components=0;conflicts=0;ambiguous=0
    for seed in 1:n
        flip[seed]>=0 && continue
        components+=1;queue=Int32[seed];flip[seed]=0;closed=true;consistent=true;index=1
        while index<=length(queue)
            face=queue[index];index+=1;closed&=manifold[face]
            for corner in 1:3
                neighbor=neighbors[corner,face];neighbor==0 && continue
                expected=Int8(xor(flip[face]==1,opposite[corner,face]))
                if flip[neighbor]<0;flip[neighbor]=expected;push!(queue,neighbor)
                elseif flip[neighbor]!=expected;consistent=false;conflicts+=1;end
            end
        end
        if closed&&consistent
            base=tris[seed][1];volume=0.0
            for face in queue
                tri=tris[face];a,b,c=flip[face]==1 ? (tri[1],tri[3],tri[2]) : tri
                a=ntuple(i->a[i]-base[i],3);b=ntuple(i->b[i]-base[i],3);c=ntuple(i->c[i]-base[i],3)
                volume+=(a[1]*(b[2]*c[3]-b[3]*c[2])+a[2]*(b[3]*c[1]-b[1]*c[3])+a[3]*(b[1]*c[2]-b[2]*c[1]))/6
            end
            if abs(volume)<=span^3*1e-15
                ambiguous+=1
            else
                closed_components+=1
                volume<0 && foreach(face->flip[face]=1-flip[face],queue)
            end
        end
    end
    repaired=Triangle[flip[i]==1 ? (tri[1],tri[3],tri[2]) : tri for (i,tri) in enumerate(tris)]
    warnings=String[]
    boundary_edges>0 && push!(warnings,"The model has $boundary_edges open boundary edges. A unique outside normal cannot be verified for open components; close the geometry or use a matching symmetry boundary.")
    nonmanifold_edges>0 && push!(warnings,"The model has $nonmanifold_edges nonmanifold edges. Repair coincident or intersecting surfaces before relying on forces.")
    conflicts>0 && push!(warnings,"Some connected faces have conflicting orientation constraints; global outward normals are unverified.")
    ambiguous>0 && push!(warnings,"Some closed components have near-zero signed volume; inspect thin or intersecting geometry.")
    components>1 && push!(warnings,"Closed components were oriented individually. Nested cavity shells still require a geometry audit to identify the material side.")
    repaired,Dict{String,Any}("triangles"=>n,"vertices"=>length(vertices),"components"=>components,
        "closed_components"=>closed_components,"boundary_edges"=>boundary_edges,
        "nonmanifold_edges"=>nonmanifold_edges,"orientation_conflicts"=>conflicts,
        "reoriented_faces"=>count(==(1),flip),"weld_tolerance_m"=>tolerance,
        "outward_orientation_verified"=>closed_components==components,"warnings"=>warnings)
end

function _model_geometry_audit(model_id::String)
    path=joinpath(MODEL_ROOT,model_id,"metadata.json")
    isfile(path) || return nothing
    metadata=_plain(JSON3.read(read(path,String)))
    get(metadata,"geometry_audit",nothing)
end

"Check that every open edge lies on the actual solver Y=0 plane after placement."
function _open_edges_match_symmetry(tris::Vector{Triangle},bounds;shift_y::Real=0.)
    lo=bounds["min"];hi=bounds["max"];tol=max(maximum(hi.-lo)*1e-8,1e-14)
    vertices=Dict{NTuple{3,Int64},Int}();edges=Dict{Tuple{Int,Int},Tuple{Int,Bool}}()
    for tri in tris, corner in 1:3
        a=tri[corner];b=tri[mod1(corner+1,3)]
        ka=ntuple(i->round(Int64,(a[i]-lo[i])/tol),3);kb=ntuple(i->round(Int64,(b[i]-lo[i])/tol),3)
        ia=get!(vertices,ka,length(vertices)+1);ib=get!(vertices,kb,length(vertices)+1)
        key=minmax(ia,ib);old=get(edges,key,(0,true))
        edges[key]=(old[1]+1,old[2]&&abs(a[2]+shift_y)<=tol&&abs(b[2]+shift_y)<=tol)
    end
    any(value[1]==1 for value in values(edges)) && all(value[1]==2 || (value[1]==1&&value[2]) for value in values(edges))
end

function _load_model(model_id::String)
    dir = joinpath(MODEL_ROOT, model_id)
    file = joinpath(dir, "source.stl")
    isfile(file) || throw(ArgumentError("Unknown model_id"))
    lock(JOB_LOCK) do
        tris,bounds = _stl_triangles(read(file))
        metadata_path = joinpath(dir,"metadata.json")
        metadata = isfile(metadata_path) ? _plain(JSON3.read(read(metadata_path,String))) : Dict{String,Any}("model_id"=>model_id)
        if !haskey(metadata,"geometry_audit")
            repaired,audit = _repair_winding(tris,bounds)
            original = joinpath(dir,"original.stl")
            isfile(original) || cp(file,original)
            temporary = joinpath(dir,"normalized.$(uuid4()).stl")
            _write_stl(temporary,repaired)
            mv(temporary,file;force=true)
            metadata["geometry_audit"] = audit
            metadata["triangles"] = length(repaired)
            metadata["bounds"] = Dict("min"=>collect(bounds.min),"max"=>collect(bounds.max))
            temporary_meta = metadata_path*".tmp"
            write(temporary_meta,JSON3.write(metadata))
            mv(temporary_meta,metadata_path;force=true)
            return repaired,bounds
        end
        tris,bounds
    end
end

function model_file(model_id::String)
    file = joinpath(MODEL_ROOT, model_id, "source.stl")
    isfile(file) || return _bad("Unknown model_id", 404)
    HTTP.Response(200, ["Content-Type" => "model/stl", "Cache-Control" => "no-store"], read(file))
end

"""Scale, then Euler rotate X/Y/Z about the source bounding-box centre.
Placement in the solver domain is configured separately from this shape transform.
"""
function _transform(tris::Vector{Triangle}, bounds, scale, angles)
    center = ntuple(i -> 0.5*(bounds.min[i] + bounds.max[i]), 3)
    ax, ay, az = deg2rad.(angles)
    sx,cx = sincos(ax); sy,cy = sincos(ay); sz,cz = sincos(az)
    function one(p)
        x = (p[1]-center[1])*scale[1]
        y = (p[2]-center[2])*scale[2]
        z = (p[3]-center[3])*scale[3]
        y,z = cx*y - sx*z, sx*y + cx*z
        x,z = cy*x + sy*z, -sy*x + cy*z
        x,y = cz*x - sz*y, sz*x + cz*y
        (x,y,z)
    end
    out = Triangle[ntuple(j -> one(t[j]), 3) for t in tris]
    out, _bounds(out)
end

"""Map source X-flow/Y-lateral/Z-up geometry to the solver's X-flow/Y-up frame.
The native ground_up_axis=z rotation has the opposite sign to the UI's world
convention, so Aero Lab performs this transform and asks the solver to keep Y up.
"""
function _ground_solver_triangles(tris::Vector{Triangle})
    Triangle[ntuple(j -> (t[j][1], t[j][3], -t[j][2]), 3) for t in tris]
end
