"""Conservative finite-volume ownership on the existing LBM Cartesian hierarchy.

The adapter changes neither the LBM block allocation nor its obstacle mask. It
removes covered parent cells, then stores each fluid/fluid face exactly once.
At a 2:1 interface a coarse face is represented by the four fine face patches.
Immersed walls are explicitly voxel/staircase walls: face areas and normals
close the full fluid voxel, not an unconstructed triangle cut cell.
"""
module PondLBMMesh

export PondLeafMesh, PondPointIndex, build_pond_lbm_mesh, build_point_index, find_leaf,
       POND_FARFIELD, POND_SLIP, POND_NOSLIP

const POND_FARFIELD = Int8(1)
const POND_SLIP = Int8(2)
const POND_NOSLIP = Int8(3)

struct PondPointIndex
    block_size::Int
    origin::NTuple{3,Float64}
    extent::NTuple{3,Float64}
    dx::Vector{Float64}
    blocks::Vector{Dict{NTuple{3,Int},Int32}}
    ids::Vector{Array{Int32,4}}
end

"""CPU topology; scalar face/cell arrays can be transferred to a GPU directly.

`neighbor == 0` means a boundary face. Face normal is `sign` on `axis`,
outward from `owner`. Signed CSR `cell_faces` contributes a positive flux to
the owner's outward sum and a negative flux to the neighbour's outward sum.
`level` indexes the supplied coarse-to-fine hierarchy; `local_index` is the
Julia linear index in its `block_size^3` interior cell array.
"""
struct PondLeafMesh
    x::Vector{Float64}; y::Vector{Float64}; z::Vector{Float64}
    dx::Vector{Float64}; volume::Vector{Float64}; wall_distance::Vector{Float64}
    level::Vector{Int16}; block::Vector{Int32}; local_index::Vector{Int16}
    owner::Vector{Int32}; neighbor::Vector{Int32}
    axis::Vector{Int8}; sign::Vector{Int8}; boundary::Vector{Int8}
    face_tag::Vector{UInt8}
    area::Vector{Float64}
    face_x::Vector{Float64}; face_y::Vector{Float64}; face_z::Vector{Float64}
    distance::Vector{Float64}
    cell_face_offsets::Vector{Int32}; cell_faces::Vector{Int32}
    origin::NTuple{3,Float64}; extent::NTuple{3,Float64}
    metadata::Dict{String,Any}
    point_index::PondPointIndex
end

"Return the retained block-local fluid cell lookup; no duplicate cell dictionary is built."
build_point_index(mesh::PondLeafMesh)=mesh.point_index

"Return the fluid leaf cell ID at a physical point; zero means solid or outside."
function find_leaf(index::PondPointIndex,x::Real,y::Real,z::Real)
    p=(Float64(x),Float64(y),Float64(z))
    all(a->isfinite(p[a]) && index.origin[a]<=p[a]<index.origin[a]+index.extent[a],1:3) || return Int32(0)
    bs=index.block_size
    for l in length(index.dx):-1:1
        g=ntuple(a->floor(Int,(p[a]-index.origin[a])/index.dx[l]),3)
        b=get(index.blocks[l],ntuple(a->fld(g[a],bs)+1,3),Int32(0))
        b==0 && continue
        i,j,k=ntuple(a->mod(g[a],bs)+1,3)
        id=@inbounds index.ids[l][i,j,k,b]
        id<0 && return Int32(0)
        id>0 && return id
    end
    Int32(0)
end
find_leaf(index::PondPointIndex,p)=find_leaf(index,p[1],p[2],p[3])
find_leaf(mesh::PondLeafMesh,args...)=find_leaf(mesh.point_index,args...)

struct _Level
    dx::Float64
    coords::Vector{NTuple{3,Int}}
    blocks::Dict{NTuple{3,Int},Int32}
    ids::Array{Int32,4} # >0 fluid leaf, -1 solid leaf, 0 covered parent
    obstacle::Array{Bool,4}
    wall_distance::Union{Nothing,Array{Float64,4}}
    covered::Union{Nothing,Array{Bool,4}}
end

@inline function _slot(level::_Level, g::NTuple{3,Int}, bs::Int)
    b = get(level.blocks, ntuple(k->fld(g[k],bs)+1,3), Int32(0))
    b == 0 && return (Int32(0), 0, 0, 0, Int32(-2))
    i,j,k = ntuple(k->mod(g[k],bs)+1,3)
    (b,i,j,k,@inbounds(level.ids[i,j,k,b]))
end

"""
    build_pond_lbm_mesh(levels; block_size=8, origin=(0,0,0),
        boundary_codes=(1,1,1,1,1,1), solid_boundary=3,
        periodic=(false,false,false), check_covered=true)

Build a shared-face mesh from actual LBM `BlockLevel`s, or objects exposing
`dx`, `active_block_coords`, `obstacle`, and optional `wall_dist`/`amr_covered`.
No population arrays are read or allocated. All lengths use the input's units.
Boundary order is x-, x+, y-, y+, z-, z+. Codes: farfield=1, slip=2, no-slip=3.
Periodic pairs have a real neighbour and code zero; their face distance is the
wrapped normal centre distance. `face_tag`: 0 internal/periodic, 1:6 outer,
7 immersed voxel wall. Existing covered masks are verified when present.

The coarsest level must cover its rectangular block envelope, all refinement
must be nested and face-balanced 2:1. Missing blocks, inconsistent ownership,
or non-closing faces are errors rather than artificial farfield boundaries.
Construction uses block lookups and local cell arrays, not all-pairs searches
or a dense grid at the finest level.
"""
function build_pond_lbm_mesh(levels; block_size::Int=8, origin=(0.,0.,0.),
        boundary_codes=(1,1,1,1,1,1), solid_boundary::Integer=3,
        periodic=(false,false,false), check_covered::Bool=true)
    bs=block_size
    bs>=2 && iseven(bs) && bs^3<=typemax(Int16) || throw(ArgumentError("block_size must be even and its cube fit Int16"))
    1<=length(levels)<=typemax(Int16) || throw(ArgumentError("empty or excessive mesh hierarchy"))
    length(origin)==3 && all(isfinite,origin) || throw(ArgumentError("origin needs three finite coordinates"))
    length(boundary_codes)==6 && all(c->c in 1:3,boundary_codes) || throw(ArgumentError("boundary_codes must contain six codes in 1:3"))
    solid_boundary in 2:3 || throw(ArgumentError("solid boundary must be slip or no-slip"))
    length(periodic)==3 || throw(ArgumentError("periodic needs three booleans"))
    org=Tuple(Float64.(origin)); per=Tuple(Bool.(periodic)); nl=length(levels)
    data=_Level[]
    for (l,raw) in enumerate(levels)
        h=Float64(raw.dx)
        isfinite(h) && h>0 || throw(ArgumentError("invalid cell size on level $l"))
        l==1 || isapprox(data[end].dx,2h;rtol=1e-12,atol=0) || throw(ArgumentError("PonD adapter requires dyadic 2:1 levels"))
        coords=NTuple{3,Int}[Tuple(Int.(c)) for c in raw.active_block_coords]
        all(c->all(>(0),c),coords) || throw(ArgumentError("LBM block coordinates must be one-based and positive"))
        length(coords)<=typemax(Int32) || throw(ArgumentError("too many mesh blocks"))
        blocks=Dict(c=>Int32(b) for (b,c) in enumerate(coords))
        length(blocks)==length(coords) || throw(ArgumentError("duplicate block coordinates on level $l"))
        shape=(bs,bs,bs,length(coords))
        obs=Array{Bool,4}(Array(raw.obstacle))
        size(obs)==shape || throw(ArgumentError("obstacle shape does not match level $l blocks"))
        wd=hasproperty(raw,:wall_dist) && length(raw.wall_dist)>1 ? Float64.(Array(raw.wall_dist)) : nothing
        wd===nothing || size(wd)==shape || throw(ArgumentError("wall distance shape does not match level $l"))
        cov=hasproperty(raw,:amr_covered) && length(raw.amr_covered)>1 ? Array{Bool,4}(Array(raw.amr_covered)) : nothing
        cov===nothing || size(cov)==shape || throw(ArgumentError("covered mask shape does not match level $l"))
        push!(data,_Level(h,coords,blocks,zeros(Int32,shape),obs,wd,cov))
    end
    isempty(data[1].coords) && throw(ArgumentError("coarsest level has no blocks"))
    dims=ntuple(k->maximum(c[k] for c in data[1].coords),3)
    prod(BigInt.(dims))==length(data[1].coords) || throw(ArgumentError("coarsest LBM block envelope has holes"))
    for l in 2:nl, c in data[l].coords
        p=ntuple(k->cld(c[k],2),3)
        haskey(data[l-1].blocks,p) || throw(ArgumentError("level $l block $c has no parent block"))
    end
    ext=ntuple(k->dims[k]*bs*data[1].dx,3)
    x=Float64[];y=Float64[];z=Float64[];dx=Float64[];volume=Float64[];wall_distance=Float64[]
    level=Int16[];block=Int32[];local_index=Int16[]
    fluid_counts=zeros(Int,nl);solid_counts=zeros(Int,nl);covered_counts=zeros(Int,nl)
    for l in 1:nl
        L=data[l];h=L.dx
        for (b,c) in enumerate(L.coords), k in 1:bs, j in 1:bs, i in 1:bs
            g=((c[1]-1)*bs+i-1,(c[2]-1)*bs+j-1,(c[3]-1)*bs+k-1)
            covered=l<nl && haskey(data[l+1].blocks,ntuple(a->fld(2g[a],bs)+1,3))
            if check_covered && L.covered!==nothing && L.covered[i,j,k,b]!=covered
                throw(ArgumentError("LBM covered mask disagrees with child topology at level $l block $b cell ($i,$j,$k)"))
            end
            if covered
                covered_counts[l]+=1
            elseif L.obstacle[i,j,k,b]
                L.ids[i,j,k,b]=-1;solid_counts[l]+=1
            else
                n=length(x)+1;n<=typemax(Int32) || throw(ArgumentError("too many finite-volume cells for Int32 topology"))
                L.ids[i,j,k,b]=Int32(n);fluid_counts[l]+=1
                push!(x,org[1]+(g[1]+.5)*h);push!(y,org[2]+(g[2]+.5)*h);push!(z,org[3]+(g[3]+.5)*h)
                push!(dx,h);push!(volume,h^3)
                push!(wall_distance,L.wall_distance===nothing ? Inf : L.wall_distance[i,j,k,b])
                push!(level,Int16(l));push!(block,Int32(b));push!(local_index,Int16(i+(j-1)*bs+(k-1)*bs^2))
            end
        end
    end
    n=length(x);n>0 || throw(ArgumentError("mesh has no fluid leaf cells"))
    owner=Int32[];neighbor=Int32[];axis=Int8[];sgn=Int8[];boundary=Int8[];face_tag=UInt8[]
    area=Float64[];face_x=Float64[];face_y=Float64[];face_z=Float64[];distance=Float64[]
    for a in (owner,neighbor,axis,sgn,boundary,face_tag,area,face_x,face_y,face_z,distance)
        sizehint!(a,3n)
    end
    function emit!(p::Int,a::Int,s::Int,q::Int32,ql::Int,qg::NTuple{3,Int},bc::Int,tag::Int)
        q>0 && !(p<q || (p==q && s>0)) && return
        h=dx[p];hq=ql>0 ? data[ql].dx : h; hf=min(h,hq)
        center=(x[p],y[p],z[p])
        fc=ntuple(d->d==a ? center[d]+s*h/2 : (hq<h ? org[d]+(qg[d]+.5)*hq : center[d]),3)
        length(owner)<typemax(Int32) || throw(ArgumentError("too many faces for Int32 topology"))
        push!(owner,Int32(p));push!(neighbor,max(q,Int32(0)));push!(axis,Int8(a));push!(sgn,Int8(s))
        push!(boundary,Int8(bc));push!(face_tag,UInt8(tag));push!(area,hf^2)
        push!(face_x,fc[1]);push!(face_y,fc[2]);push!(face_z,fc[3])
        push!(distance,q>0 ? (h+hq)/2 : h/2)
    end
    function connect!(p::Int,a::Int,s::Int,l::Int,g::NTuple{3,Int})
        _,_,_,_,q=_slot(data[l],g,bs)
        if q==-2
            l>1 || throw(ArgumentError("uncovered hole in LBM domain at cell $g"))
            connect!(p,a,s,l-1,ntuple(d->fld(g[d],2),3))
        elseif q==0
            l<nl || error("covered cell without finer level")
            tangents=filter(!=(a),(1,2,3))
            for v in 0:1,u in 0:1
                fg=ntuple(d->2g[d]+(d==a ? (s>0 ? 0 : 1) : d==tangents[1] ? u : v),3)
                connect!(p,a,s,l+1,fg)
            end
        else
            abs(Int(level[p])-l)<=1 || throw(ArgumentError("LBM mesh has an unbalanced face touching levels $(level[p]) and $l"))
            emit!(p,a,s,q,l,g,q>0 ? 0 : Int(solid_boundary),q>0 ? 0 : 7)
        end
    end
    for p in 1:n
        l=Int(level[p]);L=data[l];c=L.coords[block[p]];idx=Int(local_index[p])-1
        loc=(mod(idx,bs),mod(fld(idx,bs),bs),fld(idx,bs^2))
        g=ntuple(a->(c[a]-1)*bs+loc[a],3)
        nd=ntuple(a->dims[a]*bs*(1<<(l-1)),3)
        for a in 1:3,s in (-1,1)
            ng=ntuple(d->g[d]+(d==a ? s : 0),3)
            if ng[a]<0 || ng[a]>=nd[a]
                if per[a]
                    ng=Base.setindex(ng,mod(ng[a],nd[a]),a)
                    connect!(p,a,s,l,ng)
                else
                    tag=2a-(s<0 ? 1 : 0)
                    emit!(p,a,s,Int32(0),0,g,Int(boundary_codes[tag]),tag)
                end
            else
                connect!(p,a,s,l,ng)
            end
        end
    end
    counts=zeros(Int32,n)
    for f in eachindex(owner)
        counts[owner[f]]+=1
        neighbor[f]>0 && (counts[neighbor[f]]+=1)
    end
    sum(Int64,counts)+1<=typemax(Int32) || throw(ArgumentError("too many cell-face incidences for Int32 topology"))
    offsets=Vector{Int32}(undef,n+1);offsets[1]=1
    for p in 1:n;offsets[p+1]=offsets[p]+counts[p];end
    incidences=Vector{Int32}(undef,offsets[end]-1);cursor=copy(offsets[1:end-1])
    for f in eachindex(owner)
        p=owner[f];incidences[cursor[p]]=Int32(f);cursor[p]+=1
        if neighbor[f]>0
            q=neighbor[f];incidences[cursor[q]]=-Int32(f);cursor[q]+=1
        end
    end
    max_closure=0.0;max_area_error=0.0
    for p in 1:n
        closure=(0.0,0.0,0.0);total_area=0.0
        for k in offsets[p]:offsets[p+1]-1
            sf=incidences[k];f=abs(sf);a=axis[f]
            v=(sf>0 ? 1.0 : -1.0)*sgn[f]*area[f]
            closure=Base.setindex(closure,closure[a]+v,a);total_area+=area[f]
        end
        max_closure=max(max_closure,maximum(abs,closure)/dx[p]^2)
        max_area_error=max(max_area_error,abs(total_area/dx[p]^2-6))
    end
    max_closure<=1e-10 && max_area_error<=1e-10 || error("finite-volume mesh faces do not close (normal=$max_closure, area=$max_area_error)")
    solid_volume=sum(solid_counts[l]*data[l].dx^3 for l in 1:nl)
    domain_volume=prod(ext);leaf_volume=sum(volume)+solid_volume
    isapprox(leaf_volume,domain_volume;rtol=1e-10,atol=0) || error("leaf-cell volumes do not partition the LBM envelope")
    metadata=Dict{String,Any}("version"=>"pond_lbm_leaf_faces_v1","block_size"=>bs,
        "fluid_cells"=>n,"solid_leaf_cells"=>sum(solid_counts),"faces"=>length(owner),
        "fluid_cells_by_level"=>fluid_counts,"solid_leaf_cells_by_level"=>solid_counts,
        "covered_cells_by_level"=>covered_counts,"domain_volume"=>domain_volume,
        "fluid_volume"=>sum(volume),"solid_volume"=>solid_volume,
        "max_relative_face_closure"=>max_closure,"max_relative_face_area_error"=>max_area_error,
        "periodic"=>collect(per),"boundary_codes"=>Int.(collect(boundary_codes)),
        "wall_geometry"=>"LBM obstacle-mask voxel staircase; no triangle cut-cell reconstruction")
    point_index=PondPointIndex(bs,org,ext,[L.dx for L in data],[L.blocks for L in data],[L.ids for L in data])
    PondLeafMesh(x,y,z,dx,volume,wall_distance,level,block,local_index,
        owner,neighbor,axis,sgn,boundary,face_tag,area,face_x,face_y,face_z,distance,
        offsets,incidences,org,ext,metadata,point_index)
end

end # module
