# Deterministic opt-in normal-grid redistribution; include after cmesh_gen.jl.
# No solver/physics imports, filesystem reads, environment reads, or mesh repairs.
module CMeshFixedOuterMesh
using SHA
import ..CMeshGen
export FIXED_OUTER_VERSION, fixed_outer_mesh, fixed_outer_provenance, fixed_outer_settings, fixed_outer_restart_guard
const FIXED_OUTER_VERSION=1
const FIRST_SPACING_RTOL=1e-6
const _PROVENANCE_FIELDS=(:version,:reference_dn1,:target_dn1,:transition_j,:growth_limit,
    :reference_x_sha256,:reference_y_sha256,:result_x_sha256,:result_y_sha256,
    :frozen_x_sha256,:frozen_y_sha256)

"""
    fixed_outer_restart_guard(metadata, active_provenance)

Bind a restart to the complete active transform recipe and coordinate hashes before stepping.
Inactive runs accept legacy metadata with no fixed_outer field, including nothing, but reject any
such field even when its value is nothing. Active runs require a well-formed eleven-field provenance
and NamedTuple metadata whose fixed_outer value is exactly isequal to it. This does not replace the
solver's independent metric/state restart checks. Returns nothing on success.
"""
function fixed_outer_restart_guard(metadata,active_provenance)
    marked=hasproperty(metadata,:fixed_outer) ||
        (metadata isa AbstractDict && (haskey(metadata,:fixed_outer)||haskey(metadata,"fixed_outer")))
    if active_provenance===nothing
        !marked || throw(ArgumentError("fixed-outer checkpoint requires the matching active transform"))
        return nothing
    end
    p=active_provenance
    p isa NamedTuple && Set(keys(p))==Set(_PROVENANCE_FIELDS) ||
        throw(ArgumentError("malformed active fixed-outer provenance fields"))
    p.version isa Integer && !(p.version isa Bool) && p.version==FIXED_OUTER_VERSION &&
        p.transition_j isa Integer && !(p.transition_j isa Bool) && p.transition_j>=4 ||
        throw(ArgumentError("malformed active fixed-outer provenance version or transition"))
    all(v->v isa Real && !(v isa Bool) && isfinite(v) && v>0,
        (p.reference_dn1,p.target_dn1,p.growth_limit)) && p.reference_dn1!=p.target_dn1 && p.growth_limit>=1 ||
        throw(ArgumentError("malformed active fixed-outer provenance spacing or growth"))
    all(k->getproperty(p,k) isa AbstractString && occursin(r"^[0-9a-f]{64}$",getproperty(p,k)),
        _PROVENANCE_FIELDS[6:end]) || throw(ArgumentError("malformed active fixed-outer provenance hashes"))
    metadata isa NamedTuple && marked && metadata.fixed_outer isa NamedTuple &&
        isequal(metadata.fixed_outer,p) || throw(ArgumentError("fixed-outer restart provenance does not match active transform"))
    nothing
end

"Pure parser for the all-or-none opt-in controls; does not read ENV implicitly."
function fixed_outer_settings(environment,target_dn1,n_normal)
    names=("TOPO_FIXED_OUTER_REF_DN1","TOPO_FIXED_OUTER_J","TOPO_FIXED_OUTER_VERSION")
    present=map(k->haskey(environment,k),names)
    !any(present) && return nothing
    all(present) || throw(ArgumentError("fixed-outer controls must specify REF_DN1, J and VERSION together"))
    parsed=try
        (parse(Float64,environment[names[1]]),parse(Int,environment[names[2]]),parse(Int,environment[names[3]]))
    catch
        throw(ArgumentError("malformed fixed-outer controls"))
    end
    ref,J,version=parsed
    target_dn1 isa Real && !(target_dn1 isa Bool) && isfinite(target_dn1) && target_dn1>0 ||
        throw(ArgumentError("fixed-outer target dn1 must be finite positive"))
    isfinite(ref)&&ref>0&&ref!=target_dn1 || throw(ArgumentError("fixed-outer reference dn1 must be finite positive and differ from target"))
    n_normal isa Integer && !(n_normal isa Bool) && 4<=J<=n_normal || throw(ArgumentError("invalid fixed-outer transition node row"))
    version==FIXED_OUTER_VERSION || throw(ArgumentError("unsupported fixed-outer version"))
    (;reference_dn1=ref,target_dn1=Float64(target_dn1),transition_j=J,version)
end

_hash(a)=bytes2hex(sha256(reinterpret(UInt8,vec(a))))
_spacing(m)=hypot.(diff(m.X;dims=2),diff(m.Y;dims=2))

function _options(m,target,J,limit,version)
    m isa CMeshGen.ChannelMesh || throw(ArgumentError("fixed-outer transform requires ChannelMesh"))
    # (strip topology, 5 Oct 2026: the transform rebuilds the C block alone and would drop the strip; the driver refuses
    # the trailing-edge knobs with this control before any mesh is built)
    m.strip===nothing || throw(ArgumentError("fixed-outer transform does not support the strip topology (te_topology=strip)"))
    version isa Integer && !(version isa Bool) && version==FIXED_OUTER_VERSION ||
        throw(ArgumentError("unsupported fixed-outer version"))
    J isa Integer && !(J isa Bool) && 4<=J<=m.NJ || throw(ArgumentError("invalid fixed-outer transition node row"))
    target isa Real && !(target isa Bool) && isfinite(target) && target>0 ||
        throw(ArgumentError("fixed-outer target dn1 must be finite positive"))
    limit isa Real && !(limit isa Bool) && isfinite(limit) && limit>=1 ||
        throw(ArgumentError("fixed-outer growth limit must be finite and >=1"))
    isfinite(m.dn1) && m.dn1>0 || throw(ArgumentError("invalid fixed-outer reference dn1"))
    size(m.X)==size(m.Y)==(m.NI+1,m.NJ+1) || throw(ArgumentError("fixed-outer coordinate dimensions disagree"))
    length(m.outer_face_kind)==m.NI && length(m.growth)==m.NI+1 ||
        throw(ArgumentError("fixed-outer metadata dimensions disagree"))
    nothing
end

function _geometry(m,limit)
    all(isfinite,m.X)&&all(isfinite,m.Y) || throw(ArgumentError("fixed-outer coordinates are nonfinite"))
    all(>(0),_spacing(m)) || throw(ArgumentError("fixed-outer has zero column spacing"))
    nbad,_,_=CMeshGen._convexity_defects(m.X,m.Y,m.NI,1:m.NJ)
    nbad==0 || throw(ArgumentError("fixed-outer has nonconvex cells"))
    ratio=CMeshGen._radial_spacing_stats(m.X,m.Y,m.NI,m.NJ).ratio_symmetric_max
    ratio<=limit+512eps(Float64)*max(1.,Float64(limit)) ||
        throw(ArgumentError("fixed-outer adjacent spacing exceeds growth limit"))
    nothing
end

function _redistribute(h,target)
    N=length(h);total=sum(h)
    total>target+h[end] || throw(ArgumentError("fixed-outer target leaves no positive middle spacing"))
    target==h[1] && return copy(h)
    t=collect(range(0.,1.;length=N));w=(1 .-t).^2;v=t.*(1 .-t);A=log(target/h[1])
    evaluate(B)=h.*exp.(A.*w.+B.*v)
    residual(B)=sum(evaluate(B))-total
    lo=-1.;hi=1.
    while residual(lo)>0
        lo*=2;lo > -1e6 || throw(ArgumentError("fixed-outer lower root bracket failed"))
    end
    while residual(hi)<0
        hi*=2;hi < 1e6 || throw(ArgumentError("fixed-outer upper root bracket failed"))
    end
    for _ in 1:100
        mid=(lo+hi)/2
        residual(mid)>0 ? (hi=mid) : (lo=mid)
    end
    hp=evaluate((lo+hi)/2)
    all(isfinite,hp)&&all(>(0),hp)&&isapprox(sum(hp),total;rtol=5e-14)&&
        isapprox(hp[1],target;rtol=5e-14)&&hp[end]==h[end] ||
        throw(ArgumentError("fixed-outer spacing root failed its invariants"))
    hp
end

function _preserved(reference,result,J)
    (reference.NI,reference.NJ,reference.NIW,reference.NIF)==(result.NI,result.NJ,result.NIW,result.NIF) ||
        throw(ArgumentError("fixed-outer dimensions changed"))
    isequal(reference.X[:,1],result.X[:,1])&&isequal(reference.Y[:,1],result.Y[:,1]) ||
        throw(ArgumentError("fixed-outer wall/wake-cut nodes moved"))
    isequal(reference.X[:,J:end],result.X[:,J:end])&&isequal(reference.Y[:,J:end],result.Y[:,J:end]) ||
        throw(ArgumentError("fixed-outer frozen nodes moved"))
    reference.channel==result.channel && reference.outer_face_kind==result.outer_face_kind ||
        throw(ArgumentError("fixed-outer channel provenance changed"))
    nothing
end

"""
    fixed_outer_mesh(mesh; target_dn1=nothing, transition_j=nothing, growth_limit=1.20, version=1)

With both controls absent, return `mesh` itself without reads or changes. Otherwise redistribute nodes
2:J-1 along each reference column polyline. Every node at j>=J and j=1 stays bitwise identical. Positive
exponential segment weights preserve the reference arc-coordinate span and last inner segment.
Reconnecting sampled points can shorten the new polyline when reference corners are skipped. The actual Euclidean
first spacing must equal target_dn1 within 1e-6 relative; strict convexity and actual adjacent spacing
are enforced. No smoothing, global redistribution or rescue repair is performed.
"""
function fixed_outer_mesh(m;target_dn1=nothing,transition_j=nothing,growth_limit=1.20,version=FIXED_OUTER_VERSION)
    target_dn1===nothing && transition_j===nothing && return m
    _options(m,target_dn1,transition_j,growth_limit,version);_geometry(m,growth_limit)
    target=Float64(target_dn1);J=Int(transition_j);h=_spacing(m)
    # Identity is exactly reversible; do not round-trip existing node coordinates through interpolation.
    target==m.dn1 && maximum(abs.(h[:,1]./target.-1))<=FIRST_SPACING_RTOL && return m
    X=copy(m.X);Y=copy(m.Y)
    for i in 1:m.NI+1
        hi=vec(h[i,1:J-1]);hp=_redistribute(hi,target)
        old=[0.;cumsum(hi)];new=[0.;cumsum(hp)]
        for j in 2:J-1
            k=clamp(searchsortedlast(old,new[j]),1,J-1);f=(new[j]-old[k])/(old[k+1]-old[k])
            X[i,j]=(1-f)*m.X[i,k]+f*m.X[i,k+1]
            Y[i,j]=(1-f)*m.Y[i,k]+f*m.Y[i,k+1]
        end
    end
    # No single geometric ratio describes the transformed schedule. Store each column's measured max
    # symmetric ratio as descriptive metadata; admission independently recomputes it from coordinates.
    hh=hypot.(diff(X;dims=2),diff(Y;dims=2));ratios=hh[:,2:end]./hh[:,1:end-1]
    growth=vec(maximum(max.(ratios,1 ./ratios);dims=2))
    n=CMeshGen.ChannelMesh(m.NI,m.NJ,m.NIW,m.NIF,X,Y,target,growth,m.channel,copy(m.outer_face_kind))
    _preserved(m,n,J);_geometry(n,growth_limit)
    maximum(abs.(hh[:,1]./target.-1))<=FIRST_SPACING_RTOL ||
        throw(ArgumentError("fixed-outer actual first spacing differs from target"))
    n
end

"Return coordinate-bound deterministic transform provenance after independently checking preservation."
function fixed_outer_provenance(reference,result;transition_j,growth_limit=1.20,version=FIXED_OUTER_VERSION)
    _options(reference,result.dn1,transition_j,growth_limit,version)
    J=Int(transition_j);_preserved(reference,result,J);_geometry(result,growth_limit)
    maximum(abs.(_spacing(result)[:,1]./result.dn1.-1))<=FIRST_SPACING_RTOL ||
        throw(ArgumentError("fixed-outer provenance first spacing differs"))
    (;version=Int(version),reference_dn1=reference.dn1,target_dn1=result.dn1,transition_j=J,
      growth_limit=Float64(growth_limit),reference_x_sha256=_hash(reference.X),reference_y_sha256=_hash(reference.Y),
      result_x_sha256=_hash(result.X),result_y_sha256=_hash(result.Y),
      frozen_x_sha256=_hash(result.X[:,J:end]),frozen_y_sha256=_hash(result.Y[:,J:end]))
end
end
