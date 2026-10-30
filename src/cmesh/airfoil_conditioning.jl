# Geometry only, Julia Base. Included inside CMeshGen after AirfoilPts.
export condition_airfoil, airfoil_curvature, naca4_points

"Equation-generated NACA four-digit contour, lower TE to LE to upper TE."
function naca4_points(code::Integer; n::Int=4097)
    0 <= code <= 9999 || throw(ArgumentError("NACA code must lie in 0000:9999"))
    n >= 21 && isodd(n) || throw(ArgumentError("NACA point count must be odd and at least 21"))
    m=div(code,1000)/100; p=div(code%1000,100)/10; t=(code%100)/100
    t > 0 || throw(ArgumentError("NACA airfoil thickness must be positive"))
    nside=(n+1)÷2
    xu=zeros(nside); yu=zeros(nside); xl=zeros(nside); yl=zeros(nside)
    for k in 1:nside
        x=0.5*(1-cospi((k-1)/(nside-1)))
        yt=5t*(0.2969sqrt(x)-0.1260x-0.3516x^2+0.2843x^3-0.1015x^4)
        yc=0.0; slope=0.0
        if p>0
            if x<p
                yc=m/p^2*(2p*x-x^2); slope=2m/p^2*(p-x)
            else
                yc=m/(1-p)^2*((1-2p)+2p*x-x^2); slope=2m/(1-p)^2*(p-x)
            end
        end
        theta=atan(slope)
        xu[k]=x-yt*sin(theta); yu[k]=yc+yt*cos(theta)
        xl[k]=x+yt*sin(theta); yl[k]=yc-yt*cos(theta)
    end
    AirfoilPts(vcat(reverse(xl),xu[2:end]),vcat(reverse(yl),yu[2:end]))
end

struct _AirfoilSpline
    s::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    mx::Vector{Float64}
    my::Vector{Float64}
end

"Natural C2 cubic second derivatives for nonuniform arc-length knots."
function _airfoil_second_derivatives(s, q)
    n = length(s)
    diagonal = ones(n)
    upper = zeros(n-1)
    lower = zeros(n-1)
    rhs = zeros(n)
    for k in 2:n-1
        a = s[k]-s[k-1]; b = s[k+1]-s[k]
        lower[k-1] = a
        diagonal[k] = 2(a+b)
        upper[k] = b
        rhs[k] = 6((q[k+1]-q[k])/b - (q[k]-q[k-1])/a)
    end
    for k in 2:n
        a = lower[k-1]/diagonal[k-1]
        diagonal[k] -= a*upper[k-1]
        rhs[k] -= a*rhs[k-1]
    end
    result = zeros(n)
    result[end] = rhs[end]/diagonal[end]
    for k in n-1:-1:1
        result[k] = (rhs[k]-upper[k]*result[k+1])/diagonal[k]
    end
    result
end

function _airfoil_spline(af::AirfoilPts)
    length(af.x) == length(af.y) >= 8 ||
        throw(ArgumentError("airfoil conditioning requires at least eight coordinate pairs"))
    all(isfinite,af.x) && all(isfinite,af.y) ||
        throw(ArgumentError("airfoil coordinates must be finite"))
    # Consecutive duplicate points make arc-length parameterisation singular.
    keep = Int[1]
    for k in 2:length(af.x)
        hypot(af.x[k]-af.x[keep[end]], af.y[k]-af.y[keep[end]]) > 1e-13 && push!(keep,k)
    end
    x = af.x[keep]; y = af.y[keep]
    s = zeros(length(x))
    for k in 2:length(x)
        s[k] = s[k-1]+hypot(x[k]-x[k-1],y[k]-y[k-1])
    end
    s[end] > 0 || throw(ArgumentError("airfoil has no contour length"))
    _AirfoilSpline(s,x,y,_airfoil_second_derivatives(s,x),
                          _airfoil_second_derivatives(s,y))
end

function _airfoil_eval(curve::_AirfoilSpline, query::Real)
    s = clamp(Float64(query),0.0,curve.s[end])
    k = clamp(searchsortedlast(curve.s,s),1,length(curve.s)-1)
    h = curve.s[k+1]-curve.s[k]
    a = (curve.s[k+1]-s)/h; b = (s-curve.s[k])/h
    function evaluate(q,m)
        value = a*q[k]+b*q[k+1]+((a^3-a)*m[k]+(b^3-b)*m[k+1])*h^2/6
        derivative = (q[k+1]-q[k])/h + h*((1-3a^2)*m[k]+(3b^2-1)*m[k+1])/6
        second = a*m[k]+b*m[k+1]
        (value,derivative,second)
    end
    x,dx,ddx = evaluate(curve.x,curve.mx)
    y,dy,ddy = evaluate(curve.y,curve.my)
    (; x,y,dx,dy,ddx,ddy,s)
end

"Signed curvature along the open lower-TE to upper-TE contour; no gap-closing segment."
function airfoil_curvature(af::AirfoilPts)
    curve = _airfoil_spline(af)
    length(curve.s) == length(af.x) ||
        return [begin
            p = _airfoil_closest(curve,af.x[k],af.y[k])
            (p.dx*p.ddy-p.dy*p.ddx)/max(hypot(p.dx,p.dy)^3,1e-30)
        end for k in eachindex(af.x)]
    [begin
        p = _airfoil_eval(curve,s)
        (p.dx*p.ddy-p.dy*p.ddx)/max(hypot(p.dx,p.dy)^3,1e-30)
    end for s in curve.s]
end

"Closest spline point, with a segment bracket followed by a safeguarded Newton search."
function _airfoil_closest(curve::_AirfoilSpline,x::Real,y::Real)
    best = Inf; index = 1; query = 0.0
    for k in 1:length(curve.s)-1
        tx=curve.x[k+1]-curve.x[k]; ty=curve.y[k+1]-curve.y[k]
        t=clamp(((x-curve.x[k])*tx+(y-curve.y[k])*ty)/(tx*tx+ty*ty),0.0,1.0)
        distance=(curve.x[k]+t*tx-x)^2+(curve.y[k]+t*ty-y)^2
        if distance < best
            best=distance; index=k
            query=curve.s[k]+t*(curve.s[k+1]-curve.s[k])
        end
    end
    lo=curve.s[max(1,index-1)]; hi=curve.s[min(end,index+2)]
    for _ in 1:10
        p=_airfoil_eval(curve,query)
        f=(p.x-x)*p.dx+(p.y-y)*p.dy
        df=p.dx^2+p.dy^2+(p.x-x)*p.ddx+(p.y-y)*p.ddy
        abs(df) > 1e-15 || break
        next=clamp(query-f/df,lo,hi)
        abs(next-query) < 1e-13 && break
        query=next
    end
    _airfoil_eval(curve,query)
end

"""Reconstruct a smooth, source-interpolating open airfoil contour before meshing.

Both native trailing-edge endpoints are retained exactly. A positive requested
thickness changes the trailing edge explicitly using a C2 taper over x/c > 0.8;
zero means native geometry. The metadata records this change for comparisons.
The natural C2 arc-length spline removes the nonuniform Catmull-Rom tangent
jumps without fitting through the finite trailing-edge closing segment.
"""
function condition_airfoil(af::AirfoilPts; n::Int=4097,
                           trailing_edge_thickness::Real=0.0)
    n >= 257 || throw(ArgumentError("conditioned airfoil needs at least 257 points"))
    thickness = Float64(trailing_edge_thickness)
    isfinite(thickness) && 0 <= thickness <= 0.05 ||
        throw(ArgumentError("trailing-edge thickness must be between 0 and 0.05 chords"))
    original_gap = hypot(af.x[1]-af.x[end],af.y[1]-af.y[end])
    x=copy(af.x); y=copy(af.y)
    if thickness > 0
        ile=argmin(x)
        # Endpoints are placed about the original base midpoint in its native
        # cross-chord direction; a sharp source uses the vertical direction.
        nx,ny = original_gap > 1e-12 ?
            ((x[end]-x[1])/original_gap,(y[end]-y[1])/original_gap) : (0.0,1.0)
        change=0.5*(thickness-original_gap)
        for k in eachindex(x)
            t=clamp((af.x[k]-0.8)/0.2,0.0,1.0)
            w=t^3*(10-15t+6t^2)
            side=k <= ile ? -1.0 : 1.0
            x[k] += side*change*w*nx
            y[k] += side*change*w*ny
        end
    end
    curve=_airfoil_spline(AirfoilPts(x,y))
    # Include every source knot as well as dense stations so source ordinates
    # remain inspectable, and the true source point is never lost by preview.
    stations=sort!(unique(vcat(collect(range(0.0,curve.s[end];length=n)),curve.s)))
    points=[_airfoil_eval(curve,s) for s in stations]
    dense=AirfoilPts([p.x for p in points],[p.y for p in points])
    dense.x[1]=x[1]; dense.y[1]=y[1]
    dense.x[end]=x[end]; dense.y[end]=y[end]
    curvature=[(p.dx*p.ddy-p.dy*p.ddx)/max(hypot(p.dx,p.dy)^3,1e-30) for p in points]
    metadata=Dict{String,Any}(
        "conditioning"=>"natural C2 cubic in source cumulative arc length; source ordinates interpolated",
        "source_points"=>length(af.x),"conditioned_points"=>length(dense.x),
        "source_te_gap"=>original_gap,
        "cfd_te_gap"=>hypot(x[end]-x[1],y[end]-y[1]),
        "requested_te_thickness"=>thickness,
        "source_endpoints_preserved"=>thickness==0,
        "transform"=>thickness>0 ? "explicit finite trailing-edge thickness with C2 aft-20%-chord taper" : "none",
        "maximum_source_point_displacement_chords"=>maximum(hypot.(x.-af.x,y.-af.y)))
    (; airfoil=dense,metadata,curvature)
end
