function _safe_dat_id(id::AbstractString)
    isvalid(id) && !any(iscntrl,id) && !occursin(r"[/\\]",id) &&
        endswith(lowercase(id),".dat") && 1<=ncodeunits(id)<=128 ||
        throw(ArgumentError("Use a .dat filename of at most 128 bytes without control characters"))
    String(id)
end

"Reject repeated or intersecting panels before geometry reaches a mesher or solver."
function _validate_dat_contour(x::AbstractVector,y::AbstractVector;max_checks::Int=2_000_000)
    length(x)==length(y) || throw(ArgumentError("Airfoil coordinate lengths differ"))
    n=length(x)
    n>=3 || throw(ArgumentError("Airfoil contour needs at least three points"))
    # A sharp trailing edge may be explicitly closed. Its last point is the
    # only permitted repeat; finite-thickness trailing edges have a closing panel.
    x[1]==x[end] && y[1]==y[end] && (n-=1)
    seen=Set{Tuple{Float64,Float64}}()
    for i in 1:n
        point=(iszero(x[i]) ? 0. : Float64(x[i]),iszero(y[i]) ? 0. : Float64(y[i]))
        point in seen && throw(ArgumentError("Airfoil contour repeats a vertex at point $i; provide one traversal around the profile, with only the closing trailing-edge point repeated"))
        push!(seen,point)
    end
    tolerance=64eps(Float64)*max(1.,maximum(abs,x),maximum(abs,y))
    cross(ax,ay,bx,by)=ax*by-ay*bx
    # Adjacent panels share their endpoint by definition, but must not fold
    # back over each other along a straight line.
    for i in 1:n
        previous=i==1 ? n : i-1;next=i==n ? 1 : i+1
        ax=x[i]-x[previous];ay=y[i]-y[previous]
        bx=x[next]-x[i];by=y[next]-y[i]
        area_tolerance=tolerance*max(abs(ax),abs(ay),abs(bx),abs(by))
        abs(cross(ax,ay,bx,by))<=area_tolerance && ax*bx+ay*by<0 &&
            throw(ArgumentError("Airfoil contour doubles back at point $i; adjacent panels overlap"))
    end
    nextpoint(i)=i==n ? 1 : i+1
    xmin=[min(x[i],x[nextpoint(i)]) for i in 1:n]
    xmax=[max(x[i],x[nextpoint(i)]) for i in 1:n]
    ymin=[min(y[i],y[nextpoint(i)]) for i in 1:n]
    ymax=[max(y[i],y[nextpoint(i)]) for i in 1:n]
    active=Int[];checks=0
    # Sweep segment bounding boxes in X. Typical airfoil data has only two
    # active surfaces. A hard comparison bound also limits pathological data
    # with thousands of overlapping boxes (the input limit is 25,000 points).
    for i in sortperm(xmin)
        count=0
        for j in active
            if xmax[j]>=xmin[i]-tolerance
                count+=1;active[count]=j
            end
        end
        resize!(active,count)
        for j in active
            checks+=1
            checks<=max_checks || throw(ArgumentError("Airfoil has too many overlapping coordinate segments to validate; simplify the point list or repair its contour ordering"))
            (abs(i-j)==1 || abs(i-j)==n-1) && continue
            (ymax[j]<ymin[i]-tolerance || ymax[i]<ymin[j]-tolerance) && continue
            ii=nextpoint(i);jj=nextpoint(j)
            ax=x[ii]-x[i];ay=y[ii]-y[i];bx=x[jj]-x[j];by=y[jj]-y[j]
            o1=cross(ax,ay,x[j]-x[i],y[j]-y[i]);o2=cross(ax,ay,x[jj]-x[i],y[jj]-y[i])
            o3=cross(bx,by,x[i]-x[j],y[i]-y[j]);o4=cross(bx,by,x[ii]-x[j],y[ii]-y[j])
            ta=tolerance*max(abs(ax),abs(ay));tb=tolerance*max(abs(bx),abs(by))
            separated=(o1>ta && o2>ta)||(o1 < -ta && o2 < -ta)||
                      (o3>tb && o4>tb)||(o3 < -tb && o4 < -tb)
            separated || throw(ArgumentError("Airfoil contour intersects or touches itself between panels $i and $j; repair the coordinate ordering before importing"))
        end
        push!(active,i)
    end
    nothing
end

"Read a portable XFOIL coordinate file without writing a temporary source file."
function import_dat(text::AbstractString, filename::AbstractString)
    ncodeunits(text)<=2*1024*1024 || throw(ArgumentError("Airfoil file exceeds 2 MB"))
    isvalid(text) && !occursin('\0',text) || throw(ArgumentError("Choose a text XFOIL .dat file"))
    !any(iscntrl,filename) || throw(ArgumentError("The filename contains control characters"))
    id=_safe_dat_id(basename(replace(String(filename),'\\'=>'/')))
    rows=Tuple{Float64,Float64}[]
    header_seen=false
    for (line_number,line) in enumerate(eachsplit(replace(String(text),'\ufeff'=>""),'\n'))
        content=strip(first(split(line,r"[!#]";limit=2)))
        isempty(content) && continue
        fields=split(content)
        values=length(fields)==2 ? map(v->tryparse(Float64,replace(v,r"[dD]"=>"e")),fields) : nothing
        if values===nothing || any(isnothing,values)
            if isempty(rows) && !header_seen
                header_seen=true;continue # optional XFOIL title
            end
            throw(ArgumentError("Invalid coordinate on line $line_number; expected two numbers (x y)"))
        end
        all(isfinite,values) || throw(ArgumentError("Nonfinite coordinate on line $line_number"))
        push!(rows,(values[1],values[2]))
        length(rows)<=25001 || throw(ArgumentError("Airfoil file exceeds 25000 coordinate points"))
    end
    isempty(rows) && throw(ArgumentError("No coordinate pairs found in the airfoil file"))
    # Also accept the common two-surface Lednicer variant: counts, upper, lower.
    firstrow=rows[1]
    counted=all(v->v>=2 && v<=25000 && isinteger(v),firstrow) &&
        sum(firstrow)==length(rows)-1
    if counted
        nu,nl=Int.(firstrow)
        rows=JXfoil._stitch_surface_blocks([rows[2:nu+1],rows[nu+2:end]])
    end
    clean=Tuple{Float64,Float64}[]
    for row in rows
        (isempty(clean) || row!=last(clean)) && push!(clean,row)
    end
    8<=length(clean)<=25000 || throw(ArgumentError("Airfoil needs 8 to 25000 distinct consecutive points"))
    x=first.(clean);y=last.(clean)
    xmin,xmax=extrema(x);chord=xmax-xmin
    isfinite(chord) && chord>0 || throw(ArgumentError("Airfoil chord is degenerate"))
    x[1]>=xmax-.05chord && x[end]>=xmax-.05chord || throw(ArgumentError(
        "Order coordinates from the trailing edge around the leading edge and back to the trailing edge (XFOIL format)"))
    yle=y[argmin(x)]
    x=(x.-xmin)./chord;y=(y.-yle)./chord
    abs(JXfoil._signed_area(x,y))>1e-10 || throw(ArgumentError("Airfoil contour has zero area"))
    _validate_dat_contour(x,y)
    JXfoil._ensure_counterclockwise(x,y)
    coordinates=study_coordinates(Dict("x"=>x,"y"=>y))
    Dict("airfoil"=>Dict("kind"=>"dat","id"=>id,"coordinates"=>coordinates),
        "point_count"=>length(x),"input_chord"=>chord,
        "message"=>"Loaded $id: $(length(x)) points. Coordinates use unit chord, with the leading edge at the origin. Save the study to keep a portable copy.")
end
