"Validate optional postprocessing reference dimensions; never invent a wing area."
function _reference_dimensions(raw, boundary="free_air")
    raw === nothing && (raw=Dict())
    raw isa AbstractDict || throw(ArgumentError("reference must be an object"))
    result=Dict{String,Any}()
    for name in ("area","chord","span")
        value=get(raw,name,nothing)
        result[name]=value===nothing ? nothing : _number(value,"reference.$name";lo=1e-6,hi=1e10)
    end
    result["point"]=collect(_vec3(get(raw,"point",[0.,0.,0.]),"reference.point";lo=-1e6,hi=1e6))
    boundary=="symmetry" && abs(result["point"][2])>1e-10 && throw(ArgumentError("A symmetry case requires the moment reference point on Y=0"))
    result["stability_alpha"]=_number(get(raw,"stability_alpha",0.),"reference.stability_alpha";lo=-90,hi=90)
    result
end

"Boxer convention: positive alpha nose up; positive beta wind from the right."
function _flow_rotation(point,alpha,beta)
    sa,ca=sincosd(alpha);sb,cb=sincosd(beta)
    x,y,z=point; x,z=ca*x+sa*z,-sa*x+ca*z
    (cb*x-sb*y,sb*x+cb*y,z)
end

function _reference_moment_setting(case,bounds,chord)
    ref=get(case,"reference",nothing)
    ref===nothing && return [0.25,0.,0.]
    f=case["flow"];p=collect(_flow_rotation(ref["point"],get(f,"alpha",0.),get(f,"beta",0.)))
    lo=Float64.(bounds["min"]);hi=Float64.(bounds["max"]);boundary=case["domain"]["boundary"]
    if boundary=="ground"
        p=[p[1],p[3],-p[2]];lo,hi=[lo[1],lo[3],-hi[2]],[hi[1],hi[3],-lo[2]]
    elseif boundary=="symmetry"
        span=hi[2]-lo[2];hi[2]=get(case["domain"],"symmetry_center_y",false) ? span/2 : span;lo[2]=0.;p[2]=0.
    end
    [(p[1]-lo[1])/chord,(p[2]-(lo[2]+hi[2])/2)/chord,(p[3]-(lo[3]+hi[3])/2)/chord]
end
