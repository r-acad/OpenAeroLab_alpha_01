"""Remove tangential drift in the near-body columns, with a smooth outer transition.

The wall, its first eight rows, wake columns, TE neighbourhood and outer boundary
stay fixed. No flow solution is used to infer a streamline. Backtracking accepts
only strictly convex cells and the same air/wake quality gates as the parent.
"""
function _normal_collar!(mesh; strength::Float64=1.0)
    0 <= strength <= 1 || throw(ArgumentError("normal collar strength must be in [0,1]"))
    X,Y=mesh.X,mesh.Y; oldX,oldY=copy(X),copy(Y)
    smooth(t)=let u=clamp(t,0.0,1.0);u*u*(3-2u);end
    dx=zeros(size(X));dy=zeros(size(Y)); angles_before=Float64[]
    ni,nj=mesh.NI,mesh.NJ
    # Wall order is clockwise; the left tangent normal points into the fluid.
    for i in mesh.NIW+3:mesh.NIW+mesh.NIF-1
        x,y=X[i,1],Y[i,1]
        taper=1-smooth((x-0.72)/0.20)
        taper<=0 && continue
        tx=X[i+1,1]-X[i-1,1];ty=Y[i+1,1]-Y[i-1,1]
        len=hypot(tx,ty);len>0||continue;tx/=len;ty/=len
        nx,ny=-ty,tx
        for j in 10:nj
            rx,ry=X[i,j]-x,Y[i,j]-y
            d=rx*nx+ry*ny;d>0||continue
            w=taper*smooth((d-2e-4)/0.004)*(1-smooth((d-0.12)/0.65))
            tangent=rx*tx+ry*ty
            dx[i,j]=-w*tangent*tx;dy[i,j]=-w*tangent*ty
            x<0.72 && 0.01<d<0.10 && push!(angles_before,abs(atand(tangent,d)))
        end
    end
    before=_transonic_quality_report(mesh)
    refusals=String[]
    for gain in strength.*(1.0,0.75,0.5,0.25,0.125)
        X.=oldX.+gain.*dx;Y.=oldY.+gain.*dy
        q=try
            _transonic_quality_report(mesh)
        catch err
            push!(refusals,"strength $gain: "*sprint(showerror,err));continue
        end
        if q["geometric_gate"]["passed"] &&
           (!before["air_reference_gates"]["passed"] || q["air_reference_gates"]["passed"]) &&
           (!before["provisional_wake_gates"]["passed"] || q["provisional_wake_gates"]["passed"])
            angles=Float64[]
            for i in mesh.NIW+3:mesh.NIW+mesh.NIF-1
                oldX[i,1]<0.72||continue
                tx=oldX[i+1,1]-oldX[i-1,1];ty=oldY[i+1,1]-oldY[i-1,1];len=hypot(tx,ty);tx/=len;ty/=len
                for j in 10:nj
                    d=-(oldX[i,j]-oldX[i,1])*ty+(oldY[i,j]-oldY[i,1])*tx
                    0.01<d<0.10||continue
                    t=(X[i,j]-X[i,1])*tx+(Y[i,j]-Y[i,1])*ty
                    push!(angles,abs(atand(t,d)))
                end
            end
            return Dict{String,Any}("applied"=>true,"strength"=>gain,
                "parent_mean_normal_deviation_deg"=>isempty(angles_before) ? 0.0 : sum(angles_before)/length(angles_before),
                "mean_normal_deviation_deg"=>isempty(angles) ? 0.0 : sum(angles)/length(angles),
                "max_displacement_chords"=>maximum(hypot.(X.-oldX,Y.-oldY)),
                "preserved"=>"wall and first 8 cell rows, wake, x/c >= 0.92, outer boundary",
                "scope"=>"surface-normal collar to 0.12c; smooth return to parent by 0.77c",
                "refusals"=>refusals)
        end
        push!(refusals,"strength $gain failed parent quality gates")
    end
    X.=oldX;Y.=oldY
    Dict{String,Any}("applied"=>false,"strength"=>0.0,"refusals"=>refusals)
end
