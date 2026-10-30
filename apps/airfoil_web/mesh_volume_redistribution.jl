# Interior finite-TE collar correction. This changes only the mesh coordinates;
# the airfoil, outer boundary, two outlet columns, and paired wake cut are pinned.

"""Redistribute excess neighbouring cell area in the finite-TE collar.

Tangential edge smoothing alone does not bound area ratios: the curved corner
columns can have different normal spacings even when their edge lengths agree.
A shared i-face is translated slightly towards the larger adjacent cell. This
enlarges the smaller cell and reduces the larger cell by the same area at that
face. All proposals in a sweep use the same coordinates, avoiding an ordering
bias between the upper and lower surfaces. The downstream mesh quality gates
remain authoritative; this helper never changes their limits.
"""
function _transonic_balance_volumes!(mesh; sweeps::Int=20,
                                      ratio_target::Float64=1.8)
    sweeps >= 0 || throw(ArgumentError("volume redistribution sweeps must be nonnegative"))
    1.0 < ratio_target < 2.0 ||
        throw(ArgumentError("volume redistribution target must lie between 1 and 2"))
    X,Y=mesh.X,mesh.Y
    dX=zeros(size(X)); dY=zeros(size(Y))
    moved=falses(size(X))
    maximum_shift=0.0
    accepted_sweeps=0
    limited_sweeps=0
    area(i,j)=0.5*((X[i+1,j]-X[i,j])*(Y[i+1,j+1]-Y[i,j])-
        (Y[i+1,j]-Y[i,j])*(X[i+1,j+1]-X[i,j])+
        (X[i+1,j+1]-X[i,j])*(Y[i,j+1]-Y[i,j])-
        (Y[i+1,j+1]-Y[i,j])*(X[i,j+1]-X[i,j]))
    function positive_corners(i,j,gain)
        ax=X[i,j]+gain*dX[i,j]; ay=Y[i,j]+gain*dY[i,j]
        bx=X[i+1,j]+gain*dX[i+1,j]; by=Y[i+1,j]+gain*dY[i+1,j]
        cx=X[i+1,j+1]+gain*dX[i+1,j+1]; cy=Y[i+1,j+1]+gain*dY[i+1,j+1]
        dx=X[i,j+1]+gain*dX[i,j+1]; dy=Y[i,j+1]+gain*dY[i,j+1]
        (bx-ax)*(dy-ay)-(by-ay)*(dx-ax)>0 &&
        (cx-bx)*(ay-by)-(cy-by)*(ax-bx)>0 &&
        (dx-cx)*(by-cy)-(dy-cy)*(bx-cx)>0 &&
        (ax-dx)*(cy-dy)-(ay-dy)*(cx-dx)>0
    end
    for _ in 1:sweeps
        fill!(dX,0.0);fill!(dY,0.0);fill!(moved,false)
        proposals=0
        # Wall and first off-wall node are exact constraints: the first-cell
        # spacing/y+ is preserved. Further viscous layers may redistribute
        # tangential area where a finite-base corner fan would otherwise jump.
        for j in 3:mesh.NJ-1, i in 2:mesh.NI
            abs(X[i,j]-1.0)<0.05 && abs(Y[i,j])<0.03 || continue
            left=area(i-1,j);right=area(i,j)
            min(left,right)>0 || continue
            max(left,right)>ratio_target*min(left,right) || continue
            tx=X[i,j+1]-X[i,j];ty=Y[i,j+1]-Y[i,j]
            face_length=hypot(tx,ty)
            face_length>0 || continue
            nx=ty/face_length;ny=-tx/face_length
            logical_x=X[i+1,j]+X[i+1,j+1]-X[i-1,j]-X[i-1,j+1]
            logical_y=Y[i+1,j]+Y[i+1,j+1]-Y[i-1,j]-Y[i-1,j+1]
            if nx*logical_x+ny*logical_y<0
                nx=-nx;ny=-ny
            end
            transfer=right>ratio_target*left ?
                (right-ratio_target*left)/(ratio_target+1) :
                (ratio_target*right-left)/(ratio_target+1)
            # Move at most 2% of the smaller cell's effective width per sweep.
            limit=0.02min(left,right)/face_length
            step=clamp(0.3transfer/face_length,-limit,limit)
            for row in (j,j+1)
                dX[i,row]+=step*nx;dY[i,row]+=step*ny
                moved[i,row]=true
            end
            proposals+=1
        end
        proposals==0 && break
        affected=Tuple{Int,Int}[]
        for j in 2:mesh.NJ, i in 1:mesh.NI
            (moved[i,j] || moved[i+1,j] || moved[i,j+1] || moved[i+1,j+1]) &&
                push!(affected,(i,j))
        end
        gain=1.0
        while gain>=1/128 && !all(positive_corners(i,j,gain) for (i,j) in affected)
            gain*=0.5
        end
        gain>=1/128 || break
        gain<1 && (limited_sweeps+=1)
        for index in eachindex(X)
            moved[index] || continue
            dx=gain*dX[index];dy=gain*dY[index]
            X[index]+=dx;Y[index]+=dy
            maximum_shift=max(maximum_shift,hypot(dx,dy))
        end
        accepted_sweeps+=1
    end
    (; sweeps=accepted_sweeps,limited_sweeps,ratio_target,maximum_shift)
end
