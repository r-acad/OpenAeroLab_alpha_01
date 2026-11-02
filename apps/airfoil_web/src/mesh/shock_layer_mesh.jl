"""Resolve the off-wall shock region with one smooth, monotone row map.

The first wall layers stay exact. A common logical-row distribution avoids
introducing steps between neighbouring columns, including the finite-TE fan.
The cap covers the upper/lower shock envelope; spacing then grows smoothly
into the far field. This adds normal resolution rather than changing the
surface spacing or filtering the flow solution.
"""
function _shock_layer_mesh(mesh; height_cap::Float64=0.004,
                           resolved_distance::Float64=0.65,
                           growth_fraction::Float64=0.10, keep_rows::Int=12)
    0<height_cap<=0.02 || throw(ArgumentError("shock-layer height cap must be in (0, 0.02]"))
    0.1<=resolved_distance<=2 || throw(ArgumentError("shock envelope must be in [0.1, 2] chords"))
    0.03<=growth_fraction<=0.15 || throw(ArgumentError("shock-layer growth must be in [0.03, 0.15]"))
    X,Y=mesh.X,mesh.Y; ni,nj=mesh.NI,mesh.NJ
    nj>keep_rows+12 || throw(ArgumentError("not enough normal rows for shock-layer redistribution"))
    density=ones(nj)
    # Measure actual column lengths, not a nominal geometric growth parameter.
    # Both sides share the map; no discontinuity is introduced at the nose.
    for i in mesh.NIW+2:mesh.NIW+mesh.NIF
        .10<X[i,1]<.90 || continue
        for j in 1:nj
            h=hypot(X[i,j+1]-X[i,j],Y[i,j+1]-Y[i,j])
            d=hypot(0.5(X[i,j]+X[i,j+1])-X[i,1],0.5(Y[i,j]+Y[i,j+1])-Y[i,1])
            outer=max(0.0,d-resolved_distance)
            cap=height_cap+growth_fraction*outer^2/(outer+0.15)
            target=min(max(growth_fraction*d,1e-12),cap)
            u=clamp((j-keep_rows-1)/12,0.0,1.0)
            weight=u*u*(3-2u)
            density[j]=max(density[j],1+weight*(h/target-1))
        end
    end
    # Smooth the row-density monitor before equidistributing. This removes
    # abrupt integer subdivision changes without smoothing any field values.
    for _ in 1:4
        old=copy(density)
        for j in keep_rows+2:nj-1
            density[j]=exp((log(old[j-1])+2log(old[j])+log(old[j+1]))/4)
        end
    end
    density[1:keep_rows].=1
    cumulative=vcat(0.0,cumsum(density[keep_rows+1:end]))
    nadd=ceil(Int,cumulative[end]); newnj=keep_rows+nadd
    ni*newnj<=2_000_000 || throw(ArgumentError("resolved shock mesh exceeds two million cells"))
    # Inverse monotone interpolation gives a continuous change in cell height.
    oldrows=Float64.(keep_rows+1:nj+1)
    q=vcat(Float64.(1:keep_rows),_pchip(cumulative,oldrows,collect(range(0,cumulative[end];length=nadd+1))))
    q[1:keep_rows+1].=1:keep_rows+1; q[end]=nj+1
    Xn=zeros(ni+1,newnj+1);Yn=similar(Xn); rows=Float64.(1:nj+1)
    for i in 1:ni+1
        Xn[i,:]=_pchip(rows,collect(X[i,:]),q)
        Yn[i,:]=_pchip(rows,collect(Y[i,:]),q)
    end
    Xn[:,1:keep_rows+1].=X[:,1:keep_rows+1];Yn[:,1:keep_rows+1].=Y[:,1:keep_rows+1]
    Xn[:,end].=X[:,end];Yn[:,end].=Y[:,end]
    candidate=CMeshGen.ChannelMesh(ni,newnj,mesh.NIW,mesh.NIF,Xn,Yn,mesh.dn1,
        copy(mesh.growth),mesh.channel,mesh.outer_face_kind,mesh.strip)
    info=Dict{String,Any}("version"=>"shock_layer_v1","parent_rows"=>nj,"rows"=>newnj,
        "height_cap_chords"=>height_cap,"resolved_distance_chords"=>resolved_distance,
        "growth_fraction"=>growth_fraction,"preserved_wall_rows"=>keep_rows,
        "cell_count_factor"=>newnj/nj,"row_map"=>q,
        "scope"=>"common monotone normal-row map; unchanged wall, first 12 layers, wake cut and farfield contour")
    (;mesh=candidate,info)
end
