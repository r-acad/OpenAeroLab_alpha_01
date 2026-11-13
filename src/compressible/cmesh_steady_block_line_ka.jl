# Four-conservation-variable block tridiagonal inverses for C-mesh columns
# and streamwise row segments. The nonlinear residual still owns the wake cut.
using StaticArrays

@inline function _airfoil_block4(a,i,j)
    SMatrix{4,4,eltype(a),16}(ntuple(k->a[mod(k-1,4)+1,div(k-1,4)+1,i,j],Val(16)))
end

# Streamwise block rows use bounded segments to keep enough GPU parallelism.
# Blocks are uncoupled at their segment ends; those faces remain on the diagonal.
@kernel function _k_airfoil_block_row_solve!(out,cp,@Const(rhs),@Const(vol),
    @Const(lower),@Const(diagonal),@Const(upper),NI::Int32,NJ::Int32,
    block_length::Int32,blocks_per_row::Int32)
    index=@index(Global,Linear)
    B=eltype(diagonal)
    j=Int32(div(index-1,blocks_per_row)+1)
    @inbounds if j<=NJ
        block=Int32(mod(index-1,blocks_per_row)+1)
        first_i=(block-1)*block_length+1
        last_i=min(NI,block*block_length)
        prev_c=zero(SMatrix{4,4,B,16})
        prev_d=zero(SVector{4,B})
        for i in first_i:last_i
            l=i>first_i ? _airfoil_block4(lower,i,j) : zero(prev_c)
            d=_airfoil_block4(diagonal,i,j)-l*prev_c
            b=_airfoil_vec4(rhs,i,j,B)*vol[i,j]-l*prev_d
            di=inv(d)
            ci=i<last_i ? di*_airfoil_block4(upper,i,j) : zero(prev_c)
            di_rhs=di*b
            _airfoil_block4_store!(cp,ci,i,j)
            _airfoil_vec4_store!(out,di_rhs,i,j)
            prev_c=ci
            prev_d=di_rhs
        end
        next=_airfoil_vec4(out,last_i,j,B)
        for i in last_i-1:-1:first_i
            next=_airfoil_vec4(out,i,j,B)-_airfoil_block4(cp,i,j)*next
            _airfoil_vec4_store!(out,next,i,j)
        end
    end
end

@inline function _airfoil_block4_store!(a,m,i,j)
    for col in 1:4,row in 1:4
        a[row,col,i,j]=m[row,col]
    end
    nothing
end

@inline function _airfoil_vec4(a,i,j,::Type{B}=Float32) where {B<:AbstractFloat}
    SVector{4,B}(a[1][i,j],a[2][i,j],a[3][i,j],a[4][i,j])
end

@inline function _airfoil_vec4_store!(a,v,i,j)
    for row in 1:4
        a[row][i,j]=v[row]
    end
    nothing
end

@kernel function _k_airfoil_block_line_solve!(out,cp,@Const(rhs),@Const(vol),
    @Const(lower),@Const(diagonal),@Const(upper),NI::Int32,NJ::Int32)
    i=@index(Global,Linear)
    B=eltype(diagonal)
    @inbounds if i<=NI
        prev_c=zero(SMatrix{4,4,B,16})
        prev_d=zero(SVector{4,B})
        for j in 1:NJ
            l=_airfoil_block4(lower,i,j)
            d=_airfoil_block4(diagonal,i,j)-l*prev_c
            b=_airfoil_vec4(rhs,i,j,B)*vol[i,j]-l*prev_d
            di=inv(d)
            cj=j<NJ ? di*_airfoil_block4(upper,i,j) : zero(prev_c)
            dj=di*b
            _airfoil_block4_store!(cp,cj,i,j)
            _airfoil_vec4_store!(out,dj,i,j)
            prev_c=cj
            prev_d=dj
        end
        next=_airfoil_vec4(out,i,NJ,B)
        for j in NJ-1:-1:1
            next=_airfoil_vec4(out,i,j,B)-_airfoil_block4(cp,i,j)*next
            _airfoil_vec4_store!(out,next,i,j)
        end
    end
end
