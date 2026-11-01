# Included only after CUDA is imported by a standalone GPU worker or test.
# Keep one cuSPARSE SpSV analysis/workspace per triangular direction. Matrix
# factors and all input/output buffers are fixed for this preconditioner.
function make_cuda_projection_triangular_solver()
    cache=Dict{Char,Any}()
    function solve!(operation,L,x,y)
        C=CUDA.CUSPARSE
        if !haskey(cache,operation)
            descA=C.CuSparseMatrixDescriptor(L,'O')
            uplo=Ref{C.cusparseFillMode_t}('L');diagonal=Ref{C.cusparseDiagType_t}('N')
            C.cusparseSpMatSetAttribute(descA,'F',uplo,Csize_t(sizeof(uplo)))
            C.cusparseSpMatSetAttribute(descA,'D',diagonal,Csize_t(sizeof(diagonal)))
            descX=C.CuDenseVectorDescriptor(x);descY=C.CuDenseVectorDescriptor(y)
            descSV=C.CuSparseSpSVDescriptor();alpha=Ref{Float64}(1.)
            algo=C.CUSPARSE_SPSV_ALG_DEFAULT;sizebytes=Ref{Csize_t}(0)
            C.cusparseSpSV_bufferSize(C.handle(),operation,alpha,descA,descX,descY,Float64,algo,descSV,sizebytes)
            workspace=CUDA.zeros(UInt8,Int(sizebytes[]))
            C.cusparseSpSV_analysis(C.handle(),operation,alpha,descA,descX,descY,Float64,algo,descSV,workspace)
            cache[operation]=(;L,x,y,descA,descX,descY,descSV,alpha,algo,workspace)
        end
        state=cache[operation]
        state.L===L && state.x===x && state.y===y || error("cached CUDA triangular solve received changed factor or buffers")
        C.cusparseSpSV_solve(C.handle(),operation,state.alpha,state.descA,state.descX,state.descY,Float64,state.algo,state.descSV)
        y
    end
    solve!
end
