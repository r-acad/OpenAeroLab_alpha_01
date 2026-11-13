using Test,KernelAbstractions,StaticArrays,LinearAlgebra,TOML,SHA,Dates,Serialization
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const LBM_F=Float32
include(joinpath(ROOT,"src","collision","lattice.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_gen.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_metrics.jl"))
include(joinpath(ROOT,"src","compressible","cmesh_solver_ka.jl"))
const M=CMeshSolverKA
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const REPORT=joinpath(PRIVATE,"validation","airfoil_mesh_solver_20261112","compressible","geometric_muscl_"*Dates.format(now(UTC),"yyyymmdd_HHMMSS"))

function rectangle_case(backend=CPU();scalar_type=Float32)
    ni,nj=20,16
    X=[(i/ni)+.2*(j/nj)^2+.08*(i/ni)*(j/nj)^2 for i in 0:ni,j in 0:nj]
    Y=[(j/nj)^2+.1*(i/ni)+.1*(i/ni)^2*(j/nj)^2 for i in 0:ni,j in 0:nj]
    met=CMeshMetrics.build_metrics(X,Y,0,ni;scalar_type)
    cx,cy,cz,w,_,_=build_d3q39_static(scalar_type)
    s=M.build_sol_ka(backend,met;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w,scalar_type)
    M.set_freestream_ka!(s,.3,4.);s.nu_lam=0f0;s.gam=0f0;s.fullvisc=Int32(1)
    s,(;X,Y,NI=ni,NJ=nj,NIW=0,NIF=ni),met
end
function affine_state!(s,mesh)
    ni,nj=s.NI,s.NJ;g=s.g;X,Y=mesh.X,mesh.Y
    center(i,j)=(.25*(X[i,j]+X[i+1,j]+X[i,j+1]+X[i+1,j+1]),
        .25*(Y[i,j]+Y[i+1,j]+Y[i,j+1]+Y[i+1,j+1]))
    xx=fill(0.,size(s.rho));yy=similar(xx);fill!(yy,0.)
    for j in 1:nj,i in 1:ni
        xx[i+g,j+g],yy[i+g,j+g]=center(i,j)
    end
    for j in 1:nj
        x,y=center(1,j);xx[g,j+g]=X[1,j]+X[1,j+1]-x;yy[g,j+g]=Y[1,j]+Y[1,j+1]-y
        x,y=center(ni,j);xx[g+ni+1,j+g]=X[end,j]+X[end,j+1]-x;yy[g+ni+1,j+g]=Y[end,j]+Y[end,j+1]-y
    end
    for i in 1:ni
        x,y=center(i,1);xx[i+g,g]=X[i,1]+X[i+1,1]-x;yy[i+g,g]=Y[i,1]+Y[i+1,1]-y
        x,y=center(i,nj);xx[i+g,g+nj+1]=X[i,end]+X[i+1,end]-x;yy[i+g,g+nj+1]=Y[i,end]+Y[i+1,end]-y
    end
    coeff=((1.,.01,.02),(.2,.03,-.01),(.03,-.02,.04),(1.,.02,-.01))
    for (name,(z,a,b)) in zip((:rho,:ux,:uy,:T),coeff)
        copyto!(getproperty(s,name),eltype(s.rho).(@. z+a*xx+b*yy))
    end
    coeff
end

function affine_test(backend)
    s,mesh,met=rectangle_case(backend)
    coeff=affine_state!(s,mesh)
    w=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:barth)
    M._airfoil_muscl_prepare!(s,w);KernelAbstractions.synchronize(backend)
    states=Array(w.states);errors=zeros(4)
    for j in 1:s.NJ,i in 1:s.NI,f in 1:4
        ii,jj=f==1 ? (i,j) : f==2 ? (i+1,j) : f==3 ? (i,j) : (i,j+1)
        fx=f<=2 ? (mesh.X[ii,j]+mesh.X[ii,j+1])/2 : (mesh.X[i,jj]+mesh.X[i+1,jj])/2
        fy=f<=2 ? (mesh.Y[ii,j]+mesh.Y[ii,j+1])/2 : (mesh.Y[i,jj]+mesh.Y[i+1,jj])/2
        for k in 1:4
            z,a,b=coeff[k];errors[k]=max(errors[k],abs(states[i,j,k,f]-(z+a*fx+b*fy)))
        end
    end
    @test maximum(errors)<=2e-6
    @test w.geometry.receipt.represented_gradient_moment_error<=1e-5
    @test minimum(Array(w.status))==maximum(Array(w.status))==0
    @test minimum(states[:,:,1,:])>0&&minimum(states[:,:,4,:])>0
    @test all(isfinite,states)
    @test s.recon==Int32(3)
    M.clear_airfoil_muscl_ka!(s)
    @test s.recon==Int32(0)
    @test_throws ArgumentError M.set_airfoil_muscl_ka!(s,mesh.X.+.01,mesh.Y)
    s.cav=Int32(4)
    @test_throws ArgumentError M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y)
    errors
end

function jump_test(backend)
    s,mesh,_=rectangle_case(backend)
    for limiter in (:barth,:venkat)
        r=fill(1f0,size(s.rho));u=fill(.2f0,size(s.rho));v=fill(0f0,size(s.rho));t=fill(1f0,size(s.rho))
        for j in axes(r,2),i in axes(r,1)
            if i>s.g+s.NI÷2;r[i,j]=.125f0;t[i,j]=.2f0;u[i,j]=-.1f0;end
        end
        for (a,b) in zip((s.rho,s.ux,s.uy,s.T),(r,u,v,t));copyto!(a,b);end
        w=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter)
        M._airfoil_muscl_prepare!(s,w);KernelAbstractions.synchronize(backend)
        q=Array(w.states)
        @test minimum(q[:,:,1,:])>=.125f0-2f-6&&maximum(q[:,:,1,:])<=1f0+2f-6
        @test minimum(q[:,:,4,:])>=.2f0-2f-6&&maximum(q[:,:,4,:])<=1f0+2f-6
        @test all(isfinite,q)&&minimum(q[:,:,1,:])>0&&minimum(q[:,:,4,:])>0
        @test minimum(Array(w.theta))>=0&&maximum(Array(w.theta))<=1
        M.clear_airfoil_muscl_ka!(s)
    end
end

@testset "Physical-coordinate thermal MUSCL CPU affine/positivity" begin
    affine_test(CPU());jump_test(CPU())
end

@testset "Thermal flux retains energy and shared-face conservation" begin
    s,mesh,_=rectangle_case()
    w=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y)
    s.wall_flux=false
    M.refresh_raw_residual_ka!(s;wallmode=Int32(0))
    @test all(isfinite,s.dc_r)&&all(isfinite,s.dc_e)
    # Every interior face is stored once and cancels exactly in the sum of
    # Float64-accumulated divergences. Do not infer a small local residual.
    for (dc,fi,fj) in zip((s.dc_r,s.dc_mx,s.dc_my,s.dc_e),
            (s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e),(s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e))
        exact=sum(Float64(fi[i,j])-Float64(fi[i+1,j])+Float64(fj[i,j])-Float64(fj[i,j+1])
            for j in 1:s.NJ,i in 1:s.NI)
        boundary=sum(Float64,fi[1,:])-sum(Float64,fi[end,:])+sum(Float64,fj[:,1])-sum(Float64,fj[:,end])
        @test isapprox(exact,boundary;atol=1e-12)
    end
    # An actual shared interior face perturbation changes only its two
    # adjacent cell balances, with zero net change, in every equation.
    for (component,fi) in enumerate((s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e))
        fields=(s.dc_r,s.dc_mx,s.dc_my,s.dc_e)
        before=copy(fields[component]);i0=s.NI÷2;j0=s.NJ÷2
        old=fi[i0,j0];delta=Float32(.125component)
        fi[i0,j0]=old+delta
        M._k_residual!(s.backend)(fields...,s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e,
            s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
        change=fields[component].-before
        @test isapprox(change[i0-1,j0],delta;atol=4eps(Float32))
        @test isapprox(change[i0,j0],-delta;atol=4eps(Float32))
        @test abs(sum(Float64,change))<=4eps(Float32)
        @test count(!iszero,change)<=2
        fi[i0,j0]=old
        M._k_residual!(s.backend)(fields...,s.Fi_r,s.Fi_mx,s.Fi_my,s.Fi_e,
            s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    end
    l=(1f0,.2f0,.03f0,1f0);r=(.9f0,.15f0,.02f0,.95f0)
    a=M._airfoil_thermal_face_flux(l,r,.6f0,.8f0,1f0,s.cxa,s.cya,s.cza,s.wa,s.drot)
    b=M._airfoil_thermal_face_flux(r,l,-.6f0,-.8f0,1f0,s.cxa,s.cya,s.cza,s.wa,s.drot)
    @test all(isfinite,a)&&a[4]!=0
    @test maximum(abs.(a.+b))<=8eps(Float32)
    M.clear_airfoil_muscl_ka!(s)
    s.recon=Int32(3)
    @test_throws ArgumentError M.refresh_raw_residual_ka!(s)
end

if get(ENV,"OPENLUDWIG_TEST_CUDA","0")=="1"
    using CUDA
    @test CUDA.functional()
    @testset "Physical-coordinate thermal MUSCL CUDA" begin
        affine_test(CUDA.CUDABackend());jump_test(CUDA.CUDABackend())
    end
end

# Keep a useful validation receipt, rather than disposable output in the
# permanent validation tree. A failed testset never reaches this writer.
mkpath(REPORT)
receipt=Dict("generated_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
    "cpu_checks"=>41,"passed"=>true,"cuda_requested"=>get(ENV,"OPENLUDWIG_TEST_CUDA","0")=="1",
    "source_sha256"=>Dict(p=>bytes2hex(open(sha256,joinpath(ROOT,p))) for p in
        ("src/compressible/cmesh_geometric_reconstruction_ka.jl","src/compressible/cmesh_solver_ka.jl","src/cmesh/cmesh_flux_core.jl")),
    "scope"=>"Small stretched/skew affine, jump bounds, positivity, thermal energy and actual shared-face conservation controls; not an airfoil convergence or timing qualification.")
open(io->TOML.print(io,receipt;sorted=true),joinpath(REPORT,"receipt.toml"),"w")
