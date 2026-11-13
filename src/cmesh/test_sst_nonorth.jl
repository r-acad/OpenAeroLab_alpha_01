using Test
using KernelAbstractions
using CUDA

const LBM_F=Float32
const F=Float32
include(joinpath(@__DIR__,"..","collision","lattice.jl"))
include(joinpath(@__DIR__,"cmesh_gen.jl"))
include(joinpath(@__DIR__,"cmesh_metrics.jl"))
include(joinpath(@__DIR__,"cmesh_solver_ka.jl"))
const M=CMeshSolverKA

function fixture(backend;skew=0.0f0)
    NI=Int32(8);NJ=Int32(8);NIW=Int32(2);NIF=Int32(4);g=Int32(3)
    X=Matrix{Float32}(undef,NI+1,NJ+1);Y=similar(X)
    for j in 1:NJ+1,i in 1:NI+1
        X[i,j]=0.15f0*(i-1)+skew*0.1f0*(j-1)
        Y[i,j]=0.1f0*(j-1)
    end
    xc=Matrix{Float32}(undef,NI,NJ);yc=similar(xc)
    Six=Matrix{Float32}(undef,NI+1,NJ);Siy=similar(Six)
    Sjx=Matrix{Float32}(undef,NI,NJ+1);Sjy=similar(Sjx)
    for j in 1:NJ,i in 1:NI+1
        Six[i,j]=Y[i,j+1]-Y[i,j]
        Siy[i,j]=-(X[i,j+1]-X[i,j])
    end
    for j in 1:NJ+1,i in 1:NI
        Sjx[i,j]=-(Y[i+1,j]-Y[i,j])
        Sjy[i,j]=X[i+1,j]-X[i,j]
    end
    for j in 1:NJ,i in 1:NI
        xc[i,j]=.25f0*(X[i,j]+X[i+1,j]+X[i,j+1]+X[i+1,j+1])
        yc[i,j]=.25f0*(Y[i,j]+Y[i+1,j]+Y[i,j+1]+Y[i+1,j+1])
    end
    tk=zeros(Float32,NI+2g,NJ+2g);tom=similar(tk)
    rho=fill(1.0f0,size(tk));T=fill(1.0f0,size(tk));nut=zeros(Float32,size(tk))
    for j in 1:NJ,i in 1:NI
        tk[i+g,j+g]=1.0f0+0.2f0*xc[i,j]+0.3f0*yc[i,j]
        tom[i+g,j+g]=2.0f0+0.1f0*xc[i,j]-0.2f0*yc[i,j]
    end
    d(x)=KernelAbstractions.allocate(backend,eltype(x),size(x)...)
    function send(x)
        y=d(x);copyto!(y,x);y
    end
    zeros_dev(dims...)=send(zeros(Float32,dims...))
    (;NI,NJ,NIW,NIF,g,xc=send(xc),yc=send(yc),Six=send(Six),Siy=send(Siy),
      Sjx=send(Sjx),Sjy=send(Sjy),tk=send(tk),tom=send(tom),rho=send(rho),
      T=send(T),nut=send(nut),wdist=send(fill(1.0f0,NI,NJ)),
      gkx=zeros_dev(NI,NJ),gky=zeros_dev(NI,NJ),
      gwx=zeros_dev(NI,NJ),gwy=zeros_dev(NI,NJ),
      sigk=zeros_dev(NI,NJ),sigw=zeros_dev(NI,NJ),nu=zeros_dev(NI,NJ),
      Fik=zeros_dev(NI+1,NJ),Fiw=zeros_dev(NI+1,NJ),
      Fjk=zeros_dev(NI,NJ+1),Fjw=zeros_dev(NI,NJ+1))
end

function check_sst_step(backend,label; characteristic_i_outlet::Bool=false)
    xs=[.5*(1-cospi(k/32)) for k in 0:32]
    yt(x)=x==1.0 ? 0.0 : 5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),
        vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=64,n_wake=16,n_normal=24,
        dn1=.002,growth_max=1.8,R_ff=25.0,L_wake=25.0,
        smooth_iters=80,bl_freeze=6)
    met=CMeshMetrics.build_metrics(mesh)
    cx,cy,cz,w,_,_=build_d3q39_static()
    s=M.build_sol_ka(backend,met;drot=2.0f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    s.wall_flux=true;s.recon=Int32(1);s.fullvisc=Int32(1)
    M.set_freestream_ka!(s,.2,2.0)
    s.nu_lam=1.0f-4
    M.set_sst_ka!(s)
    fill!(s.dtf,1.0f-6)
    if characteristic_i_outlet
        s.characteristic_i_outlet=Int32(1)
        M.applyBC_ka!(s,Int32(2))
    end
    @test_throws ArgumentError M.sst_step_ka!(s;corrected_wall_metric=false,nonorth_diffusion=true)
    M.sst_step_ka!(s;corrected_wall_metric=true,nonorth_diffusion=true)
    KernelAbstractions.synchronize(backend)
    @testset "$label full SST opt-in smoke" begin
        @test all(isfinite,Array(s.tk))
        @test all(isfinite,Array(s.tom))
        @test all(isfinite,Array(s.nut))
        @test s.characteristic_i_outlet == Int32(characteristic_i_outlet)
        fk=Array(s.sst_Fjk);fw=Array(s.sst_Fjw)
        for i in 1:s.NIW
            ip=s.NI+1-i
            @test fk[i,1]==-fk[ip,1]
            @test fw[i,1]==-fw[ip,1]
        end
    end
end

function check_backend(label,backend;skew)
    f=fixture(backend;skew)
    # Finish host-to-device fixture copies before KA kernels use those arrays.
    KernelAbstractions.synchronize(backend)
    M._k_sst_nonorth_prep!(backend)(f.gkx,f.gky,f.gwx,f.gwy,
        f.sigk,f.sigw,f.nu,f.tk,f.tom,f.rho,f.T,f.xc,f.yc,f.Sjx,f.Sjy,f.wdist,
        .01f0,1.0f0,0.0f0,Int32(0),0.0f0,
        f.NI,f.NJ,f.NIW,f.NIF,f.g,Int32(0);ndrange=(f.NI,f.NJ))
    M._k_sst_nonorth_i!(backend)(f.Fik,f.Fiw,f.tk,f.tom,f.nut,
        f.gkx,f.gky,f.gwx,f.gwy,f.sigk,f.sigw,f.nu,
        f.xc,f.yc,f.Six,f.Siy,f.NI,f.NJ,f.g;ndrange=(f.NI+1,f.NJ))
    M._k_sst_nonorth_j!(backend)(f.Fjk,f.Fjw,f.tk,f.tom,f.nut,
        f.gkx,f.gky,f.gwx,f.gwy,f.sigk,f.sigw,f.nu,
        f.xc,f.yc,f.Sjx,f.Sjy,f.NI,f.NJ,f.NIW,f.NIF,f.g;
        ndrange=(f.NI,f.NJ+1))
    KernelAbstractions.synchronize(backend)
    gkx,gky,gwx,gwy=map(Array,(f.gkx,f.gky,f.gwx,f.gwy))
    Fik,Fiw,Fjk,Fjw=map(Array,(f.Fik,f.Fiw,f.Fjk,f.Fjw))
    Six,Siy,Sjx,Sjy=map(Array,(f.Six,f.Siy,f.Sjx,f.Sjy))
    @testset "$label skew=$skew" begin
        for j in 3:Int(f.NJ)-2,i in 3:Int(f.NI)-2
            @test isapprox(gkx[i,j],.2f0;atol=3e-5)
            @test isapprox(gky[i,j],.3f0;atol=3e-5)
            @test isapprox(gwx[i,j],.1f0;atol=3e-5)
            @test isapprox(gwy[i,j],-.2f0;atol=3e-5)
        end
        for j in 3:Int(f.NJ)-2,i in 4:Int(f.NI)-2
            @test isapprox(Fik[i,j],.01f0*(.2f0*Six[i,j]+.3f0*Siy[i,j]);atol=3e-7)
            @test isapprox(Fiw[i,j],.01f0*(.1f0*Six[i,j]-.2f0*Siy[i,j]);atol=3e-7)
        end
        for j in 4:Int(f.NJ)-2,i in 3:Int(f.NI)-2
            @test isapprox(Fjk[i,j],.01f0*(.2f0*Sjx[i,j]+.3f0*Sjy[i,j]);atol=3e-7)
            @test isapprox(Fjw[i,j],.01f0*(.1f0*Sjx[i,j]-.2f0*Sjy[i,j]);atol=3e-7)
        end
        for i in 1:Int(f.NIW)
            ip=Int(f.NI)+1-i
            @test Fjk[i,1] == -Fjk[ip,1]
            @test Fjw[i,1] == -Fjw[ip,1]
        end
        for j in 3:Int(f.NJ)-2,i in 3:Int(f.NI)-2
            @test isapprox(Fik[i+1,j]-Fik[i,j]+Fjk[i,j+1]-Fjk[i,j],0.0f0;atol=5e-7)
            @test isapprox(Fiw[i+1,j]-Fiw[i,j]+Fjw[i,j+1]-Fjw[i,j],0.0f0;atol=5e-7)
        end
    end
end

@testset "staged KA nonorth diffusion" begin
    check_backend("KA CPU",CPU();skew=0.0f0)
    check_backend("KA CPU",CPU();skew=0.7f0)
    check_sst_step(CPU(),"KA CPU")
    check_sst_step(CPU(),"KA CPU characteristic i outlet";
                   characteristic_i_outlet=true)
    if get(ENV,"NONORTH_TEST_CUDA","0")=="1"
        CUDA.functional() || error("CUDA unavailable for requested smoke")
        check_backend("CUDA",CUDA.CUDABackend();skew=0.0f0)
        check_backend("CUDA",CUDA.CUDABackend();skew=0.7f0)
        check_sst_step(CUDA.CUDABackend(),"CUDA")
        check_sst_step(CUDA.CUDABackend(),"CUDA characteristic i outlet";
                       characteristic_i_outlet=true)
    end
end
