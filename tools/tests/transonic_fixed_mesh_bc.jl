# Run with --project=src. CPU only; the uniform-state control is not an airfoil solution.
using Test,KernelAbstractions,StaticArrays
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const LBM_F=Float32
include(joinpath(ROOT,"src","collision","lattice.jl"))
include(joinpath(ROOT,"apps","airfoil_web","src","backends","cfd_backend.jl"))
include(joinpath(ROOT,"src","compressible","cmesh_solver_ka.jl"))
const G=CFDBackend.CMeshGen
const M=CMeshSolverKA

@testset "Fixed mesh frame retains the physical flow angle" begin
    af=G.naca4_points(2412;n=401)
    dn1=.7/(3e6*sqrt((.026*(3e6)^(-1/7))/2))
    args=(n_foil=384,n_wake=64,n_normal=112,dn1,alpha_deg=12.,trip_upper=0.,trip_lower=0.,
        n_base=2,mesh_refinement=:aplus_uniform_fixed,normal_collar=false,shock_layer=false)
    built=CFDBackend._build_transonic_mesh(af;args...)
    @test built.metadata["mesh_angle_deg"]==0.
    @test built.metadata["flow_alpha_deg"]==12.
    @test built.metadata["quality_report"]["air_reference_gates"]["passed"]
    @test CFDBackend.CMeshMetrics.gcl_check(built.metrics)[1] <= 2eps(Float32)
    @test_throws ArgumentError CFDBackend._build_transonic_mesh(af;args...,farfield_angle_deg=12.)
    cx,cy,cz,w,_,_=build_d3q39_static()
    s=M.build_sol_ka(CPU(),built.metrics;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    s.gam=0f0;s.nu_lam=0f0;s.recon=Int32(1);s.fullvisc=Int32(1)
    for alpha in (0.,4.,12.)
        M.set_freestream_ka!(s,.3,alpha;pg_correction=true)
        @test isapprox(atand(Float64(s.uyi),Float64(s.uxi)),alpha;atol=2e-6)
        M.applyBC_ka!(s,Int32(2));KernelAbstractions.synchronize(CPU())
        cols=s.g+1:s.g+s.NI;outer=s.g+s.NJ+1
        @test maximum(abs.(s.ux[cols,outer].-s.uxi))<=2e-6
        @test maximum(abs.(s.uy[cols,outer].-s.uyi))<=2e-6
        wall=s.g+s.NIW+1:s.g+s.NIW+s.NIF
        @test maximum(abs.(s.ux[wall,s.g].+s.uxi))<=eps(Float32)
        @test maximum(abs.(s.uy[wall,s.g].+s.uyi))<=eps(Float32)
        # Transparent boundaries isolate the constant-state operator/GCL test.
        # A physical no-slip airfoil cannot have a uniform solution at nonzero speed.
        M.set_freestream_ka!(s,.3,alpha;pg_correction=true);s.wall_flux=false
        M.refresh_raw_residual_ka!(s;wallmode=Int32(0))
        U=hypot(Float64(s.uxi),Float64(s.uyi));p=Float64(s.rho_inf)*M.CS2*s.T_inf
        refs=(Float64(s.rho_inf)*U,Float64(s.rho_inf)*U^2+p,Float64(s.rho_inf)*U^2+p,
            U*(Float64(s.rho_inf)*(s.eint*s.T_inf+.5U^2)+p))
        face_error=zeros(4);volume_error=zeros(4);met=built.metrics
        for j in 1:s.NJ,i in 1:s.NI
            length_sum=hypot(met.Six[i,j],met.Siy[i,j])+hypot(met.Six[i+1,j],met.Siy[i+1,j])+
                hypot(met.Sjx[i,j],met.Sjy[i,j])+hypot(met.Sjx[i,j+1],met.Sjy[i,j+1])
            for (k,a) in enumerate((s.dc_r,s.dc_mx,s.dc_my,s.dc_e))
                face_error[k]=max(face_error[k],abs(Float64(a[i,j]))/(length_sum*refs[k]))
                volume_error[k]=max(volume_error[k],abs(Float64(a[i,j]))/met.vol[i,j])
            end
        end
        @test maximum(face_error)<=128eps(Float32)
        @test all(isfinite,volume_error)
        println("Uniform control alpha=",alpha," local-face-scaled error=",face_error,
            "; volume-scaled residual=",volume_error," (Float32 amplification; not a physical flow convergence test)")
    end
end
