using Test,Printf
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"apps","airfoil_web","src","backends","cfd_backend.jl"))
const G=CFDBackend.CMeshGen
const Q=G.CMeshQualityMetrics
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","transonic_airfoil_geometry_tests");mkpath(SCRATCH)

function write_naca(path,m,p,t)
    upper=NTuple{2,Float64}[];lower=NTuple{2,Float64}[]
    for k in 0:200
        x=.5*(1-cospi(k/200))
        yt=5t*(.2969sqrt(x)-.126x-.3516x^2+.2843x^3-.1015x^4)
        yc=m==0 ? 0. : x<p ? m/p^2*(2p*x-x*x) : m/(1-p)^2*((1-2p)+2p*x-x*x)
        slope=m==0 ? 0. : x<p ? 2m/p^2*(p-x) : 2m/(1-p)^2*(p-x)
        a=atan(slope)
        push!(upper,(x-yt*sin(a),yc+yt*cos(a)))
        push!(lower,(x+yt*sin(a),yc-yt*cos(a)))
    end
    open(path,"w") do io
        println(io,"Finite trailing-edge XFOIL regression fixture")
        for (x,y) in vcat(reverse(upper),lower[2:end])
            @printf(io,"%.17g %.17g\n",x,y)
        end
    end
end

@testset "Strict finite-base A-plus uniform quality profile" begin
    re=3e6;dn1=.7/(re*sqrt((.026*re^(-1/7))/2))
    for (name,camber) in (("0012",0.),("2412",.02),("4412",.04))
        path=joinpath(SCRATCH,"naca"*name*".dat")
        write_naca(path,camber,.4,.12)
        af=G.read_airfoil_dat(path)
        built=CFDBackend._build_transonic_mesh(af;n_foil=384,n_wake=64,n_normal=112,dn1,
            alpha_deg=4.,trip_upper=0.,trip_lower=0.,n_base=2,farfield_angle_deg=4.,
            mesh_refinement=:aplus_uniform_quality,normal_collar=false,shock_layer=false)
        mesh=built.mesh
        summary=Q.quality_summary(mesh.X,mesh.Y,mesh.NIW,mesh.NIF)
        @test isempty(Q.gate_failures(Q.quality_gates(summary;cavitating=false)))
        defects,jac,_=G._convexity_defects(mesh.X,mesh.Y,mesh.NI,1:mesh.NJ)
        @test defects==0 && jac>0
        @test built.metadata["quality_report"]["provisional_wake_gates"]["passed"]
        @test built.metadata["wake_partner_max_distance"] <= 1e-12
        @test built.metadata["cfd_te_gap"] == built.metadata["source_te_gap"]
        @test built.metadata["conditioning"]["maximum_source_point_displacement_chords"]==0
        @test built.metadata["te_wall_fan_intervals"]==6
        @test summary.cells==173376
        h=[hypot(mesh.X[i,2]-mesh.X[i,1],mesh.Y[i,2]-mesh.Y[i,1])
           for i in mesh.NIW+1:mesh.NIW+mesh.NIF+1]
        # Nested interpolation slightly shortens the new children's normal
        # vectors; the original parent columns retain their requested spacing.
        @test minimum(h) >= .99dn1
        @test maximum(h) <= 1.000001dn1
    end
    af=G.read_airfoil_dat(joinpath(SCRATCH,"naca2412.dat"))
    previewdir=joinpath(SCRATCH,"strict_profile_preview");mkpath(previewdir)
    preview=CFDBackend.preview_mesh(joinpath(SCRATCH,"naca2412.dat"),
        Dict{String,Any}("cfd_mesh"=>"aplus_uniform_quality","cfd_te_topology"=>"aplus",
            "reynolds"=>re,"alpha_deg"=>4.,"trip_upper"=>0.,"trip_lower"=>0.),
        previewdir)
    @test preview.metadata["mesh_refinement"]=="aplus_uniform_quality"
    @test preview.metadata["source_n_foil"]==384
    @test preview.metadata["source_n_wake"]==64
    @test preview.metadata["source_n_base"]==2
    @test preview.metadata["quality_report"]["air_reference_gates"]["passed"]
    @test isfile(preview.mesh["artifact"])&&isfile(preview.mesh["preview_artifact"])
    for angle in (0.,4.,12.)
        @test CFDBackend._transonic_mesh_angle("aplus_uniform_fixed",Dict(),angle)==0
        @test CFDBackend._transonic_mesh_angle("aplus_uniform_quality",Dict(),angle)==angle
    end
    @test_throws ArgumentError CFDBackend._transonic_mesh_angle("aplus_uniform_fixed",Dict("cfd_farfield_angle_deg"=>4.),12.)
    fixed_dir=joinpath(SCRATCH,"fixed_profile_preview");mkpath(fixed_dir)
    fixed=CFDBackend.preview_mesh(joinpath(SCRATCH,"naca4412.dat"),
        Dict{String,Any}("cfd_mesh"=>"aplus_uniform_fixed","cfd_te_topology"=>"aplus",
            "reynolds"=>re,"alpha_deg"=>12.,"trip_upper"=>0.,"trip_lower"=>0.),fixed_dir)
    @test fixed.metadata["mesh_angle_deg"]==0
    @test fixed.metadata["flow_alpha_deg"]==12.
    @test fixed.metadata["mesh_orientation_policy"]=="fixed chord frame"
    @test fixed.metadata["quality_report"]["air_reference_gates"]["passed"]
    @test isfile(fixed.mesh["artifact"])
    for bad_fan in (NaN,Inf,0.5)
        @test_throws ArgumentError G.generate_cmesh(af;te_fan_cells=bad_fan)
    end
end
