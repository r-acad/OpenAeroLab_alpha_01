using Test,Printf,KernelAbstractions
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"solvers","pbcav_s1030","src","pbcav","PBCav.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_quality_metrics.jl"))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","pbcav_airfoil_geometry_tests");mkpath(SCRATCH)

function write_naca_dat(path,m,p,t)
    xx=(1 .- cos.(range(0,pi;length=201)))./2
    upper=Tuple{Float64,Float64}[];lower=Tuple{Float64,Float64}[]
    for x in xx
        yt=5t*(.2969sqrt(x)-.1260x-.3516x^2+.2843x^3-.1015x^4)
        yc=m==0 ? 0. : x<p ? m/p^2*(2p*x-x^2) : m/(1-p)^2*((1-2p)+2p*x-x^2)
        dy=m==0 ? 0. : x<p ? 2m/p^2*(p-x) : 2m/(1-p)^2*(p-x)
        a=atan(dy);push!(upper,(x-yt*sin(a),yc+yt*cos(a)));push!(lower,(x+yt*sin(a),yc-yt*cos(a)))
    end
    open(path,"w") do io
        println(io,"XFOIL cosine-spaced NACA regression fixture")
        for (x,y) in vcat(reverse(upper),lower[2:end]);@printf(io,"%.17g %.17g\n",x,y);end
    end
end

@testset "Arc-length interpolation uses nonuniform knots" begin
    s=[0.,.0001,.01,.1,.4,1.]
    x=2 .* s .+ 1;y=-3 .* s .+ .2
    for q in range(0,1;length=101)
        a,b=PBCav.CMeshGen._cr_eval(x,y,s,q)
        @test isapprox(a,2q+1;atol=2e-15)
        @test isapprox(b,-3q+.2;atol=2e-15)
    end
    for j in eachindex(s)
        @test PBCav.CMeshGen._cr_eval(x,y,s,s[j])==(x[j],y[j])
    end
end

@testset "Native DAT C+H meshes and exact Gurney retain their quality gates" begin
    mktempdir(SCRATCH) do dir
        for (name,m,p,t) in (("0012",0.,.4,.12),("2412",.02,.4,.12),("4412",.04,.4,.12))
            path=joinpath(dir,"naca$name.dat");write_naca_dat(path,m,p,t)
            af=PBCav.CMeshGen.read_airfoil_dat(path);gap=hypot(af.x[1]-af.x[end],af.y[1]-af.y[end]);nb=ceil(Int,gap/(3.9*3e-5))
            env=Dict("PB_DOMAIN"=>"freeair","PB_FOIL"=>path,"PB_MESH_LE_DS"=>"0","PB_MESH"=>"192,128,3e-5,1.10,80,64",
                "PB_MESH_TE_BASE"=>string(nb),"PB_MESH_TE_DS"=>string(1.01gap/(2nb)),"PB_MESH_TE_TOPOLOGY"=>"strip",
                "PB_MESH_REMAP"=>"common","PB_MESH_REMAP_KEEP"=>"2,24","PB_MESH_REMAP_LAW"=>"128,1.10,5",
                "PB_MESH_REMAP_WAKE"=>"300,500","PB_MESH_REMAP_STRIP"=>"graded")
            k,src=PBCav.read_knobs(env)
            gen=PBCav.generate_v1(PBCav.mesh_call(k,src));b=PBCav.cblocks(gen)
            parent=b
            remapped=PBCav.remap_columns(b,PBCav.remap_spec(k))
            @test remapped.info.version=="PBMeshRemapCommonV4"
            foil=b.NIW+1:b.NIW+b.NIF+1
            @test remapped.blocks.X[foil,1:3]==parent.X[foil,1:3]
            @test remapped.blocks.Y[foil,1:3]==parent.Y[foil,1:3]
            @test remapped.info.delta[b.NIW+1]==0
            @test remapped.info.delta[b.NIW+b.NIF+1]==0
            # The added strip width must start smoothly from the physical base.
            # A nonzero fourth-power offset followed by a forced base station
            # used to create a >20-degree boundary-cell kink here.
            added=remapped.info.strip_W-remapped.info.strip_W0
            ds=hypot(parent.X[b.NIW,1]-parent.X[b.NIW+1,1],parent.Y[b.NIW,1]-parent.Y[b.NIW+1,1])
            @test added[1]==0
            @test added[2]/ds<0.02
            b=PBCav.freeair_blocks(PBCav.subdivide(remapped.blocks,1))
            cases=name!="0012" ? ((false,:lower,.01),(true,:lower,.01),(true,:upper,.01),(true,:symmetric,.04)) : ((false,:lower,.01),(true,:lower,.01))
            for (flap,side,height) in cases
                plan=flap ? PBCav.baffle_plan(b,PBCav.BaffleSpec(;side,h=height)) : nothing
                blocks=plan===nothing ? b : plan.blocks
                mesh=PBCav.build_faces(blocks;backend=CPU(),baffle=plan)
                audit=PBCav.mesh_audit(mesh)
                @test all(last(g) for g in PBCav.mesh_audit_gates(audit))
                quality=CMeshQualityMetrics.quality_summary(blocks.X,blocks.Y,blocks.NIW,blocks.NIF;strip=(X=blocks.Xs,Y=blocks.Ys))
                @test isempty(CMeshQualityMetrics.gate_failures(CMeshQualityMetrics.quality_gates(quality;cavitating=false,strip_grade=true)))
                @test blocks.X[:,1]==b.X[:,1] && blocks.Y[:,1]==b.Y[:,1]
                # The app measures every geometric face; PBCav removes faces
                # cut into baffle walls from its internal-skew population.
                @test flap ? audit.skew_of_max<=quality.skew_of_max+1e-10 : isapprox(audit.skew_of_max,quality.skew_of_max;rtol=1e-10,atol=1e-10)
                if flap
                    @test isempty(PBCav.baffle_gate_failures(plan))
                    @test all(q->isapprox(q.h_real,side===:symmetric ? height/2 : height;atol=1e-14),plan.plates)
                end
                println("NACA$name flap=$flap side=$side height=$height cells=$(mesh.nc) app_gates=pass native_gates=pass")
            end
        end
    end
end

@testset "Graded-strip widening meets the base without an offset" begin
    W0=.00252;h0=3e-5;n=44;cw=1.02
    width(h)=PBCav._graded_strip_width(W0,h,h0,n,cw)
    @test width(h0)===W0
    @test width(prevfloat(h0))===W0
    @test width(nextfloat(h0))>=W0
    s=1e-4;kap=.01
    added(a)=width(hypot(h0,kap*a))-W0
    @test isapprox(added(s)/added(s/2),4.;rtol=.002)
    for h in (1e-4,1e-3,.01)
        @test width(h)>W0
        @test width(h)>n*h
    end
end
