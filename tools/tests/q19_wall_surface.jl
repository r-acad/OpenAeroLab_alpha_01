# Independent output-only tests. No mesh generation, simulation advance or production data.
# julia --startup-file=no --project=src/octree_q19 tools/tests/q19_wall_surface.jl
using Test, OctreeQ19, SHA, TOML
const G=OctreeQ19
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT"))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","q19_wall_surface_tests")
mkpath(SCRATCH)
const OUT=mktempdir(SCRATCH)
const CSHA=repeat("a",64)
const SSHA=repeat("b",64)

function fixture(;reversed=false)
    # A planar source facet has cells on both sides, with two distinct pressures
    # on its upper side. Cell 5 also cuts another, perpendicular source facet.
    a=((0.,0.,1.),(4.,0.,1.),(0.,4.,1.))
    reversed && (a=(a[1],a[3],a[2]))
    tri=G.Tri[a,((1.,0.,0.),(1.,4.,0.),(1.,0.,4.)),
        ((0.,-2.,1.),(1.,-2.,1.),(0.,-1.,1.))] # excluded/unsampled geometry
    lv=G.OctLevel(1,1.,(4,4,4));lv.nocts=1
    lv.origin=zeros(Int32,3,1);lv.kind=fill(G.K_FLUID,8,1)
    lv.cut_slot=Int32[1,5,6];lv.cut_off=Int32[1,2,6,7]
    lv.cut_dir=UInt8[6,7,14,2,4,7];lv.cut_q=fill(0.5f0,6)
    lv.cut_tri=Int32[1,1,1,2,0,1]
    mesh=G.OctMesh((4.,4.,4.),(0.,0.,0.),(false,false,false),[lv],tri,CSHA,SSHA,Dict{String,Any}())
    spec=G.MeshSpec(domain=(4.,4.,4.),levels=1,dx_finest=1.,faces=(:inlet,:outlet,:symmetry,:slip,:slip,:slip))
    initial=G.test_case_config(spec;velocity=10.,u_lattice=.1)
    cfg=G.CaseConfig((name===:config_sha ? CSHA : name===:stl_sha ? SSHA : getfield(initial,name) for name in fieldnames(G.CaseConfig))...)
    mesh,cfg
end

function groups(mesh)
    rows=NamedTuple[]
    stats=G.foreach_wall_group(mesh) do l,ci,s,t,p,n,dx,count,flags
        push!(rows,(level=l,ci=ci,slot=s,parent=t,position=p,normal=n,dx=dx,count=count,flags=flags))
    end
    rows,stats
end

function geometry_records(path)
    open(path,"r") do io
        @test String(read(io,8))=="LBMWGEO\0"
        @test read(io,UInt32)==1
        @test read(io,UInt32)==36
        count=Int(read(io,UInt64));rows=NamedTuple[]
        for _ in 1:count
            p=ntuple(_->read(io,Float32),3);n=ntuple(_->read(io,Float32),3)
            push!(rows,(position=p,normal=n,dx=read(io,Float32),parent=read(io,UInt32),
                level=read(io,UInt8),count=read(io,UInt8),flags=read(io,UInt16)))
        end
        @test eof(io)
        rows
    end
end

function field_record(path)
    open(path,"r") do io
        @test String(read(io,8))=="LBMWFLD\0"
        @test read(io,UInt32)==1
        @test read(io,UInt32)==1
        count=Int(read(io,UInt64));sha=String(read(io,64));step=read(io,Int64)
        time=read(io,Float64);rho0=read(io,Float64);speed=read(io,Float64);qref=read(io,Float64)
        @test position(io)==128
        pressure=read!(io,Vector{Float32}(undef,count));@test eof(io)
        (;count,sha,step,time,rho0,speed,qref,pressure)
    end
end

@testset "Native geometric support, independent pressure locations and winding" begin
    mesh,cfg=fixture();rows,stats=groups(mesh)
    @test stats["samples"]==4
    @test stats["contributing_links"]==5
    @test stats["unmapped_links"]==1
    @test stats["invalid_links"]==stats["outside_domain_links"]==0
    @test [(r.slot,r.parent,r.count) for r in rows]==[(1,1,1),(5,1,2),(5,2,1),(6,1,1)]
    @test rows[2].position==(.75,.5,1.) # mean of vertical and diagonal exact intersections
    @test rows[3].position==(1.,.5,1.5)
    @test rows[1].normal==(0.,0.,-1.)
    @test rows[2].normal==rows[4].normal==(0.,0.,1.)
    @test rows[3].normal==(-1.,0.,0.)
    @test all(r->r.position[2]>=0,rows) # no fabricated mirrored/excluded half
    @test all(r->r.parent!=3,rows) # no fabricated samples on unsupported facet
    @test all(r->G.vdot(r.normal,G.vsub(G.slot_centre(mesh.levels[1],r.slot),r.position))>0,rows)
    reverse_mesh,_=fixture(reversed=true)
    @test first(groups(reverse_mesh))==rows
    masked=deepcopy(mesh);masked.levels[1].kind[1]=G.K_SOLID
    @test length(first(groups(masked)))==3
    bad=deepcopy(mesh);bad.levels[1].cut_q[1]=NaN32
    @test last(groups(bad))["invalid_links"]==1
    @test length(first(groups(bad)))==3
end

@testset "Fine pressures vary within original facets and files bind immutable geometry" begin
    mesh,cfg=fixture();rhos=Dict(1=>1.04f0,5=>1.01f0,6=>1.02f0)
    out=joinpath(OUT,"variation")
    path=G.write_wall_surface!(mesh,cfg,(l,ci,s)->rhos[s],out,12;source=Dict("kind"=>"independent_fixture"))
    f=field_record(path);metadata=TOML.parsefile(path*".toml")
    geo=joinpath(dirname(path),metadata["geometry_file"]);rows=geometry_records(geo)
    @test length(rows)==f.count==4
    @test filesize(geo)==24+36f.count
    @test filesize(path)==128+4f.count
    @test f.sha==bytes2hex(open(sha256,geo))==metadata["geometry_sha256"]
    @test f.step==12
    @test f.time≈12G.lattice_scaling(cfg).dt_finest
    @test f.qref≈.5cfg.flow.density*cfg.flow.velocity^2
    scale=cfg.flow.density*G.lattice_scaling(cfg).velocity_scale^2/3
    expected=Float32[(Float64(rhos[s])-1)*scale for s in (1,5,5,6)]
    @test f.pressure≈expected
    @test f.pressure[2]!=f.pressure[4] # SAME original parent, distinct cell pressures
    @test f.pressure[2]==f.pressure[3] # SAME fluid cell, separate geometric facets
    @test f.pressure[1]!=f.pressure[2] # opposite sides stay independent
    @test metadata["original_facets_with_pressure_variation"]==1
    @test metadata["finite_samples"]==4
    @test metadata["shear_available"]===false
    @test metadata["convergence_assessed"]===false
    @test occursin("No facet averaging",metadata["pressure_note"])
    old=read(path)
    @test G.write_wall_surface!(mesh,cfg,(l,ci,s)->rhos[s],out,12)==path
    @test TOML.parsefile(path*".toml")["source"]["kind"]=="independent_fixture"
    @test_throws ErrorException G.write_wall_surface!(mesh,cfg,(l,ci,s)->1.2,out,12)
    @test read(path)==old
    @test !any(endswith(".tmp"),readdir(dirname(path)))
    invalid=G.write_wall_surface!(mesh,cfg,(l,ci,s)->s==5 ? NaN32 : rhos[s],out,13)
    missing=field_record(invalid)
    @test missing.count==f.count && missing.sha==f.sha
    @test isnan(missing.pressure[2]) && isnan(missing.pressure[3])
    @test missing.pressure[[1,4]]==f.pressure[[1,4]]
    uniform=field_record(G.write_wall_surface!(mesh,cfg,(l,ci,s)->1f0,out,0))
    @test all(iszero,uniform.pressure)
    @test uniform.step==0
    # A same-length corruption must not silently reuse a cached geometry hash.
    before=read(geo)
    open(geo,"r+") do io;seek(io,24);write(io,UInt8(before[25]⊻0x01));end
    @test_throws ErrorException G.wall_surface_geometry(mesh,out)
    write(geo,before)
end

function checkpoint_fixture(path,mesh;T=Float32)
    density=T[1.04,1.005,1.006,1.007,1.01,1.02,1.008,1.009]
    open(path,"w") do io
        write(io,codeunits(G.CHECKPOINT_MAGIC),Int64(1),Int64(sizeof(T)),codeunits(mesh.config_sha),codeunits(mesh.stl_sha))
        write(io,Int64(12),Float64(.1),Float64(8),Int64(2),zeros(Float64,9),Int64(1),Int64(1))
        write(io,zeros(T,8*(2G.Q+6)),density,density,Int64(3),zeros(T,3))
    end
    density
end

@testset "Read-only checkpoint sampling preserves stored state and rejects malformed identity" begin
    mesh,cfg=fixture();path=joinpath(OUT,"checkpoint.bin")
    for T in (Float32,Float64)
        density=checkpoint_fixture(path,mesh;T)
        matching=G.CaseConfig((name===:precision ? T : getfield(cfg,name) for name in fieldnames(G.CaseConfig))...)
        before=read(path);before_hash=bytes2hex(sha256(before));step,wall=G.checkpoint_wall_densities(path,mesh,matching)
        @test step==12
        @test length(wall)==1 && eltype(wall[1])===T
        @test wall[1]==density[[1,5,6]]
        @test read(path)==before
        @test bytes2hex(open(sha256,path))==before_hash
        T===Float64 && @test_throws ErrorException G.checkpoint_wall_densities(path,mesh,cfg)
    end
    checkpoint_fixture(path,mesh)
    original=read(path)
    for (name,offset,value) in (("precision",16,Int64(3)),("version",8,Int64(2)),
            ("levels",256,Int64(2)),("oct_count",264,Int64(2)))
        invalid=joinpath(OUT,name*".bin");write(invalid,original)
        open(invalid,"r+") do io;seek(io,offset);write(io,value);end
        @test_throws ErrorException G.checkpoint_wall_densities(invalid,mesh,cfg)
    end
    invalid=joinpath(OUT,"wrong_hash.bin");write(invalid,original)
    open(invalid,"r+") do io;seek(io,24);write(io,codeunits(repeat("c",64)));end
    @test_throws ErrorException G.checkpoint_wall_densities(invalid,mesh,cfg)
    for count in (80,300,length(original)-1)
        invalid=joinpath(OUT,"truncated_$count.bin");write(invalid,original[1:count])
        @test_throws Exception G.checkpoint_wall_densities(invalid,mesh,cfg)
    end
    invalid=joinpath(OUT,"trailing.bin");write(invalid,vcat(original,0x00))
    @test_throws ErrorException G.checkpoint_wall_densities(invalid,mesh,cfg)
end
println("WALL_SURFACE_TEST_OUTPUT ",OUT)
