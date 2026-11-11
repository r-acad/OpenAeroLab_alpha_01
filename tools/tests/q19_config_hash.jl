# Configuration identities must invalidate cached meshes when the setup changes.
# Run: julia --startup-file=no --project=src/octree_q19 -t2 tools/tests/q19_config_hash.jl
using Test, SHA, OctreeQ19
const Q=OctreeQ19
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","q19_config_hash")
mkpath(SCRATCH)
const FIXTURE=mktempdir(SCRATCH;prefix="identity_")
const GEOMETRY=joinpath(FIXTURE,"model.stl")
const CONFIG=joinpath(FIXTURE,"config.yaml")
open(GEOMETRY,"w") do io
    println(io,"solid test_sphere")
    for tri in sphere_tris(6,12;r=.25)
        println(io,"facet normal 0 0 0\nouter loop")
        for v in tri;println(io,"vertex ",join(v," "));end
        println(io,"endloop\nendfacet")
    end
    println(io,"endsolid test_sphere")
end
const BASE="""
basic:
  stl_file: [model.stl]
  stl_scale: 1.0
  reference_chord: 1.0
  reference_length_for_meshing: 1.0
  fluid: {density: 1.225, kinematic_viscosity: 0.000015, speed_of_sound: 343.0}
  flow: {velocity: 10.0}
  simulation:
    build_only: true
    steps: 10
    ramp_steps: 2
    output_freq: 11
    slice_output_freq: 10
advanced:
  domain: {upstream: 1.0, downstream: 1.0, lateral: 1.0, height: 1.0}
  refinement: {symmetric_analysis: false}
  forces:
    output_freq: 10
    moment_center: [0.25, 0.0, 0.0]
  slice: {planes: []}
octree_q19:
  precision: f32
  mesh:
    levels: 1
    finest_dx_m: 0.25
    seal_cavities: false
    band_fine: 3
    band: 3
    wake: {length: 0.0}
  numerics: {collision: rr, sgs: wale, u_lattice: 0.08}
  boundaries: {y_min: slip}
"""
load_text(text;kwargs...)=(write(CONFIG,text);Q.load_case(FIXTURE;kwargs...))
@testset "Q19 fingerprints retain the parsed source text" begin
    base=load_text(BASE)
    @test base.config_sha!=bytes2hex(sha256(UInt8[]))
    @test length(base.config_sha)==64
    @test base.stl_sha==bytes2hex(open(sha256,GEOMETRY))
    @test load_text(BASE).config_sha==base.config_sha
    # These run controls are intentionally excluded to permit continuing the
    # same physical setup with a different duration or output cadence.
    continued=replace(BASE,"build_only: true"=>"build_only: false","steps: 10"=>"steps: 30",
        "ramp_steps: 2"=>"ramp_steps: 4","output_freq: 11"=>"output_freq: 31",
        "slice_output_freq: 10"=>"slice_output_freq: 5")
    @test load_text(continued).config_sha==base.config_sha
    @test load_text(BASE;build_only=false).config_sha==base.config_sha
    formatted=replace(BASE,"343.0"=>"343","velocity: 10.0"=>"velocity: 10",
        "length: 0.0"=>"length: 0","stl_scale: 1.0"=>"stl_scale: 1")
    @test load_text(formatted).config_sha==base.config_sha
    reordered=Dict(reverse(collect(pairs(base.raw))))
    @test Q._configuration_fingerprint(reordered)==base.config_sha
    @test Q._configuration_fingerprint(Dict("x"=>1))==Q._configuration_fingerprint(Dict("x"=>1.0))
    @test Q._configuration_fingerprint(Dict("x"=>1))!=Q._configuration_fingerprint(Dict("x"=>true))
    @test Q._configuration_fingerprint(Dict("x"=>1.0))!=Q._configuration_fingerprint(Dict("x"=>nextfloat(1.0)))
    @test Q._configuration_fingerprint(Dict("x"=>Int64(2)^53+1))!=Q._configuration_fingerprint(Dict("x"=>Float64(2)^53))
    changed=("finest_dx_m: 0.25"=>"finest_dx_m: 0.2",
        "levels: 1"=>"levels: 2","seal_cavities: false"=>"seal_cavities: true",
        "upstream: 1.0"=>"upstream: 2.0","velocity: 10.0"=>"velocity: 12.0",
        "sgs: wale"=>"sgs: none","y_min: slip"=>"y_min: freestream",
        "precision: f32"=>"precision: f64","stl_scale: 1.0"=>"stl_scale: 1.1",
        "moment_center: [0.25, 0.0, 0.0]"=>"moment_center: [0.3, 0.0, 0.0]")
    hashes=String[]
    for replacement in changed
        cfg=load_text(replace(BASE,replacement))
        @test cfg.config_sha!=base.config_sha
        push!(hashes,cfg.config_sha)
    end
    @test length(unique(hashes))==length(changed)
    # Geometry contents have their own independent fingerprint.
    original=read(GEOMETRY)
    open(GEOMETRY,"a") do io;println(io," ");end
    altered=load_text(BASE)
    @test altered.config_sha==base.config_sha
    @test altered.stl_sha!=base.stl_sha
    write(GEOMETRY,original)
end
@testset "Changed setup replaces the cached mesh" begin
    cfg=load_text(BASE)
    mesh=Q.build_only_case(cfg)
    @test mesh.config_sha==cfg.config_sha
    continued=load_text(replace(BASE,"build_only: true"=>"build_only: false","steps: 10"=>"steps: 20"))
    cached=Q.case_mesh(continued)
    @test cached.config_sha==cfg.config_sha
    @test cached.levels[end].dx==.25
    finer=load_text(replace(BASE,"finest_dx_m: 0.25"=>"finest_dx_m: 0.2"))
    rebuilt=Q.case_mesh(finer)
    @test rebuilt.config_sha==finer.config_sha
    @test rebuilt.config_sha!=mesh.config_sha
    @test rebuilt.levels[end].dx==.2
    # Old files remain readable for inspection, but their empty-source identity
    # must never authorize reuse by the corrected loader.
    legacy=Q.OctMesh(rebuilt.domain,rebuilt.offset,rebuilt.periodic,rebuilt.levels,
        rebuilt.triangles,bytes2hex(sha256(UInt8[])),rebuilt.stl_sha,rebuilt.counts)
    Q.write_mesh(joinpath(FIXTURE,"mesh.oct"),legacy)
    replaced=Q.case_mesh(finer)
    @test replaced.config_sha==finer.config_sha
    @test replaced.config_sha!=legacy.config_sha
    @test replaced.levels[end].dx==.2
end
@testset "Old cavity-sealed mesh is never reused or overwritten" begin
    sealed=load_text(replace(BASE,"seal_cavities: false"=>"seal_cavities: true"))
    mesh=Q.build_only_case(sealed)
    legacy_counts=copy(mesh.counts)
    delete!(legacy_counts,"cavity_seal_policy_version")
    legacy=Q.OctMesh(mesh.domain,mesh.offset,mesh.periodic,mesh.levels,
        mesh.triangles,mesh.config_sha,mesh.stl_sha,legacy_counts)
    path=joinpath(FIXTURE,"mesh.oct")
    Q.write_mesh(path,legacy)
    old_sha=bytes2hex(open(sha256,path))
    err=try Q.case_mesh(sealed); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("Create a new mesh",sprint(showerror,err))
    @test bytes2hex(open(sha256,path))==old_sha
    # A changed setup must not bypass the guard and overwrite the old file.
    changed=load_text(replace(BASE,"seal_cavities: false"=>"seal_cavities: true",
        "finest_dx_m: 0.25"=>"finest_dx_m: 0.2"))
    @test_throws ErrorException Q.case_mesh(changed)
    @test bytes2hex(open(sha256,path))==old_sha
    for invalid in (true,"2",nothing,-1)
        legacy_counts["cavity_seal_policy_version"]=invalid
        @test_throws ErrorException Q._require_current_cavity_policy(sealed,legacy)
    end
    legacy_counts["cavity_seal_policy_version"]=2
    @test Q._require_current_cavity_policy(sealed,legacy) === nothing
end
@testset "Unimplemented numerical options fail before a run" begin
    @test Q._require_implemented_numerics((walls=(wall_function=false,),numerics=(temporal=:linear,))) === nothing
    @test_throws ArgumentError Q._require_implemented_numerics((walls=(wall_function=true,),numerics=(temporal=:linear,)))
    @test_throws ArgumentError Q._require_implemented_numerics((walls=(wall_function=false,),numerics=(temporal=:quadratic,)))
end
println("Fixture retained under disposable root: ",FIXTURE)
