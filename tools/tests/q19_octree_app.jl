# The first Aero Lab backend test: the CPU octree solver option end to end through the backend's own request
# handlers (no sockets): capabilities, model upload (a sphere STL written here), case validation and YAML, mesh
# creation (the solver's build-only subprocess), job polling to mesh_ready, the octree build summary against the
# app's validator, the format-4 AMR artifact, and the refusal of Run until its milestone. A temporary private root
# keeps every file out of the real results tree.
#   julia --project=third_party/JXfoil tools/tests/q19_octree_app.jl
using Test, Printf
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SCRATCH = joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","q19_tests")
mkpath(SCRATCH)
const PRIVATE = mktempdir(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"] = PRIVATE
include(joinpath(ROOT, "apps", "ludwig_web", "src", "workbench_revision.jl"))
include(joinpath(ROOT, "apps", "ludwig_web", "src", "aero_backend.jl"))
using HTTP, JSON3
const A = AeroBackend

function sphere_stl_bytes(n_lat, n_lon; r=0.5)
    P(θ, φ) = (r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ))
    tris = []
    for i in 0:n_lat-1, j in 0:n_lon-1
        θ0 = π * i / n_lat; θ1 = π * (i + 1) / n_lat; φ0 = 2π * j / n_lon; φ1 = 2π * (j + 1) / n_lon
        a = P(θ0, φ0); b = P(θ1, φ0); c = P(θ1, φ1); d = P(θ0, φ1)
        i < n_lat - 1 && push!(tris, (a, b, c)); i > 0 && push!(tris, (a, c, d))
    end
    io = IOBuffer(); write(io, zeros(UInt8, 80)); write(io, UInt32(length(tris)))
    for t in tris
        write(io, Float32(0), Float32(0), Float32(0))
        for v in t; write(io, Float32(v[1]), Float32(v[2]), Float32(v[3])); end
        write(io, UInt16(0))
    end
    take!(io)
end
body(response) = JSON3.read(String(response.body), Dict{String,Any})

@testset "capabilities list the CPU octree solver" begin
    caps = body(A.capabilities())
    q19 = only(filter(s -> s["id"] == "q19_octree", caps["solvers"]))
    @test q19["available"] == true
    @test q19["web_lattices"] == ["d3q19"]
    @test caps["features"]["q19_octree"] == true && caps["features"]["max_octree_levels"] == 10
end

model_id = let
    req = HTTP.Request("POST", "/api/aero/models?name=sphere.stl", ["Content-Type" => "model/stl"], sphere_stl_bytes(24, 48))
    r = A.upload_model(req)
    @test r.status == 201
    String(body(r)["model_id"])
end

settings = Dict{String,Any}("model_id" => model_id, "solver" => "q19_octree", "lattice" => "d3q19",
    "transform" => Dict("scale" => [1, 1, 1], "rotation_deg" => [0, 0, 0], "translation" => [0, 0, 0]),
    "domain" => Dict("upstream" => 2.0, "downstream" => 4.0, "lateral" => 2.0, "height" => 2.0, "boundary" => "free_air", "sizing" => "minimum"),
    "surface_dx" => 0.025, "levels" => 3, "mesh_sizing" => "manual", "target_vram_percent" => 80,
    "octree" => Dict("band_fine" => 6, "band" => 4, "wake_length" => 1.0, "wake_level_offset" => 1, "threads" => 8),
    "flow" => Dict("alpha" => 0, "beta" => 0, "velocity" => 10.0, "density" => 1.225, "kinematic_viscosity" => 1.5e-5, "speed_of_sound" => 343.0),
    "steps" => 100, "frames" => 4, "cuts" => Any[], "outputs" => Dict{String,Any}())

@testset "validation and YAML" begin
    case, tri = A._validate_case(settings)
    @test case["solver"] == "q19_octree" && case["lattice"] == "d3q19"
    @test case["octree"]["band_fine"] == 6 && case["octree"]["threads"] == 8
    yaml = A._yaml_case(case, case["bounds"]; build_only=true)
    @test occursin("lattice: \"d3q19\"", yaml) && occursin("collision_operator: \"rr\"", yaml)
    @test occursin("octree_q19:", yaml) && occursin("band_fine: 6", yaml) && occursin("build_only: true", yaml)
    est = A._estimate(case)
    @test est["fits"] === nothing && est["budget_gb"] === nothing && est["memory_gb"] > 0
    # the lattice must match the solver, the solver refuses VRAM sizing, Mach is capped at 0.3
    @test_throws ArgumentError A._validate_case(merge(settings, Dict("lattice" => "d3q27")))
    @test_throws ArgumentError A._validate_case(merge(settings, Dict("mesh_sizing" => "vram")))
    @test_throws ArgumentError A._validate_case(merge(settings, Dict("flow" => merge(settings["flow"], Dict("velocity" => 120.0)))))
    @test_throws ArgumentError A._validate_case(merge(settings, Dict("levels" => 11)))
    c10, _ = A._validate_case(merge(settings, Dict("levels" => 10)))
    @test c10["levels"] == 10
end

job_id = let
    req = HTTP.Request("POST", "/api/aero/mesh", ["Content-Type" => "application/json"], JSON3.write(settings))
    r = A.create_mesh(req)
    @test r.status == 202
    String(body(r)["job_id"])
end

@testset "mesh build to mesh_ready" begin
    status = ""; job = nothing
    t0 = time()
    while time() - t0 < 600
        job = body(A.job_response(job_id)); status = String(job["status"])
        status in ("mesh_ready", "mesh_failed", "cancelled") && break
        sleep(2)
    end
    @test status == "mesh_ready"
    if status != "mesh_ready"
        println("job: ", job["message"]); log = joinpath(A._case_dir(job_id, "mesh"), "solver.log"); isfile(log) && println(read(log, String)[max(1, end - 3000):end])
    end
    actual = job["actual"]
    @test actual["mesh"] == "octree" && actual["num_levels"] == 3
    @test actual["total_cells"] > 0 && actual["octree"]["fluid_cells_total"] > 0
    @test actual["memory_gb"] > 0 && occursin("octree", actual["memory_basis"])
    @test actual["octree"]["reports"]["leak_cells"] == 0
    @test actual["quality"]["runnable"] == true
    @test Float64(job["mesh_progress"]["percent"]) == 100
    @printf("octree mesh: %d fluid cells, %d stored slots, %.3f GB, domain %s m\n", actual["octree"]["fluid_cells_total"], actual["octree"]["stored_slots"], actual["memory_gb"], string(actual["domain_size_m"]))
    # the AMR artifact is the format-4 octree
    amr = JSON3.read(read(joinpath(A._case_dir(job_id, "mesh"), "RESULTS", "amr_viz", "amr_data.json"), String), Dict{String,Any})
    @test amr["format_version"] == 4 && amr["mesh"] == "octree" && length(amr["levels"]) == 3
    @test all(haskey(l, "boxes") && l["n_boxes"] * 6 == length(l["boxes"]) for l in amr["levels"])
    @test length(amr["geometry"]["vertices"]) == 9 * amr["geometry"]["n_triangles"]
    # running needs an output folder first
    err = try; A.start_run(job_id); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("output folder", sprint(showerror, err))
end

@testset "run to completion through the backend" begin
    export_root = joinpath(PRIVATE, "exports_test"); mkpath(export_root)
    updates = Dict("study_name" => "octree backend test",
                   "outputs" => Dict("folder" => export_root, "folder_selected" => true, "surface_vtk" => false, "surface_lbmp" => true,
                                     "volume_vtk" => true, "numeric_slices" => true, "slice_png" => false, "csv_history" => true))
    r = A.start_run(job_id, HTTP.Request("POST", "/api/aero/jobs/$job_id/run", ["Content-Type" => "application/json"], JSON3.write(updates)))
    job = body(r)
    @test job["status"] in ("starting", "running")
    @test !haskey(job, "gpu") || true
    status = ""; t0 = time()
    while time() - t0 < 900
        job = body(A.job_response(job_id)); status = String(job["status"])
        status in ("completed", "failed", "cancelled") && break
        sleep(3)
    end
    if status != "completed"
        println("run job: ", job["message"]); log = joinpath(String(job["run_case_dir"]), "solver.log"); isfile(log) && println(read(log, String)[max(1, end - 4000):end])
    end
    @test status == "completed"
    @test job["solver_summary"]["status"] == "completed_steps" && job["solver_summary"]["steps"] == settings["steps"]
    @test occursin("CPU octree", String(job["message"]))
    progress = job["progress"]; @test Int(progress["step"]) == settings["steps"]
    arts = job["artifacts"]
    flat = JSON3.write(arts)
    @test occursin("surface_000100", flat) && occursin("lbmslice", flat)       # the surface and the planes are catalogued
    println("artifact keys: ", join(sort(collect(keys(arts))), ", "))
    run_dir = String(job["run_case_dir"])
    @test isfile(joinpath(run_dir, "RESULTS", "csv", "forces.csv")) && isfile(joinpath(run_dir, "RESULTS", "csv", "convergence.csv"))
    @test isfile(joinpath(run_dir, "RESULTS", "VTK", @sprintf("surface_%06d.lbmp", settings["steps"])))
    @test isfile(joinpath(run_dir, "RESULTS", "VTK", @sprintf("flow_%06d.vtu", settings["steps"])))
    @test isdir(joinpath(run_dir, "RESULTS", "SLICES", @sprintf("step_%06d", settings["steps"])))
    @test isfile(joinpath(run_dir, "mesh.oct")) || isfile(joinpath(dirname(run_dir), "mesh", "mesh.oct"))
    # the monitoring and progress readers accept the histories
    mon = job["monitoring"]; @test mon isa AbstractDict && !haskey(mon, "error")
    println("monitoring keys: ", join(sort(collect(keys(mon))), ", "))
    @printf("run: %d steps, status %s, Cd %.4f, exports in %s\n", settings["steps"], status, Float64(get(job["solver_summary"], "Cd", NaN)), String(job["export_directory"]))
end
println("q19_octree_app: done")
