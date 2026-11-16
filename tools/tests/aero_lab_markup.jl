# Aero Lab page, scripts and requests, without a browser or a solver run:
#   - the page markup: unique ids, the solver choice above numbered steps, closed Advanced groups, step actions;
#   - every control id the scripts name exists in the page, and no removed control is named anywhere;
#   - for each solver the page offers, the request the page's own builder makes from the page's default values
#     (tools/tests/fixtures/aero_request_dom.cjs, Node), validated by the backend and written as a case file that
#     carries only keys that solver reads; the octree block for default settings, exactly;
#   - the page's own scripts run on the page markup with a small DOM and no 3D engine
#     (tools/tests/fixtures/aero_page_dom.cjs, Node): steps, solver choice, restoring a saved case, the request.
# Layout, styles and the 3D view are not checked here.
#   julia --startup-file=no --project=third_party/JXfoil tools/tests/aero_lab_markup.jl
using Test, HTTP, JSON3
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const STATIC = joinpath(ROOT, "apps", "ludwig_web", "static")
const SCRATCH = joinpath(get(ENV, "LUDWIG_PRIVATE_ROOT", joinpath(ROOT, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")), "TEMP_DELETEABLE", "aero_lab_markup_tests")
mkpath(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"] = joinpath(mktempdir(SCRATCH), "app")
include(joinpath(ROOT, "apps", "ludwig_web", "src", "aero_backend.jl"))
include(joinpath(@__DIR__, "aero_case_file_keys.jl"))
const A = AeroBackend

const PAGE = read(joinpath(STATIC, "aero_lab.html"), String)
const SCRIPT_NAMES = sort!(filter(name -> startswith(name, "aero_") && endswith(name, ".js"), readdir(STATIC)))
const SCRIPTS = Dict(name => read(joinpath(STATIC, name), String) for name in SCRIPT_NAMES)
const PAGE_IDS = [m.captures[1] for m in eachmatch(r"\bid\s*=\s*\"([^\"]+)\"", PAGE)]
const ASIDE = match(r"(?s)<aside\b[^>]*class=\"setup panel\"[^>]*>(.*?)</aside>", PAGE).captures[1]
const NAV = match(r"(?s)<nav\b[^>]*workflow-bar[^>]*>(.*?)</nav>", PAGE).captures[1]
panel(name) = match(Regex("(?s)<section\\b[^>]*data-parameter-panel=\"$name\"[^>]*>(.*?)</section>"), ASIDE).captures[1]
select_options(id) = [m.captures[1] for m in eachmatch(r"<option\b[^>]*\bvalue=\"([^\"]*)\"",
    match(Regex("(?s)<select\\b[^>]*\\bid=\"" * id * "\"[^>]*>(.*?)</select>"), PAGE).captures[1])]

# Controls removed in the clean-up of 14 Nov 2026. None may come back by name in the page or in a script.
const REMOVED_IDS = ["lattice", "lattice-label", "lattice-title", "lattice-note", "frames", "flow-input-mode",
    "octree-collision", "octree-absorbing", "octree-ramp-steps", "symmetry-auto-center-y", "geometry-audit",
    "surface-budget-note", "study-save-menu", "run-live-results", "results-menu-summary",
    "menu-file", "menu-model", "menu-domain", "menu-solver", "menu-analysis", "menu-results", "menu-view", "menu-help"]

@testset "Aero Lab page: ids, solver choice, numbered steps, Advanced groups" begin
    @test !isempty(PAGE_IDS) && allunique(PAGE_IDS)
    # The solver decides which inputs exist: one selector, above every step panel, offering the runnable solvers.
    @test first(findfirst("id=\"solver\"", ASIDE)) < first(findfirst("data-parameter-panel=", ASIDE))
    @test occursin(r"<span>Solver</span>\s*<select id=\"solver\"", ASIDE)
    @test select_options("solver") == [String(choice.id) for choice in A.SOLVER_CHOICES] == ["main_d3q27", "main_d3q39", "pond_thermal", "q19_octree"]
    # The navigation is a flat row of numbered steps in working order, then the other panels after a separator.
    @test !occursin("<details", NAV) && !occursin("menu-popover", NAV) && !occursin("top-menu", PAGE)
    steps = [(m.captures[1], m.captures[2]) for m in
        eachmatch(r"workflow-step\"[^>]*data-parameter-target=\"([^\"]+)\"[^>]*><span class=\"step-number\">(\d)</span>", NAV)]
    @test steps == [("model", "1"), ("domain", "2"), ("flow", "3"), ("mesh", "4"), ("run", "5"), ("results", "6")]
    separator = findfirst("workflow-separator", NAV)
    @test separator !== nothing
    @test all(first(findfirst("data-parameter-target=\"$name\"", NAV)) > first(separator) for name in ("study", "display", "help"))
    # Every destination exists exactly once.
    targets = unique(m.captures[1] for m in eachmatch(r"data-parameter-target=\"([^\"]+)\"", PAGE))
    panels = [m.captures[1] for m in eachmatch(r"data-parameter-panel=\"([^\"]+)\"", PAGE)]
    @test allunique(panels) && sort(panels) == sort(["model", "domain", "flow", "mesh", "run", "results", "study", "display", "help"])
    @test isempty(setdiff(targets, panels))
    focus = unique(m.captures[1] for m in eachmatch(r"data-focus-control=\"([^\"]+)\"", PAGE))
    @test isempty(setdiff(focus, PAGE_IDS))
    # Rarely needed settings sit in one Advanced group per step, closed in the page source; no other disclosure in the setup pane.
    details = [m.match for m in eachmatch(r"<details\b[^>]*>", ASIDE)]
    @test [match(r"id=\"([^\"]+)\"", d).captures[1] for d in details] ==
        ["advanced-model", "advanced-domain", "advanced-flow", "advanced-mesh", "advanced-results", "advanced-display"]
    @test all(d -> occursin("advanced-group", d) && !occursin(r"\sopen\b", d), details)
    # Each step ends with its action, and each quantity has one control.
    @test occursin("id=\"build-mesh\"", panel("mesh")) && occursin("id=\"run-solver\"", panel("run")) && occursin("id=\"cancel-job\"", panel("run"))
    @test occursin("id=\"model-file\"", panel("model")) && occursin("id=\"steps\"", panel("run")) && occursin("id=\"study-save\"", panel("study"))
    for id in ("octree-controls", "octree-wall-function", "octree-threads", "octree-seal-cavities", "gpu-mesh-controls", "target-vram")
        @test occursin("id=\"$id\"", panel("mesh"))
    end
    for id in ("octree-u-lattice", "octree-hrr-sigma", "octree-checkpoint-keep", "octree-plane-samples")   # Advanced, octree only
        @test occursin("id=\"$id\"", match(r"(?s)<details id=\"advanced-mesh\"[^>]*>(.*?)</details>", ASIDE).captures[1])
    end
    @test occursin(r"<input id=\"octree-wall-function\" type=\"checkbox\" title=[^>]*> Wall function \(experimental, not validated\)", ASIDE)
    @test isempty(intersect(PAGE_IDS, REMOVED_IDS))
    @test !occursin(r"(?i)>[^<>]*\blegacy\b[^<>]*<", replace(PAGE, "ASCII legacy VTK" => "ASCII VTK"))   # "legacy VTK" is a file format name
    @test !occursin("lbmtau", PAGE) && !occursin("data-workflow-help", PAGE)
    # The page loads only scripts and styles the server serves.
    server = read(joinpath(ROOT, "apps", "ludwig_web", "src", "server.jl"), String)
    for m in eachmatch(r"\"(/workbench-static/[^\"?]+)", PAGE)
        served = match(Regex("\"" * m.captures[1] * "\" => \\((?:joinpath\\(\"vendor\", )?\"([^\"]+)\""), server)
        @test served !== nothing
        served === nothing || @test isfile(joinpath(STATIC, occursin("vendor", m.captures[1]) ? joinpath("vendor", served.captures[1]) : served.captures[1]))
    end
end

@testset "Aero Lab scripts name only controls the page has" begin
    # Ids a script creates itself, and the output controls built by the shared output script.
    created = Set(m.captures[1] for (_, text) in SCRIPTS for m in eachmatch(r"\.id\s*=\s*'([^']+)'", text))
    union!(created, ["output-folder"])
    known = union(Set(PAGE_IDS), created)
    for (name, text) in SCRIPTS
        named = Set(m.captures[1] for m in eachmatch(r"(?:\$|getElementById)\('([A-Za-z0-9_-]+)'\)", text))
        union!(named, (m.captures[1] for m in eachmatch(r"\bio\.(?:value|checked)\('([A-Za-z0-9_-]+)'\)", text)))
        union!(named, (m.captures[1] for m in eachmatch(r"\b(?:number\(io, |inputNumber\()'([A-Za-z0-9_-]+)',", text)))
        # Ids built per axis ('scale-' + axis): all three must exist.
        for m in eachmatch(r"\b(?:number\(io, |inputNumber\()'([a-z-]+-)' ?\+", text), axis in ("x", "y", "z")
            push!(named, m.captures[1] * axis)
        end
        @testset "$name" begin
            @test isempty(setdiff(named, known))
            @test all(id -> !occursin("'" * id * "'", text) && !occursin("#" * id, text), REMOVED_IDS)
        end
    end
    # Tooltips are attached by id: each id must exist.
    tips = match(r"(?s)const tips = \{(.*?)\n    \};", SCRIPTS["aero_layout.js"]).captures[1]
    @test isempty(setdiff([m.captures[1] for m in eachmatch(r"'([a-z0-9-]+)':'", tips)], known))
    # Removed request keys and server checks are gone from the scripts.
    all_scripts = join(values(SCRIPTS), "\n")
    for token in ("symmetry_center_y", "input_mode", "absorbing_faces", "ramp_steps", "wall_law", "outdatedServer", "web_lattices", "lbmtau", "closeMenus")
        @test !occursin(token, all_scripts)
    end
end

# The request the page would send: the page's default control values through the page's own builder.
function page_request(choice; overrides=Dict{String,Any}())
    node = Sys.which("node")
    node === nothing && error("Node.js is required for the Aero Lab request fixture")
    fixture = joinpath(@__DIR__, "fixtures", "aero_request_dom.cjs")
    JSON3.read(read(`$node $fixture $ROOT $choice $(JSON3.write(overrides))`, String), Dict{String,Any})
end
plain(x) = A._plain(x)

const MODEL_ID = let
    path = joinpath(ENV["LUDWIG_PRIVATE_ROOT"], "sphere.stl"); mkpath(dirname(path))
    # a closed sphere of radius 0.25 m
    point(i, j) = (θ = π * i / 8; φ = 2π * j / 16; (.25sin(θ) * cos(φ), .25sin(θ) * sin(φ), .25cos(θ)))
    triangles = A.Triangle[]
    for i in 0:7, j in 0:15
        a, b, c, d = point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)
        i < 7 && push!(triangles, (a, b, c)); i > 0 && push!(triangles, (a, c, d))
    end
    A._write_stl(path, triangles)
    response = A.upload_model(HTTP.Request("POST", "/api/aero/models", ["Content-Type" => "application/octet-stream"], read(path)))
    JSON3.read(String(response.body), Dict{String,Any})["model_id"]
end

@testset "Requests of the page: each solver gets its own settings and a case file it reads" begin
    gpu_reader = gpu_reader_text(); octree_reader = octree_reader_text()
    common = ["model_id", "solver", "lattice", "geometry_handling", "transform", "domain", "flow", "surface_dx", "levels", "steps", "display_every", "outputs", "cuts"]
    gpu_only = ["mesh_sizing", "target_vram_percent", "surface_triangle_max_edge_m"]
    for choice in A.SOLVER_CHOICES
        @testset "$(choice.id)" begin
            built = page_request(String(choice.id))
            @test built["solvers"] == built["page_solvers"] == [String(c.id) for c in A.SOLVER_CHOICES]
            request = plain(built["request"]); request["model_id"] = MODEL_ID
            @test request["solver"] == choice.solver && request["lattice"] == choice.lattice
            octree = choice.solver == "q19_octree"; thermal = choice.solver == "pond_thermal"
            @test sort(collect(keys(request))) == sort(vcat(common, octree ? ["octree"] : gpu_only))
            @test sort(collect(keys(request["flow"]))) == sort(vcat(["alpha", "beta", "density", "kinematic_viscosity"],
                thermal ? ["mach", "temperature_k", "cfl", "wall_model"] : ["velocity", "speed_of_sound"]))
            @test sort(collect(keys(request["domain"]))) == ["boundary", "downstream", "height", "lateral", "sizing", "upstream", "x_offset_m"]
            @test haskey(built["run"], "target_vram_percent") == !octree
            octree && @test sort(collect(keys(request["octree"]))) == ["band", "band_fine", "checkpoint_keep", "hrr_sigma", "max_plane_samples",
                "seal_cavities", "threads", "u_lattice", "wake_length", "wake_level_offset", "wall_function"]
            # The backend accepts the request as sent and writes the case file of a mesh build and of a run.
            case, _ = A._validate_case(request)
            @test A._check_study_setup(deepcopy(request)) isa AbstractDict       # the same request is a valid study setup
            for build_only in (true, false)
                yaml = A._yaml_case(case, case["bounds"]; build_only)
                @test isempty(unread_case_keys(yaml, octree ? octree_reader : gpu_reader))
                paths = case_file_paths(YAML.load(yaml))
                if octree
                    @test isempty(setdiff([p for p in paths if !startswith(p, "octree_q19.")], OCTREE_CASE_PATHS))
                    @test endswith(yaml, octree_block_text())                    # the block for default settings, exactly
                    @test !any(startswith(p, "mesh.") || startswith(p, "advanced.numerics") || startswith(p, "advanced.gpu") for p in paths)
                else
                    # No octree block, no auto-sizer block, and the settings of the other GPU solvers only where they belong.
                    @test !any(startswith(p, "octree_q19") || startswith(p, "mesh.") || endswith(p, "_chords") || endswith(p, "hrr_sigma_per_level") for p in paths)
                    @test any(startswith(p, "advanced.numerics.pond_thermal.") for p in paths) == thermal
                    @test any(startswith(p, "advanced.numerics.shifted_lattice.") for p in paths) == (choice.lattice == "d3q39")
                    @test ("advanced.initial_condition.type" in paths) == (choice.lattice == "d3q39")
                    @test "basic.simulation.ramp_steps" in paths && "advanced.gpu.async_depth" in paths
                end
            end
            # The run settings reach a built case without a new mesh.
            state = Dict{String,Any}("case" => deepcopy(case))
            A._apply_run_output_updates!(state, merge(plain(built["run"]), Dict{String,Any}("steps" => 240, "display_every" => 60)))
            @test state["case"]["steps"] == 240 && state["case"]["effective_display_every"] == 60
            @test occursin("    steps: 240\n", A._yaml_case(state["case"], case["bounds"]; build_only=false))
        end
    end
    # Chosen octree values, a symmetry plane and proportions travel from the controls to the case file.
    built = page_request("q19_octree"; overrides=Dict{String,Any}("octree-wall-function" => true, "octree-hrr-sigma" => 1, "octree-threads" => 4,
        "octree-seal-cavities" => true, "boundary" => "symmetry", "symmetry-placement" => "center", "domain-sizing" => "proportions", "domain-ratio-x" => 3))
    request = plain(built["request"]); request["model_id"] = MODEL_ID
    @test request["domain"]["symmetry_placement"] == "center" && request["domain"]["proportions"] == [3, 1, 1] && !haskey(request["domain"], "symmetry_offset_y_m")
    case, _ = A._validate_case(request)
    yaml = A._yaml_case(case, case["bounds"]; build_only=true)
    @test endswith(yaml, octree_block_text(threads=4, seal=true, hrr_sigma="1", wall_function=true))
    @test occursin("symmetric_analysis: true", yaml)
    # A solver the page does not offer is refused by the fixture and by the backend.
    @test !success(pipeline(`$(Sys.which("node")) $(joinpath(@__DIR__, "fixtures", "aero_request_dom.cjs")) $ROOT lbmtau`; stderr=devnull))
    request["solver"] = "lbmtau"; request["lattice"] = "d3q27"; delete!(request, "octree")
    @test_throws ArgumentError A._validate_case(request)
end

@testset "The page scripts run on the page markup (no browser, no 3D engine)" begin
    fixture = joinpath(@__DIR__, "fixtures", "aero_page_dom.cjs")
    output = IOBuffer(); errors = IOBuffer()
    process = run(pipeline(ignorestatus(`$(Sys.which("node")) $fixture $ROOT`); stdout=output, stderr=errors))
    process.exitcode == 0 || println(String(take!(errors)))
    @test process.exitcode == 0
    report = process.exitcode == 0 ? JSON3.read(String(take!(output)), Dict{String,Any}) : Dict{String,Any}("checks" => 0)
    @test report["checks"] >= 60
end
println("aero_lab_markup: done")
