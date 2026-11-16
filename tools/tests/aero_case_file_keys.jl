# Helper for the Aero Lab tests: which keys a generated case file (config.yaml) carries, and whether the solver
# that receives it reads them. Included by q19_octree_app.jl and aero_lab_markup.jl; not a test on its own.
const CASE_KEYS_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
# YAML is a dependency of the octree solver package, as in the other Aero Lab tests that parse case files.
joinpath(CASE_KEYS_ROOT, "src", "octree_q19") in LOAD_PATH || push!(LOAD_PATH, joinpath(CASE_KEYS_ROOT, "src", "octree_q19"))
using YAML

"Every leaf of a parsed case file as a dotted path (a list is one leaf)."
function case_file_paths(node, prefix="")
    node isa AbstractDict || return [prefix]
    paths = String[]
    for (key, value) in node
        append!(paths, case_file_paths(value, isempty(prefix) ? String(key) : prefix * "." * String(key)))
    end
    sort!(paths)
end

"Source text in which a solver names the case-file keys it reads."
solver_reader_text(files) = join((read(joinpath(CASE_KEYS_ROOT, file), String) for file in files), "\n")

"The CPU octree D3Q19 solver: its package sources (the loader is src/octree_q19/src/config.jl)."
octree_reader_text() = solver_reader_text([joinpath("src", "octree_q19", "src", file)
    for file in readdir(joinpath(CASE_KEYS_ROOT, "src", "octree_q19", "src")) if endswith(file, ".jl")])

"The GPU lattice Boltzmann and thermal finite-volume solvers: the config loader, the output options, the entry point."
gpu_reader_text() = solver_reader_text(vcat(
    [joinpath("src", "io", file) for file in ("config_loader.jl", "lbm_io_opts.jl", "case_id_contract.jl")],
    [joinpath("src", "main.jl")],
    [joinpath("src", "collision", "pond", file)
        for file in readdir(joinpath(CASE_KEYS_ROOT, "src", "collision", "pond")) if endswith(file, ".jl")]))

"Case-file keys whose name the reader text never quotes: keys the solver cannot be reading."
function unread_case_keys(yaml_text::AbstractString, reader::AbstractString)
    names = unique(String(last(split(path, '.'))) for path in case_file_paths(YAML.load(yaml_text)))
    sort!([name for name in names if !occursin("\"" * name * "\"", reader)])
end

"""
The case-file keys the CPU octree solver reads outside its own block, as `src/octree_q19/src/config.jl` names them.
Placement keys appear only when the model is moved; `preserve_symmetry_y` only with a retained symmetry position.
"""
const OCTREE_CASE_PATHS = Set([
    "basic.stl_file", "basic.stl_scale", "basic.alpha", "basic.beta",
    "basic.reference_chord", "basic.reference_length_for_meshing", "basic.reference_area_of_full_model",
    "basic.fluid.density", "basic.fluid.kinematic_viscosity", "basic.fluid.speed_of_sound", "basic.flow.velocity",
    "basic.simulation.steps", "basic.simulation.output_freq", "basic.simulation.slice_output_freq",
    "basic.simulation.output_startup_slices", "basic.simulation.build_only",
    "advanced.domain.upstream", "advanced.domain.downstream", "advanced.domain.lateral", "advanced.domain.height",
    "advanced.domain.sponge_thickness", "advanced.domain.preserve_symmetry_y",
    "advanced.domain.object_center_x_m", "advanced.domain.object_center_y_m", "advanced.domain.object_center_z_m",
    "advanced.refinement.symmetric_analysis", "advanced.boundary.transverse_freestream",
    "advanced.diagnostics.freq", "advanced.forces.moment_center", "advanced.forces.output_freq",
    "advanced.slice.export_fields", "advanced.slice.planes"])

"The `octree_q19:` block exactly as the app writes it (the solver contract of 14 Nov 2026)."
octree_block_text(; threads=0, levels=3, dx="0.02", band_fine=8, band=6, seal=false, wake_length=1.0, wake_offset=1,
                  hrr_sigma="0.98", u_lattice=0.08, wall_function=false, surface_vtk=true, surface_lbmp=true,
                  plane_samples=4000000, keep=2) = """
octree_q19:
  precision: f32
  threads: $threads
  mesh:
    levels: $levels
    finest_dx_m: $dx
    band_fine: $band_fine
    band: $band
    seal_cavities: $seal
    wake: {length: $wake_length, level_offset: $wake_offset, margin: 0.1}
  numerics: {hrr_sigma: $hrr_sigma, sgs: wale, u_lattice: $u_lattice}
  walls: {wall_function: {enabled: $wall_function, sampling: normal_1, filter_steps: 50, pressure_gradient: false}}
  sponge: {strength: 0.05}
  outputs: {surface_vtk: $surface_vtk, surface_lbmp: $surface_lbmp, max_plane_samples: $plane_samples}
  checkpoint: {every_minutes: 30, keep: $keep}
"""
