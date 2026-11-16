"CPU octree D3Q19 backend (src/octree_q19): its own cell-level octree mesher and solver, no GPU, any platform."
_q19_case(case::AbstractDict) = get(case,"solver","")=="q19_octree"
_q19_available() = isfile(joinpath(Q19_SOLVER,"main.jl")) && isfile(joinpath(Q19_SOLVER,"Project.toml"))

_memory_lattice(case::AbstractDict) = _pond_case(case) ? "pond_thermal" : _q19_case(case) ? "q19_octree" : get(case,"lattice","d3q27")
const Q19_BYTES_PER_SLOT = 195.0   # two population buffers, two velocity buffers, kind, cut index, oct table, at FP32
const Q19_MAX_LEVELS = 10
const Q19_MAX_MACH = 0.3

# The cell octree has 2×2×2 sibling groups, not the GPU mesher's 8-cell blocks.
_q19_coarse_axis(length,h)=max(2,2ceil(Int,(Float64(length)/h-1e-9)/2))
_mesh_surface_area(case)=get(case,_q19_case(case) ? "solved_surface_area_m2" : "surface_area_m2",get(case,"surface_area_m2",0.))

"""
Validated octree settings of a request. The solver has one collision (hybrid regularised, `hrr_sigma`; 1 means no
hybrid part), absorbing layers on every far-field face and a uniform start, so none of these is a choice here.
Keys this function does not name are ignored.
"""
function _q19_settings(raw::AbstractDict)
    input=get(raw,"octree",Dict{String,Any}())
    input isa AbstractDict || throw(ArgumentError("octree must be an object"))
    seal=get(input,"seal_cavities",false)
    seal isa Bool || throw(ArgumentError("octree.seal_cavities must be true or false"))
    # Equilibrium wall function of src/octree_q19: experimental and not validated, off unless switched on.
    wall_function=get(input,"wall_function",false)
    wall_function isa Bool || throw(ArgumentError("octree.wall_function must be true or false"))
    settings=Dict{String,Any}(
        "seal_cavities"=>seal,
        "wall_function"=>wall_function,
        "hrr_sigma"=>_number(get(input,"hrr_sigma",.98),"octree.hrr_sigma";lo=.5,hi=1),
        "u_lattice"=>_number(get(input,"u_lattice",.08),"octree.u_lattice";lo=1e-5,hi=.12),
        "checkpoint_keep"=>_integer(get(input,"checkpoint_keep",2),"octree.checkpoint_keep";lo=1,hi=10),
        "max_plane_samples"=>_integer(get(input,"max_plane_samples",4_000_000),"octree.max_plane_samples";lo=10_000,hi=16_000_000),
        "band_fine"=>_integer(get(input,"band_fine",8),"octree.band_fine";lo=3,hi=64),
        "band"=>_integer(get(input,"band",6),"octree.band";lo=3,hi=64),
        "wake_length"=>_number(get(input,"wake_length",1.0),"octree.wake_length";lo=0,hi=20),
        "wake_level_offset"=>_integer(get(input,"wake_level_offset",1),"octree.wake_level_offset";lo=0,hi=4),
        "threads"=>_integer(get(input,"threads",0),"octree.threads";lo=0,hi=256))
    # A refinement stage keeps its source stage's meshing reference (and with it the sponge length).
    reference=get(input,"stage_reference_length_m",nothing)
    reference===nothing || (settings["stage_reference_length_m"]=_number(reference,"octree.stage_reference_length_m";lo=1e-8,hi=1e6))
    settings
end

_q19_threads(case::AbstractDict) = (t=Int(get(get(case,"octree",Dict{String,Any}()),"threads",0)); t>0 ? t : max(1,Sys.CPU_THREADS÷2))

"The `octree_q19:` block of the case file: exactly the keys src/octree_q19/src/config.jl reads from it."
function _q19_yaml_lines(case::AbstractDict)
    o=merge(_q19_settings(Dict{String,Any}()),get(case,"octree",Dict{String,Any}()))
    outputs=get(case,"outputs",OUTPUT_DEFAULTS)
    ["octree_q19:","  precision: f32","  threads: $(o["threads"])",
     "  mesh:","    levels: $(case["levels"])","    finest_dx_m: $(@sprintf("%.12g",case["surface_dx"]))",
     "    band_fine: $(o["band_fine"])","    band: $(o["band"])",
     "    seal_cavities: $(o["seal_cavities"])",
     "    wake: {length: $(o["wake_length"]), level_offset: $(o["wake_level_offset"]), margin: 0.1}",
     "  numerics: {hrr_sigma: $(@sprintf("%.6g",o["hrr_sigma"])), sgs: wale, u_lattice: $(o["u_lattice"])}",
     "  walls: {wall_function: {enabled: $(o["wall_function"]), sampling: normal_1, filter_steps: 50, pressure_gradient: false}}",
     "  sponge: {strength: 0.05}",
     "  outputs: {surface_vtk: $(outputs["surface_vtk"]), surface_lbmp: $(outputs["surface_lbmp"]), max_plane_samples: $(o["max_plane_samples"])}",
     "  checkpoint: {every_minutes: 30, keep: $(o["checkpoint_keep"])}"]
end

"""
Envelope preview of the octree cell count: the coarse background plus surface bands on each side of the body
surface (band_fine cells at the finest level, band cells on each coarser level, one side solid-trimmed), replaced
by the generator's counts once the mesh is built.
"""
function _q19_cells_estimate(case::AbstractDict,coarse::Real,surface_area::Real,dx::Real,levels::Integer)
    o=get(case,"octree",nothing); o isa AbstractDict || (o=_q19_settings(Dict{String,Any}()))
    cells=Float64(coarse)
    area=Float64(surface_area)
    for l in 2:levels
        h=dx*2.0^(levels-l)
        band=(l==levels ? o["band_fine"]+2 : o["band"])*h       # outside band plus about two cells inside
        cells+=area*band/h^3
    end
    ceil(Int,cells)
end
