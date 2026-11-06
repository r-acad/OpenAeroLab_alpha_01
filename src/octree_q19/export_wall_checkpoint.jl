# julia --project=src/octree_q19 src/octree_q19/export_wall_checkpoint.jl --case CASE --checkpoint FILE [--mesh FILE] [--output RESULTS]
using OctreeQ19
function main(args)
    options=Dict{String,String}();i=1
    while i<=length(args)
        arg=args[i]
        if arg in ("--help","-h")
            println("Read-only wall-pressure export; never runs a solver or builds a mesh.")
            println("usage: export_wall_checkpoint.jl --case CASE --checkpoint FILE [--mesh mesh.oct] [--output RESULTS]")
            return
        end
        arg in ("--case","--checkpoint","--mesh","--output") || error("Unknown option $arg")
        i<length(args) || error("$arg requires a path")
        haskey(options,arg) && error("Repeated option $arg")
        options[arg]=args[i+1];i+=2
    end
    all(k->haskey(options,k),("--case","--checkpoint")) || error("--case and --checkpoint are required; see --help")
    case=abspath(options["--case"])
    path=OctreeQ19.export_checkpoint_wall_surface(case,abspath(options["--checkpoint"]);
        mesh_path=abspath(get(options,"--mesh",joinpath(case,"mesh.oct"))),
        out=abspath(get(options,"--output",joinpath(case,"RESULTS"))))
    println("[WallSurface] Complete: $path")
end
main(ARGS)
