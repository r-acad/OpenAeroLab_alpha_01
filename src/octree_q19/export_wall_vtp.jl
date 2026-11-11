# julia --project=src/octree_q19 src/octree_q19/export_wall_vtp.jl --frame FILE
using OctreeQ19
function main(args)
    if args in (["--help"],["-h"])
        println("Export a saved fine native wall-pressure frame as ParaView VTP, without advancing or remeshing.")
        println("usage: export_wall_vtp.jl --frame /path/to/surface_NNNNNN.lbmw")
        return
    end
    length(args)==2 && args[1]=="--frame" || error("Expected --frame FILE; see --help")
    path=OctreeQ19.write_wall_surface_vtp(abspath(args[2]))
    println("ParaView wall data: $path")
end
main(ARGS)
