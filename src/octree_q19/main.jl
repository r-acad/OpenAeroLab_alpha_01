# Entry point of the CPU octree D3Q19 solver. Run from its own environment:
#   julia --project=src/octree_q19 -t <threads> --check-bounds=no -O3 src/octree_q19/main.jl --case <dir> [--build-only] [--resume]
using OctreeQ19
exit(OctreeQ19.main(ARGS))
