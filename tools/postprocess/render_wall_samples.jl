using CairoMakie,JSON3,SHA
include(joinpath(@__DIR__,"WallSamplePlots.jl"))
length(ARGS)==1||error("Usage: render_wall_samples.jl REQUEST.json")
r=JSON3.read(read(only(ARGS),String),Dict{String,Any})
result=WallSamplePlots.plot_wall_samples(r["geometry"],r["field"],r["output"];title=get(r,"title","Native LBM link-patch Cp"),
    legacy=Bool(get(r,"show_original_context",false)) ? get(r,"legacy_surface",nothing) : nothing)
write(r["output"]*".audit.json",JSON3.write(Dict("representation"=>"native_wall_samples","step"=>result.frame.step,"samples"=>result.frame.count,
    "finite_samples"=>result.image.finite_samples,"geometry_sha256"=>result.frame.geometry_sha256,"field_sha256"=>bytes2hex(open(sha256,r["field"])),
    "colour_limits"=>result.limits,"full_cp_range"=>extrema(filter(isfinite,result.frame.pressure)./result.frame.q),"converged"=>false,
    "display_mesh"=>"native_link_patches","patch_triangles"=>2*result.frame.count,
    "original_facet_underlay"=>Bool(get(r,"show_original_context",false)),"capture_kind"=>"native saved-field software render")))
