"""Make the selected GR runtime's plugin dependencies visible to Windows' DLL loader.

GR registers its selected artifact directories with AddDllDirectory, but versions
of the native GKS plugin loader use LoadLibrary without the matching search flag.
Prepending those same directories to this process's PATH lets PNG/SVG plugins find
their dependencies. This changes no machine or user environment settings.
"""
function configure_gr_windows_path!(plots::Module)
    Sys.iswindows() || return nothing
    isdefined(plots,:GR) || error("Cannot configure GR rendering: Plots.GR is unavailable.")
    gr=getfield(plots,:GR)
    isdefined(gr,:GRPreferences) || error("Cannot configure GR rendering: GRPreferences is unavailable in the selected GR version.")
    preferences=getfield(gr,:GRPreferences)
    isdefined(preferences,:libpath) || error("Cannot configure GR rendering: the selected GR runtime does not expose its library search path.")
    paths=filter(!isempty,split(String(getfield(preferences,:libpath)[]),';'))
    isempty(paths) && error("Cannot configure GR rendering: the selected GR runtime has no library search path.")
    existing=split(get(ENV,"PATH",""),';';keepempty=false)
    key(s)=lowercase(normpath(s))
    additions=String[]
    selected=Set{String}()
    for path in paths
        key(path) in selected && continue
        push!(additions,path);push!(selected,key(path))
    end
    remaining=filter(path->!(key(path) in selected),existing)
    ENV["PATH"]=join(vcat(additions,remaining),';')
    nothing
end
