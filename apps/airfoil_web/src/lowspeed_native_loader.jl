module AirfoilLowSpeedNative

using TOML

const PACKAGE_NAME="PBCavNativeCache"
const PACKAGE_UUID="6f0a2905-8233-4ec0-9ceb-66eb6ae42f8a"

"Use a disposable private depot for newly compiled code, retaining installed dependency depots."
function _prepare_host_depot!()
    private=abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",
        joinpath(@__DIR__,"..","..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
    depot=joinpath(private,"TEMP_DELETEABLE","airfoil_native_host_cache")
    mkpath(depot)
    filter!(p->normpath(abspath(p))!=depot,DEPOT_PATH)
    pushfirst!(DEPOT_PATH,depot)
    ENV["JULIA_DEPOT_PATH"]=join(DEPOT_PATH,Sys.iswindows() ? ';' : ':')
    depot
end

"Load the checked named host package, or a legacy snapshot without that package."
function load_native(snapshot::AbstractString)
    native_dir=normpath(joinpath(snapshot,"src","pbcav"))
    project_path=joinpath(native_dir,"Project.toml")
    isfile(project_path)||throw(ArgumentError("The requested native solver environment is missing"))
    project=TOML.parsefile(project_path)
    entry=joinpath(native_dir,"src",PACKAGE_NAME*".jl")
    if get(project,"name","")==PACKAGE_NAME
        get(project,"uuid","")==PACKAGE_UUID||throw(ArgumentError("The native host package has an unexpected identity"))
        isfile(entry)||throw(ArgumentError("The named native host package entry point is missing"))
        depot=_prepare_host_depot!()
        package=Base.require(Base.PkgId(Base.UUID(PACKAGE_UUID),PACKAGE_NAME))
        return Base.invokelatest() do
            provenance=package.validate_provenance!(;native_dir)
            (native=package.PBCav,mode="named_package",provenance=provenance,cache_depot=depot)
        end
    end
    isfile(entry)&&throw(ArgumentError("The native host package environment is inconsistent"))
    # The older pinned snapshots predate the wrapper. Preserve their own module
    # and arithmetic; errors in a named package are never hidden by this path.
    native=Base.include(@__MODULE__,joinpath(native_dir,"PBCav.jl"))
    source=Base.invokelatest() do
        isdefined(native,:freeair_source_signature) ? native.freeair_source_signature() : ""
    end
    (native=native,mode="legacy_include",provenance=(schema="legacy",native_sha256=source,
        bundle_sha256="",workload="none",julia_version=string(VERSION)),cache_depot="")
end

end
