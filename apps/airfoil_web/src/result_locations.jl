module AirfoilResultLocations

using Dates, TOML

const JOB_ID = r"^[0-9a-f-]{36}$"
const FOLDER_NAME = r"^\d{8}_\d{6}_\d{3}_UTC_airfoil_([0-9a-f-]{36})$"
const POINTER_FILE = "result_location.toml"

function checked_id(id::AbstractString)
    occursin(JOB_ID,id) || throw(ArgumentError("Invalid result job id"))
    String(id)
end

"Resolve only a server-created pointer; study documents never supply file access."
function registered_directory(root::AbstractString,id::AbstractString)
    id=checked_id(id);pointer=joinpath(root,id,POINTER_FILE)
    isfile(pointer) || return nothing
    base=realpath(root);canonical(p)=Sys.iswindows() ? lowercase(normpath(p)) : normpath(p)
    startswith(canonical(realpath(pointer)),canonical(base)*string(Sys.iswindows() ? '\\' : '/')) ||
        throw(ArgumentError("Saved result pointer escapes private storage"))
    data=try TOML.parsefile(pointer) catch; throw(ArgumentError("Saved result location is damaged")) end
    get(data,"job_id",nothing)==id || throw(ArgumentError("Saved result location has a different job id"))
    path=get(data,"directory",nothing)
    path isa AbstractString && isabspath(path) && !occursin('\0',path) || throw(ArgumentError("Invalid saved result location"))
    match=Base.match(FOLDER_NAME,basename(normpath(path)))
    match!==nothing && match.captures[1]==id || throw(ArgumentError("Invalid saved result folder name"))
    isdir(path) && canonical(realpath(path))!=canonical(path) &&
        throw(ArgumentError("Saved result folder was replaced by a link"))
    # Keep the registered absolute location if a disk is temporarily unavailable:
    # never silently open/create another folder for the same saved job.
    normpath(String(path))
end

"Create a unique native run folder and its durable private lookup pointer."
function create_directory(root::AbstractString,id::AbstractString;
                          study_path::AbstractString="",parent::AbstractString="",clock::DateTime=now(UTC))
    id=checked_id(id)
    isempty(study_path) || isabspath(study_path) && isfile(study_path) || throw(ArgumentError("Saved study file is unavailable"))
    destination_parent=isempty(parent) ? (isempty(study_path) ? root : dirname(study_path)) : parent
    isabspath(destination_parent) && !occursin('\0',destination_parent) || throw(ArgumentError("Choose an absolute results folder"))
    isfile(destination_parent) && throw(ArgumentError("The results folder is a file"))
    mkpath(destination_parent)
    name=Dates.format(clock,"yyyymmdd_HHMMSS_sss")*"_UTC_airfoil_"*id
    target=joinpath(realpath(destination_parent),name)
    pointerdir=joinpath(root,id)
    (ispath(target)||ispath(pointerdir)) && throw(ArgumentError("Refusing to replace an existing result folder"))
    mkdir(target)
    try
        mkpath(pointerdir)
        open(joinpath(pointerdir,POINTER_FILE),"w") do io
            TOML.print(io,Dict("job_id"=>id,"directory"=>target,"created_at_utc"=>string(clock)*"Z");sorted=true)
        end
    catch
        # Do not remove a folder that may already contain useful output.
        rethrow()
    end
    target
end

end
