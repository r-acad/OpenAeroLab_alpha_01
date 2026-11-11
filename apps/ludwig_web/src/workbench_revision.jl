using SHA

const WORKBENCH_API_REVISION = "2026-11-11-q19-refinement-v23"

function _workbench_repository_id(root::AbstractString)
    path = joinpath(splitpath(normpath(isdir(root) ? realpath(root) : abspath(root)))...)
    Sys.iswindows() && (path=lowercase(path))
    bytes2hex(sha256(path))
end
