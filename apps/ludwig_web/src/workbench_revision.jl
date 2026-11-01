using SHA

const WORKBENCH_API_REVISION = "2026-11-01-airfoil-studies-v9"

function _workbench_repository_id(root::AbstractString)
    path = joinpath(splitpath(normpath(isdir(root) ? realpath(root) : abspath(root)))...)
    Sys.iswindows() && (path=lowercase(path))
    bytes2hex(sha256(path))
end
