module CaseIDContract

using SHA

export FileSnapshot, CaseSelection, StartupSelection, OutputTarget, ClaimedOutput,
       STLInputSnapshot, RuntimeSelection, SummarySinkTarget, ClaimedSummarySink,
       SummaryFileClaim, validate_case_id, parse_case_ids,
       load_case_ids, checked_case_paths, capture_case_selection, capture_startup_selection,
       capture_file_snapshot, verify_file_location, verify_file_snapshot,
       verify_case_location, verify_case_selection, verify_startup_selection,
       capture_runtime_selection, verify_runtime_config, verify_runtime_inputs, verify_runtime_selection,
       claim_output_directory, verify_claimed_output,
       capture_summary_sink, verify_summary_sink_target, claim_summary_sink,
       verify_claimed_summary_sink, verify_summary_file_claim,
       write_summary_file_exclusive,
       validate_stl_declarations, validate_stl_declaration,
       LaunchSelector, resolve_launch_selector, program_is

const CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
const TEMPLATE_CASE_ID = "TEMPLATE_starter"
const WINDOWS_RESERVED_CASE_IDS = Set(vcat(
    ["CON", "PRN", "AUX", "NUL"],
    ["COM$i" for i in 1:9],
    ["LPT$i" for i in 1:9],
))
const FileIdentity = NTuple{3,UInt64}

_path_key(path::AbstractString) = Sys.iswindows() ? lowercase(normpath(path)) : normpath(path)

function _safe_leaf(raw; source::AbstractString)
    raw isa AbstractString ||
        throw(ArgumentError("$source must be a string; got $(typeof(raw))"))
    leaf = String(raw)
    isempty(leaf) && throw(ArgumentError("$source must not be empty"))
    leaf in (".", "..") && throw(ArgumentError("$source must be one direct child"))
    any(iscntrl, leaf) && throw(ArgumentError("$source must not contain control characters"))
    (occursin('/', leaf) || occursin('\\', leaf) || occursin(':', leaf)) &&
        throw(ArgumentError("$source must not contain a path separator or drive prefix"))
    (endswith(leaf, '.') || endswith(leaf, ' ')) &&
        throw(ArgumentError("$source must not end with a dot or space"))
    occursin(CASE_ID_PATTERN, leaf) ||
        throw(ArgumentError("$source must match $(CASE_ID_PATTERN.pattern); got $(repr(leaf))"))
    reserved_stem = uppercase(first(split(leaf, '.'; limit=2)))
    reserved_stem in WINDOWS_RESERVED_CASE_IDS &&
        throw(ArgumentError("$source uses a Windows reserved device name: $(repr(leaf))"))
    return leaf
end

struct FileSnapshot
    path::String
    realpath::String
    text::String
    sha256::String
    identity::FileIdentity
end

struct CaseSelection
    case_id::String
    case_dir::String
    config_path::String
    config_text::String
    config_sha256::String
    case_identity::FileIdentity
    config_identity::FileIdentity
    root_realpath::String
    cases_realpath::String
    root_identity::FileIdentity
    cases_identity::FileIdentity
end

struct StartupSelection
    root_dir::String
    cases_file::String
    cases_text::String
    cases_sha256::String
    selections::Tuple
    root_realpath::String
    cases_realpath::String
    cases_file_realpath::String
    root_identity::FileIdentity
    cases_identity::FileIdentity
    cases_file_identity::FileIdentity
end

"""A validated, still-unclaimed direct output child of one selected case."""
struct OutputTarget
    leaf::String
    path::String
    case_realpath::String
    case_identity::FileIdentity
end

"""The generation created by `claim_output_directory`."""
struct ClaimedOutput
    target::OutputTarget
    realpath::String
    identity::FileIdentity
end

"""Immutable bytes and namespace identity for one configured STL input."""
struct STLInputSnapshot
    declared::String
    path::String
    realpath::String
    bytes::Vector{UInt8}
    sha256::String
    size::Int64
    identity::FileIdentity
    parent_realpath::String
    parent_identity::FileIdentity
end

"""All config-derived filesystem authority for one selected case."""
struct RuntimeSelection
    case::CaseSelection
    output::OutputTarget
    stls::Tuple
end

"""Internal typed state for the fixed direct-root `CASE_LOGS` sink."""
struct SummarySinkTarget
    root_dir::String
    root_realpath::String
    root_identity::FileIdentity
    path::String
    existed::Bool
    sink_realpath::Union{Nothing,String}
    sink_identity::Union{Nothing,FileIdentity}
end

"""Internal typed state for a verified `CASE_LOGS` generation."""
struct ClaimedSummarySink
    target::SummarySinkTarget
    realpath::String
    identity::FileIdentity
    created::Bool
end

"""Internal typed state for one final-name file created by an exclusive open."""
struct SummaryFileClaim
    sink::ClaimedSummarySink
    filename::String
    path::String
    realpath::String
    identity::FileIdentity
end

"""
    validate_case_id(raw; source="case selection") -> String

Return an unchanged, canonical case ID. A case ID is exactly one direct child
name below `CASES`: ASCII alphanumerics plus `_`, `.`, and `-`, beginning with
an alphanumeric. Paths, traversal tokens, control/whitespace characters, and
the starter template are rejected rather than normalized.
"""
function validate_case_id(raw; source::AbstractString="case selection")
    raw isa AbstractString ||
        throw(ArgumentError("$source must be a string case ID; got $(typeof(raw))"))

    case_id = String(raw)
    isempty(case_id) &&
        throw(ArgumentError("$source must not be empty"))
    case_id in (".", "..") &&
        throw(ArgumentError("$source must name one direct CASES child, not $(repr(case_id))"))
    any(iscntrl, case_id) &&
        throw(ArgumentError("$source must not contain control characters: $(repr(case_id))"))
    (occursin('/', case_id) || occursin('\\', case_id)) &&
        throw(ArgumentError("$source must not contain path separators: $(repr(case_id))"))
    endswith(case_id, '.') &&
        throw(ArgumentError("$source must not end with a dot: $(repr(case_id))"))
    occursin(CASE_ID_PATTERN, case_id) ||
        throw(ArgumentError("$source must match $(CASE_ID_PATTERN.pattern); got $(repr(case_id))"))
    lowercase(case_id) == lowercase(TEMPLATE_CASE_ID) &&
        throw(ArgumentError("$source must not select $TEMPLATE_CASE_ID"))
    reserved_stem = uppercase(first(split(case_id, '.'; limit=2)))
    reserved_stem in WINDOWS_RESERVED_CASE_IDS &&
        throw(ArgumentError("$source uses a Windows reserved device name: $(repr(case_id))"))

    return case_id
end

function _case_scalar(raw::AbstractString, source::AbstractString)
    token = strip(raw)
    if length(token) >= 2 && first(token) in ('\'', '"')
        last(token) == first(token) || throw(ArgumentError("$source has mismatched quotes"))
        token = token[nextind(token, firstindex(token)):prevind(token, lastindex(token))]
    elseif occursin('\'', token) || occursin('"', token)
        throw(ArgumentError("$source has malformed quoting"))
    end
    return validate_case_id(token; source=source)
end

"""Parse the strict one-key selection contract from immutable text."""
function parse_case_ids(text::AbstractString; source::AbstractString="cases_to_run.yaml")
    meaningful = Tuple{Int,String}[]
    for (line_no, line) in enumerate(eachline(IOBuffer(text)))
        text = strip(first(split(line, '#'; limit=2)))
        isempty(text) || push!(meaningful, (line_no, text))
    end
    isempty(meaningful) && error("$source must contain a nonempty case_folders selection")

    header_no, header = first(meaningful)
    prefix = "case_folders:"
    startswith(header, prefix) ||
        error("$source line $header_no must be the exact case_folders mapping")
    rest = strip(header[nextind(header, firstindex(header), length(prefix)):end])
    cases = String[]

    if isempty(rest)
        length(meaningful) > 1 || error("$source case_folders must not be empty")
        for (line_no, line) in meaningful[2:end]
            m = match(r"^-\s+(.+)$", line)
            m === nothing && error("$source line $line_no is not a case_folders list item")
            push!(cases, _case_scalar(m.captures[1], "$source case_folders[$(length(cases) + 1)]"))
        end
    else
        (startswith(rest, '[') && endswith(rest, ']')) ||
            error("$source line $header_no has malformed inline case_folders")
        body = strip(rest[nextind(rest, firstindex(rest)):prevind(rest, lastindex(rest))])
        isempty(body) && error("$source case_folders must not be empty")
        length(meaningful) == 1 ||
            error("$source contains content outside the case_folders mapping")
        for raw in split(body, ',')
            push!(cases, _case_scalar(raw, "$source case_folders[$(length(cases) + 1)]"))
        end
    end

    length(unique(lowercase.(cases))) == length(cases) ||
        error("$source case_folders must not contain case-insensitive duplicates")
    return cases
end

"""Read and parse the strict one-key `cases_to_run.yaml` selection contract."""
function load_case_ids(cases_file::AbstractString; missing_ok::Bool=false)
    if !isfile(cases_file)
        missing_ok && return nothing
        error("cases_to_run.yaml not found: $(abspath(cases_file))")
    end
    islink(cases_file) && error("cases_to_run.yaml must not be a symlink: $(abspath(cases_file))")
    stat(cases_file).nlink == 1 ||
        error("cases_to_run.yaml must not be hard-linked: $(abspath(cases_file))")
    bytes = read(cases_file)
    return parse_case_ids(String(bytes); source=basename(cases_file))
end

"""
    checked_case_paths(root_dir, raw; source="case selection")

Validate `raw`, then require its direct `CASES` child and `config.yaml` to be
ordinary, present filesystem entries. Returns `(case_id, case_dir, config_path)`.
The raw selection is validated before it participates in any path operation.
"""
function checked_case_paths(root_dir::AbstractString, raw;
                            source::AbstractString="case selection")
    case_id = validate_case_id(raw; source=source)
    cases_dir = abspath(joinpath(root_dir, "CASES"))
    isdir(cases_dir) || error("CASES directory not found: $cases_dir")
    islink(cases_dir) && error("CASES directory must not be a symlink: $cases_dir")
    root_real = realpath(abspath(root_dir))
    cases_real = realpath(cases_dir)
    path_key(path) = Sys.iswindows() ? lowercase(normpath(path)) : normpath(path)
    path_key(cases_real) == path_key(joinpath(root_real, "CASES")) ||
        error("CASES directory must not be a linked directory: $cases_dir")
    case_id in readdir(cases_dir) ||
        error("Selected case ID does not exactly match a direct CASES child: $(repr(case_id))")
    case_dir = joinpath(cases_dir, case_id)

    isdir(case_dir) || error("Selected case folder not found: $case_dir")
    islink(case_dir) && error("Selected case folder must not be a symlink: $case_dir")
    expected_case_dir = joinpath(cases_real, case_id)
    actual_case_dir = realpath(case_dir)
    path_key(actual_case_dir) == path_key(expected_case_dir) ||
        error("Selected case folder must not be a linked directory: $case_dir")

    config_path = joinpath(case_dir, "config.yaml")
    isfile(config_path) || error("Selected case config.yaml not found: $config_path")
    islink(config_path) && error("Selected case config.yaml must not be a symlink: $config_path")
    stat(config_path).nlink == 1 || error("Selected case config.yaml must not be hard-linked: $config_path")
    path_key(realpath(config_path)) == path_key(joinpath(actual_case_dir, "config.yaml")) ||
        error("Selected case config.yaml must not be linked: $config_path")

    return (case_id=case_id, case_dir=case_dir, config_path=config_path)
end

sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(text)))
function _windows_file_identity(handle)::FileIdentity
    bytes = Vector{UInt8}(undef, 24)
    ok = ccall((:GetFileInformationByHandleEx, "kernel32"), stdcall, Int32,
               (Ptr{Cvoid}, Int32, Ptr{UInt8}, UInt32),
               handle, 18, bytes, UInt32(length(bytes)))
    ok != 0 || error("Unable to read FILE_ID_INFO")
    return Tuple(reinterpret(UInt64, bytes))
end

function file_identity(path::AbstractString)::FileIdentity
    if !Sys.iswindows()
        info = stat(path)
        return (UInt64(info.device), UInt64(info.inode), 0x0000000000000000)
    end

    # Julia's Windows stat().inode can be immediately reused after deletion.
    # FILE_ID_INFO supplies the volume serial plus NTFS FILE_ID_128 generation,
    # which distinguishes an object recreated at the same path.
    handle = ccall((:CreateFileW, "kernel32"), stdcall, Ptr{Cvoid},
                   (Cwstring, UInt32, UInt32, Ptr{Cvoid}, UInt32, UInt32, Ptr{Cvoid}),
                   path, 0x00000080, 0x00000007, C_NULL, 0x00000003,
                   0x02000000, C_NULL)
    handle == Ptr{Cvoid}(-1) && error("Unable to open filesystem identity handle: $path")
    try
        return _windows_file_identity(handle)
    finally
        ccall((:CloseHandle, "kernel32"), stdcall, Int32, (Ptr{Cvoid},), handle)
    end
end

function file_identity(io::Base.Filesystem.File)::FileIdentity
    if !Sys.iswindows()
        info = stat(io)
        return (UInt64(info.device), UInt64(info.inode), 0x0000000000000000)
    end
    return _windows_file_identity(Base.Libc._get_osfhandle(fd(io)))
end

function _read_stable(path::AbstractString, label::AbstractString)
    first_text = read(path, String)
    second_text = read(path, String)
    first_text == second_text || error("$label changed while being captured: $path")
    return first_text
end

function capture_file_snapshot(path::AbstractString;
                               label::AbstractString="selected file")
    absolute = abspath(path)
    isfile(absolute) || error("$label is missing or not an ordinary file: $absolute")
    islink(absolute) && error("$label must not be a symlink: $absolute")
    stat(absolute).nlink == 1 || error("$label must not be hard-linked: $absolute")
    text = _read_stable(absolute, label)
    snapshot = FileSnapshot(absolute, realpath(absolute), text, sha256_hex(text),
                            file_identity(absolute))
    verify_file_snapshot(snapshot; label=label)
    return snapshot
end

function verify_file_snapshot(snapshot::FileSnapshot;
                              label::AbstractString="selected file")
    verify_file_location(snapshot; label=label)
    read(snapshot.path, String) == snapshot.text || error("$label bytes changed")
    sha256_hex(snapshot.text) == snapshot.sha256 || error("$label captured digest mismatch")
    return snapshot
end

function verify_file_location(snapshot::FileSnapshot;
                              label::AbstractString="selected file")
    isfile(snapshot.path) || error("$label is missing or not an ordinary file: $(snapshot.path)")
    islink(snapshot.path) && error("$label must not be a symlink: $(snapshot.path)")
    stat(snapshot.path).nlink == 1 || error("$label must not be hard-linked: $(snapshot.path)")
    realpath(snapshot.path) == snapshot.realpath || error("$label physical path changed")
    file_identity(snapshot.path) == snapshot.identity || error("$label filesystem identity changed")
    return snapshot
end

"""Capture the queue and every selected config as one immutable startup value."""
function capture_case_selection(root_dir::AbstractString, raw;
                                source::AbstractString="case selection")
    root = abspath(root_dir)
    paths = checked_case_paths(root_dir, raw; source=source)
    cases_dir = joinpath(root, "CASES")
    config_text = _read_stable(paths.config_path, "case config.yaml")
    selection = CaseSelection(paths.case_id, paths.case_dir, paths.config_path,
                              config_text, sha256_hex(config_text),
                              file_identity(paths.case_dir), file_identity(paths.config_path),
                              realpath(root), realpath(cases_dir),
                              file_identity(root), file_identity(cases_dir))
    verify_case_selection(root, selection)
    return selection
end

# -- lbm-io-opts (27 Oct 2026): launch-time case selector ----------------------
# LUDWIG_CASES_FILE=<file> or --cases-file <file>: a queue file in the cases_to_run.yaml
#   format anywhere; its directory is the root (cases resolve under <dir>/CASES, the batch
#   summary goes to <dir>/CASE_LOGS). <solver root>/cases_to_run.yaml is not read.
# LUDWIG_CASE=<id|dir> or --case <id|dir>: one case, no queue file. A value with a path
#   separator is a case directory, which must be a direct child of a directory named CASES
#   (root = its grandparent); otherwise it is a case ID under <solver root>/CASES.
# With none of them set (an empty variable counts as unset) the queue is
# <solver root>/cases_to_run.yaml, as before. At most one selector may be given.
const LAUNCH_ENV_CASES_FILE = "LUDWIG_CASES_FILE"
const LAUNCH_ENV_CASE = "LUDWIG_CASE"
const _NO_FILE_IDENTITY = (0x0000000000000000, 0x0000000000000000, 0x0000000000000000)

"""Where the case queue comes from: `kind` is :default, :cases_file or :case."""
struct LaunchSelector
    kind::Symbol
    value::String
    source::String
end

function capture_startup_selection(root_dir::AbstractString; missing_ok::Bool=false,
                                   selector::Union{Nothing,LaunchSelector}=nothing)
    # lbm-io-opts: a launch selector other than :default replaces the queue file.
    if selector !== nothing && selector.kind !== :default
        return _capture_selected_startup(root_dir, selector)
    end
    root = abspath(root_dir)
    return _capture_startup_from_file(root, joinpath(root, "cases_to_run.yaml"), "cases_to_run.yaml";
                                      missing_ok=missing_ok)
end

# The body of the original capture_startup_selection, with the queue file and its label as
# parameters (for the default queue: <root>/cases_to_run.yaml, label "cases_to_run.yaml").
function _capture_startup_from_file(root::AbstractString, cases_file::AbstractString,
                                    label::AbstractString; missing_ok::Bool=false)
    if !isfile(cases_file)
        missing_ok && return nothing
        error("$label not found: $cases_file")
    end
    islink(cases_file) && error("$label must not be a symlink: $cases_file")
    stat(cases_file).nlink == 1 || error("$label must not be hard-linked: $cases_file")
    root_real = realpath(root)
    cases_dir = joinpath(root, "CASES")
    isdir(cases_dir) || error("CASES directory not found: $cases_dir")
    islink(cases_dir) && error("CASES directory must not be a symlink: $cases_dir")
    cases_real = realpath(cases_dir)
    cases_file_real = realpath(cases_file)
    cases_text = _read_stable(cases_file, label)
    case_ids = parse_case_ids(cases_text; source=label)
    selections = CaseSelection[]
    for (i, case_id) in enumerate(case_ids)
        push!(selections, capture_case_selection(
            root, case_id; source="$label case_folders[$i]"))
    end
    # Close the capture window over the selector after all config captures.
    read(cases_file, String) == cases_text || error("$label changed during startup capture")
    return StartupSelection(root, cases_file, cases_text, sha256_hex(cases_text), Tuple(selections),
                            root_real, cases_real, cases_file_real,
                            file_identity(root), file_identity(cases_dir),
                            file_identity(cases_file))
end

"""True when `program_file` (PROGRAM_FILE) names the file `this_file` (@__FILE__)."""
function program_is(program_file::AbstractString, this_file::AbstractString)
    isempty(program_file) && return false
    a = abspath(program_file); b = abspath(this_file)
    if isfile(a) && isfile(b)
        return _path_key(realpath(a)) == _path_key(realpath(b))
    end
    return _path_key(normpath(a)) == _path_key(normpath(b))
end

"""
    resolve_launch_selector(env, args) -> LaunchSelector

Read LUDWIG_CASES_FILE / LUDWIG_CASE from `env` and `--cases-file <file>` / `--case <x>`
(or `--cases-file=<file>`, `--case=<x>`) from `args`. Refuses unknown arguments, a
missing value and more than one selector.
"""
function resolve_launch_selector(env, args)
    found = Tuple{Symbol,String,String}[]
    for (name, kind) in ((LAUNCH_ENV_CASES_FILE, :cases_file), (LAUNCH_ENV_CASE, :case))
        value = String(get(env, name, ""))
        isempty(value) || push!(found, (kind, value, name))
    end
    i = 1
    while i <= length(args)
        arg = String(args[i])
        m = match(r"^--(cases-file|case)(?:=(.*))?$", arg)
        m === nothing && throw(ArgumentError(
            "main.jl: unknown argument $(repr(arg)); accepted: --cases-file <file>, --case <case id or directory>"))
        kind = m.captures[1] == "cases-file" ? :cases_file : :case
        if m.captures[2] !== nothing
            value = String(m.captures[2])
        else
            i < length(args) || throw(ArgumentError("main.jl: $arg needs a value"))
            i += 1
            value = String(args[i])
        end
        isempty(value) && throw(ArgumentError("main.jl: $arg needs a non-empty value"))
        push!(found, (kind, value, "--" * m.captures[1]))
        i += 1
    end
    length(found) > 1 && throw(ArgumentError(
        "main.jl: more than one case selector given ($(join([f[3] for f in found], ", "))); " *
        "give at most one of LUDWIG_CASES_FILE, LUDWIG_CASE, --cases-file, --case"))
    isempty(found) && return LaunchSelector(:default, "", "default")
    return LaunchSelector(found[1]...)
end

function _capture_selected_startup(default_root::AbstractString, selector::LaunchSelector)
    if selector.kind === :cases_file
        cases_file = abspath(selector.value)
        isfile(cases_file) || error("$(selector.source): cases file not found: $cases_file")
        return _capture_startup_from_file(dirname(cases_file), cases_file,
                                          "$(selector.source) $(basename(cases_file))")
    end
    selector.kind === :case || error("unknown launch selector kind $(repr(selector.kind))")
    value = selector.value
    if occursin('/', value) || occursin('\\', value)
        case_dir = rstrip(normpath(abspath(value)), ['/', '\\'])
        isdir(case_dir) || error("$(selector.source): case directory not found: $case_dir")
        parent = dirname(case_dir)
        (Sys.iswindows() ? uppercase(basename(parent)) : basename(parent)) == "CASES" ||
            error("$(selector.source): a case directory must be a direct child of a directory named CASES: $case_dir")
        root = dirname(parent)
        case_id = basename(case_dir)
    else
        root = abspath(default_root)
        case_id = value
    end
    validate_case_id(case_id; source=selector.source)
    root_real = realpath(root)
    cases_dir = joinpath(root, "CASES")
    isdir(cases_dir) || error("CASES directory not found: $cases_dir")
    islink(cases_dir) && error("CASES directory must not be a symlink: $cases_dir")
    selection = capture_case_selection(root, case_id; source="$(selector.source) case")
    text = "case_folders:\n  - \"$case_id\"\n"
    return StartupSelection(root, "", text, sha256_hex(text), (selection,),
                            root_real, realpath(cases_dir), "",
                            file_identity(root), file_identity(cases_dir), _NO_FILE_IDENTITY)
end

"""The checks of verify_startup_selection for a single-case selection without a queue file."""
function _verify_fileless_startup_selection(snapshot::StartupSelection)
    isdir(snapshot.root_dir) || error("Captured root directory is missing: $(snapshot.root_dir)")
    islink(snapshot.root_dir) && error("Captured root directory must not be a symlink: $(snapshot.root_dir)")
    cases_dir = joinpath(snapshot.root_dir, "CASES")
    isdir(cases_dir) || error("Captured CASES directory is missing: $cases_dir")
    islink(cases_dir) && error("Captured CASES directory must not be a symlink: $cases_dir")
    realpath(snapshot.root_dir) == snapshot.root_realpath || error("Captured root physical path changed")
    realpath(cases_dir) == snapshot.cases_realpath || error("Captured CASES physical path changed")
    file_identity(snapshot.root_dir) == snapshot.root_identity || error("Captured root filesystem identity changed")
    file_identity(cases_dir) == snapshot.cases_identity || error("Captured CASES filesystem identity changed")
    isempty(snapshot.cases_file_realpath) && snapshot.cases_file_identity == _NO_FILE_IDENTITY ||
        error("A single-case selection must not carry a queue file")
    length(snapshot.selections) == 1 || error("A single-case selection must hold exactly one case")
    parse_case_ids(snapshot.cases_text; source="captured single-case selection") ==
        [selection.case_id for selection in snapshot.selections] ||
        error("captured single-case selection does not match its case")
    verify_case_selection(snapshot.root_dir, snapshot.selections[1]; position=1)
    sha256_hex(snapshot.cases_text) == snapshot.cases_sha256 || error("Captured selection digest mismatch")
    return snapshot
end

"""Fail if queue identity, case paths, or any config bytes changed since capture."""
function verify_case_location(root_dir::AbstractString, selection::CaseSelection;
                              position::Integer=1)
    root = abspath(root_dir)
    cases_dir = joinpath(root, "CASES")
    isdir(root) || error("Captured root directory is missing: $root")
    islink(root) && error("Captured root directory must not be a symlink: $root")
    isdir(cases_dir) || error("Captured CASES directory is missing: $cases_dir")
    islink(cases_dir) && error("Captured CASES directory must not be a symlink: $cases_dir")
    realpath(root) == selection.root_realpath || error("Captured case root physical path changed")
    realpath(cases_dir) == selection.cases_realpath || error("Captured CASES physical path changed")
    file_identity(root) == selection.root_identity || error("Captured case root identity changed")
    file_identity(cases_dir) == selection.cases_identity || error("Captured CASES identity changed")
    paths = checked_case_paths(root_dir, selection.case_id;
                               source="captured case_folders[$position]")
    paths.case_dir == selection.case_dir ||
        error("Selected case directory changed after startup capture: $(selection.case_id)")
    paths.config_path == selection.config_path ||
        error("Selected config path changed after startup capture: $(selection.case_id)")
    file_identity(selection.case_dir) == selection.case_identity ||
        error("Selected case directory identity changed after startup capture: $(selection.case_id)")
    file_identity(selection.config_path) == selection.config_identity ||
        error("Selected config identity changed after startup capture: $(selection.case_id)")
    return selection
end

function verify_case_selection(root_dir::AbstractString, selection::CaseSelection;
                               position::Integer=1)
    verify_case_location(root_dir, selection; position=position)
    read(selection.config_path, String) == selection.config_text ||
        error("Selected config bytes changed after startup capture: $(selection.case_id)")
    sha256_hex(selection.config_text) == selection.config_sha256 ||
        error("Captured config digest mismatch: $(selection.case_id)")
    return selection
end

function verify_startup_selection(snapshot::StartupSelection)
    isempty(snapshot.cases_file) && return _verify_fileless_startup_selection(snapshot)   # lbm-io-opts
    isdir(snapshot.root_dir) || error("Captured root directory is missing: $(snapshot.root_dir)")
    islink(snapshot.root_dir) && error("Captured root directory must not be a symlink: $(snapshot.root_dir)")
    cases_dir = joinpath(snapshot.root_dir, "CASES")
    isdir(cases_dir) || error("Captured CASES directory is missing: $cases_dir")
    islink(cases_dir) && error("Captured CASES directory must not be a symlink: $cases_dir")
    isfile(snapshot.cases_file) || error("Captured cases_to_run.yaml is missing: $(snapshot.cases_file)")
    islink(snapshot.cases_file) && error("Captured cases_to_run.yaml must not be a symlink: $(snapshot.cases_file)")
    stat(snapshot.cases_file).nlink == 1 ||
        error("Captured cases_to_run.yaml must not be hard-linked: $(snapshot.cases_file)")
    realpath(snapshot.root_dir) == snapshot.root_realpath ||
        error("Captured root physical path changed")
    realpath(cases_dir) == snapshot.cases_realpath ||
        error("Captured CASES physical path changed")
    realpath(snapshot.cases_file) == snapshot.cases_file_realpath ||
        error("Captured cases_to_run.yaml physical path changed")
    file_identity(snapshot.root_dir) == snapshot.root_identity ||
        error("Captured root filesystem identity changed")
    file_identity(cases_dir) == snapshot.cases_identity ||
        error("Captured CASES filesystem identity changed")
    file_identity(snapshot.cases_file) == snapshot.cases_file_identity ||
        error("Captured cases_to_run.yaml filesystem identity changed")
    read(snapshot.cases_file, String) == snapshot.cases_text ||
        error("cases_to_run.yaml changed after startup capture")
    parse_case_ids(snapshot.cases_text; source="captured cases_to_run.yaml") ==
        [selection.case_id for selection in snapshot.selections] ||
        error("captured case queue does not match captured selections")
    for (i, selection) in enumerate(snapshot.selections)
        verify_case_selection(snapshot.root_dir, selection; position=i)
    end
    sha256_hex(snapshot.cases_text) == snapshot.cases_sha256 ||
        error("Captured cases_to_run digest mismatch")
    return snapshot
end

function _case_realpath(selection::CaseSelection)
    case_real = realpath(selection.case_dir)
    expected = joinpath(selection.cases_realpath, selection.case_id)
    _path_key(case_real) == _path_key(expected) ||
        error("Selected case physical path changed: $(selection.case_dir)")
    return case_real
end

function _output_target(selection::CaseSelection, raw)
    leaf = _safe_leaf(raw; source="basic.simulation.output_dir")
    case_real = _case_realpath(selection)
    path = joinpath(selection.case_dir, leaf)
    _path_key(normpath(path)) == _path_key(joinpath(case_real, leaf)) ||
        error("Configured output directory is not a direct selected-case child")
    any(entry -> lowercase(entry) == lowercase(leaf), readdir(selection.case_dir)) &&
        error("Configured output directory already exists; archive it before running: $path")
    return OutputTarget(leaf, path, case_real, selection.case_identity)
end

function validate_stl_declarations(raw)
    declarations = if raw isa AbstractString
        String[String(raw)]
    elseif raw isa AbstractVector
        isempty(raw) && error("basic.stl_file must not be an empty list")
        all(value -> value isa AbstractString, raw) ||
            error("every basic.stl_file list member must be a string")
        String[String(value) for value in raw]
    else
        error("basic.stl_file must be a string or nonempty list of strings; got $(typeof(raw))")
    end
    isempty(declarations) && error("basic.stl_file must not be empty")
    return declarations
end

"""Validate one STL declaration without touching the filesystem."""
function validate_stl_declaration(raw; source::AbstractString="basic.stl_file")
    raw isa AbstractString || error("$source member must be a string; got $(typeof(raw))")
    declared = String(raw)
    occursin('\\', declared) && error("STL reference must use a local leaf or ../leaf: $(repr(declared))")
    local_leaf = !occursin('/', declared)
    shared_leaf = startswith(declared, "../") && count(==('/'), declared) == 1
    (local_leaf || shared_leaf) ||
        error("STL reference must be a local leaf or one direct CASES child via ../leaf: $(repr(declared))")
    leaf_raw = shared_leaf ? declared[nextind(declared, firstindex(declared), 3):end] : declared
    leaf = _safe_leaf(leaf_raw; source="basic.stl_file")
    lowercase(splitext(leaf)[2]) == ".stl" ||
        error("basic.stl_file must name an .stl file: $(repr(declared))")
    return (declared=declared, scope=shared_leaf ? :cases : :case, leaf=leaf)
end

function _stl_path(selection::CaseSelection, declared::String)
    validated = validate_stl_declaration(declared)
    shared_leaf = validated.scope == :cases
    leaf = validated.leaf
    parent = shared_leaf ? dirname(selection.case_dir) : selection.case_dir
    path = joinpath(parent, leaf)
    expected_parent = shared_leaf ? selection.cases_realpath : _case_realpath(selection)
    _path_key(realpath(parent)) == _path_key(expected_parent) ||
        error("STL parent physical path changed: $parent")
    return path, parent, leaf
end

function _capture_stl(selection::CaseSelection, declared::String)
    path, parent, leaf = _stl_path(selection, declared)
    isdir(parent) || error("STL parent directory is missing: $parent")
    islink(parent) && error("STL parent directory must not be linked: $parent")
    leaf in readdir(parent) || error("STL path spelling does not exactly match a directory entry: $path")
    count(entry -> lowercase(entry) == lowercase(leaf), readdir(parent)) == 1 ||
        error("Configured STL has a case-fold alias in its parent directory: $path")
    isfile(path) || error("Configured STL is missing or not an ordinary file: $path")
    islink(path) && error("Configured STL must not be a symlink: $path")
    stat(path).nlink == 1 || error("Configured STL must not be hard-linked: $path")
    parent_real = realpath(parent)
    expected_parent = parent == selection.case_dir ? _case_realpath(selection) : selection.cases_realpath
    _path_key(parent_real) == _path_key(expected_parent) ||
        error("Configured STL parent escapes the captured CASES tree: $path")
    expected_real = joinpath(parent_real, leaf)
    _path_key(realpath(path)) == _path_key(expected_real) ||
        error("Configured STL resolves through a linked path: $path")

    parent_identity = file_identity(parent)
    identity_before = file_identity(path)
    first_bytes = read(path)
    identity_middle = file_identity(path)
    second_bytes = read(path)
    identity_after = file_identity(path)
    identity_before == identity_middle == identity_after ||
        error("Configured STL identity changed while being captured: $path")
    first_bytes == second_bytes || error("Configured STL bytes changed while being captured: $path")
    file_identity(parent) == parent_identity || error("Configured STL parent changed while being captured: $parent")
    return STLInputSnapshot(declared, path, realpath(path), copy(first_bytes),
                            bytes2hex(sha256(first_bytes)), Int64(length(first_bytes)),
                            identity_after, parent_real, parent_identity)
end

function _verify_stl(snapshot::STLInputSnapshot)
    parent = dirname(snapshot.path)
    isdir(parent) || error("Configured STL parent directory is missing: $parent")
    islink(parent) && error("Configured STL parent directory must not be linked: $parent")
    _path_key(realpath(parent)) == _path_key(snapshot.parent_realpath) ||
        error("Configured STL parent physical path changed")
    file_identity(parent) == snapshot.parent_identity || error("Configured STL parent identity changed")
    basename(snapshot.path) in readdir(parent) || error("Configured STL exact path spelling changed")
    count(entry -> lowercase(entry) == lowercase(basename(snapshot.path)), readdir(parent)) == 1 ||
        error("Configured STL acquired a case-fold alias")
    isfile(snapshot.path) || error("Configured STL is missing or not an ordinary file: $(snapshot.path)")
    islink(snapshot.path) && error("Configured STL must not be a symlink: $(snapshot.path)")
    stat(snapshot.path).nlink == 1 || error("Configured STL must not be hard-linked: $(snapshot.path)")
    _path_key(realpath(snapshot.path)) == _path_key(snapshot.realpath) ||
        error("Configured STL physical path changed")
    file_identity(snapshot.path) == snapshot.identity || error("Configured STL identity changed")
    Int64(filesize(snapshot.path)) == snapshot.size || error("Configured STL size changed")
    bytes = read(snapshot.path)
    bytes == snapshot.bytes || error("Configured STL bytes changed")
    bytes2hex(sha256(bytes)) == snapshot.sha256 || error("Configured STL digest changed")
    return snapshot
end

function _mapping_value(mapping::AbstractDict, key::String, source::String)
    haskey(mapping, key) || error("$source is missing required key $key")
    return mapping[key]
end

function _runtime_config_values(config::AbstractDict)
    basic = _mapping_value(config, "basic", "captured config.yaml")
    basic isa AbstractDict || error("captured config.yaml basic must be a mapping")
    simulation = _mapping_value(basic, "simulation", "captured config.yaml basic")
    simulation isa AbstractDict || error("captured config.yaml basic.simulation must be a mapping")
    output_leaf = _safe_leaf(
        _mapping_value(simulation, "output_dir", "captured config.yaml basic.simulation");
        source="basic.simulation.output_dir")
    declarations = validate_stl_declarations(
        _mapping_value(basic, "stl_file", "captured config.yaml basic"))
    return output_leaf, declarations
end

"""Bind a runtime selection to the values parsed from its captured config bytes."""
function verify_runtime_config(runtime::RuntimeSelection, config::AbstractDict)
    output_leaf, declarations = _runtime_config_values(config)
    output_leaf == runtime.output.leaf ||
        error("runtime output target differs from captured config.yaml")
    declarations == [stl.declared for stl in runtime.stls] ||
        error("runtime STL selection differs from captured config.yaml")
    return runtime
end

"""Capture all config-derived filesystem paths before any run-side mutation."""
function capture_runtime_selection(root_dir::AbstractString, selection::CaseSelection,
                                   config::AbstractDict)
    verify_case_selection(root_dir, selection)
    output_leaf, declarations = _runtime_config_values(config)
    output = _output_target(selection, output_leaf)
    stls = Tuple(_capture_stl(selection, declared) for declared in declarations)
    keys = [_path_key(stl.realpath) for stl in stls]
    length(unique(keys)) == length(keys) ||
        error("basic.stl_file contains duplicate or case-fold-alias targets")
    runtime = RuntimeSelection(selection, output, stls)
    verify_runtime_config(runtime, config)
    verify_runtime_selection(root_dir, runtime)
    return runtime
end

function verify_runtime_inputs(root_dir::AbstractString, runtime::RuntimeSelection)
    verify_case_selection(root_dir, runtime.case)
    isempty(runtime.stls) && error("runtime STL selection must not be empty")
    keys = String[]
    for stl in runtime.stls
        expected_path, expected_parent, _ = _stl_path(runtime.case, stl.declared)
        _path_key(expected_path) == _path_key(stl.path) ||
            error("runtime STL path differs from its declaration")
        _path_key(realpath(expected_parent)) == _path_key(stl.parent_realpath) ||
            error("runtime STL parent differs from its declaration")
        push!(keys, _path_key(stl.realpath))
    end
    length(unique(keys)) == length(keys) ||
        error("runtime STL selection contains duplicate or case-fold-alias targets")
    foreach(_verify_stl, runtime.stls)
    return runtime
end

function _verify_output_available(runtime::RuntimeSelection)
    target = runtime.output
    _safe_leaf(target.leaf; source="runtime output leaf") == target.leaf ||
        error("runtime output leaf is not canonical")
    case_real = _case_realpath(runtime.case)
    _path_key(case_real) == _path_key(target.case_realpath) || error("output target case binding mismatch")
    runtime.case.case_identity == target.case_identity || error("output target case identity mismatch")
    _path_key(realpath(runtime.case.case_dir)) == _path_key(target.case_realpath) ||
        error("output target parent physical path changed")
    file_identity(runtime.case.case_dir) == target.case_identity || error("output target parent identity changed")
    _path_key(target.path) == _path_key(joinpath(target.case_realpath, target.leaf)) ||
        error("output target is not the captured direct selected-case child")
    _path_key(dirname(normpath(target.path))) == _path_key(target.case_realpath) ||
        error("output target parent differs from the selected case")
    any(entry -> lowercase(entry) == lowercase(target.leaf), readdir(runtime.case.case_dir)) &&
        error("Configured output directory already exists; archive it before running: $(target.path)")
    return target
end

function verify_runtime_selection(root_dir::AbstractString, runtime::RuntimeSelection)
    verify_runtime_inputs(root_dir, runtime)
    _verify_output_available(runtime)
    return runtime
end

"""Atomically claim the previously absent output leaf with `mkdir`.

`before_mkdir` is a deterministic CPU-test seam. Production callers leave it
at its inert default; the availability check is repeated by the atomic mkdir.
"""
function claim_output_directory(root_dir::AbstractString, runtime::RuntimeSelection;
                                before_mkdir::Function=() -> nothing)
    verify_runtime_selection(root_dir, runtime)
    before_mkdir()
    mkdir(runtime.output.path)
    claim = nothing
    try
        isdir(runtime.output.path) || error("claimed output is not a directory: $(runtime.output.path)")
        islink(runtime.output.path) && error("claimed output must not be linked: $(runtime.output.path)")
        claim = ClaimedOutput(runtime.output, realpath(runtime.output.path),
                              file_identity(runtime.output.path))
        verify_claimed_output(runtime, claim)
        return claim
    catch
        # Clean up only the exact unchanged, still-empty directory generation
        # created above. Never recurse and never touch a replacement/nonempty path.
        if claim !== nothing
            try
                verify_claimed_output(runtime, claim)
                isempty(readdir(claim.target.path)) && rm(claim.target.path)
            catch
            end
        end
        rethrow()
    end
end

function verify_claimed_output(runtime::RuntimeSelection, claim::ClaimedOutput)
    _safe_leaf(runtime.output.leaf; source="runtime output leaf") == runtime.output.leaf ||
        error("runtime output leaf is not canonical")
    claim.target.leaf == runtime.output.leaf &&
        _path_key(claim.target.path) == _path_key(runtime.output.path) &&
        _path_key(claim.target.case_realpath) == _path_key(runtime.output.case_realpath) &&
        claim.target.case_identity == runtime.output.case_identity ||
        error("claimed output target differs from runtime selection")
    case_real = _case_realpath(runtime.case)
    _path_key(realpath(runtime.case.case_dir)) == _path_key(case_real) || error("claimed output parent moved")
    file_identity(runtime.case.case_dir) == runtime.case.case_identity || error("claimed output parent identity changed")
    isdir(claim.target.path) || error("claimed output directory is missing")
    islink(claim.target.path) && error("claimed output directory must not be linked")
    _path_key(realpath(claim.target.path)) == _path_key(claim.realpath) ||
        error("claimed output physical path changed")
    file_identity(claim.target.path) == claim.identity || error("claimed output identity changed")
    count(entry -> lowercase(entry) == lowercase(runtime.output.leaf),
          readdir(runtime.case.case_dir)) == 1 || error("claimed output acquired a case-fold alias")
    _path_key(claim.realpath) == _path_key(joinpath(case_real, runtime.output.leaf)) ||
        error("claimed output escaped the selected case")
    return claim
end

const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = UInt32(0x00000400)
const WINDOWS_INVALID_FILE_ATTRIBUTES = typemax(UInt32)
const SUMMARY_SINK_LEAF = "CASE_LOGS"
const SUMMARY_FILENAME_PATTERN = r"^cases_summary_[0-9]{8}_[0-9]{6}\.md$"

function _is_reparse_point(path::AbstractString)
    Sys.iswindows() || return false
    attributes = ccall((:GetFileAttributesW, "kernel32"), stdcall, UInt32,
                       (Cwstring,), path)
    attributes == WINDOWS_INVALID_FILE_ATTRIBUTES &&
        error("Unable to read Windows file attributes: $path")
    return (attributes & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT) != 0
end

_casefold_entries(parent::AbstractString, leaf::AbstractString) =
    filter(entry -> lowercase(entry) == lowercase(leaf), readdir(parent))

function _verify_summary_root(target::SummarySinkTarget)
    isdir(target.root_dir) || error("Captured project root is missing: $(target.root_dir)")
    islink(target.root_dir) && error("Captured project root must not be linked")
    _is_reparse_point(target.root_dir) && error("Captured project root must not be a reparse point")
    _path_key(realpath(target.root_dir)) == _path_key(target.root_realpath) ||
        error("Captured project root physical path changed")
    file_identity(target.root_dir) == target.root_identity ||
        error("Captured project root identity changed")
    _path_key(target.path) == _path_key(joinpath(target.root_realpath, SUMMARY_SINK_LEAF)) ||
        error("CASE_LOGS target is not the captured root's direct child")
    return target
end

function _verify_existing_summary_sink(target::SummarySinkTarget,
                                       real::String, identity::FileIdentity)
    _verify_summary_root(target)
    matches = _casefold_entries(target.root_dir, SUMMARY_SINK_LEAF)
    matches == [SUMMARY_SINK_LEAF] ||
        error("CASE_LOGS must have one exact direct-root spelling")
    isdir(target.path) || error("CASE_LOGS is missing or not a directory: $(target.path)")
    islink(target.path) && error("CASE_LOGS must not be linked: $(target.path)")
    _is_reparse_point(target.path) && error("CASE_LOGS must not be a reparse point: $(target.path)")
    _path_key(realpath(target.path)) == _path_key(real) ||
        error("CASE_LOGS physical path changed")
    _path_key(real) == _path_key(joinpath(target.root_realpath, SUMMARY_SINK_LEAF)) ||
        error("CASE_LOGS escapes the captured project root")
    file_identity(target.path) == identity || error("CASE_LOGS identity changed")
    return target
end

"""Capture the exact fixed project-level summary sink before run-side mutation."""
function capture_summary_sink(snapshot::StartupSelection)
    verify_startup_selection(snapshot)
    root = snapshot.root_dir
    path = joinpath(root, SUMMARY_SINK_LEAF)
    matches = _casefold_entries(root, SUMMARY_SINK_LEAF)
    length(matches) <= 1 || error("CASE_LOGS has case-fold aliases")
    if isempty(matches)
        target = SummarySinkTarget(root, snapshot.root_realpath, snapshot.root_identity,
                                   path, false, nothing, nothing)
    else
        matches == [SUMMARY_SINK_LEAF] || error("CASE_LOGS spelling must be exact")
        isdir(path) || error("CASE_LOGS is not a directory: $path")
        islink(path) && error("CASE_LOGS must not be linked: $path")
        _is_reparse_point(path) && error("CASE_LOGS must not be a reparse point: $path")
        sink_real = realpath(path)
        sink_identity = file_identity(path)
        target = SummarySinkTarget(root, snapshot.root_realpath, snapshot.root_identity,
                                   path, true, sink_real, sink_identity)
    end
    verify_startup_selection(snapshot)
    verify_summary_sink_target(snapshot, target)
    return target
end


function verify_summary_sink_target(snapshot::StartupSelection, target::SummarySinkTarget)
    verify_startup_selection(snapshot)
    target.root_dir == snapshot.root_dir &&
        _path_key(target.root_realpath) == _path_key(snapshot.root_realpath) &&
        target.root_identity == snapshot.root_identity ||
        error("CASE_LOGS target differs from the captured startup root")
    _verify_summary_root(target)
    if target.existed
        target.sink_realpath isa String && target.sink_identity isa FileIdentity ||
            error("Existing CASE_LOGS target lacks a captured generation")
        _verify_existing_summary_sink(target, target.sink_realpath, target.sink_identity)
    else
        target.sink_realpath === nothing && target.sink_identity === nothing ||
            error("Absent CASE_LOGS target carries a forged generation")
        isempty(_casefold_entries(target.root_dir, SUMMARY_SINK_LEAF)) ||
            error("CASE_LOGS appeared after startup capture")
    end
    return target
end

"""Claim the startup-captured CASE_LOGS generation, creating it only if it was absent."""
function claim_summary_sink(snapshot::StartupSelection, target::SummarySinkTarget;
                            before_mkdir::Function=() -> nothing,
                            after_mkdir::Function=() -> nothing)
    verify_summary_sink_target(snapshot, target)
    if target.existed
        claim = ClaimedSummarySink(target, target.sink_realpath, target.sink_identity, false)
        return verify_claimed_summary_sink(claim)
    end

    before_mkdir()
    verify_summary_sink_target(snapshot, target)
    mkdir(target.path)
    claim = nothing
    try
        identity = file_identity(target.path)
        claim = ClaimedSummarySink(target,
                                   joinpath(target.root_realpath, SUMMARY_SINK_LEAF),
                                   identity, true)
        after_mkdir()
        return verify_claimed_summary_sink(claim)
    catch
        if claim !== nothing
            try
                verify_claimed_summary_sink(claim)
                isempty(readdir(claim.target.path)) && rm(claim.target.path)
            catch
            end
        end
        rethrow()
    end
end


function verify_claimed_summary_sink(claim::ClaimedSummarySink)
    target = claim.target
    _verify_summary_root(target)
    if target.existed
        target.sink_realpath isa String && target.sink_identity isa FileIdentity ||
            error("Existing CASE_LOGS target lacks its startup generation")
        _path_key(claim.realpath) == _path_key(target.sink_realpath) &&
            claim.identity == target.sink_identity ||
            error("CASE_LOGS claim differs from the startup-captured generation")
    end
    _verify_existing_summary_sink(target, claim.realpath, claim.identity)
    claim.created == !target.existed || error("CASE_LOGS claim creation state is forged")
    return claim
end

function _validate_summary_filename(raw)
    raw isa AbstractString || error("Summary filename must be a string")
    filename = _safe_leaf(raw; source="summary filename")
    occursin(SUMMARY_FILENAME_PATTERN, filename) ||
        error("Summary filename must match $(SUMMARY_FILENAME_PATTERN.pattern)")
    return filename
end

function _verify_summary_file_available(sink::ClaimedSummarySink, filename::String)
    verify_claimed_summary_sink(sink)
    isempty(_casefold_entries(sink.target.path, filename)) ||
        error("Summary file already exists; refusing to overwrite: $(joinpath(sink.target.path, filename))")
    return nothing
end

function verify_summary_file_claim(claim::SummaryFileClaim)
    verify_claimed_summary_sink(claim.sink)
    _validate_summary_filename(claim.filename) == claim.filename ||
        error("Summary filename is not canonical")
    expected = joinpath(claim.sink.realpath, claim.filename)
    _path_key(claim.path) == _path_key(joinpath(claim.sink.target.path, claim.filename)) &&
        _path_key(claim.realpath) == _path_key(expected) ||
        error("Summary file target differs from its sink")
    _casefold_entries(claim.sink.target.path, claim.filename) == [claim.filename] ||
        error("Summary file acquired a case-fold alias")
    isfile(claim.path) || error("Summary file is missing or not an ordinary file")
    islink(claim.path) && error("Summary file must not be linked")
    _is_reparse_point(claim.path) && error("Summary file must not be a reparse point")
    stat(claim.path).nlink == 1 || error("Summary file must not be hard-linked")
    _path_key(realpath(claim.path)) == _path_key(claim.realpath) ||
        error("Summary file physical path changed")
    file_identity(claim.path) == claim.identity || error("Summary file identity changed")
    return claim
end

function _cleanup_summary_failure(claim::Union{Nothing,SummaryFileClaim},
                                  sink::ClaimedSummarySink)
    if claim !== nothing
        try
            verify_summary_file_claim(claim)
            rm(claim.path)
        catch
        end
    end
    if sink.created
        try
            verify_claimed_summary_sink(sink)
            isempty(readdir(sink.target.path)) && rm(sink.target.path)
        catch
        end
    end
    return nothing
end

"""Exclusively create and render one final-name summary.

The final name is claimed atomically and is never overwritten. Content can be
visible while `renderer` is still writing; this is not an atomic-publication
contract. Hostile concurrent namespace replacement remains outside this
path-relative boundary.
"""
function write_summary_file_exclusive(renderer::Function, sink::ClaimedSummarySink,
                                      raw_filename; before_open::Function=() -> nothing)
    filename = _validate_summary_filename(raw_filename)
    _verify_summary_file_available(sink, filename)
    before_open()
    _verify_summary_file_available(sink, filename)
    path = joinpath(sink.target.path, filename)
    flags = Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT |
            Base.Filesystem.JL_O_EXCL
    io = nothing
    claim = nothing
    try
        io = Base.Filesystem.open(path, flags, 0o600)
        handle_identity = file_identity(io)
        claim = SummaryFileClaim(sink, filename, path,
                                 joinpath(sink.realpath, filename), handle_identity)
        stat(io).nlink == 1 || error("Exclusive summary stream became hard-linked")
        verify_summary_file_claim(claim)
        renderer(io, claim)
        flush(io)
        file_identity(io) == claim.identity || error("Exclusive summary stream identity changed")
        stat(io).nlink == 1 || error("Exclusive summary stream became hard-linked")
        verify_summary_file_claim(claim)
        close(io)
        io = nothing
        verify_summary_file_claim(claim)
        return claim
    catch
        if io !== nothing
            try
                close(io)
            catch
            end
        end
        _cleanup_summary_failure(claim, sink)
        rethrow()
    end
end

end # module CaseIDContract
