# ==============================================================================
# SOLVER-PRESET BOOTSTRAP (2026-06-10b)
# ==============================================================================
# `advanced.solver_preset: pristine` gates three D3Q19 seam features whose
# flags are COMPILE-TIME consts baked into the GPU kernels at module load:
#   _D3Q19_BSPLINE_MACRO_TIME_CONSISTENT  (physics_interpolation.jl — branch
#       folded into the prolongation device functions)
#   _D3Q19_RESIDUAL_PARENT_FILL           (timestep.jl — Val{} kernel gate)
#   _D3Q19_BSPLINE_TILE                   (timestep.jl — kernel selection)
# They must therefore be resolved BEFORE dependencies.jl is included — this
# file is included (and `preset_bootstrap!` called) at the very top of main.jl.
#
# The pre-scan reads cases_to_run.yaml → each case's config.yaml →
# advanced.solver_preset and maps the preset onto the env gates. EXPLICIT env
# vars always win (only keys the user has NOT set are filled in). Mixed
# pristine/non-pristine presets in one multi-case run cannot share compile-time
# gates and are rejected before env mutation or dependency loading. Quick and
# vanilla share the non-pristine gate class and may be queued together.
# config_loader.jl re-checks per case at load time.
#
# The case queue and config bytes are captured before this file (and YAML.jl)
# is loaded. A narrow text preflight rejects YAML constructs whose duplicate or
# merge semantics could hide preset keys; the actual preset value is then read
# with YAML.load, the same parser used by config_loader.jl at runtime.
#
# Unit test: tools/test_preset_bootstrap.jl (drives this file standalone
# against synthetic case trees and an injected env Dict).
# ==============================================================================

if !isdefined(@__MODULE__, :CaseIDContract)
    include(joinpath(@__DIR__, "io", "case_id_contract.jl"))
end
using YAML

const _PRESET_SEAM_GATE_DEFAULTS = (
    ("LUDWIG_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT", "0"),
    ("LUDWIG_D3Q19_RESIDUAL_PARENT_FILL", "1"),
    ("LUDWIG_D3Q19_BSPLINE_TILE", "1"),
)

"""Scan one checked case config.yaml for `solver_preset:`; absent key →
"pristine" (the production default since 2026-06-11 — must match
config_loader.jl's SOLVER_PRESET default so the compile-time gates agree with
the per-case parse)."""
function _preset_boot_case_preset(config_text::AbstractString,
                                  source::AbstractString="config.yaml")
    section = nothing
    advanced_count = 0
    preset_count = 0
    for (line_no, raw_line) in enumerate(eachline(IOBuffer(config_text)))
        line = first(split(raw_line, '#'; limit=2))
        isempty(strip(line)) && continue
        strip(line) in ("---", "...") &&
            error("$source line $line_no uses unsupported YAML document markers")
        (occursin('&', line) || occursin('*', line) || occursin(r"^\s*<<\s*:", line)) &&
            error("$source line $line_no uses unsupported YAML anchor/alias/merge syntax")
        occursin(r"^\s*[\"']", line) &&
            error("$source line $line_no uses unsupported quoted/escaped mapping keys")
        top = match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if top !== nothing
            section = top.captures[1]
            top.captures[1] == "solver_preset" &&
                error("$source line $line_no places solver_preset outside advanced")
            if section == "advanced"
                advanced_count += 1
                advanced_count == 1 || error("$source contains duplicate top-level advanced mappings")
                isempty(strip(top.captures[2])) ||
                    error("$source line $line_no must use a block-form advanced mapping")
            end
            continue
        end
        key = match(r"^\s+solver_preset\s*:\s*(.*?)\s*$", line)
        if key !== nothing
            section == "advanced" ||
                error("$source line $line_no places solver_preset outside advanced")
            match(r"^  solver_preset", line) !== nothing ||
                error("$source line $line_no must make solver_preset a direct advanced child")
            preset_count += 1
            preset_count == 1 || error("$source contains duplicate advanced.solver_preset")
        elseif occursin("solver_preset", line)
            error("$source line $line_no has malformed solver_preset")
        end
    end

    parsed = try
        YAML.load(config_text)
    catch err
        error("$source YAML parse failed: $(sprint(showerror, err))")
    end
    parsed isa AbstractDict || error("$source must parse to a top-level mapping")
    advanced = get(parsed, "advanced", nothing)
    (advanced === nothing || advanced isa AbstractDict) ||
        error("$source advanced must be a mapping")
    value = advanced === nothing ? "pristine" :
            lowercase(string(get(advanced, "solver_preset", "pristine")))
    value in ("quick", "pristine", "vanilla") ||
        error("$source has unknown advanced.solver_preset $(repr(value))")
    return value
end

"""Read `advanced.boundary.periodic_y_amr` (Bool, default false) from one
checked case config. Called only after `_preset_boot_case_preset` accepted the
same text (its narrow preflight rejects anchors/merges/duplicates), so the plain
YAML.load is the same parse config_loader.jl uses at runtime."""
function _preset_boot_case_periodic_y_amr(config_text::AbstractString,
                                          source::AbstractString="config.yaml")
    parsed = YAML.load(config_text)
    parsed isa AbstractDict || error("$source must parse to a top-level mapping")
    advanced = get(parsed, "advanced", nothing)
    advanced isa AbstractDict || return false
    boundary = get(advanced, "boundary", nothing)
    boundary isa AbstractDict || return false
    value = get(boundary, "periodic_y_amr", false)
    value isa Bool || error("$source advanced.boundary.periodic_y_amr must be true or false (got $(repr(value)))")
    return value
end

"""
    preset_bootstrap!(root_dir; env=ENV) -> Union{String, Nothing}

Resolve the captured run's solver-preset gate class and fill in the pristine
seam-gate env defaults when every selected case is pristine. Returns the first
case's preset (quick and vanilla share a gate class), or `nothing` from the
path overload when `cases_to_run.yaml` is absent. `env` is injectable for tests.
"""
function preset_bootstrap!(snapshot::CaseIDContract.StartupSelection; env=ENV)
    CaseIDContract.verify_startup_selection(snapshot)
    presets = [_preset_boot_case_preset(selection.config_text,
                                        "$(selection.case_id)/config.yaml")
               for selection in snapshot.selections]
    first_preset = presets[1]
    if any(p -> (p == "pristine") != (first_preset == "pristine"), presets)
        error("Mixed pristine and non-pristine solver_preset values cannot share " *
              "compile-time seam gates: $(join(presets, ", "))")
    end
    # PERIODIC-Y AMR (27 Oct 2026): the kernels' seam wrap is a compile-time
    # gate (src/collision/periodic_y_amr.jl). Resolve it from the queue here,
    # before dependencies.jl compiles the kernels. All queued cases must agree.
    # Cases without the key leave the env untouched (gate off ⇒ legacy kernels);
    # config_loader.jl refuses any case that disagrees with the compiled gate.
    pyamr = [_preset_boot_case_periodic_y_amr(selection.config_text,
                                              "$(selection.case_id)/config.yaml")
             for selection in snapshot.selections]
    if any(pyamr) && !all(pyamr)
        error("Mixed advanced.boundary.periodic_y_amr values cannot share the compile-time " *
              "periodic-Y AMR kernel gate: $(join(string.(pyamr), ", ")). Queue them in separate runs.")
    end
    if first_preset == "pristine"
        # PRISTINE mode (HANDOVER_2026-06-10): monolith-quality seam at
        # split-kernel speed (~1.55× flow with tile + ranged residual).
        for (key, val) in _PRESET_SEAM_GATE_DEFAULTS
            get!(env, key, val)
        end
        println("[PresetBoot] solver_preset=pristine → D3Q19 seam gates ON " *
                "(bspline-always macro, residual parent-fill, bspline tile; " *
                "explicit env vars override).")
    end
    if !isempty(pyamr) && all(pyamr)
        get!(env, "LUDWIG_PERIODIC_Y_AMR", "1")
        println("[PresetBoot] advanced.boundary.periodic_y_amr=true → periodic-Y AMR kernel gate " *
                "LUDWIG_PERIODIC_Y_AMR=$(env["LUDWIG_PERIODIC_Y_AMR"]).")
    end
    return first_preset
end

function preset_bootstrap!(root_dir::AbstractString; env=ENV)
    snapshot = CaseIDContract.capture_startup_selection(root_dir; missing_ok=true)
    snapshot === nothing && return nothing
    return preset_bootstrap!(snapshot; env=env)
end
