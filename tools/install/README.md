# Setup and runtime tools

Use the root setup launcher before opening an app. Setup follows the
[OpenJFEM installation workflow](https://github.com/r-acad/JF_2026_05_25): one
root setup, then platform-specific application launchers. It installs the
environments actually used by each isolated OpenAeroLab worker:

| Environment | Purpose | Platforms installed |
| --- | --- | --- |
| `third_party/JXfoil` | HTTP/JSON, workbenches, JXfoil, plotting | Windows, Linux, Mac |
| `src` | Main LBM, C-mesh, cavitation workers | Windows, Linux |
| `solvers/lbm_lbmtau/src` | Pinned LBM worker | Windows, Linux |
| `solvers/pbcav_s1030/src/pbcav` | Pinned pressure worker | Windows, Linux |
| `tools/postprocess` | CairoMakie charts, reports and renders | Windows, Linux, Mac |

`Pkg.instantiate` uses each checked-in manifest; setup does not run `Pkg.update`
or replace solver sources. Package precompilation is automatic. Verification
loads the applications and runtime imports, renders a small Makie figure, and
checks a small GPU operation when CUDA is available. No simulation is started.
There is no requirement for a machine-specific sysimage. First installation may
take tens of minutes depending on downloads and CPU speed; detailed progress is
written to the log path printed by setup.

## Options

Pass these to `setup.cmd`, `bash setup.sh` or `bash setup.command`:

- `--help`: describe the available options (requires Julia to read the detailed help).
- `--dry-run`: show the plan without installing or changing files; requires the pinned Julia.
- `--offline`: use the existing Julia, packages and artifacts only; fail if something is missing.
- `--no-precompile`: skip the explicit precompile stage.
- `--no-verify`: skip import/application checks. Combine with `--no-precompile` for download-only setup.

Unknown options fail with a nonzero exit status. Failed runs keep their logs
and downloaded caches; rerun to resume. A setup lock prevents two installers
from modifying environments concurrently.

Set `LUDWIG_JULIA` to a Julia **executable path** to override discovery.
`AIRFOIL_JULIA` remains a compatible alias. The version must be 1.13.2 because
the pinned pressure environment requires that exact version. Setup does not
replace another Julia on PATH. Download URLs and SHA-256 values are recorded in
`runtimes.tsv`, from the official
[Julia release index](https://julialang-s3.julialang.org/bin/versions.json).
Unverified archives are never executed. Runtime downloads use HTTPS.

## Files and disk space

`LUDWIG_PRIVATE_ROOT` selects the private data folder. Its default is the sibling
`02_PRIVATE_PROJECT_DEVELOPMENT` directory; this also preserves the established
development workspace arrangement. Set it before setup and use the same value
when launching. Paths containing spaces are supported.

- `<private>/runtime/`: optional downloaded Julia and the installation receipt.
  Deleting a downloaded runtime requires running setup again before launch.
- `<private>/TEMP_DELETEABLE/setup/`: download staging and installation logs.
  Safe to delete after setup exits.
- `<private>/TEMP_DELETEABLE/launchers/`: per-launch logs. Safe to delete after
  the corresponding server stops.
- Julia's standard depot (normally `~/.julia`): downloaded packages, artifacts
  and compiled caches, shared by the environments. Use Julia's `Pkg.gc()` for
  unused packages; deleting active packages requires setup again.
- Study/output folders selected in the application: persistent user data.
  These are not installer caches and must be retained to reopen results.

Other app caches under `TEMP_DELETEABLE` can contain the only copy of an unsaved
job. Save the study and persistent outputs before cleaning those folders. Setup
does not delete studies or results, and no generated files belong in GitHub.

## System requirements and troubleshooting

An internet connection is needed on first installation. Linux bootstrap uses
the system's Bash, curl/wget, tar and SHA-256 tool; macOS includes the required
utilities. Windows bootstrap uses PowerShell 5.1 or newer. Application browser
assets are bundled: no npm, pip, browser CDN or build server is required.

GPU solves need an installed NVIDIA driver compatible with CUDA 13.0. Windows
also needs the Microsoft Visual C++ x64 runtime; setup downloads and installs it
if missing, verifying its Microsoft signature before executing it. Windows may
request administrator authorization. Drivers remain a system prerequisite;
see [CUDA.jl's installation guide](https://cuda.juliagpu.org/stable/installation/overview/).
Do not disable TLS verification to work around a corporate proxy; configure its
trusted certificates for Julia/your system and rerun setup.

The standalone external XFOIL executable and optional reference datasets are
not needed for the bundled JXfoil solver or web apps. Set `AIRFOIL_XFOIL_EXE` to
an existing XFOIL executable if you want that additional backend. Optional
reference comparisons remain unavailable when their datasets are not installed.

If a launch fails, use the printed log to identify the missing dependency and
rerun setup. `LUDWIG_WEB_NO_BROWSER=1` disables browser opening;
`LUDWIG_WEB_PORT` selects the preferred loopback port. The workbench tries nearby
ports and reuses a server only when its source revision and repository match.
On Windows set `LUDWIG_WEB_NO_PAUSE=1` for unattended scripts.
