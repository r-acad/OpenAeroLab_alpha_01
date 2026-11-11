# Setup and runtime tools

Use the root setup launcher before opening an app. Setup follows the
[OpenJFEM installation workflow](https://github.com/r-acad/JF_2026_05_25): one
root setup, then platform-specific application launchers. It installs the
environments actually used by each isolated OpenAeroLab worker:

| Environment | Purpose | Platforms installed |
| --- | --- | --- |
| `third_party/JXfoil` | HTTP/JSON, workbenches, JXfoil, plotting | Windows, Linux, Mac |
| `src/octree_q19` | CPU D3Q19 octree solver | Windows, Linux, Mac |
| `src` | Main LBM, C-mesh, cavitation workers | Windows, Linux |
| `solvers/lbm_lbmtau/src` | Pinned LBM worker | Windows, Linux |
| `solvers/pbcav_s1030/src/pbcav` | Pinned pressure worker | Windows, Linux |
| `tools/postprocess` | CairoMakie charts, reports and renders | Windows, Linux, Mac |

`Pkg.instantiate` uses each checked-in manifest; setup does not run `Pkg.update`
or replace solver sources. Package precompilation is strict: a compile failure
stops setup in that environment and records the failed stage. Verification
loads the applications and runtime imports, renders a small Makie figure, and
checks a small GPU operation when CUDA is available. No simulation is started.
There is no requirement for a machine-specific sysimage. First installation may
take tens of minutes depending on downloads and CPU speed; detailed progress is
written to the log path printed by setup.

## Options

Pass these to `setup.cmd`, `bash setup.sh` or `bash setup.command`:

- `--help`: describe the available options (on Windows, this does not require Julia).
- `--dry-run`: show the plan without installing or changing files; requires the pinned Julia.
- `--offline`: use cached packages and artifacts only; fail if something is missing.
  Windows can also install a supplied Julia ZIP as described below, without network access.
- `--no-precompile`: skip the explicit precompile stage.
- `--no-verify`: skip import/application checks. Combine with `--no-precompile` for download-only setup.
- Windows only: `--install-julia` allows the verified Julia download without a prompt;
  `--no-install-julia` requires an existing Julia. When neither is passed,
  `setup.cmd` asks if the pinned Julia is missing. Choosing no stops setup until
  Julia 1.13.2 is installed manually. `--offline` and `--dry-run` never download Julia.
  `--install-julia --offline` requires `LUDWIG_JULIA_ARCHIVE`.

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
- `<private>/runtime/install_logs/`: persistent bootstrap and package installation logs,
  including progress from failed attempts. Keep these when reporting a problem.
- `<private>/runtime/installation_attempt.toml`: the latest package setup status,
  failed stage, and exact `log` path. Read this path instead of choosing the newest
  file in `install_logs`, which can be a separate Julia bootstrap log.
- `<private>/runtime/launch_logs/`: persistent Windows startup/server logs, including
  missing dependencies and import failures before the web server opens.
- `<private>/TEMP_DELETEABLE/setup/`: reusable downloads and extraction staging.
  Safe to delete after setup exits.
- `<private>/TEMP_DELETEABLE/runtime/`: temporary files used by Windows app
  launchers. Safe to delete after the corresponding server stops.
- Julia's standard depot (normally `~/.julia`): downloaded packages, artifacts
  and compiled caches, shared by the environments. Use Julia's `Pkg.gc()` for
  unused packages; deleting active packages requires setup again.
- Study/output folders selected in the application: persistent user data.
  These are not installer caches and must be retained to reopen results.

Jobs, uploaded geometry, imported results and exports are retained in
`<private>/results/`; studies and user-selected output folders are persistent.
`TEMP_DELETEABLE` contains only disposable data. Stop the apps and reply bridge
before clearing it. Setup does not delete studies or results, and no generated
files belong in GitHub.

## System requirements and troubleshooting

An internet connection is needed on first installation. Linux bootstrap uses
the system's Bash, curl/wget, tar and SHA-256 tool; macOS includes the required
utilities. Windows bootstrap uses CMD, `curl.exe`, `certutil.exe` and `tar.exe`;
it does not execute unsigned `.ps1` files. If a corporate policy requires signed
PowerShell scripts, use the root `setup.cmd` from the latest ZIP. The Visual C++
installer is checked with an inline Authenticode query. Application browser
assets are bundled: no npm, pip, browser CDN or build server is required.

GPU solves need an installed NVIDIA driver compatible with CUDA 13.0. Windows
also needs the Microsoft Visual C++ x64 runtime; setup downloads and installs it
if missing, verifying its Microsoft signature before executing it. Windows may
request administrator authorization. Drivers remain a system prerequisite;
see [CUDA.jl's installation guide](https://cuda.juliagpu.org/stable/installation/overview/).
Do not disable TLS verification to work around a corporate proxy; configure its
trusted certificates for Julia/your system and rerun setup.

## Windows extraction and interrupted setup

The Windows installer verifies the complete Julia ZIP before extracting it. It
uses native Windows `tar` rather than PowerShell's `Expand-Archive`, checks the
extracted runtime with `julia --version`, and only
then publishes it. It checks Julia again at its final location. A failed
replacement retains the previous runtime; an interrupted download is never used
as an installation. A runtime installation lock prevents concurrent installers.

If an earlier installer stopped with a missing `LICENSE.md` during extraction,
update the **complete repository**, then rerun `setup.cmd`. The old extraction
folders are disposable; studies and results are not. Use a writable local folder
with a reasonably short path, for example `C:\OpenAeroLab`, where company policy
permits it. Long-path extraction does not override antivirus quarantine or
application execution policies. If the verified executable is removed or cannot
run, give IT the installer log and ask for an approved installation location or
an approved Julia installation. Do not disable endpoint protection.

## Company networks and IT-managed installations

The OpenJFEM approach remains available: install Julia **1.13.2** using the
organization's approved installer, then point OpenAeroLab to it. In Command Prompt:

```bat
set "LUDWIG_JULIA=C:\ApprovedApps\Julia-1.13.2\bin\julia.exe"
setup.cmd --no-install-julia
```

The normal launcher reuses that runtime and makes no system PATH changes. Set the
same override when launching. The normal `.cmd` setup already avoids PowerShell
script files. An alternative is to run the package installer directly with the
approved Julia executable from the repository root:

```bat
"C:\ApprovedApps\Julia-1.13.2\bin\julia.exe" --startup-file=no tools\install\setup.jl
```

This direct route installs and verifies the Julia packages; it does not install
Julia itself, the NVIDIA driver, or the Microsoft Visual C++ runtime. IT must
provide those system prerequisites as applicable. The workbench can likewise be
started directly with approved Julia:

```bat
"C:\ApprovedApps\Julia-1.13.2\bin\julia.exe" --startup-file=no --project=third_party/JXfoil --threads=auto apps\ludwig_web\run.jl
```

Windows runtime downloads use the system's trusted certificates. Set
`HTTPS_PROXY` / `HTTP_PROXY` and `NO_PROXY` when a company proxy is required;
browser PAC settings are not automatically imported. `LUDWIG_PROXY` provides an
explicit HTTPS proxy for both the Windows bootstrap and the Julia setup process.
Downloads have bounded retries and preserve completed, verified archives for
another attempt. Package setup retries transient transfer failures, but reports
authentication, certificate and integrity failures for correction.

| Setting | Use |
| --- | --- |
| `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY` | IT-approved proxy and bypass list; inherited by Julia |
| `JULIA_PKG_SERVER` | Organization-approved Julia package server; existing settings are preserved |
| `JULIA_SSL_CA_ROOTS_PATH` | IT-provided PEM certificate bundle when Julia needs an explicit CA bundle |
| `JULIA_DEPOT_PATH` | Optional prepared package/artifact depot; use the same value for setup and launch |
| `LUDWIG_PRIVATE_ROOT` | Writable private data location; use the same value for setup and launch |

For example, in Command Prompt (replace the example values with those supplied by IT):

```bat
set "HTTPS_PROXY=http://proxy.company.example:8080"
set "HTTP_PROXY=%HTTPS_PROXY%"
set "NO_PROXY=localhost,127.0.0.1,::1"
set "JULIA_SSL_CA_ROOTS_PATH=C:\CompanyCertificates\trusted-ca-bundle.pem"
setup.cmd
```

Do not put passwords into commands or share proxy credentials in logs. An
authenticated proxy may require organization-specific configuration or an
IT-provided archive and package depot.
Setup does not switch package servers or disable certificate validation to
circumvent a denial.

Julia documents HTTPS access to `julialang.org`, `julialang.net` and their
subdomains on port 443. The Julia ZIP comes from `julialang-s3.julialang.org`;
packages normally use `pkg.julialang.org` and redirected storage hosts. Packages
or artifacts may also require GitHub or their upstream download hosts. The
optional Microsoft runtime download uses `aka.ms` and Microsoft redirect targets.
Provide the failing hostname from the log to IT rather than assuming one hostname
covers every dependency. See Julia's [official infrastructure domains](https://julialang.org/downloads/#official_domains)
and [certificate configuration](https://docs.julialang.org/en/v1/stdlib/NetworkOptions/).

### Local Julia archive and offline setup on Windows

An IT administrator can obtain the official **Julia 1.13.2 Windows x64 ZIP** listed
in [runtimes.tsv](runtimes.tsv), then transfer it to the target computer. The
installer checks it against the repository's pinned SHA-256; local files are
subject to the same verification as downloads.

```bat
set "LUDWIG_JULIA_ARCHIVE=C:\ApprovedInstallers\julia-1.13.2-win64.zip"
setup.cmd --install-julia
```

That command avoids the Julia download but still downloads missing packages.
For a fully offline installation, also prepare the Julia depot for the **same
operating system, architecture, Julia version and repository manifests**, install
the required system prerequisites, and use:

```bat
set "JULIA_DEPOT_PATH=C:\ApprovedInstallers\openaerolab-depot"
set "LUDWIG_VCREDIST_INSTALLER=C:\ApprovedInstallers\vc_redist.x64.exe"
setup.cmd --install-julia --offline
```

The Visual C++ file is needed only if that runtime is missing; its Microsoft
signature is checked before execution. Offline mode does not silently go online
when a package, artifact or prerequisite is missing. On all platforms, an
already installed Julia and prepared depot can use `--offline` directly.

The standalone external XFOIL executable and optional reference datasets are
not needed for the bundled JXfoil solver or web apps. Set `AIRFOIL_XFOIL_EXE` to
an existing XFOIL executable if you want that additional backend. Optional
reference comparisons remain unavailable when their datasets are not installed.

If a launch fails, use the printed log to identify the missing dependency and
rerun setup. `LUDWIG_WEB_NO_BROWSER=1` disables browser opening;
`LUDWIG_WEB_PORT` selects the preferred loopback port. The workbench tries nearby
ports and reuses a server only when its source revision and repository match.
On Windows set `LUDWIG_WEB_NO_PAUSE=1` for unattended scripts.
