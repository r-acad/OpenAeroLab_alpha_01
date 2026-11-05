# Windows

1. Extract the complete GitHub ZIP to a writable folder. Do not run inside the ZIP.
2. Double-click **`setup.cmd`** at the repository root. If Julia 1.13.2 is missing,
   setup asks whether to download and install it automatically. Choose **Y** to
   install the verified Julia runtime privately, or **N** to install Julia 1.13.2
   yourself and rerun setup. Setup then downloads, precompiles and checks the
   required web, solver and reporting packages. Wait for **Setup complete**.
3. Double-click **`OpenAeroLab.cmd`** in this folder. The default browser opens
   when the local server is ready. Keep its console open while working.

Use **`Airfoil-Lab.cmd`** for Airfoil Lab alone. **`LBM-solver.cmd`** runs the CLI
case list after you prepare `cases_to_run.yaml`; it also accepts `--case PATH`.
The existing `apps/ludwig_web/run.cmd` and `apps/airfoil_web/run.cmd` shortcuts
use the same runtime resolver.

For unattended setup, run `setup.cmd --install-julia` to allow the Julia download
without a prompt, or `setup.cmd --no-install-julia` to require an existing Julia.
If the pinned Julia is already available, setup reuses it without asking.

`setup.cmd` and the Windows app shortcuts do not load unsigned PowerShell
script files. This also works on machines where an organization enforces
`AllSigned` for `.ps1` files. If you see a `setup.ps1 is not digitally signed`
error, you have an older ZIP: download the latest repository ZIP, extract it,
and run its root `setup.cmd`. No execution-policy change is needed. Automated
Julia installation uses the built-in Windows `curl.exe`, `certutil.exe` and
`tar.exe` tools and verifies the archive SHA-256 before extracting it.
You can also set `LUDWIG_JULIA` to an IT-approved Julia 1.13.2 executable, or set
`LUDWIG_JULIA_ARCHIVE` to a downloaded official Windows ZIP. The archive is always
checksum verified. No download is required for that runtime when a local ZIP is
provided. See the advanced guide for a fully offline package depot.

If an older setup failed during extraction with a missing `LICENSE.md`, update
the complete repository and rerun setup. The installer now handles long Windows
extraction paths and checks Julia before activating it. Installation logs are
kept under `<private>/runtime/install_logs`; previous failed staging folders are
disposable. If endpoint protection blocks execution, ask IT for an approved Julia
installation rather than disabling protection. The advanced guide includes a
direct Julia setup command for machines where PowerShell scripts are restricted.

A 64-bit NVIDIA driver compatible with CUDA 13.0 and the
[Microsoft Visual C++ x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe)
are needed for GPU solves. Setup installs the signed Microsoft runtime when missing
(Windows may ask for administrator authorization); install the NVIDIA driver
separately. Julia installs the CUDA toolkit libraries;
you do not need a separate CUDA toolkit installation. JXfoil and result viewing
do not require a GPU. OpenAeroLab does not change your GPU driver or system PATH.

The terminal shows startup output and the path of a persistent startup/server
log under `<private>/runtime/launch_logs`. If launch fails, inspect that log and
rerun setup. Network proxies must permit Julia's package and artifact downloads.
See [advanced setup and troubleshooting](../../tools/install/README.md).
