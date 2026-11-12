# Linux

1. Extract the complete GitHub ZIP to a writable folder and open a terminal there.
2. Run **`bash setup.sh`**. Julia 1.13.2 is detected or downloaded automatically
   for x86-64 or ARM64. Setup downloads, precompiles and verifies the required
   packages. Wait for **Setup complete**.
3. Run **`bash "__Start here!__/Linux/OpenAeroLab.sh"`** from the repository root.
   Keep the terminal open. `xdg-open` opens your default browser; if unavailable,
   open the printed localhost URL yourself.

The same folder contains `Airfoil-Lab.sh` for standalone Airfoil Lab and
`LBM-solver.sh` for the CLI. Run them with `bash` in the same way. You can make
the scripts executable with `chmod +x` if you prefer direct launching.

Automatic Julia download needs `curl` or `wget`, `tar`, and `sha256sum` or `shasum`,
normally included with desktop Linux. Install missing system utilities using your
distribution's package manager. Standard glibc Linux is supported; musl/Alpine
requires a separate compatible Julia installation. No `sudo` is used by setup.

For native file dialogs, install `zenity` (GNOME) or `kdialog` (KDE). Without one,
use the browser file selector and explicit destination paths. NVIDIA GPU solves
need a driver compatible with the pinned CUDA 12.9 runtime. AMD GPUs and Apple
GPU backends are not implemented. CPU-only machines can use JXfoil and viewers;
the installer reports whether CUDA is functional.

See [advanced setup and troubleshooting](../../tools/install/README.md).
