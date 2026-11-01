# Mac

1. Extract the complete GitHub ZIP to a writable folder and open Terminal there.
2. Run **`bash setup.command`**. Julia 1.13.2 is detected or downloaded automatically
   for Apple Silicon or Intel. Setup installs and checks the workbenches, JXfoil
   and Makie. Wait for **Setup complete**.
3. Run **`bash "__Start here!__/Mac/OpenAeroLab.command"`**. Your default browser
   opens when the local server is ready. Keep Terminal open while working.

Use **`bash "__Start here!__/Mac/Airfoil-Lab.command"`** for Airfoil Lab alone.
To launch by double-clicking in Finder, first grant the `.command` launchers
execute permission with `chmod +x` on those files. Terminal invocation works
even when ZIP extraction did not preserve executable permissions.

macOS supports JXfoil, model placement, studies, saved native/VTK result viewing,
and Makie reports. Native open/save dialogs use macOS's built-in chooser.
The current LBM, pressure and C-mesh runtimes depend on CUDA, which is not
available on modern macOS. Setup skips those environments; use Windows or Linux
with an NVIDIA GPU for those runs and open their saved results here.

See [advanced setup and troubleshooting](../../tools/install/README.md).
