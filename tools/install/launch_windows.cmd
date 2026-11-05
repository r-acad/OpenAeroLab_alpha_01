@echo off
setlocal DisableDelayedExpansion
call "%~dp0resolve_julia.cmd"
if errorlevel 2 goto failed
if not defined OA_JULIA goto missing_julia
set "OA_PROJECT=%OA_ROOT%\third_party\JXfoil"
if /I "%OA_APP%"=="workbench" set "OA_ENTRY=%OA_ROOT%\apps\ludwig_web\run.jl"
if /I "%OA_APP%"=="airfoil" set "OA_ENTRY=%OA_ROOT%\apps\airfoil_web\run.jl"
if /I "%OA_APP%"=="solver" set "OA_ENTRY=%OA_ROOT%\src\main.jl"
if /I "%OA_APP%"=="solver" set "OA_PROJECT=%OA_ROOT%\src"
if not defined OA_ENTRY goto bad_app
set "OA_SCRATCH=%LUDWIG_PRIVATE_ROOT%\TEMP_DELETEABLE\runtime"
mkdir "%OA_SCRATCH%" 2>nul
set "TEMP=%OA_SCRATCH%"
set "TMP=%OA_SCRATCH%"
set "TMPDIR=%OA_SCRATCH%"
cd /d "%OA_ROOT%"
echo Starting OpenAeroLab (%OA_APP%) using %OA_JULIA%
echo Keep this console open while using the app.
"%OA_JULIA%" --startup-file=no "%~dp0launch_windows.jl" "%OA_APP%" "%OA_PROJECT%" "%OA_ENTRY%" %*
set "OA_EXIT=%ERRORLEVEL%"
if not "%OA_EXIT%"=="0" echo OpenAeroLab exited with code %OA_EXIT%. Run setup.cmd if dependencies are missing.
goto finish
:missing_julia
echo Julia %OA_VERSION% is missing. Run setup.cmd at the repository root first. 1>&2
goto failed
:bad_app
echo Unknown OpenAeroLab Windows app. 1>&2
:failed
set "OA_EXIT=1"
:finish
if not "%OA_EXIT%"=="0" if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause
exit /b %OA_EXIT%
