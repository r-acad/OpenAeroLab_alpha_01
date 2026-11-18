@echo off
setlocal DisableDelayedExpansion
REM Native Windows bootstrap: works even when policy requires signed PowerShell scripts.
set "OA_INSTALL_PREF="
set "OA_OFFLINE="
set "OA_DRY_RUN="
set "OA_SETUP_ARGS="
:parse_options
if "%~1"=="" goto options_done
if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help
if /I "%~1"=="--install-julia" goto option_install
if /I "%~1"=="--no-install-julia" goto option_no_install
if /I "%~1"=="--offline" set "OA_OFFLINE=1"
if /I "%~1"=="--dry-run" set "OA_DRY_RUN=1"
if /I "%~1"=="--offline" goto option_forward
if /I "%~1"=="--dry-run" goto option_forward
if /I "%~1"=="--no-precompile" goto option_forward
if /I "%~1"=="--no-verify" goto option_forward
echo Unknown setup option: %~1. Use --help. 1>&2
goto fail
:option_install
if "%OA_INSTALL_PREF%"=="no" goto conflicting_options
set "OA_INSTALL_PREF=yes"
shift /1
goto parse_options
:option_no_install
if "%OA_INSTALL_PREF%"=="yes" goto conflicting_options
set "OA_INSTALL_PREF=no"
shift /1
goto parse_options
:option_forward
set "OA_SETUP_ARGS=%OA_SETUP_ARGS% %~1"
shift /1
goto parse_options
:conflicting_options
echo Choose only one of --install-julia and --no-install-julia. 1>&2
goto fail
:options_done
if "%OA_INSTALL_PREF%"=="yes" if defined OA_OFFLINE if not defined LUDWIG_JULIA_ARCHIVE goto offline_archive_required
if "%OA_INSTALL_PREF%"=="yes" if defined OA_DRY_RUN goto incompatible_options
call "%~dp0tools\install\resolve_julia.cmd"
if errorlevel 2 goto fail
if defined OA_JULIA goto run_setup
if defined OA_OFFLINE if "%OA_INSTALL_PREF%"=="yes" goto download_julia
if defined OA_OFFLINE goto missing_julia
if defined OA_DRY_RUN goto missing_julia
if "%OA_INSTALL_PREF%"=="yes" goto download_julia
if "%OA_INSTALL_PREF%"=="no" goto missing_julia
choice /C YN /M "No Julia %OA_MIN_VERSION% or later was found. Download and install Julia %OA_VERSION% automatically"
if errorlevel 2 goto declined_julia
if errorlevel 1 goto download_julia
goto declined_julia
:incompatible_options
echo --install-julia cannot be combined with --dry-run. 1>&2
goto fail
:offline_archive_required
echo Offline Julia installation requires LUDWIG_JULIA_ARCHIVE pointing to the official Windows ZIP. 1>&2
goto fail
:missing_julia
echo No Julia %OA_MIN_VERSION% or later was found. Install one manually, or rerun setup.cmd with --install-julia. 1>&2
goto fail
:declined_julia
echo Install Julia %OA_MIN_VERSION% or later manually, then rerun setup.cmd. 1>&2
goto fail
:download_julia
if defined OA_OFFLINE (call "%~dp0tools\install\install_julia.cmd" --offline) else call "%~dp0tools\install\install_julia.cmd"
if errorlevel 1 goto fail
call "%~dp0tools\install\resolve_julia.cmd"
if errorlevel 1 goto fail
:run_setup
echo Starting package setup. The first Julia startup with an empty depot can take a few minutes.
"%OA_JULIA%" --startup-file=no "%~dp0tools\install\setup_windows.jl" %OA_SETUP_ARGS%
set "SETUP_EXIT=%ERRORLEVEL%"
goto finish
:help
set "OA_MIN_VERSION="
for /f "usebackq delims=" %%V in ("%~dp0tools\install\julia_min_version.txt") do if not defined OA_MIN_VERSION set "OA_MIN_VERSION=%%V"
echo OpenAeroLab Windows setup: setup.cmd [options]
echo   Any Julia %OA_MIN_VERSION% or later is used. If none is found, setup asks before downloading the verified archive.
echo   --install-julia     Download Julia automatically, without a prompt
echo   --no-install-julia  Require Julia to be installed already
echo   --offline           Use only existing Julia, packages and artifacts
echo   --dry-run           Show the setup plan without installing anything
echo   --no-precompile     Skip explicit package precompilation
echo   --no-verify         Skip runtime and application checks
echo   LUDWIG_JULIA may point to an approved existing Julia %OA_MIN_VERSION% or later executable.
echo   LUDWIG_JULIA_ARCHIVE may point to the official Julia ZIP for offline installation.
echo   HTTPS_PROXY or LUDWIG_PROXY may specify a proxy; TLS remains verified.
set "SETUP_EXIT=0"
goto finish
:fail
set "SETUP_EXIT=1"
:finish
echo.
if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause
exit /b %SETUP_EXIT%
