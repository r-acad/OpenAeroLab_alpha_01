@echo off
setlocal

for %%I in ("%~dp0..\..") do set "WORKSPACE=%%~fI"
set "JX_PROJECT=%WORKSPACE%\third_party\JXfoil"
if not defined AIRFOIL_WEB_PORT set "AIRFOIL_WEB_PORT=8058"

if not exist "%JX_PROJECT%\Project.toml" (
    echo JXfoil project not found: "%JX_PROJECT%"
    pause
    exit /b 1
)

rem The app, JXfoil and the CFD projects are built for Julia 1.13.2 (24 Oct 2026; 1.12.5 was removed
rem from this machine). Another julia.exe on PATH (for example a juliaup alias) may be a different
rem version without their precompiled packages, so prefer the installed 1.13.2. Set AIRFOIL_JULIA to
rem a julia.exe path to override.
if not defined AIRFOIL_JULIA if exist "%LOCALAPPDATA%\Programs\Julia-1.13.2\bin\julia.exe" set "AIRFOIL_JULIA=%LOCALAPPDATA%\Programs\Julia-1.13.2\bin\julia.exe"
if not defined AIRFOIL_JULIA (
    where julia.exe >nul 2>&1
    if errorlevel 1 (
        echo Julia was not found on PATH. Install Julia or add julia.exe to PATH.
        pause
        exit /b 1
    )
    set "AIRFOIL_JULIA=julia.exe"
)

pushd "%WORKSPACE%"
echo Starting Airfoil Lab with "%AIRFOIL_JULIA%". Julia will open the browser at the available local port.
"%AIRFOIL_JULIA%" --startup-file=no --project="%JX_PROJECT%" --threads=auto "%~dp0run.jl"
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
