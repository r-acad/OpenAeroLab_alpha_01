@echo off
REM Shared Windows Julia resolver. Called from setup.cmd and app launchers.
REM Any Julia from the release in julia_min_version.txt on is accepted (1.13 means 1.13.0, 1.13.5, 1.14, 2.0 ...).
REM julia_version.txt names the release that setup downloads when no acceptable Julia is found.
REM Do not use SETLOCAL: the caller needs OA_JULIA, OA_VERSION, OA_MIN_VERSION and LUDWIG_PRIVATE_ROOT.
for %%D in ("%~dp0..\..") do set "OA_ROOT=%%~fD"
if not defined LUDWIG_PRIVATE_ROOT for %%D in ("%OA_ROOT%\..\02_PRIVATE_PROJECT_DEVELOPMENT") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
for %%D in ("%LUDWIG_PRIVATE_ROOT%") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
set "OA_VERSION="
for /f "usebackq delims=" %%V in ("%~dp0julia_version.txt") do if not defined OA_VERSION set "OA_VERSION=%%V"
if not defined OA_VERSION exit /b 3
set "OA_MIN_VERSION="
for /f "usebackq delims=" %%V in ("%~dp0julia_min_version.txt") do if not defined OA_MIN_VERSION set "OA_MIN_VERSION=%%V"
if not defined OA_MIN_VERSION exit /b 3
set "OA_MIN_MAJOR="
set "OA_MIN_MINOR="
for /f "tokens=1,2 delims=." %%A in ("%OA_MIN_VERSION%") do call :set_minimum "%%A" "%%B"
if not defined OA_MIN_MAJOR exit /b 3
if not defined OA_MIN_MINOR exit /b 3
set "OA_JULIA="
set "OA_JULIA_VERSION="
if defined LUDWIG_JULIA goto explicit_ludwig
if defined AIRFOIL_JULIA goto explicit_airfoil
REM The managed runtime and the recommended release first, then any other installed release, then PATH.
call :probe "%LUDWIG_PRIVATE_ROOT%\runtime\julia-%OA_VERSION%\julia-%OA_VERSION%\bin\julia.exe"
if defined OA_JULIA exit /b 0
call :probe "%LOCALAPPDATA%\Programs\Julia-%OA_VERSION%\bin\julia.exe"
if defined OA_JULIA exit /b 0
for /d %%D in ("%LOCALAPPDATA%\Programs\Julia-*") do if not defined OA_JULIA call :probe "%%~fD\bin\julia.exe"
if defined OA_JULIA exit /b 0
for /f "delims=" %%J in ('where julia.exe 2^>nul') do if not defined OA_JULIA call :probe "%%J"
if defined OA_JULIA exit /b 0
exit /b 1

:explicit_ludwig
call :probe "%LUDWIG_JULIA%"
if defined OA_JULIA exit /b 0
echo LUDWIG_JULIA must point to Julia %OA_MIN_VERSION% or later. 1>&2
exit /b 2

:explicit_airfoil
call :probe "%AIRFOIL_JULIA%"
if defined OA_JULIA exit /b 0
echo AIRFOIL_JULIA must point to Julia %OA_MIN_VERSION% or later. 1>&2
exit /b 2

:set_minimum
set "OA_MIN_MAJOR=%~1"
set "OA_MIN_MINOR=%~2"
exit /b 0

:probe
REM Accept the executable when it reports "julia version X.Y..." with X.Y at or above the minimum.
if not exist "%~1" exit /b 1
set "OA_DETECTED="
for /f "usebackq delims=" %%V in (`"%~1" --startup-file=no --version 2^>nul`) do if not defined OA_DETECTED set "OA_DETECTED=%%V"
if not defined OA_DETECTED exit /b 1
set "OA_DETECTED_NUMBER="
for /f "tokens=1,2,3" %%P in ("%OA_DETECTED%") do if /I "%%P %%Q"=="julia version" set "OA_DETECTED_NUMBER=%%R"
if not defined OA_DETECTED_NUMBER exit /b 1
set "OA_DETECTED_MAJOR="
set "OA_DETECTED_MINOR="
for /f "tokens=1,2 delims=.-+" %%A in ("%OA_DETECTED_NUMBER%") do call :set_detected "%%A" "%%B"
if not defined OA_DETECTED_MAJOR exit /b 1
if not defined OA_DETECTED_MINOR exit /b 1
set "OA_ACCEPT="
if %OA_DETECTED_MAJOR% GTR %OA_MIN_MAJOR% set "OA_ACCEPT=1"
if %OA_DETECTED_MAJOR% EQU %OA_MIN_MAJOR% if %OA_DETECTED_MINOR% GEQ %OA_MIN_MINOR% set "OA_ACCEPT=1"
if not defined OA_ACCEPT exit /b 0
set "OA_JULIA=%~1"
set "OA_JULIA_VERSION=%OA_DETECTED_NUMBER%"
exit /b 0

:set_detected
set "OA_DETECTED_MAJOR=%~1"
set "OA_DETECTED_MINOR=%~2"
exit /b 0
