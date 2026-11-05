@echo off
REM Shared Windows Julia resolver. Called from setup.cmd and app launchers.
REM Do not use SETLOCAL: the caller needs OA_JULIA and LUDWIG_PRIVATE_ROOT.
for %%D in ("%~dp0..\..") do set "OA_ROOT=%%~fD"
if not defined LUDWIG_PRIVATE_ROOT for %%D in ("%OA_ROOT%\..\02_PRIVATE_PROJECT_DEVELOPMENT") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
for %%D in ("%LUDWIG_PRIVATE_ROOT%") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
set "OA_VERSION="
for /f "usebackq delims=" %%V in ("%~dp0julia_version.txt") do if not defined OA_VERSION set "OA_VERSION=%%V"
if not defined OA_VERSION exit /b 3
set "OA_JULIA="
if defined LUDWIG_JULIA goto explicit_ludwig
if defined AIRFOIL_JULIA goto explicit_airfoil
call :probe "%LUDWIG_PRIVATE_ROOT%\runtime\julia-%OA_VERSION%\julia-%OA_VERSION%\bin\julia.exe"
if defined OA_JULIA exit /b 0
call :probe "%LOCALAPPDATA%\Programs\Julia-%OA_VERSION%\bin\julia.exe"
if defined OA_JULIA exit /b 0
for /f "delims=" %%J in ('where julia.exe 2^>nul') do if not defined OA_JULIA call :probe "%%J"
if defined OA_JULIA exit /b 0
exit /b 1

:explicit_ludwig
call :probe "%LUDWIG_JULIA%"
if defined OA_JULIA exit /b 0
echo LUDWIG_JULIA must point to Julia %OA_VERSION%. 1>&2
exit /b 2

:explicit_airfoil
call :probe "%AIRFOIL_JULIA%"
if defined OA_JULIA exit /b 0
echo AIRFOIL_JULIA must point to Julia %OA_VERSION%. 1>&2
exit /b 2

:probe
if not exist "%~1" exit /b 1
set "OA_DETECTED="
for /f "usebackq delims=" %%V in (`"%~1" --startup-file=no --version 2^>nul`) do if not defined OA_DETECTED set "OA_DETECTED=%%V"
if /I "%OA_DETECTED%"=="julia version %OA_VERSION%" set "OA_JULIA=%~1"
exit /b 0
