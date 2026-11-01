@echo off
rem run.jl reuses the current API revision and ignores older server instances.
setlocal
for %%I in ("%~dp0..\..") do set "REPOSITORY=%%~fI"
set "JX_PROJECT=%REPOSITORY%\third_party\JXfoil"
if not defined LUDWIG_WEB_PORT set "LUDWIG_WEB_PORT=8058"

rem Resolve a relative private-root override before changing the working folder.
if defined LUDWIG_PRIVATE_ROOT (
  for %%I in ("%LUDWIG_PRIVATE_ROOT%") do set "LUDWIG_PRIVATE_ROOT=%%~fI"
) else (
  for %%I in ("%REPOSITORY%\..\02_PRIVATE_PROJECT_DEVELOPMENT") do set "LUDWIG_PRIVATE_ROOT=%%~fI"
)
set "LAUNCH_LOG_DIR=%LUDWIG_PRIVATE_ROOT%\TEMP_DELETEABLE\ludwig_web"
if not exist "%LAUNCH_LOG_DIR%\" mkdir "%LAUNCH_LOG_DIR%" >nul 2>&1
if not exist "%LAUNCH_LOG_DIR%\" (
  echo Cannot create the launcher log folder: "%LAUNCH_LOG_DIR%"
  set "EXIT_CODE=1"
  goto finish
)

rem A running server keeps stdout open. Every invocation needs its own log.
set "LAUNCH_LOG=%LAUNCH_LOG_DIR%\launcher_%RANDOM%_%RANDOM%.log"
> "%LAUNCH_LOG%" echo Ludwig Workbench startup
>> "%LAUNCH_LOG%" echo Repository: "%REPOSITORY%"
if not exist "%JX_PROJECT%\Project.toml" (
  >> "%LAUNCH_LOG%" echo JXfoil environment not found: "%JX_PROJECT%"
  set "EXIT_CODE=1"
  goto finish
)

if not defined AIRFOIL_JULIA if exist "%LOCALAPPDATA%\Programs\Julia-1.13.2\bin\julia.exe" set "AIRFOIL_JULIA=%LOCALAPPDATA%\Programs\Julia-1.13.2\bin\julia.exe"
if not defined AIRFOIL_JULIA set "AIRFOIL_JULIA=julia.exe"
if exist "%AIRFOIL_JULIA%" goto start
where "%AIRFOIL_JULIA%" >nul 2>&1
if not errorlevel 1 goto start
>> "%LAUNCH_LOG%" echo Julia executable not found: "%AIRFOIL_JULIA%"
>> "%LAUNCH_LOG%" echo Install Julia or set AIRFOIL_JULIA to the full path of julia.exe.
set "EXIT_CODE=9009"
goto finish

:start
echo Starting Ludwig Workbench with "%AIRFOIL_JULIA%".
echo Startup log: "%LAUNCH_LOG%"
echo Julia will open or reuse the local app in your browser.
echo A new server will use this console; an existing server will be reused.
pushd "%REPOSITORY%"
if errorlevel 1 (
  >> "%LAUNCH_LOG%" echo Cannot enter the repository folder: "%REPOSITORY%"
  set "EXIT_CODE=1"
  goto finish
)
"%AIRFOIL_JULIA%" --startup-file=no --project="%JX_PROJECT%" --threads=auto "%~dp0run.jl" >> "%LAUNCH_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
popd

:finish
echo.
if defined LAUNCH_LOG if exist "%LAUNCH_LOG%" type "%LAUNCH_LOG%"
if "%EXIT_CODE%"=="0" goto success
echo.
echo Ludwig Workbench could not start. Exit code: %EXIT_CODE%
if defined LAUNCH_LOG echo Startup details: "%LAUNCH_LOG%"
if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause

goto exit

:success
rem A successful reuse ends Julia normally, but does not stop the existing server.
findstr /l /c:"Reusing current Ludwig Workbench at " "%LAUNCH_LOG%" >nul 2>&1
if errorlevel 1 goto exit
echo.
echo Ludwig Workbench is already online at the URL shown above.
echo Closing this launcher will not stop the existing server.
if "%LUDWIG_WEB_NO_PAUSE%"=="1" goto exit
echo Press any key to close this launcher; the existing server stays running.
pause >nul

:exit
exit /b %EXIT_CODE%
