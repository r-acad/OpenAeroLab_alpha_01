@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\tools\install\launch.ps1" airfoil %*
set "LAUNCH_EXIT=%ERRORLEVEL%"
if not "%LAUNCH_EXIT%"=="0" if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause
exit /b %LAUNCH_EXIT%
