@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install\setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"
echo.
if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause
exit /b %SETUP_EXIT%
