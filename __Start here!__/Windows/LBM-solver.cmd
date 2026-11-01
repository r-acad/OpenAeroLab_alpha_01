@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\tools\install\launch.ps1" solver %*
set "RESULT=%ERRORLEVEL%"
if not "%LUDWIG_WEB_NO_PAUSE%"=="1" pause
exit /b %RESULT%
