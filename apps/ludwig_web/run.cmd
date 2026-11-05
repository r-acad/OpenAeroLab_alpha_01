@echo off
setlocal DisableDelayedExpansion
set "OA_APP=workbench"
call "%~dp0..\..\tools\install\launch_windows.cmd" %*
set "LAUNCH_EXIT=%ERRORLEVEL%"
exit /b %LAUNCH_EXIT%
