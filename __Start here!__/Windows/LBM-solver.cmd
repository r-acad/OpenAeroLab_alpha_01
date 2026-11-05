@echo off
setlocal DisableDelayedExpansion
set "OA_APP=solver"
call "%~dp0..\..\tools\install\launch_windows.cmd" %*
set "RESULT=%ERRORLEVEL%"
exit /b %RESULT%
