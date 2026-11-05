@echo off
setlocal DisableDelayedExpansion
REM Install-only native bootstrap; no PowerShell scripts or policy changes.
set "OA_OFFLINE="
if /I "%~1"=="--offline" set "OA_OFFLINE=1"
if not "%~1"=="" if /I not "%~1"=="--offline" exit /b 2
for %%D in ("%~dp0..\..") do set "OA_ROOT=%%~fD"
if not defined LUDWIG_PRIVATE_ROOT for %%D in ("%OA_ROOT%\..\02_PRIVATE_PROJECT_DEVELOPMENT") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
for %%D in ("%LUDWIG_PRIVATE_ROOT%") do set "LUDWIG_PRIVATE_ROOT=%%~fD"
set "OA_VERSION="
for /f "usebackq delims=" %%V in ("%~dp0julia_version.txt") do if not defined OA_VERSION set "OA_VERSION=%%V"
if not defined OA_VERSION goto missing_record
set "OA_URL="
set "OA_SHA="
for /f "usebackq tokens=1,2,3 delims=|" %%A in ("%~dp0runtimes.tsv") do call :read_record "%%A" "%%B" "%%C"
if not defined OA_URL goto missing_record
if not defined OA_SHA goto missing_record
set "OA_RUNTIME=%LUDWIG_PRIVATE_ROOT%\runtime\julia-%OA_VERSION%"
set "OA_STAGE=%LUDWIG_PRIVATE_ROOT%\TEMP_DELETEABLE\setup\j-%RANDOM%-%RANDOM%"
set "OA_CACHE=%LUDWIG_PRIVATE_ROOT%\TEMP_DELETEABLE\setup\julia-%OA_VERSION%-win64.zip"
set "OA_LOG=%LUDWIG_PRIVATE_ROOT%\runtime\install_logs\julia-native-%RANDOM%-%RANDOM%.log"
if not exist "%LUDWIG_PRIVATE_ROOT%\runtime\install_logs" mkdir "%LUDWIG_PRIVATE_ROOT%\runtime\install_logs"
if not exist "%LUDWIG_PRIVATE_ROOT%\runtime\install_logs" goto unwritable
if not exist "%SystemRoot%\System32\tar.exe" goto missing_tools
if not exist "%SystemRoot%\System32\certutil.exe" goto missing_tools
set "OA_LOCK_ACQUIRED="
call :install_locked 9>"%LUDWIG_PRIVATE_ROOT%\runtime\julia-install.lock"
set "OA_INSTALL_EXIT=%ERRORLEVEL%"
if not defined OA_LOCK_ACQUIRED goto locked
exit /b %OA_INSTALL_EXIT%

:install_locked
set "OA_LOCK_ACQUIRED=1"
echo Julia %OA_VERSION% bootstrap. Diagnostics: "%OA_LOG%"
echo Julia %OA_VERSION% bootstrap started.>"%OA_LOG%"
if exist "%OA_STAGE%" goto stage_collision
mkdir "%OA_STAGE%\u"
if errorlevel 1 goto unwritable
if not exist "%OA_RUNTIME%\julia-%OA_VERSION%\bin\julia.exe" goto choose_archive
"%OA_RUNTIME%\julia-%OA_VERSION%\bin\julia.exe" --startup-file=no --version >"%OA_STAGE%\existing-version.txt" 2>&1
if errorlevel 1 goto choose_archive
set "OA_DETECTED="
for /f "usebackq delims=" %%V in ("%OA_STAGE%\existing-version.txt") do if not defined OA_DETECTED set "OA_DETECTED=%%V"
if not "%OA_DETECTED%"=="julia version %OA_VERSION%" goto choose_archive
echo The managed Julia %OA_VERSION% runtime already starts correctly; reusing it.
exit /b 0
:choose_archive
set "OA_ARCHIVE=%OA_CACHE%"
if defined LUDWIG_JULIA_ARCHIVE for %%F in ("%LUDWIG_JULIA_ARCHIVE%") do set "OA_ARCHIVE=%%~fF"
call :verify_archive
if not errorlevel 1 goto extract_runtime
if defined LUDWIG_JULIA_ARCHIVE goto local_archive_invalid
if defined OA_OFFLINE goto offline_missing
if not exist "%SystemRoot%\System32\curl.exe" goto missing_tools
if exist "%OA_CACHE%\" goto cache_failure
if defined LUDWIG_PROXY set "HTTPS_PROXY=%LUDWIG_PROXY%"
echo Downloading the official Julia %OA_VERSION% archive. Keep this window open.
echo Downloading verified Julia archive.>>"%OA_LOG%"
set "OA_ARCHIVE=%OA_STAGE%\julia.zip"
REM Schannel uses Windows certificate trust. Explicit proxy settings are inherited.
REM Never disable TLS, permit HTTP redirects, or echo proxy credentials.
"%SystemRoot%\System32\curl.exe" --fail --location --no-insecure --no-proxy-insecure --proto =https --proto-redir =https --retry 3 --retry-delay 2 --retry-max-time 900 --connect-timeout 30 --speed-limit 1024 --speed-time 120 --silent --show-error --output "%OA_ARCHIVE%" "%OA_URL%" >>"%OA_LOG%" 2>&1
if errorlevel 1 goto download_failure
call :verify_archive
if errorlevel 1 goto checksum_failure
REM Preserve verified downloads across retries. This is a file move inside TEMP_DELETEABLE.
move /Y "%OA_ARCHIVE%" "%OA_CACHE%" >nul
if errorlevel 1 goto cache_failure
set "OA_ARCHIVE=%OA_CACHE%"
:extract_runtime
echo Checksum verified. Extracting Julia using Windows tar...
echo Checksum verified; extracting native ZIP.>>"%OA_LOG%"
"%SystemRoot%\System32\tar.exe" -xf "%OA_ARCHIVE%" -C "%OA_STAGE%\u" >>"%OA_LOG%" 2>&1
if errorlevel 1 goto extraction_failure
if not exist "%OA_STAGE%\u\julia-%OA_VERSION%\bin\julia.exe" goto extraction_failure
if not exist "%OA_STAGE%\u\julia-%OA_VERSION%\LICENSE.md" goto extraction_failure
"%OA_STAGE%\u\julia-%OA_VERSION%\bin\julia.exe" --startup-file=no --version >"%OA_STAGE%\version.txt" 2>&1
if errorlevel 1 goto execution_failure
set "OA_DETECTED="
for /f "usebackq delims=" %%V in ("%OA_STAGE%\version.txt") do if not defined OA_DETECTED set "OA_DETECTED=%%V"
if not "%OA_DETECTED%"=="julia version %OA_VERSION%" goto execution_failure
echo Julia starts correctly. Installing and checking its final location...
"%OA_STAGE%\u\julia-%OA_VERSION%\bin\julia.exe" --startup-file=no "%~dp0promote_runtime.jl" "%LUDWIG_PRIVATE_ROOT%" "%OA_STAGE%" "%OA_VERSION%" >>"%OA_LOG%" 2>&1
if errorlevel 1 goto promotion_failure
echo Julia %OA_VERSION% installed and verified. Cached downloads are safe to delete with TEMP_DELETEABLE.
exit /b 0

:verify_archive
set "OA_ACTUAL_SHA="
if not exist "%OA_ARCHIVE%" exit /b 1
"%SystemRoot%\System32\certutil.exe" -hashfile "%OA_ARCHIVE%" SHA256 >"%OA_STAGE%\hash.txt" 2>>"%OA_LOG%"
if errorlevel 1 exit /b 1
for /f "usebackq skip=1 tokens=1" %%H in ("%OA_STAGE%\hash.txt") do if not defined OA_ACTUAL_SHA set "OA_ACTUAL_SHA=%%H"
if /I "%OA_ACTUAL_SHA%"=="%OA_SHA%" exit /b 0
exit /b 1
:read_record
if /I "%~1"=="windows-x86_64" set "OA_URL=%~2"
if /I "%~1"=="windows-x86_64" set "OA_SHA=%~3"
exit /b 0
:locked
echo Another Julia installation is active, or the runtime folder is not writable. Close the other setup and retry. 1>&2
echo The lock is released automatically when its process exits; the remaining lock file is harmless. 1>&2
exit /b 1
:unwritable
echo Cannot write the private installation folders. Choose an IT-approved writable LUDWIG_PRIVATE_ROOT. 1>&2
exit /b 1
:stage_collision
echo A staging-name collision occurred. Rerun setup; existing files were preserved. 1>&2
exit /b 1
:missing_tools
echo Windows tar.exe and certutil.exe are required; downloading also needs Windows curl.exe. 1>&2
echo Ask IT to install Julia %OA_VERSION%, then set LUDWIG_JULIA to its executable and rerun setup. 1>&2
exit /b 1
:missing_record
echo The pinned Windows Julia version or archive record is missing. Download a complete repository ZIP. 1>&2
exit /b 1
:local_archive_invalid
echo LUDWIG_JULIA_ARCHIVE is missing or has the wrong SHA-256 for Julia %OA_VERSION%. 1>&2
echo Use the official Windows ZIP listed in tools/install/runtimes.tsv. Your supplied file was not modified. 1>&2
goto diagnostics
:offline_missing
echo Offline installation requires a verified cached ZIP or LUDWIG_JULIA_ARCHIVE pointing to the official ZIP. 1>&2
goto diagnostics
:checksum_failure
echo Julia archive checksum mismatch. No runtime was executed or installed. 1>&2
echo A network filter may have returned a login page. Ask IT for the official Windows ZIP. 1>&2
goto diagnostics
:download_failure
echo Julia download failed. Check HTTPS_PROXY or LUDWIG_PROXY and ask IT to allow julialang-s3.julialang.org. 1>&2
echo Windows certificate trust is used; ask IT to install its inspection certificate if required. 1>&2
echo You can use LUDWIG_JULIA_ARCHIVE for an approved ZIP, or LUDWIG_JULIA for an existing installation. 1>&2
goto diagnostics
:cache_failure
echo Could not retain the downloaded ZIP. Check folder permissions or security-software quarantine. 1>&2
goto diagnostics
:extraction_failure
echo Julia ZIP extraction failed. Existing runtimes are unchanged; the verified ZIP is retained for retry. 1>&2
echo Check the log, available disk space and Windows Security quarantine with IT. 1>&2
echo If the path is restricted or too long, select a shorter approved LUDWIG_PRIVATE_ROOT. 1>&2
goto diagnostics
:execution_failure
if exist "%OA_STAGE%\version.txt" type "%OA_STAGE%\version.txt" >>"%OA_LOG%"
echo Julia extracted but could not start. Existing runtimes are unchanged. 1>&2
echo Ask IT to check AppLocker/WDAC, Windows Security quarantine and the Visual C++ runtime. 1>&2
echo Keep protection enabled; an IT-approved Julia can be selected with LUDWIG_JULIA. 1>&2
goto diagnostics
:promotion_failure
echo Julia could not be installed at its final location. The promotion log identifies any preserved previous runtime. 1>&2
goto diagnostics
:diagnostics
echo Diagnostic log: "%OA_LOG%" 1>&2
echo Staging files: "%OA_STAGE%" 1>&2
echo Correct the cause and rerun setup; verified cached downloads are reused. 1>&2
exit /b 1
