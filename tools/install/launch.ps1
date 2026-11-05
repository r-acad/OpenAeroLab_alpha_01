# Compatibility entry point. Prefer the .cmd launchers on managed computers.
param([ValidateSet('workbench','airfoil','solver')][string]$App = 'workbench')
$previousApp = $env:OA_APP
try {
    $env:OA_APP = $App
    & (Join-Path $PSScriptRoot 'launch_windows.cmd') @args
    $launcherExit = $LASTEXITCODE
} finally { $env:OA_APP = $previousApp }
exit $launcherExit
