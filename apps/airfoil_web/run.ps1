$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$jxProject = Join-Path $workspace 'third_party\JXfoil'
if (-not $env:AIRFOIL_WEB_PORT) { $env:AIRFOIL_WEB_PORT = '8058' }
# The app's projects are built for Julia 1.13.2 (24 Oct 2026); prefer it over another julia on PATH (see run.cmd).
$julia = $env:AIRFOIL_JULIA
if (-not $julia) {
    $pinned = Join-Path $env:LOCALAPPDATA 'Programs\Julia-1.13.2\bin\julia.exe'
    $julia = if (Test-Path $pinned) { $pinned } else { 'julia' }
}
& $julia --startup-file=no --project=$jxProject -t auto (Join-Path $PSScriptRoot 'run.jl')
