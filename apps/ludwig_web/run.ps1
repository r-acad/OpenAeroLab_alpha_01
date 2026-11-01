$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$jxProject = Join-Path $repository 'third_party\JXfoil'
if (-not $env:LUDWIG_WEB_PORT) { $env:LUDWIG_WEB_PORT = '8058' }
$julia = $env:AIRFOIL_JULIA
if (-not $julia) {
    $pinned = Join-Path $env:LOCALAPPDATA 'Programs\Julia-1.13.2\bin\julia.exe'
    $julia = if (Test-Path -LiteralPath $pinned) { $pinned } else { 'julia' }
}
& $julia --startup-file=no --project=$jxProject -t auto (Join-Path $PSScriptRoot 'run.jl')
