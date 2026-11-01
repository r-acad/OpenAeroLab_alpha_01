$ErrorActionPreference = 'Stop'
try {
    $allowed = @('--offline','--dry-run','--help','-h','--no-precompile','--no-verify')
    foreach ($option in $args) { if ($option -notin $allowed) { throw "Unknown setup option: $option. Use --help." } }
    $repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    . (Join-Path $PSScriptRoot 'bootstrap.ps1')
    $install = -not (@($args | Where-Object { $_ -in @('--offline','--dry-run','--help','-h') }).Count)
    $julia = Get-LudwigJulia -Repository $repository -Install:$install
    if (-not (@($args | Where-Object { $_ -in @('--dry-run','--help','-h') }).Count)) {
        Install-LudwigVisualCpp -Offline:($args -contains '--offline')
    }
    & $julia --startup-file=no (Join-Path $PSScriptRoot 'setup.jl') @args
    exit $LASTEXITCODE
} catch { Write-Error $_; exit 1 }
