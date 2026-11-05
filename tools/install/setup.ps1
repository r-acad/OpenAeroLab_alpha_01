$ErrorActionPreference = 'Stop'
try {
    $allowed = @('--offline','--dry-run','--help','-h','--no-precompile','--no-verify','--install-julia','--no-install-julia')
    foreach ($option in $args) { if ($option -notin $allowed) { throw "Unknown setup option: $option. Use --help." } }
    $repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    $version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'julia_version.txt') -Raw).Trim()
    if ($args -contains '--help' -or $args -contains '-h') {
        Write-Host 'OpenAeroLab Windows setup: setup.cmd [options]'
        Write-Host "  Julia $version is required. If missing, setup asks before downloading it."
        Write-Host '  --install-julia     Install Julia automatically if missing; no prompt'
        Write-Host '  --no-install-julia  Require Julia to be installed already; no prompt'
        Write-Host '  --offline           Use only existing Julia, packages and artifacts'
        Write-Host '  --dry-run           Show the setup plan without installing anything'
        Write-Host '  --no-precompile     Skip explicit package precompilation'
        Write-Host '  --no-verify         Skip runtime and application checks'
        exit 0
    }
    if ($args -contains '--install-julia' -and $args -contains '--no-install-julia') {
        throw 'Choose only one of --install-julia and --no-install-julia.'
    }
    if ($args -contains '--install-julia' -and (@($args | Where-Object { $_ -in @('--offline','--dry-run') }).Count)) {
        throw '--install-julia cannot be combined with --offline or --dry-run.'
    }
    . (Join-Path $PSScriptRoot 'bootstrap.ps1')
    $julia = Get-LudwigJulia -Repository $repository -AllowMissing
    if (-not $julia) {
        $noInstall = @($args | Where-Object { $_ -in @('--offline','--dry-run','--no-install-julia') }).Count -gt 0
        if ($noInstall) {
            throw "Julia $version is missing. Install it manually, or rerun setup.cmd with --install-julia."
        }
        $install = $args -contains '--install-julia'
        if (-not $install) {
            while ($true) {
                try { $answer = Read-Host "Julia $version is missing. Download and install it automatically? [Y/N]" }
                catch { throw 'No interactive answer was available. Rerun setup.cmd with --install-julia or --no-install-julia.' }
                if ($null -eq $answer) { throw 'No answer was supplied. Rerun setup.cmd with --install-julia or --no-install-julia.' }
                $answer = $answer.Trim().ToLowerInvariant()
                if ($answer -in @('y','yes')) { $install = $true; break }
                if ($answer -in @('n','no','')) { break }
                Write-Host 'Please answer Y or N.'
            }
        }
        if (-not $install) { throw "Julia $version is required. Install it manually, then rerun setup.cmd." }
        $julia = Get-LudwigJulia -Repository $repository -Install
    }
    if ($args -notcontains '--dry-run') {
        Install-LudwigVisualCpp -Offline:($args -contains '--offline')
    }
    $setupArgs = @($args | Where-Object { $_ -notin @('--install-julia','--no-install-julia') })
    & $julia --startup-file=no (Join-Path $PSScriptRoot 'setup.jl') @setupArgs
    exit $LASTEXITCODE
} catch { Write-Error $_; exit 1 }
