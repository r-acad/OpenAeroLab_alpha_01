param([ValidateSet('workbench','airfoil','solver')][string]$App = 'workbench')
$ErrorActionPreference = 'Stop'
try {
    $repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    . (Join-Path $PSScriptRoot 'bootstrap.ps1')
    $julia = Get-LudwigJulia -Repository $repository
    $project = if ($App -eq 'solver') { Join-Path $repository 'src' } else { Join-Path $repository 'third_party\JXfoil' }
    $entry = switch ($App) { 'workbench' {'apps\ludwig_web\run.jl'} 'airfoil' {'apps\airfoil_web\run.jl'} 'solver' {'src\main.jl'} }
    $logRoot = Join-Path $env:LUDWIG_PRIVATE_ROOT 'TEMP_DELETEABLE\launchers'
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $log = Join-Path $logRoot ($App + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $PID + '.log')
    Write-Host "Starting OpenAeroLab ($App) using $julia"
    Write-Host "Startup / server log: $log"
    Write-Host 'The browser opens when the service is ready. Keep this console open while using the app.'
    $writer = New-Object IO.StreamWriter($log, $false, (New-Object Text.UTF8Encoding($false)))
    $writer.AutoFlush = $true
    Push-Location -LiteralPath $repository
    try {
        # Native warnings on stderr are log output, not PowerShell terminating errors.
        $ErrorActionPreference = 'Continue'
        & $julia --startup-file=no --project=$project --threads=auto (Join-Path $repository $entry) @args 2>&1 |
            ForEach-Object { $line = $_.ToString(); $writer.WriteLine($line); Write-Host $line }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
    } finally { $writer.Dispose(); Pop-Location }
    if ($result -ne 0) { Write-Host "Launch failed ($result). Run setup.cmd if dependencies are missing. See $log" }
    exit $result
} catch { Write-Host "OpenAeroLab: $_"; exit 1 }
