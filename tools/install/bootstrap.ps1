# Used by setup and every Windows launcher. Julia needs no PATH edits or elevation.
function Install-LudwigVisualCpp {
    param([switch]$Offline)
    $key = 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    if ((Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue).Installed -eq 1) { return }
    if ($Offline) { throw 'The Microsoft Visual C++ x64 runtime is missing. Run setup online once.' }
    $folder = Join-Path $env:LUDWIG_PRIVATE_ROOT 'TEMP_DELETEABLE\setup'
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $installer = Join-Path $folder ('vc_redist_' + [guid]::NewGuid().ToString('N') + '.exe')
    Write-Host 'Downloading the Microsoft Visual C++ x64 runtime required by CUDA...'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $installer
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
        throw 'The Visual C++ installer did not have a valid Microsoft signature; it was not run.'
    }
    Write-Host 'Installing the signed runtime. Windows may request administrator authorization.'
    $process = Start-Process -FilePath $installer -ArgumentList '/install','/quiet','/norestart' -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -notin @(0,1638,3010)) { throw "Visual C++ installation failed ($($process.ExitCode)). Install the x64 runtime from Microsoft, then rerun setup." }
    if ($process.ExitCode -eq 3010) { Write-Host 'The runtime installer requests a Windows restart before GPU use.' }
}

function Get-LudwigJulia {
    param([string]$Repository, [switch]$Install)
    $version = (Get-Content -LiteralPath (Join-Path $Repository 'tools\install\julia_version.txt') -Raw).Trim()
    if (-not $env:LUDWIG_PRIVATE_ROOT) {
        $env:LUDWIG_PRIVATE_ROOT = [IO.Path]::GetFullPath((Join-Path $Repository '..\02_PRIVATE_PROJECT_DEVELOPMENT'))
    } else { $env:LUDWIG_PRIVATE_ROOT = [IO.Path]::GetFullPath($env:LUDWIG_PRIVATE_ROOT) }
    $runtime = Join-Path $env:LUDWIG_PRIVATE_ROOT "runtime\julia-$version"
    $managed = Join-Path $runtime "julia-$version\bin\julia.exe"
    $explicit = if ($env:LUDWIG_JULIA) { $env:LUDWIG_JULIA } else { $env:AIRFOIL_JULIA }
    $candidates = if ($explicit) { @($explicit) } else {
        @($managed, (Join-Path $env:LOCALAPPDATA "Programs\Julia-$version\bin\julia.exe"), 'julia.exe')
    }
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            $detected = & $command.Source --startup-file=no --version 2>$null
            if ($LASTEXITCODE -eq 0 -and "$detected" -eq "julia version $version") { return $command.Source }
        }
    }
    if ($explicit) { throw "LUDWIG_JULIA / AIRFOIL_JULIA must point to Julia $version. Remove the override to let setup install it." }
    if (-not $Install) { throw "Julia $version is missing. Run setup.cmd at the repository root first." }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'OpenAeroLab requires a 64-bit operating system.' }
    $record = Get-Content -LiteralPath (Join-Path $Repository 'tools\install\runtimes.tsv') | Where-Object { $_.StartsWith('windows-x86_64|') }
    $parts = $record.Split('|')
    $tempRoot = Join-Path $env:LUDWIG_PRIVATE_ROOT 'TEMP_DELETEABLE\setup'
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    $staging = Join-Path $tempRoot ('julia-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    $archive = Join-Path $staging 'julia.zip'
    Write-Host "Downloading Julia $version from julialang.org..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $parts[1] -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $parts[2]) {
        throw "Julia download checksum mismatch. No files were installed; rerun setup. Download: $archive"
    }
    Write-Host 'Checksum verified. Extracting the private Julia runtime...'
    $unpacked = Join-Path $staging 'unpacked'
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
    $binary = Join-Path $unpacked "julia-$version\bin\julia.exe"
    if (-not (Test-Path -LiteralPath $binary)) { throw 'The Julia archive has an unexpected layout.' }
    New-Item -ItemType Directory -Force -Path (Split-Path $runtime) | Out-Null
    if (Test-Path -LiteralPath $runtime) { throw "An incomplete runtime already exists at $runtime. Rename it, then rerun setup." }
    # Both source and destination are fixed descendants of the selected private root.
    $privatePrefix = [IO.Path]::GetFullPath($env:LUDWIG_PRIVATE_ROOT).TrimEnd('\') + '\'
    foreach ($target in @($unpacked, $runtime)) {
        if (-not [IO.Path]::GetFullPath($target).StartsWith($privatePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime path escapes the private folder.' }
    }
    Move-Item -LiteralPath $unpacked -Destination $runtime
    Remove-Item -LiteralPath $archive
    return $managed
}
