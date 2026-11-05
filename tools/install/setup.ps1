# Compatibility entry point. Normal Windows setup uses setup.cmd directly and
# does not load unsigned PowerShell scripts or change execution policy.
$launcher = Join-Path $PSScriptRoot '..\..\setup.cmd'
& $launcher @args
exit $LASTEXITCODE
