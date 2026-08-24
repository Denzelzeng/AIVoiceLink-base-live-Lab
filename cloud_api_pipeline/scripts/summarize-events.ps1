param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$EventsPath
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
& $pythonExe (Join-Path $PSScriptRoot 'summarize_events.py') $EventsPath
exit $LASTEXITCODE
