param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$EventsPath
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = 'D:\ProgramData\miniforge3\envs\aivoicelink\python.exe'
& $pythonExe (Join-Path $PSScriptRoot 'summarize_events.py') $EventsPath
exit $LASTEXITCODE
