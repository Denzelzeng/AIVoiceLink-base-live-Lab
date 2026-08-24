$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = 'D:\ProgramData\miniforge3\envs\aivoicelink\python.exe'

& $pythonExe -m simultrans_baseline @args
exit $LASTEXITCODE
