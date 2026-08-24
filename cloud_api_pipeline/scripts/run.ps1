$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourcePath = Join-Path $projectRoot 'src'
$vendorPath = Join-Path $projectRoot '.vendor'
$env:PYTHONPATH = "$vendorPath;$sourcePath"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python -ErrorAction Stop).Source }

& $pythonExe -m simultrans_baseline @args
exit $LASTEXITCODE
