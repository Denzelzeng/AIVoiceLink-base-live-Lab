$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python -ErrorAction Stop).Source }

& $pythonExe -m unittest discover -s (Join-Path $projectRoot 'tests') -v
exit $LASTEXITCODE
