$ErrorActionPreference = 'Stop'

$appDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExecutable = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python -ErrorAction Stop).Source }

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "找不到 Python 环境: $pythonExecutable"
}

& $pythonExecutable (Join-Path $appDirectory 'main.py') @args
exit $LASTEXITCODE
