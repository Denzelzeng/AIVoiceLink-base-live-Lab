$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = 'D:\ProgramData\miniforge3\envs\aivoicelink\python.exe'
$vendorDir = Join-Path $projectRoot '.vendor'
$modelDir = Join-Path $projectRoot 'models'

New-Item -ItemType Directory -Force -Path $vendorDir, $modelDir | Out-Null

& $pythonExe -m pip install --upgrade --target $vendorDir 'sherpa-onnx==1.13.6'
if ($LASTEXITCODE -ne 0) {
    throw 'sherpa-onnx installation failed'
}

function Install-VerifiedModel {
    param(
        [Parameter(Mandatory)] [string] $Uri,
        [Parameter(Mandatory)] [string] $Destination,
        [Parameter(Mandatory)] [string] $Sha256
    )

    if (Test-Path -LiteralPath $Destination) {
        $existing = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash
        if ($existing -eq $Sha256) {
            Write-Host "Model ready: $Destination"
            return
        }
    }

    $download = "$Destination.download"
    Invoke-WebRequest -Uri $Uri -OutFile $download
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $download).Hash
    if ($actual -ne $Sha256) {
        Remove-Item -LiteralPath $download -Force
        throw "Model checksum mismatch for $Uri"
    }
    Move-Item -LiteralPath $download -Destination $Destination -Force
    Write-Host "Installed model: $Destination"
}

Install-VerifiedModel `
    -Uri 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx' `
    -Destination (Join-Path $modelDir 'silero_vad.onnx') `
    -Sha256 '9E2449E1087496D8D4CABA907F23E0BD3F78D91FA552479BB9C23AC09CBB1FD6'

Install-VerifiedModel `
    -Uri 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx' `
    -Destination (Join-Path $modelDir 'campplus_zh_en.onnx') `
    -Sha256 'AA3CFC16963A10586A9393F5035D6D6B57E98D358B347F80C2A30BF4F00CEBA2'

Write-Host 'Local VAD and speaker-change models are ready.'
