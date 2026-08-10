$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelName = "vits-piper-en_US-lessac-medium"
$Archive = Join-Path $AppDir "models\$ModelName.tar.bz2"
$ModelDir = Join-Path $AppDir "models\$ModelName"
$ModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$ModelName.tar.bz2"

New-Item -ItemType Directory -Force (Join-Path $AppDir "models") | Out-Null
if (-not (Test-Path (Join-Path $ModelDir "en_US-lessac-medium.onnx"))) {
    Write-Host "Downloading the default English voice (about 60 MB)..."
    Invoke-WebRequest -Uri $ModelUrl -OutFile $Archive
    tar -xjf $Archive -C (Join-Path $AppDir "models")
    Remove-Item $Archive
}

$Config = @{
    engine = "sherpa-onnx-vits"
    executable = (Join-Path $AppDir "runtime\bin\sherpa-onnx-offline-tts.exe")
    model = (Join-Path $ModelDir "en_US-lessac-medium.onnx")
    tokens = (Join-Path $ModelDir "tokens.txt")
    data_dir = (Join-Path $ModelDir "espeak-ng-data")
    provider = "cpu"
    num_threads = 4
    speaker_id = 0
}
$Config | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $AppDir "say-the-rest.json")

$ServiceExe = Join-Path $AppDir "say-the-rest-service.exe"
$ConfigPath = Join-Path $AppDir "say-the-rest.json"
$Action = New-ScheduledTaskAction -Execute $ServiceExe -Argument "--config `"$ConfigPath`"" -WorkingDirectory $AppDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "Say the Rest" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Private local text-to-speech service" -Force | Out-Null
$DesktopAction = New-ScheduledTaskAction -Execute (Join-Path $AppDir "say-the-rest-desktop.exe") -WorkingDirectory $AppDir
Register-ScheduledTask -TaskName "Say the Rest Desktop" -Action $DesktopAction -Trigger $Trigger -Settings $Settings -Description "Global offline text-to-speech shortcuts" -Force | Out-Null
Start-Process -FilePath $ServiceExe -ArgumentList @("--config", $ConfigPath) -WorkingDirectory $AppDir
Start-Process -FilePath (Join-Path $AppDir "say-the-rest-desktop.exe") -WorkingDirectory $AppDir

Write-Host "Setup complete. Say the Rest is running in your system tray."
