$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServiceExe = Join-Path $AppDir "say-the-rest-service.exe"
$Action = New-ScheduledTaskAction -Execute $ServiceExe -WorkingDirectory $AppDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Say the Rest" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Private local text-to-speech service" -Force | Out-Null
$DesktopAction = New-ScheduledTaskAction -Execute (Join-Path $AppDir "say-the-rest-desktop.exe") -WorkingDirectory $AppDir
Register-ScheduledTask -TaskName "Say the Rest Desktop" -Action $DesktopAction -Trigger $Trigger -Settings $Settings -Description "Global offline text-to-speech shortcuts" -Force | Out-Null
Start-Process -FilePath $ServiceExe -WorkingDirectory $AppDir
Start-Process -FilePath (Join-Path $AppDir "say-the-rest-desktop.exe") -WorkingDirectory $AppDir

Write-Host "Setup complete. Say the Rest is running in your system tray. Choose a model in onboarding to download it with integrity verification."
