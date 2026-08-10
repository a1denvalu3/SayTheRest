#ifndef Stage
  #error Stage must point to the staged release directory
#endif
#ifndef OutputDir
  #error OutputDir must point to the release output directory
#endif

[Setup]
AppId={{81EF8C0F-B9DE-45AE-90DF-302D59AEE36E}
AppName=Say the Rest
AppVersion=0.1.0
AppPublisher=a1denvalu3
DefaultDirName={localappdata}\Programs\Say the Rest
DefaultGroupName=Say the Rest
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SayTheRest-Setup-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#Stage}\icons\icon.ico
UninstallDisplayIcon={app}\say-the-rest-desktop.exe

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Say the Rest"; Filename: "{app}\say-the-rest-desktop.exe"
Name: "{group}\Uninstall Say the Rest"; Filename: "{uninstallexe}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"""; WorkingDir: "{app}"; Description: "Download the default voice and start Say the Rest"; Flags: postinstall skipifsilent waituntilterminated

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Get-Process say-the-rest-service,say-the-rest-desktop -ErrorAction SilentlyContinue | Stop-Process -Force; Unregister-ScheduledTask -TaskName 'Say the Rest','Say the Rest Desktop' -Confirm:$false -ErrorAction SilentlyContinue"""; Flags: runhidden; RunOnceId: "SayTheRestCleanup"
