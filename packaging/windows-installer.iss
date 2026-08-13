#ifndef Stage
  #error Stage must point to the staged release directory
#endif
#ifndef OutputDir
  #error OutputDir must point to the release output directory
#endif

[Setup]
AppId={{81EF8C0F-B9DE-45AE-90DF-302D59AEE36E}
AppName=sayIt
AppVersion=0.1.1
AppPublisher=a1denvalu3
DefaultDirName={localappdata}\Programs\sayIt
DefaultGroupName=sayIt
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=sayIt-Setup-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#Stage}\icons\icon.ico
UninstallDisplayIcon={app}\sayit-desktop.exe

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\sayIt"; Filename: "{app}\sayit-desktop.exe"
Name: "{group}\Uninstall sayIt"; Filename: "{uninstallexe}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"""; WorkingDir: "{app}"; Description: "Start sayIt and enable launch at sign-in"; Flags: postinstall waituntilterminated

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Get-Process sayit-service,sayit-desktop,say-the-rest-service,say-the-rest-desktop -ErrorAction SilentlyContinue | Stop-Process -Force; Unregister-ScheduledTask -TaskName 'sayIt','sayIt Desktop','Say the Rest','Say the Rest Desktop' -Confirm:$false -ErrorAction SilentlyContinue"""; Flags: runhidden; RunOnceId: "sayItCleanup"
