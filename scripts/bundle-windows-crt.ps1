param([Parameter(Mandatory = $true)][string]$Stage)
$ErrorActionPreference = 'Stop'
$Stage = (Resolve-Path -LiteralPath $Stage).Path
$required = @('msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')

# Use Visual Studio's redistributable directory, never arbitrary DLLs from
# System32. App-local deployment preserves the per-user/portable installation.
# https://learn.microsoft.com/cpp/windows/deployment-in-visual-cpp
$redistRoots = @()
if ($env:VCToolsRedistDir) { $redistRoots += $env:VCToolsRedistDir }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio discovery failed' }
    if ($installation) {
        $redist = Join-Path $installation 'VC\Redist\MSVC'
        if (Test-Path -LiteralPath $redist) {
            $redistRoots += Get-ChildItem -LiteralPath $redist -Directory |
                Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
                Sort-Object { [version]$_.Name } -Descending |
                ForEach-Object { $_.FullName }
        }
    }
}
$crtDirectory = $null
foreach ($root in $redistRoots) {
    foreach ($directory in Get-ChildItem -Path (Join-Path $root 'x64\Microsoft.VC*.CRT') -Directory -ErrorAction SilentlyContinue) {
        $missing = @($required | Where-Object { !(Test-Path -LiteralPath (Join-Path $directory.FullName $_) -PathType Leaf) })
        if ($missing.Count -eq 0) { $crtDirectory = $directory.FullName; break }
    }
    if ($crtDirectory) { break }
}
if (!$crtDirectory) {
    throw 'Install the Visual Studio x64 C++ redistributable build component before packaging'
}

$manifest = @()
foreach ($file in Get-ChildItem -LiteralPath $crtDirectory -Filter '*.dll' -File) {
    $destination = Join-Path $Stage $file.Name
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    $manifest += [ordered]@{
        file = $file.Name
        version = $file.VersionInfo.FileVersion
        sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifestPath = Join-Path $Stage 'microsoft-crt-manifest.json'
[IO.File]::WriteAllText($manifestPath, (ConvertTo-Json -InputObject $manifest -Depth 3), (New-Object Text.UTF8Encoding $false))
$notice = @'
Microsoft Visual C++ Redistributable runtime libraries
Copyright Microsoft Corporation. All rights reserved.

These application-local runtime DLLs come from the Visual Studio x64
redistributable directory. The adjacent manifest records versions and SHA-256
digests. Application updates replace these copies along with the native runtime.

https://learn.microsoft.com/cpp/windows/deployment-in-visual-cpp
https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files
'@
[IO.File]::WriteAllText((Join-Path $Stage 'MICROSOFT-CRT-NOTICE.txt'), $notice, (New-Object Text.UTF8Encoding $false))

# A fresh process in the application directory proves that dependency loading
# uses the staged files even though the CI host also has the global runtime.
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (!(Test-Path -LiteralPath $compiler)) { throw 'The Windows packaging audit needs the .NET Framework C# compiler' }
$probe = Join-Path $Stage 'sayit-crt-probe.exe'
try {
    & $compiler /nologo /platform:x64 "/out:$probe" (Join-Path $PSScriptRoot 'windows-crt-probe.cs')
    if ($LASTEXITCODE -ne 0) { throw 'C++ runtime audit compilation failed' }
    & $probe (Join-Path $Stage 'runtime\native\lib\onnxruntime.dll')
    if ($LASTEXITCODE -ne 0) { throw 'Packaged ONNX Runtime did not use the application-local C++ runtime' }
} finally {
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
}
Write-Host "Verified application-local C++ runtime: $crtDirectory"
