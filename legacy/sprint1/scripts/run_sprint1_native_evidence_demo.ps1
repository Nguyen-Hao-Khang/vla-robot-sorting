# Windows-only, read-only evidence replay; NEVER claims live RoboTwin or VLA integration.
param(
    [string]$BridgeRoot = 'D:\DAMH-Bridge',
    [string]$Output = '',
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    # The demo builder uses only the standard library; use the system interpreter.
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $command) { throw 'No Python available for evidence replay' }
    $python = $command.Source
}
if (-not $Output) {
    $Output = Join-Path $BridgeRoot 'incoming\sprint1-native-evidence-demo.html'
}
if (Test-Path $Output) {
    throw "Refusing overwrite of existing replay: $Output. Supply -Output with a new path."
}
$native = Join-Path $BridgeRoot 'incoming\sprint1_round2_20261015T030716Z\native-strict'
$vision = Join-Path $BridgeRoot 'incoming\s1v5_depth_repair_20261013T154351Z'
$accepted = Join-Path $repo 'experiments\accepted\native_s1_6_manipulation.json'
$env:PYTHONPATH = Join-Path $repo 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python -B (Join-Path $repo 'scripts\build_sprint1_native_evidence_demo.py') --native-strict $native --vision-export $vision --accepted-manifest $accepted --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Evidence replay validation failed' }
if ($Open) { Start-Process -FilePath $Output }
Write-Output "S1_DEMO_HISTORICAL_REPLAY_READY=$Output"
Write-Output 'LIVE_SIMULATION=false'
Write-Output 'ONLINE_IMAGE_TO_ACTION=false'
