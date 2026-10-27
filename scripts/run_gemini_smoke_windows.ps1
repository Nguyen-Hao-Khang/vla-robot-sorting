# One Vietnamese/English semantic fixture request with real API, opt-in only.
# No secret values or raw request headers are written to output artifacts.
param(
    [switch]$AllowLiveApi,
    [ValidateSet('BC-001','BC-002')]
    [string]$CaseId = 'BC-002'
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = 'D:\DAMH-Bridge\envs\damh-windows\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Verified D: Python environment missing' }
Set-Location $Repo

if (-not $AllowLiveApi) {
    Write-Output 'LIVE_API_DISABLED: No network call made. Add -AllowLiveApi explicitly.'
    exit 0
}
$key = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','Process')
if ([string]::IsNullOrWhiteSpace($key)) {
    $key = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')
}
if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Output 'MISSING_KEY: Set GEMINI_API_KEY as a Windows User environment variable.'
    exit 2
}
# Make the key available only in this child process. Never print the value.
$env:GEMINI_API_KEY = $key
if ([string]::IsNullOrWhiteSpace($env:VLM_PRIMARY_MODEL)) {
    $env:VLM_PRIMARY_MODEL = 'gemini-robotics-er-2-preview'
}
if ([string]::IsNullOrWhiteSpace($env:VLM_FALLBACK_MODEL)) {
    $env:VLM_FALLBACK_MODEL = 'gemini-3.5-flash-lite'
}
$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$Output = Join-Path $Repo "experiments\results\gemini_smoke_$Stamp"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Results = Join-Path $Output 'results.jsonl'
$Summary = Join-Path $Output 'summary.json'

# The semantic evaluation runner can return 0 even when an API case failed.
# Always validate the recorded model response and structured decision.
& $Python -B scripts/run_bilingual_semantic_evaluation.py `
    --mode live --case-id $CaseId --max-cases 1 `
    --output-results $Results --output-summary $Summary | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output 'GEMINI_SMOKE_RUNNER_ERROR'
    exit 3
}
$records = @(Get-Content $Results | ForEach-Object { $_ | ConvertFrom-Json })
if ($records.Count -ne 1) {
    Write-Output 'GEMINI_SMOKE_FAIL: Expected exactly one record'
    exit 4
}
$r = $records[0]
if ($null -ne $r.error -or [string]::IsNullOrWhiteSpace($r.observed_behavior)) {
    Write-Output 'GEMINI_SMOKE_FAIL: API/semantic validation did not succeed'
    Write-Output ('Result file: ' + $Results)
    exit 5
}
Write-Output 'GEMINI_SMOKE_PASS'
Write-Output ('Case: ' + $r.case_id)
Write-Output ('Model: ' + $r.model_id)
Write-Output ('Fallback used: ' + $r.fallback_used)
Write-Output ('Decision: ' + $r.observed_behavior)
Write-Output ('Semantic result saved: ' + $Results)
Write-Output 'LIMITATION: Single-case API smoke only; not a 66-case benchmark or robot demo.'
