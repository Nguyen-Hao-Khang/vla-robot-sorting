# Reproducible Windows-only midpoint checkpoint; do not claim native pick/place PASS.
param(
    [string]$OutputDir = "",
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$ruff = Join-Path $repo ".venv\Scripts\ruff.exe"
if (-not (Test-Path $python -PathType Leaf)) {
    # Worktrees reuse the canonical Windows checkout virtual environment.
    $gitCommon = (git -C $repo rev-parse --path-format=absolute --git-common-dir).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Git common directory missing" }
    $canonicalRepo = Split-Path $gitCommon -Parent
    $python = Join-Path $canonicalRepo ".venv\Scripts\python.exe"
    $ruff = Join-Path $canonicalRepo ".venv\Scripts\ruff.exe"
}
if (-not (Test-Path $python -PathType Leaf)) {
    throw "Missing Windows .venv Python. Run documented bootstrap."
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repo "experiments\results\midterm_20261011_demo"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_ADDOPTS = "-p no:cacheprovider"
Push-Location $repo
try {
    if (-not $SkipTests) {
        & $ruff check --no-cache .
        if ($LASTEXITCODE -ne 0) { throw "Ruff regression failed" }
        & $python -B -m pytest -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Pytest regression failed" }
    }
    & $python -B scripts/run_sprint1_offline_scenarios.py --output (Join-Path $OutputDir "offline-scenarios.json")
    if ($LASTEXITCODE -ne 0) { throw "Offline scenarios failed" }
    & $python -B scripts/build_sprint1_evaluation.py --output-dir (Join-Path $OutputDir "evaluation") --require-complete
    if ($LASTEXITCODE -ne 0) { throw "Fixed 12-case evaluation failed" }
    & $python -B scripts/audit_sprint1_readiness.py --output (Join-Path $OutputDir "readiness.json")
    if ($LASTEXITCODE -ne 0) { throw "Readiness audit failed" }

    $offline = @(Get-Content (Join-Path $OutputDir "offline-scenarios.json") |
        Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
    $coverage = Get-Content (Join-Path $OutputDir "evaluation\coverage.json") -Raw | ConvertFrom-Json
    $ready = Get-Content (Join-Path $OutputDir "readiness.json") -Raw | ConvertFrom-Json
    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Git provenance unavailable" }

    if ($offline.Count -ne 9 -or @($offline | Where-Object { $_.outcome -ne "pass" }).Count -ne 0) {
        throw "Offline suite is not exact 9/9 PASS"
    }
    if ($coverage.record_total -ne 12 -or $coverage.outcomes.pass -ne 12 -or
        $coverage.runtimes.offline_fake -ne 9 -or $coverage.runtimes.native_sim -ne 3 -or
        $coverage.complete -ne $true) {
        throw "Evaluation matrix is not exact 12/12 (9 offline, 3 accepted native)"
    }
    $durable = ($ready.native_closure_candidate.source -eq "durable_manifest" -and
        $ready.native_closure_candidate.classification -eq "PASS_AUDITED" -and
        $ready.native_pending_count -eq 0)
    $expectedAccepted = if ($durable) { 12 } else { 11 }
    $expectedPending = if ($durable) { 0 } else { 1 }
    if ($ready.accepted_count -ne $expectedAccepted -or
        $ready.accepted_total -ne $expectedAccepted -or
        $ready.native_pending_count -ne $expectedPending) {
        throw "Unexpected readiness; review rather than show stale success claims"
    }
    $nativeStatus = if ($durable) { "PASS_NATIVE_3_OF_3_NOT_FULL_VLA" } else { "PENDING_NATIVE_3_OF_3" }
    $summary = [ordered]@{
        schema = "midterm-checkpoint-v1"
        evidence_date = "2026-10-10"
        git_head = $head
        windows_offline_passed = 9
        fixed_matrix_total = 12
        fixed_matrix_passed = 12
        fixed_matrix_native_accepted = 3
        fixed_matrix_offline = 9
        accepted_baseline_gates = 11
        accepted_baseline_total = 11
        manipulation_end_to_end = $nativeStatus
        native_readiness_gates = $ready.accepted_count
        native_readiness_total = $ready.accepted_total
        live_vla_image_to_action = "NOT_ACCEPTED"
        live_gemini = "PENDING_API"
        native_manipulation_evidence_date = if ($durable) { "2026-10-15" } else { $null }
        claim_boundary = "The 12-case scenario matrix is separate from the 3-seed physical task and never proves full live image-to-action VLA."
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "MIDTERM_SUMMARY.json") -Encoding UTF8
    Write-Output "MIDTERM_DEMO_VALIDATED"
    Write-Output "Git HEAD: $head"
    Write-Output "Offline: 9/9 PASS; 12-case matrix: 12/12 PASS (9 offline + 3 accepted native)"
    Write-Output "Baseline: 11/11 accepted; native readiness: $($ready.accepted_count)/$($ready.accepted_total); manipulation: $nativeStatus"
    Write-Output "Live Gemini/VLA image-to-action: NOT ACCEPTED"
    Write-Output "Output: $OutputDir"
}
finally {
    Pop-Location
}
