$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv before continuing."
}

uv sync --dev
uv run pytest
uv run ruff check .

Write-Host "Windows bootstrap and verification completed successfully."
