"""Credential preflight contract: no network or secret disclosure."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "scripts/check_sprint1_environment.py"
)
SMOKE = (
    Path(__file__).resolve().parents[2] / "scripts/run_gemini_smoke_windows.ps1"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sprint1_env_preflight", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_never_includes_key_material(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_git", lambda *args: (
        "" if args[0] == "status" else "main" if args[0] == "branch" else "1234abcd"
    ))
    key = "fictional-example-secret-" + "do-not-print"
    report = module.make_report(environ={
        "GEMINI_API_KEY": key,
        "GOOGLE_API_KEY": "another-secret",
        "VLM_PRIMARY_MODEL": "model-a",
        "VLM_FALLBACK_MODEL": "model-b",
    })
    assert report["network_requests"] == 0
    assert report["secrets_printed"] is False
    assert report["gates"]["gemini_live_response"] == "NOT_RUN"
    assert report["environment"]["gemini_key_present_in_process"] is True
    assert report["environment"]["ambiguous_sdk_key_precedence"] is True
    assert report["environment"]["repo_dotenv_autoloaded"] is False
    assert key not in str(report)
    assert "another-secret" not in str(report)


def test_preflight_missing_api_key_does_not_block_offline_gate(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_git", lambda *args: (
        "" if args[0] == "status" else "main" if args[0] == "branch" else "1234abcd"
    ))
    report = module.make_report(environ={})
    assert report["gates"]["gemini_credentials"] == "AWAIT_USER_KEY"
    assert report["gates"]["gemini_live_response"] == "NOT_RUN"
    assert report["gates"]["ubuntu_native"] == "NOT_RUN"
    assert report["gates"]["sprint1_end_to_end"] == "NOT_ACCEPTED"


def test_live_smoke_opt_in_and_failure_is_not_success():
    source = SMOKE.read_text(encoding="utf-8")
    assert "if (-not $AllowLiveApi)" in source
    assert "LIVE_API_DISABLED" in source
    assert "'GEMINI_API_KEY','User'" in source
    assert "if ($records.Count -ne 1)" in source
    assert "$null -ne $r.error" in source
    assert "$r.observed_behavior" in source
    assert "--max-cases 1" in source
    assert "GEMINI_SMOKE_PASS" in source
    assert "--mode live" in source
    assert "$key" not in source.split("Write-Output 'GEMINI_SMOKE_PASS'")[1]
