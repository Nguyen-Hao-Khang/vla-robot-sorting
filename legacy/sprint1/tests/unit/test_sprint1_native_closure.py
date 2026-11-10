from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_sprint1_native_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sprint1_native_closure_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readiness(*, pending: int = 0, closure_ready: bool = True) -> dict:
    return {
        "schema": "sprint1-readiness-v1",
        "accepted": {f"gate-{index}": True for index in range(11)},
        "accepted_count": 11,
        "accepted_total": 11,
        "native_pending": {} if pending == 0 else {"manip": "pending"},
        "native_pending_count": pending,
        "native_closure_candidate": {
            "provided": closure_ready,
            "ready": closure_ready,
            "source": "provisional_candidate" if closure_ready else None,
            "classification": (
                "PASS_AUDITED_CANDIDATE" if closure_ready else None
            ),
            "linux_git_head": "abc123" if closure_ready else None,
            "promotion_required_on_windows": closure_ready,
        },
        "cloud_api_pending": {
            "gemini_live_semantic_evaluation": {
                "key_present_in_current_environment": False,
            }
        },
    }


def test_native_closure_audit_passes_provisional_candidate(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(_readiness()), encoding="utf-8")

    result = _load_module().audit_native_closure(path)

    assert result["passed"] is True
    assert result["classification"] == "PASS_CANDIDATE"
    assert all(result["checks"].values())
    assert result["windows_promotion_required"] is True


def test_native_closure_audit_rejects_remaining_native_pending(
    tmp_path: Path,
) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(_readiness(pending=1)), encoding="utf-8")

    result = _load_module().audit_native_closure(path)

    assert result["passed"] is False
    assert result["classification"] == "FAIL"
    assert result["checks"]["native_pending_cleared"] is False


def test_native_closure_audit_keeps_cloud_api_separate(tmp_path: Path) -> None:
    payload = _readiness()
    payload["cloud_api_pending"] = {}
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _load_module().audit_native_closure(path)

    assert result["passed"] is False
    assert result["checks"]["cloud_api_kept_separate"] is False


def test_native_closure_audit_accepts_durable_12_of_12_state(
    tmp_path: Path,
) -> None:
    payload = _readiness()
    payload["accepted"]["project-owned-manipulation"] = True
    payload["accepted_count"] = 12
    payload["accepted_total"] = 12
    payload["native_closure_candidate"] = {
        "provided": True,
        "ready": True,
        "source": "durable_manifest",
        "classification": "PASS_AUDITED",
        "linux_git_head": "abc123",
        "promotion_required_on_windows": False,
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _load_module().audit_native_closure(path)

    assert result["passed"] is True
    assert result["classification"] == "PASS_DURABLE"
    assert result["windows_promotion_required"] is False
