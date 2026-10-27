from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "promote_manipulation_closure_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "promote_manipulation_closure_candidate_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_candidate(root: Path) -> Path:
    seeds = []
    for seed in (0, 1, 2):
        probe = root / f"manipulation-seed-{seed}.json"
        audit = root / f"manipulation-seed-{seed}-audit.log"
        probe.write_text(
            json.dumps(
                {
                    "passed": True,
                    "final_phase": "complete",
                    "task_success_final": True,
                }
            ),
            encoding="utf-8",
        )
        audit.write_text("PASS summary\nAUDIT=PASS\n", encoding="utf-8")
        seeds.append(
            {
                "seed": seed,
                "probe_relative": probe.name,
                "probe_sha256": _sha(probe),
                "audit_log_relative": audit.name,
                "audit_log_sha256": _sha(audit),
                "probe_passed": True,
                "independent_audit_passed": True,
                "final_phase": "complete",
                "task_success_final": True,
            }
        )
    candidate = root / "manipulation-closure-candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema": "native-manipulation-closure-candidate-v1",
                "gate_id": "S1.6-MANIPULATION",
                "classification": "PASS_AUDITED_CANDIDATE",
                "linux_git_head": "deadbeef",
                "result_dir": root.name,
                "fixed_seeds": [0, 1, 2],
                "probe_pass_count": 3,
                "independent_audit_pass_count": 3,
                "seeds": seeds,
                "promotion_required_on_windows": True,
            }
        ),
        encoding="utf-8",
    )
    return candidate


def test_promotion_rehashes_exported_evidence_before_accepting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    candidate = _prepare_candidate(tmp_path)

    manifest = module.promote(
        candidate,
        tmp_path,
        source_bridge_relative="incoming/session",
        review="docs/reviews/manipulation-pass.md",
        accepted_at="2026-10-08",
    )

    assert manifest["schema"] == "accepted-native-gate-v1"
    assert manifest["gate_id"] == "S1.6-MANIPULATION"
    assert manifest["classification"] == "PASS_AUDITED"
    assert manifest["accepted_at"] == "2026-10-08"
    assert manifest["source_bridge_relative"] == "incoming/session"
    assert manifest["linux_git_head"] == "deadbeef"
    assert manifest["fixed_seeds"] == [0, 1, 2]
    assert manifest["probe_pass_count"] == 3
    assert manifest["independent_audit_pass_count"] == 3
    assert len(manifest["candidate_sha256"]) == 64
    assert [row["seed"] for row in manifest["seeds"]] == [0, 1, 2]


def test_promotion_rejects_tampered_probe_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    candidate = _prepare_candidate(tmp_path)
    (tmp_path / "manipulation-seed-1.json").write_text(
        json.dumps(
            {
                "passed": True,
                "final_phase": "complete",
                "task_success_final": True,
                "tampered": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed 1 probe SHA256 mismatch"):
        module.promote(
            candidate,
            tmp_path,
            source_bridge_relative="incoming/session",
            review="docs/reviews/manipulation-pass.md",
            accepted_at="2026-10-08",
        )


def test_promotion_rejects_failed_windows_reaudit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    candidate = _prepare_candidate(tmp_path)

    def fake_audit(path: Path):
        if str(path).endswith("manipulation-seed-2.json"):
            return SimpleNamespace(
                passed=False,
                failures=(SimpleNamespace(name="summary"),),
            )
        return SimpleNamespace(passed=True, failures=())

    monkeypatch.setattr(module, "audit_project_manipulation_probe", fake_audit)

    with pytest.raises(ValueError, match="seed 2 independent re-audit failed"):
        module.promote(
            candidate,
            tmp_path,
            source_bridge_relative="incoming/session",
            review="docs/reviews/manipulation-pass.md",
            accepted_at="2026-10-08",
        )


def test_promotion_preserves_recovered_seed1_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    candidate = _prepare_candidate(tmp_path)

    recovery_label = "manipulation-seed-1-recovery-1"
    recovery_probe = tmp_path / f"{recovery_label}.json"
    recovery_audit = tmp_path / f"{recovery_label}-audit.log"
    recovery_probe.write_text(
        json.dumps(
            {
                "passed": True,
                "final_phase": "complete",
                "task_success_final": True,
                "B5": {
                    "selected": {
                        "candidate_id": "pillbottle:cyl:side:2:s-1:up75"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    recovery_audit.write_text("PASS summary\nAUDIT=PASS\n", encoding="utf-8")

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["seeds"][1] = {
        "seed": 1,
        "source_label": recovery_label,
        "recovered": True,
        "candidate_id": "pillbottle:cyl:side:2:s-1:up75",
        "probe_relative": recovery_probe.name,
        "probe_sha256": _sha(recovery_probe),
        "audit_log_relative": recovery_audit.name,
        "audit_log_sha256": _sha(recovery_audit),
        "probe_passed": True,
        "independent_audit_passed": True,
        "final_phase": "complete",
        "task_success_final": True,
    }
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    manifest = module.promote(
        candidate,
        tmp_path,
        source_bridge_relative="incoming/session",
        review="docs/reviews/manipulation-pass.md",
        accepted_at="2026-10-08",
    )

    seed1 = manifest["seeds"][1]
    assert seed1["seed"] == 1
    assert seed1["source_label"] == recovery_label
    assert seed1["recovered"] is True
    assert seed1["candidate_id"] == "pillbottle:cyl:side:2:s-1:up75"
    assert seed1["probe_relative"] == recovery_probe.name
    assert seed1["audit_log_relative"] == recovery_audit.name
