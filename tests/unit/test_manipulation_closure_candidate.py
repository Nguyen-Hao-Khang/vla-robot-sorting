from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_manipulation_closure_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_manipulation_closure_candidate_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_seed(root: Path, seed: int, *, passed: bool = True) -> None:
    (root / f"manipulation-seed-{seed}.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "final_phase": "complete" if passed else "recovery_required",
                "task_success_final": True if passed else None,
            }
        ),
        encoding="utf-8",
    )
    (root / f"manipulation-seed-{seed}-audit.log").write_text(
        "PASS summary\nAUDIT=PASS\n",
        encoding="utf-8",
    )


def test_build_candidate_requires_three_passed_fixed_seeds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    for seed in (0, 1, 2):
        _write_seed(tmp_path, seed)

    candidate = module.build_candidate(tmp_path, linux_git_head="deadbeef")

    assert candidate["schema"] == "native-manipulation-closure-candidate-v1"
    assert candidate["gate_id"] == "S1.6-MANIPULATION"
    assert candidate["classification"] == "PASS_AUDITED_CANDIDATE"
    assert candidate["fixed_seeds"] == [0, 1, 2]
    assert candidate["probe_pass_count"] == 3
    assert candidate["independent_audit_pass_count"] == 3
    assert candidate["linux_git_head"] == "deadbeef"
    assert candidate["promotion_required_on_windows"] is True
    assert [row["seed"] for row in candidate["seeds"]] == [0, 1, 2]
    assert all(len(row["probe_sha256"]) == 64 for row in candidate["seeds"])
    assert all(len(row["audit_log_sha256"]) == 64 for row in candidate["seeds"])


def test_build_candidate_rejects_partial_probe(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    _write_seed(tmp_path, 0)
    _write_seed(tmp_path, 1, passed=False)
    _write_seed(tmp_path, 2)

    with pytest.raises(ValueError, match="seed 1 probe is not PASS"):
        module.build_candidate(tmp_path, linux_git_head="deadbeef")


def test_build_candidate_rejects_failed_independent_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    for seed in (0, 1, 2):
        _write_seed(tmp_path, seed)

    def fake_audit(path: Path):
        if str(path).endswith("manipulation-seed-1.json"):
            return SimpleNamespace(
                passed=False,
                failures=(SimpleNamespace(name="lift"),),
            )
        return SimpleNamespace(passed=True, failures=())

    monkeypatch.setattr(module, "audit_project_manipulation_probe", fake_audit)

    with pytest.raises(ValueError, match="seed 1 independent audit failed"):
        module.build_candidate(tmp_path, linux_git_head="deadbeef")


def test_build_candidate_rejects_missing_audit_pass_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    for seed in (0, 1, 2):
        _write_seed(tmp_path, seed)
    (tmp_path / "manipulation-seed-2-audit.log").write_text(
        "AUDIT=FAIL\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not report AUDIT=PASS"):
        module.build_candidate(tmp_path, linux_git_head="deadbeef")


def _write_named_probe(
    root: Path,
    label: str,
    *,
    passed: bool = True,
    candidate_id: str | None = None,
) -> None:
    payload = {
        "passed": passed,
        "final_phase": "complete" if passed else "recovery_required",
        "task_success_final": True if passed else None,
    }
    if candidate_id is not None:
        payload["B5"] = {"selected": {"candidate_id": candidate_id}}
    (root / f"{label}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (root / f"{label}-audit.log").write_text(
        "PASS summary\nAUDIT=PASS\n",
        encoding="utf-8",
    )


def _write_resolution(
    root: Path,
    *,
    seed1_label: str,
    recovered: bool = True,
    candidate_id: str | None = "candidate-1",
) -> Path:
    path = root / "manipulation-seed-resolution.json"
    rows = []
    for seed in (0, 1, 2):
        label = seed1_label if seed == 1 else f"manipulation-seed-{seed}"
        rows.append(
            {
                "seed": seed,
                "source_label": label,
                "probe_relative": f"{label}.json",
                "audit_log_relative": f"{label}-audit.log",
                "recovered": seed == 1 and recovered,
                "candidate_id": candidate_id if seed == 1 and recovered else None,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "manipulation-seed-resolution-v1",
                "policy": "deterministic-ranked-recovery-v1",
                "seeds": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_candidate_accepts_independently_audited_recovered_seed1(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    _write_seed(tmp_path, 0)
    _write_seed(tmp_path, 2)
    recovery_label = "manipulation-seed-1-recovery-1"
    recovery_candidate = "pillbottle:cyl:side:2:s-1:up75"
    _write_named_probe(
        tmp_path,
        recovery_label,
        candidate_id=recovery_candidate,
    )
    resolution = _write_resolution(
        tmp_path,
        seed1_label=recovery_label,
        candidate_id=recovery_candidate,
    )

    candidate = module.build_candidate(
        tmp_path,
        linux_git_head="deadbeef",
        seed_resolution=resolution,
    )

    seed1 = candidate["seeds"][1]
    assert seed1["seed"] == 1
    assert seed1["source_label"] == recovery_label
    assert seed1["recovered"] is True
    assert seed1["candidate_id"] == "pillbottle:cyl:side:2:s-1:up75"
    assert seed1["probe_relative"] == f"{recovery_label}.json"
    assert seed1["audit_log_relative"] == f"{recovery_label}-audit.log"
    assert len(seed1["probe_sha256"]) == 64
    assert len(seed1["audit_log_sha256"]) == 64


def test_build_candidate_reaudits_recovered_seed1(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    _write_seed(tmp_path, 0)
    _write_seed(tmp_path, 2)
    recovery_label = "manipulation-seed-1-recovery-1"
    _write_named_probe(
        tmp_path,
        recovery_label,
        candidate_id="candidate-1",
    )
    resolution = _write_resolution(tmp_path, seed1_label=recovery_label)

    def fake_audit(path: Path):
        if path.name == f"{recovery_label}.json":
            return SimpleNamespace(
                passed=False,
                failures=(SimpleNamespace(name="support-escape"),),
            )
        return SimpleNamespace(passed=True, failures=())

    monkeypatch.setattr(module, "audit_project_manipulation_probe", fake_audit)

    with pytest.raises(ValueError, match="seed 1 independent audit failed"):
        module.build_candidate(
            tmp_path,
            linux_git_head="deadbeef",
            seed_resolution=resolution,
        )


def test_seed_resolution_rejects_path_traversal(tmp_path: Path) -> None:
    module = _load_module()
    resolution = tmp_path / "manipulation-seed-resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "schema": "manipulation-seed-resolution-v1",
                "seeds": [
                    {
                        "seed": 0,
                        "probe_relative": "../outside.json",
                        "audit_log_relative": "manipulation-seed-0-audit.log",
                    },
                    {
                        "seed": 1,
                        "probe_relative": "manipulation-seed-1.json",
                        "audit_log_relative": "manipulation-seed-1-audit.log",
                    },
                    {
                        "seed": 2,
                        "probe_relative": "manipulation-seed-2.json",
                        "audit_log_relative": "manipulation-seed-2-audit.log",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resolve directly inside"):
        module.build_candidate(
            tmp_path,
            linux_git_head="deadbeef",
            seed_resolution=resolution,
        )


def _seed2_recovered_fixture(root: Path) -> Path:
    _write_seed(root, 0)
    _write_seed(root, 1)
    _write_seed(root, 2, passed=False)
    old = root / "manipulation-seed-2.json"
    old_data = json.loads(old.read_text(encoding="utf-8"))
    old_data["B5"] = {"selected": {"candidate_id": "grasp-original"}}
    old.write_text(json.dumps(old_data), encoding="utf-8")
    name = "manipulation-seed-2-ranked-alternative"
    alternative_id = "grasp-ranked-alternative"
    _write_named_probe(root, name, candidate_id=alternative_id)
    alternative = root / (name + ".json")
    data = json.loads(alternative.read_text(encoding="utf-8"))
    data["B5"]["ranking"] = [{"candidate_id": alternative_id, "eligible": True}]
    alternative.write_text(json.dumps(data), encoding="utf-8")
    resolution = root / "manipulation-seed-resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "schema": "manipulation-seed-resolution-v1",
                "policy": "deterministic-ranked-recovery-v1",
                "seeds": [
                    {
                        "seed": seed,
                        "source_label": name if seed == 2 else f"manipulation-seed-{seed}",
                        "probe_relative": (name if seed == 2 else f"manipulation-seed-{seed}")
                        + ".json",
                        "audit_log_relative": (name if seed == 2 else f"manipulation-seed-{seed}")
                        + "-audit.log",
                        "recovered": seed == 2,
                        "candidate_id": alternative_id if seed == 2 else None,
                    }
                    for seed in (0, 1, 2)
                ],
            }
        ),
        encoding="utf-8",
    )
    return resolution


def test_candidate_builder_accepts_audited_distinct_ranked_seed2(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    resolution = _seed2_recovered_fixture(tmp_path)
    result = module.build_candidate(tmp_path, linux_git_head="deadbeef", seed_resolution=resolution)
    assert result["seeds"][2]["recovered"] is True
    assert result["seeds"][2]["candidate_id"] == "grasp-ranked-alternative"


def test_candidate_builder_seed2_rejects_forged_candidate_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "audit_project_manipulation_probe",
        lambda path: SimpleNamespace(passed=True, failures=()),
    )
    resolution = _seed2_recovered_fixture(tmp_path)
    path = tmp_path / "manipulation-seed-2-ranked-alternative.json"
    original_data = json.loads(path.read_text(encoding="utf-8"))
    for bad in ("not_eligible", "same_as_original", "missing_original"):
        data = json.loads(json.dumps(original_data))
        if bad == "not_eligible":
            data["B5"]["ranking"][0]["eligible"] = False
        elif bad == "same_as_original":
            data["B5"]["selected"]["candidate_id"] = "grasp-original"
            data["B5"]["ranking"] = [{"candidate_id": "grasp-original", "eligible": True}]
            raw = json.loads(resolution.read_text(encoding="utf-8"))
            raw["seeds"][2]["candidate_id"] = "grasp-original"
            resolution.write_text(json.dumps(raw), encoding="utf-8")
        else:
            (tmp_path / "manipulation-seed-2.json").unlink()
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="seed 2 recovery"):
            module.build_candidate(tmp_path, linux_git_head="deadbeef", seed_resolution=resolution)
        if bad == "missing_original":
            break
