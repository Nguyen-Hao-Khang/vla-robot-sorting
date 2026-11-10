from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    ScenarioOutcome,
)
from vla_robot_sorting.evaluation.native_evidence import (
    accepted_a007_scenario_record,
    s1_1_scenario_records,
)
from vla_robot_sorting.evaluation.sprint1 import (
    load_scenario_matrix,
    scenario_coverage_report,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "experiments" / "scenarios" / "sprint1_matrix.json"
S11 = ROOT / "experiments" / "accepted" / "native_s1_1_20261005.json"
A007 = ROOT / "experiments" / "accepted" / "native_s1_a007_20261006.json"
RUNNER = ROOT / "scripts" / "run_sprint1_offline_scenarios.py"
ASSEMBLER = ROOT / "scripts" / "build_sprint1_evaluation.py"


def _offline_records() -> tuple[object, ...]:
    spec = importlib.util.spec_from_file_location("offline_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.run())


def test_accepted_s1_1_converts_to_two_native_pass_records() -> None:
    r004, r005 = s1_1_scenario_records(S11)

    assert (r004.scenario_id, r005.scenario_id) == ("S1-R-004", "S1-R-005")
    assert r004.runtime is EvaluationRuntime.NATIVE_SIM
    assert r005.runtime is EvaluationRuntime.NATIVE_SIM
    assert r004.outcome is ScenarioOutcome.PASS
    assert r005.outcome is ScenarioOutcome.PASS
    assert r005.resources.peak_vram_mib == 2279.0
    assert r005.metadata["classification"] == "PASS_AUDITED"
    assert r005.metadata["raw_result_classification"] == "FAIL"


def test_current_matrix_coverage_is_complete_with_accepted_a007() -> None:
    matrix = load_scenario_matrix(MATRIX)
    records = (
        *_offline_records(),
        *s1_1_scenario_records(S11),
        accepted_a007_scenario_record(A007),
    )

    report = scenario_coverage_report(matrix, records)

    assert report["matrix_total"] == 12
    assert report["outcomes"] == {"pass": 12}
    assert report["runtimes"] == {"native_sim": 3, "offline_fake": 9}
    assert report["native_required_total"] == 3
    assert report["native_pass_count"] == 3
    assert report["native_pass_ids"] == ["S1-A-007", "S1-R-004", "S1-R-005"]
    assert report["pending_ids"] == []
    assert report["complete"] is True


def test_assembler_writes_honest_current_report(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["outcomes"] == {"pass": 12}
    assert report["coverage"]["pending_ids"] == []
    assert report["coverage"]["complete"] is True
    assert report["summary"]["native_sim_total"] == 3
    assert report["summary"]["offline_total"] == 9
    assert report["accepted_native_gates"]["S1-A-007"]["classification"] == "PASS_AUDITED"


def test_require_complete_uses_accepted_a007_evidence(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ASSEMBLER),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--require-complete",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_accepted_a007_converts_to_native_pass_record() -> None:
    record = accepted_a007_scenario_record(A007)

    assert record.scenario_id == "S1-A-007"
    assert record.runtime is EvaluationRuntime.NATIVE_SIM
    assert record.outcome is ScenarioOutcome.PASS
    assert record.recovery_succeeded is True
    assert record.metadata["classification"] == "PASS_AUDITED"
    assert record.metadata["probe"]["executor_calls"] == 0
    assert record.metadata["probe"]["physics_steps"] == 0
    assert record.metadata["probe"]["qpos_unchanged"] is True
