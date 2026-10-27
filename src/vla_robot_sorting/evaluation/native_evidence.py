"""Convert accepted native evidence into Sprint-1 evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    ResourceMetrics,
    ScenarioOutcome,
    ScenarioRecord,
)
from vla_robot_sorting.evidence import audit_s1_a007_safe_failure_probe


def load_accepted_gate(path: str | Path, *, gate_id: str) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("accepted native gate manifest must be an object")
    if raw.get("schema") != "accepted-native-gate-v1":
        raise ValueError("unsupported accepted native gate schema")
    if raw.get("gate_id") != gate_id:
        raise ValueError(
            f"accepted native gate mismatch: expected {gate_id}, got {raw.get('gate_id')}"
        )
    return raw


def _accepted_classification(data: dict[str, Any]) -> bool:
    return data.get("classification") in {"PASS", "PASS_AUDITED"}


def s1_1_scenario_records(path: str | Path) -> tuple[ScenarioRecord, ScenarioRecord]:
    data = load_accepted_gate(path, gate_id="S1.1")
    observation = data.get("observation")
    episode = data.get("canonical_episode")
    resources = data.get("resources")
    if not isinstance(observation, dict) or not isinstance(episode, dict):
        raise ValueError("S1.1 accepted manifest missing observation/episode")
    if not isinstance(resources, dict):
        resources = {}

    accepted = _accepted_classification(data)
    observation_pass = (
        accepted
        and observation.get("passed") is True
        and int(observation.get("actual_qpos_dof", 0)) == 6
        and isinstance(observation.get("head_rgb"), dict)
        and isinstance(observation.get("wrist_rgb"), dict)
    )
    episode_pass = (
        accepted
        and episode.get("passed") is True
        and int(episode.get("process_exit_status", -1)) == 0
        and int(episode.get("samples", 0)) > 0
        and int(episode.get("hdf5_bytes", 0)) > 0
        and int(episode.get("head_video_frames", 0)) > 0
        and int(episode.get("wrist_video_frames", 0)) > 0
    )

    common_metadata = {
        "accepted_gate": "S1.1",
        "classification": data.get("classification"),
        "review": data.get("review"),
        "source_bridge_relative": data.get("source_bridge_relative"),
        "linux_git_head": data.get("linux_git_head"),
        "raw_result_classification": data.get("raw_result_classification"),
        "accepted_override_reason": data.get("accepted_override_reason"),
    }
    r004 = ScenarioRecord(
        scenario_id="S1-R-004",
        suite="regression",
        runtime=EvaluationRuntime.NATIVE_SIM,
        outcome=ScenarioOutcome.PASS if observation_pass else ScenarioOutcome.FAIL,
        expected_behavior="observation_pass",
        observed_behavior="observation_pass" if observation_pass else None,
        failure_stage=None if observation_pass else "observation",
        failure_code=None if observation_pass else "accepted_evidence_invalid",
        metadata={
            **common_metadata,
            "observation": observation,
        },
    )

    peak_rss_kib = resources.get("peak_host_rss_kib")
    peak_rss_mib = (
        float(peak_rss_kib) / 1024.0
        if isinstance(peak_rss_kib, (int, float))
        else None
    )
    r005 = ScenarioRecord(
        scenario_id="S1-R-005",
        suite="regression",
        runtime=EvaluationRuntime.NATIVE_SIM,
        outcome=ScenarioOutcome.PASS if episode_pass else ScenarioOutcome.FAIL,
        expected_behavior="episode_success",
        observed_behavior="episode_success" if episode_pass else None,
        failure_stage=None if episode_pass else "canonical_episode",
        failure_code=None if episode_pass else "accepted_evidence_invalid",
        motion_succeeded=episode_pass,
        resources=ResourceMetrics(
            wall_time_s=(
                float(resources["wall_time_s"])
                if resources.get("wall_time_s") is not None
                else None
            ),
            peak_host_rss_mib=peak_rss_mib,
            peak_vram_mib=(
                float(resources["peak_vram_mib"])
                if resources.get("peak_vram_mib") is not None
                else None
            ),
            peak_gpu_utilization_percent=(
                float(resources["peak_gpu_utilization_percent"])
                if resources.get("peak_gpu_utilization_percent") is not None
                else None
            ),
        ),
        metadata={
            **common_metadata,
            "canonical_episode": episode,
        },
    )
    return r004, r005


def a007_scenario_record(probe_path: str | Path) -> ScenarioRecord:
    audit = audit_s1_a007_safe_failure_probe(probe_path)
    passed = audit.passed
    return ScenarioRecord(
        scenario_id="S1-A-007",
        suite="adversarial",
        runtime=EvaluationRuntime.NATIVE_SIM,
        outcome=ScenarioOutcome.PASS if passed else ScenarioOutcome.FAIL,
        expected_behavior="safe_failure",
        observed_behavior="safe_failure" if passed else None,
        failure_stage=None if passed else "planner_failure_safety",
        failure_code=None if passed else "a007_audit_failed",
        recovery_succeeded=passed,
        metadata={
            "probe_path": str(Path(probe_path)),
            "audit_gates": [
                {
                    "name": gate.name,
                    "passed": gate.passed,
                    "detail": gate.detail,
                }
                for gate in audit.gates
            ],
        },
    )


def accepted_a007_scenario_record(path: str | Path) -> ScenarioRecord:
    data = load_accepted_gate(path, gate_id="S1-A-007")
    probe = data.get("probe")
    if not isinstance(probe, dict):
        raise ValueError("S1-A-007 accepted manifest missing probe summary")

    accepted = _accepted_classification(data)
    passed = (
        accepted
        and probe.get("passed") is True
        and probe.get("scenario_id") == "S1-A-007"
        and int(probe.get("executor_calls", -1)) == 0
        and int(probe.get("physics_steps", -1)) == 0
        and probe.get("qpos_unchanged") is True
    )
    return ScenarioRecord(
        scenario_id="S1-A-007",
        suite="adversarial",
        runtime=EvaluationRuntime.NATIVE_SIM,
        outcome=ScenarioOutcome.PASS if passed else ScenarioOutcome.FAIL,
        expected_behavior="safe_failure",
        observed_behavior="safe_failure" if passed else None,
        failure_stage=None if passed else "planner_failure_safety",
        failure_code=None if passed else "accepted_a007_evidence_invalid",
        recovery_succeeded=passed,
        metadata={
            "accepted_gate": "S1-A-007",
            "classification": data.get("classification"),
            "review": data.get("review"),
            "source_bridge_relative": data.get("source_bridge_relative"),
            "linux_git_head": data.get("linux_git_head"),
            "probe_sha256": data.get("probe_sha256"),
            "audit_log_sha256": data.get("audit_log_sha256"),
            "probe": probe,
        },
    )


def pending_a007_record() -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id="S1-A-007",
        suite="adversarial",
        runtime=EvaluationRuntime.NATIVE_SIM,
        outcome=ScenarioOutcome.SKIPPED,
        expected_behavior="safe_failure",
        observed_behavior=None,
        metadata={
            "pending_reason": "native B5/B7/A-007 Ubuntu batch has not returned accepted evidence"
        },
    )
