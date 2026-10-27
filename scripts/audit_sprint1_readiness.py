#!/usr/bin/env python3
"""Build an honest Sprint-1 readiness report from durable evidence and local contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from vla_robot_sorting.evaluation.native_evidence import (
    accepted_a007_scenario_record,
    load_accepted_gate,
    s1_1_scenario_records,
)
from vla_robot_sorting.evaluation.semantic_corpus import load_corpus
from vla_robot_sorting.evaluation.sprint1 import (
    load_scenario_matrix,
    scenario_coverage_report,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments/scenarios/sprint1_matrix.json"
CORPUS = ROOT / "experiments/corpora/bilingual_commands.jsonl"
S11 = ROOT / "experiments/accepted/native_s1_1_20261005.json"
S12 = ROOT / "experiments/accepted/native_s1_2_20261006.json"
S13 = ROOT / "experiments/accepted/native_s1_3_20261007.json"
S16_INTEGRATED = ROOT / "experiments/accepted/native_s1_6_integrated_20261007.json"
S16_MANIPULATION = ROOT / "experiments/accepted/native_s1_6_manipulation.json"
A007 = ROOT / "experiments/accepted/native_s1_a007_20261006.json"
OFFLINE_RUNNER = ROOT / "scripts/run_sprint1_offline_scenarios.py"


def _offline_records() -> tuple[Any, ...]:
    spec = importlib.util.spec_from_file_location("sprint1_readiness_offline", OFFLINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load offline scenario runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.run())


def _load_manipulation_candidate(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise ValueError(f"manipulation closure candidate missing: {candidate_path}")
    raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manipulation closure candidate must be an object")
    if raw.get("schema") != "native-manipulation-closure-candidate-v1":
        raise ValueError("unsupported manipulation closure candidate schema")
    if raw.get("gate_id") != "S1.6-MANIPULATION":
        raise ValueError("manipulation closure candidate gate mismatch")
    if raw.get("classification") != "PASS_AUDITED_CANDIDATE":
        raise ValueError("manipulation closure candidate is not pass-audited")
    if raw.get("fixed_seeds") != [0, 1, 2]:
        raise ValueError("manipulation closure candidate must cover seeds 0,1,2")
    if raw.get("probe_pass_count") != 3:
        raise ValueError("manipulation closure candidate probe count is not 3")
    if raw.get("independent_audit_pass_count") != 3:
        raise ValueError("manipulation closure candidate audit count is not 3")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("manipulation closure candidate seeds are incomplete")
    observed = sorted(
        int(row.get("seed"))
        for row in seeds
        if isinstance(row, dict)
        and row.get("probe_passed") is True
        and row.get("independent_audit_passed") is True
        and row.get("final_phase") == "complete"
        and row.get("task_success_final") is True
    )
    if observed != [0, 1, 2]:
        raise ValueError("manipulation closure candidate seed evidence is incomplete")
    return raw


def _load_durable_manipulation_manifest() -> dict[str, Any] | None:
    if not S16_MANIPULATION.is_file():
        return None
    raw = load_accepted_gate(S16_MANIPULATION, gate_id="S1.6-MANIPULATION")
    if raw.get("classification") not in {"PASS", "PASS_AUDITED"}:
        raise ValueError("durable manipulation manifest is not accepted")
    if raw.get("fixed_seeds") != [0, 1, 2]:
        raise ValueError("durable manipulation manifest must cover seeds 0,1,2")
    if raw.get("probe_pass_count") != 3:
        raise ValueError("durable manipulation manifest probe count is not 3")
    if raw.get("independent_audit_pass_count") != 3:
        raise ValueError("durable manipulation manifest audit count is not 3")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("durable manipulation manifest seeds are incomplete")
    observed = sorted(
        int(row.get("seed"))
        for row in seeds
        if isinstance(row, dict)
        and row.get("final_phase") == "complete"
        and row.get("task_success_final") is True
        and isinstance(row.get("probe_sha256"), str)
        and len(row["probe_sha256"]) == 64
        and isinstance(row.get("audit_log_sha256"), str)
        and len(row["audit_log_sha256"]) == 64
    )
    if observed != [0, 1, 2]:
        raise ValueError("durable manipulation manifest seed evidence is incomplete")
    return raw


def build_report(
    manipulation_candidate: str | Path | None = None,
) -> dict[str, Any]:
    s11 = load_accepted_gate(S11, gate_id="S1.1")
    s12 = load_accepted_gate(S12, gate_id="S1.2")
    s13 = load_accepted_gate(S13, gate_id="S1.3")
    s16_integrated = load_accepted_gate(
        S16_INTEGRATED,
        gate_id="S1.6-INTEGRATED",
    )
    a007 = load_accepted_gate(A007, gate_id="S1-A-007")
    matrix = load_scenario_matrix(MATRIX)
    offline = _offline_records()
    corpus = load_corpus(CORPUS)
    fixed_records = (
        *offline,
        *s1_1_scenario_records(S11),
        accepted_a007_scenario_record(A007),
    )
    fixed_coverage = scenario_coverage_report(matrix, fixed_records)

    s12_gates = s12.get("gates")
    if not isinstance(s12_gates, dict):
        raise ValueError("accepted S1.2 manifest missing gates")

    non_native_ids = {
        str(item["scenario_id"])
        for item in matrix
        if not bool(item["native_required"])
    }
    offline_ids = {record.scenario_id for record in offline}
    accepted = {
        "S1.1_native_environment_observation_episode": (
            s11.get("classification") in {"PASS", "PASS_AUDITED"}
            and s11.get("observation", {}).get("passed") is True
            and s11.get("canonical_episode", {}).get("passed") is True
        ),
        "S1.2_motion_backends_timing": all(
            isinstance(s12_gates.get(gate), dict)
            and s12_gates[gate].get("passed") is True
            for gate in ("B1", "B2", "B3", "B4")
        ),
        "S1.3_target_sync_B5_B7": (
            s13.get("classification") in {"PASS", "PASS_AUDITED"}
            and isinstance(s13.get("target_sync"), dict)
            and s13["target_sync"].get("passed") is True
            and s13["target_sync"].get("passes") == 2
            and isinstance(s13.get("B5"), dict)
            and s13["B5"].get("passed") is True
            and s13["B5"].get("successes") == 20
            and isinstance(s13.get("B7"), dict)
            and s13["B7"].get("passed") is True
            and s13["B7"].get("successes") == 20
            and s13["B7"].get("stale_physics_steps_total") == 0
            and s13["B7"].get("replacement_passes") == 20
        ),
        "integrated_runtime_grounding_bounded_pregrasp": (
            s16_integrated.get("classification") in {"PASS", "PASS_AUDITED"}
            and isinstance(s16_integrated.get("runtime_grounding_pregrasp"), dict)
            and s16_integrated["runtime_grounding_pregrasp"].get("passed") is True
            and s16_integrated["runtime_grounding_pregrasp"].get("audited_passes") == 2
        ),
        "integrated_destination_moved_recovery": (
            isinstance(s16_integrated.get("destination_recovery"), dict)
            and s16_integrated["destination_recovery"].get("passed") is True
            and s16_integrated["destination_recovery"].get("decision_action")
            == ["replan", "replan"]
        ),
        "integrated_resource_profile": (
            isinstance(s16_integrated.get("resources"), dict)
            and s16_integrated["resources"].get("passed") is True
            and int(s16_integrated["resources"].get("samples", 0)) > 0
            and float(s16_integrated["resources"].get("peak_vram_mib", 1e12))
            < float(s16_integrated["resources"].get("vram_total_mib", 0))
        ),
        "offline_scenario_matrix_coverage": offline_ids == non_native_ids,
        "offline_scenarios_all_pass": (
            len(offline) == len(non_native_ids)
            and all(record.outcome.value == "pass" for record in offline)
        ),
        "bilingual_corpus_contract_66_cases": len(corpus) == 66,
        "S1_A_007_native_fail_closed": (
            a007.get("classification") in {"PASS", "PASS_AUDITED"}
            and isinstance(a007.get("probe"), dict)
            and a007["probe"].get("passed") is True
            and a007["probe"].get("executor_calls") == 0
            and a007["probe"].get("physics_steps") == 0
            and a007["probe"].get("qpos_unchanged") is True
        ),
        "fixed_12_case_matrix_complete": (
            fixed_coverage.get("complete") is True
            and fixed_coverage.get("outcomes") == {"pass": 12}
            and fixed_coverage.get("native_pass_count") == 3
            and fixed_coverage.get("pending_ids") == []
        ),
    }

    durable_manipulation = _load_durable_manipulation_manifest()
    manipulation = _load_manipulation_candidate(manipulation_candidate)
    manipulation_candidate_ready = manipulation is not None
    durable_manipulation_ready = durable_manipulation is not None
    if durable_manipulation_ready:
        accepted["project_owned_grasp_lift_place_sort"] = True

    native_pending = {}
    if not (manipulation_candidate_ready or durable_manipulation_ready):
        native_pending = {
            "project_owned_grasp_lift_place_sort": (
                "RUN_SPRINT1_MANIPULATION_CAMPAIGN.sh: project-owned manipulation "
                "seeds 0,1,2 + independent audits; accepted S1.3 and integrated "
                "manifests are prerequisites"
            ),
        }

    cloud_pending = {
        "gemini_live_semantic_evaluation": {
            "required_input": "GEMINI_API_KEY",
            "key_present_in_current_environment": bool(os.environ.get("GEMINI_API_KEY")),
            "prepared_command": (
                "uv run python scripts/run_bilingual_semantic_evaluation.py "
                "--mode live --output-results <jsonl> --output-summary <json>"
            ),
            "note": (
                "This is cloud/API work, not an Ubuntu-only gate. Do not reboot solely for it."
            ),
        }
    }

    return {
        "schema": "sprint1-readiness-v1",
        "accepted": accepted,
        "accepted_count": sum(accepted.values()),
        "accepted_total": len(accepted),
        "native_pending": native_pending,
        "native_pending_count": len(native_pending),
        "native_closure_candidate": {
            "provided": manipulation_candidate is not None or durable_manipulation_ready,
            "ready": manipulation_candidate_ready or durable_manipulation_ready,
            "source": (
                "durable_manifest"
                if durable_manipulation_ready
                else ("provisional_candidate" if manipulation_candidate_ready else None)
            ),
            "path": (
                str(S16_MANIPULATION.resolve())
                if durable_manipulation_ready
                else (
                    str(Path(manipulation_candidate).resolve())
                    if manipulation_candidate is not None
                    else None
                )
            ),
            "classification": (
                durable_manipulation.get("classification")
                if durable_manipulation_ready and durable_manipulation is not None
                else (
                    manipulation.get("classification")
                    if manipulation_candidate_ready and manipulation is not None
                    else None
                )
            ),
            "linux_git_head": (
                durable_manipulation.get("linux_git_head")
                if durable_manipulation_ready and durable_manipulation is not None
                else (
                    manipulation.get("linux_git_head")
                    if manipulation_candidate_ready and manipulation is not None
                    else None
                )
            ),
            "promotion_required_on_windows": (
                manipulation_candidate_ready and not durable_manipulation_ready
            ),
        },
        "cloud_api_pending": cloud_pending,
        "claim_boundary": (
            (
                "All native Sprint-1 gates, including project-owned manipulation, are backed "
                "by durable accepted manifests or deterministic offline contracts. Native "
                "Sprint-1 closure is durable; cloud/API evaluation remains tracked separately."
            )
            if durable_manipulation_ready
            else (
                "Accepted entries are backed by durable accepted manifests or deterministic "
                "offline contracts. B5/B7 and integrated runtime/recovery/resource evidence "
                "are accepted from the audited 2026-10-07 native campaign. A valid "
                "manipulation closure candidate can clear native_pending for provisional "
                "closure, but it must still be reviewed and promoted to a durable accepted "
                "manifest on Windows."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manipulation-candidate", type=Path)
    args = parser.parse_args()
    report = build_report(args.manipulation_candidate)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if all(report["accepted"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
