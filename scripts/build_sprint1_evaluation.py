#!/usr/bin/env python3
"""Assemble current Sprint-1 offline + accepted native scenario evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from vla_robot_sorting.evaluation.aggregate import (
    record_to_dict,
    summarize,
    summary_to_dict,
)
from vla_robot_sorting.evaluation.native_evidence import (
    a007_scenario_record,
    accepted_a007_scenario_record,
    load_accepted_gate,
    s1_1_scenario_records,
)
from vla_robot_sorting.evaluation.sprint1 import (
    load_scenario_matrix,
    scenario_coverage_report,
    validate_records_against_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments" / "scenarios" / "sprint1_matrix.json"
S11_ACCEPTED = ROOT / "experiments" / "accepted" / "native_s1_1_20261005.json"
S12_ACCEPTED = ROOT / "experiments" / "accepted" / "native_s1_2_20261006.json"
A007_ACCEPTED = ROOT / "experiments" / "accepted" / "native_s1_a007_20261006.json"
OFFLINE_RUNNER = ROOT / "scripts" / "run_sprint1_offline_scenarios.py"


def _offline_records() -> tuple[Any, ...]:
    spec = importlib.util.spec_from_file_location(
        "sprint1_offline_runner_for_assembly",
        OFFLINE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Sprint-1 offline runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.run()
    return tuple(records)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record_to_dict(record),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a007-probe", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_scenario_matrix(MATRIX)
    offline = _offline_records()
    native_s11 = s1_1_scenario_records(S11_ACCEPTED)
    a007 = (
        a007_scenario_record(args.a007_probe)
        if args.a007_probe is not None
        else accepted_a007_scenario_record(A007_ACCEPTED)
    )
    records = (*offline, *native_s11, a007)
    validate_records_against_matrix(matrix, records)

    coverage = scenario_coverage_report(matrix, records)
    summary = summary_to_dict(summarize(records))
    s11 = load_accepted_gate(S11_ACCEPTED, gate_id="S1.1")
    s12 = load_accepted_gate(S12_ACCEPTED, gate_id="S1.2")
    a007_accepted = load_accepted_gate(A007_ACCEPTED, gate_id="S1-A-007")

    output_dir = args.output_dir.resolve()
    records_path = output_dir / "scenario-records.jsonl"
    summary_path = output_dir / "summary.json"
    coverage_path = output_dir / "coverage.json"
    _write_jsonl(records_path, records)
    _write_json(summary_path, summary)
    _write_json(coverage_path, coverage)

    report = {
        "schema": "sprint1-evaluation-assembly-v1",
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "coverage_path": str(coverage_path),
        "accepted_native_gates": {
            "S1.1": {
                "classification": s11["classification"],
                "review": s11["review"],
                "source_bridge_relative": s11["source_bridge_relative"],
            },
            "S1.2": {
                "classification": s12["classification"],
                "review": s12["review"],
                "source_bridge_relative": s12["source_bridge_relative"],
            },
            "S1-A-007": {
                "classification": a007_accepted["classification"],
                "review": a007_accepted["review"],
                "source_bridge_relative": a007_accepted["source_bridge_relative"],
            },
        },
        "coverage": coverage,
        "summary": summary,
        "claim_boundary": (
            "Offline records prove project logic only. R-004/R-005 use accepted native "
            "S1.1 evidence and A-007 uses its accepted independently audited native "
            "fail-closed evidence unless an explicit --a007-probe overrides it. "
            "B5/B7 are native research gates outside the fixed 12-scenario matrix."
        ),
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))

    if args.require_complete and not coverage["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
