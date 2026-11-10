#!/usr/bin/env python3
"""Audit preserved seed2 approach evidence without modifying native records."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Pin imports to this checkout, not a possibly stale editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.planning.approach_safety_audit import (
    ApproachAuditError,
    audit_seed2_approach_failure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the checksummed S1-V4 seed2 export and summarize its "
            "plan-only approach collision failure."
        )
    )
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON report path outside the immutable evidence directory",
    )
    return parser


def run(evidence_dir: Path, output: Path | None = None) -> dict:
    evidence_root = evidence_dir.resolve()
    report = audit_seed2_approach_failure(evidence_root)
    if output is not None:
        output_path = output.resolve()
        if output_path == evidence_root or evidence_root in output_path.parents:
            raise ApproachAuditError("output path cannot be inside immutable evidence")
        if output_path.exists():
            raise ApproachAuditError("refusing to overwrite an existing audit report")
        if not output_path.parent.is_dir():
            raise ApproachAuditError("output parent directory must already exist")
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report["audit_report_path"] = str(output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args.evidence_dir, args.output)
    except (ApproachAuditError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "s1m.seed2-approach-failure-audit.v1",
                    "classification": "SEED2_APPROACH_AUDIT_REJECTED",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "controller_eligible": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
