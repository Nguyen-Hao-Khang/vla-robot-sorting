#!/usr/bin/env python3
"""Audit a checksummed S1-M plan-only diagnostic export offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.planning.approach_safety_audit import (
    ApproachAuditError,
    audit_plan_only_trial_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a source-locked, checksummed S1-M plan-only export. "
            "The audit never grants controller or native task acceptance."
        )
    )
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_plan_only_trial_bundle(args.evidence_dir)
    except (ApproachAuditError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "s1m.approach-plan-only-bundle-audit.v1",
                    "classification": "PLAN_ONLY_DIAGNOSTIC_REJECTED",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "controller_eligible": False,
                    "native_acceptance": False,
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
