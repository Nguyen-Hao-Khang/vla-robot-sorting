#!/usr/bin/env python3
"""Audit an extracted S1.1 evidence directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_robot_sorting.evidence import audit_s1_1_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()

    result = audit_s1_1_evidence(args.evidence_dir)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.name}: {gate.detail}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
