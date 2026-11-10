#!/usr/bin/env python3
"""Audit one native Sprint-1 A-007 safe-failure probe JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_robot_sorting.evidence import audit_s1_a007_safe_failure_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_json", type=Path)
    args = parser.parse_args()

    result = audit_s1_a007_safe_failure_probe(args.probe_json)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.name}: {gate.detail}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
