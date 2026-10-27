#!/usr/bin/env python3
"""Audit native integrated grounding/pregrasp/recovery probe JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_robot_sorting.evidence import audit_integrated_pregrasp_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_json", type=Path)
    args = parser.parse_args()
    result = audit_integrated_pregrasp_probe(args.probe_json)
    for gate in result.gates:
        print(f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
