#!/usr/bin/env python3
"""Audit native project-owned Panthera manipulation evidence."""

from __future__ import annotations

import argparse

from vla_robot_sorting.evidence import audit_project_manipulation_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe")
    args = parser.parse_args()

    result = audit_project_manipulation_probe(args.probe)
    for gate in result.gates:
        print(f"{'PASS' if gate.passed else 'FAIL'} {gate.name}: {gate.detail}")
    print(f"AUDIT={'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
