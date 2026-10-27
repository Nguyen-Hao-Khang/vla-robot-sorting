"""Offline audit for native Sprint-1 A-007 planner-failure safety evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class S16ProbeGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class S1A007SafeFailureAudit:
    probe_path: Path
    gates: tuple[S16ProbeGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[S16ProbeGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def _finite_joint_vector(value: Any, *, length: int = 6) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(isfinite(item) for item in numbers)


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _case_gate(case: dict[str, Any], expected_fault: str) -> S16ProbeGate:
    good = (
        case.get("fault") == expected_fault
        and case.get("passed") is True
        and case.get("motion_cycle_status") == "planning_failed"
        and case.get("planner_id") == "a007-controlled-fault-injection"
        and case.get("planner_failure_code") == expected_fault
        and case.get("trajectory_created") is False
        and case.get("execution_created") is False
        and _zero(case.get("parameterizer_calls"))
        and _zero(case.get("path_safety_calls"))
        and _zero(case.get("trajectory_safety_calls"))
        and _zero(case.get("executor_calls"))
        and _zero(case.get("physics_steps"))
        and case.get("qpos_unchanged") is True
    )
    detail = (
        f"fault={case.get('fault')!r}, status={case.get('motion_cycle_status')!r}, "
        f"executor_calls={case.get('executor_calls')!r}, "
        f"physics_steps={case.get('physics_steps')!r}"
    )
    return S16ProbeGate(f"fault-{expected_fault}", good, detail)


def audit_s1_a007_safe_failure_probe(
    probe_path: str | Path,
) -> S1A007SafeFailureAudit:
    path = Path(probe_path).resolve()
    if not path.is_file():
        return S1A007SafeFailureAudit(
            path,
            (S16ProbeGate("file", False, "safe-failure JSON missing"),),
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return S1A007SafeFailureAudit(
            path,
            (S16ProbeGate("json", False, f"invalid JSON: {exc}"),),
        )
    if not isinstance(data, dict):
        return S1A007SafeFailureAudit(
            path,
            (S16ProbeGate("json-object", False, "top-level JSON must be an object"),),
        )

    gates: list[S16ProbeGate] = [
        S16ProbeGate(
            "schema",
            data.get("schema") == "panthera-native-safe-failure-v1",
            f"schema={data.get('schema')!r}",
        ),
        S16ProbeGate(
            "scenario-id",
            data.get("scenario_id") == "S1-A-007",
            f"scenario_id={data.get('scenario_id')!r}",
        ),
        S16ProbeGate(
            "actual-qpos",
            _finite_joint_vector(data.get("actual_joint_positions_rad")),
            "requires six finite measured arm joint values",
        ),
    ]

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        gates.append(S16ProbeGate("cases", False, "cases array missing"))
        return S1A007SafeFailureAudit(path, tuple(gates))

    by_fault = {
        item.get("fault"): item
        for item in raw_cases
        if isinstance(item, dict) and isinstance(item.get("fault"), str)
    }
    gates.append(
        S16ProbeGate(
            "case-set",
            len(raw_cases) == 2 and set(by_fault) == {"timeout", "collision"},
            f"faults={sorted(str(key) for key in by_fault)}",
        )
    )
    for fault in ("timeout", "collision"):
        case = by_fault.get(fault)
        if not isinstance(case, dict):
            gates.append(S16ProbeGate(f"fault-{fault}", False, "case missing"))
        else:
            gates.append(_case_gate(case, fault))

    gates.append(
        S16ProbeGate(
            "summary",
            data.get("pass") is True,
            f"pass={data.get('pass')!r}",
        )
    )
    return S1A007SafeFailureAudit(path, tuple(gates))
