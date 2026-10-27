"""Offline audit of the native S1.2 Panthera project-motion planning probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class S12ProbeGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class S12MotionProbeAudit:
    probe_path: Path
    gates: tuple[S12ProbeGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[S12ProbeGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def _finite_numeric_sequence(value: Any, *, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(isfinite(item) for item in numbers)


def _joint_limit_gate(data: dict[str, Any]) -> S12ProbeGate:
    limits = data.get("joint_limits")
    if not isinstance(limits, dict):
        return S12ProbeGate("joint-limits", False, "joint_limits object missing")

    required = (
        "lower_position_rad",
        "upper_position_rad",
        "max_velocity_rad_s",
        "max_acceleration_rad_s2",
        "max_jerk_rad_s3",
    )
    if not all(_finite_numeric_sequence(limits.get(key), length=6) for key in required):
        return S12ProbeGate(
            "joint-limits",
            False,
            "joint limit arrays must contain six finite values",
        )

    lower = tuple(float(value) for value in limits["lower_position_rad"])
    upper = tuple(float(value) for value in limits["upper_position_rad"])
    velocity = tuple(float(value) for value in limits["max_velocity_rad_s"])
    acceleration = tuple(float(value) for value in limits["max_acceleration_rad_s2"])
    jerk = tuple(float(value) for value in limits["max_jerk_rad_s3"])
    good = (
        all(left < right for left, right in zip(lower, upper, strict=True))
        and all(value > 0 for value in velocity)
        and all(value > 0 for value in acceleration)
        and all(value > 0 for value in jerk)
    )
    return S12ProbeGate("joint-limits", good, str(limits.get("source", "unknown")))


def _probe_gate(data: dict[str, Any], label: str) -> S12ProbeGate:
    probes = data.get("probes")
    if not isinstance(probes, list):
        return S12ProbeGate(label, False, "probes array missing")
    matches = [probe for probe in probes if isinstance(probe, dict) and probe.get("label") == label]
    if len(matches) != 1:
        return S12ProbeGate(label, False, f"expected exactly one {label!r} probe")

    probe = matches[0]
    if not probe.get("succeeded"):
        failure = probe.get("failure")
        return S12ProbeGate(label, False, f"planner failed: {failure!r}")
    try:
        path_samples = int(probe["path_samples"])
        trajectory_samples = int(probe["trajectory_samples"])
        duration_s = float(probe["trajectory_duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        return S12ProbeGate(label, False, f"invalid numeric evidence: {exc}")

    violations = probe.get("violations")
    good = (
        bool(probe.get("joint_limit_safe"))
        and path_samples >= 2
        and trajectory_samples >= 2
        and isfinite(duration_s)
        and duration_s > 0
        and violations == []
    )
    detail = (
        f"path_samples={path_samples}, trajectory_samples={trajectory_samples}, "
        f"duration_s={duration_s}"
    )
    return S12ProbeGate(label, good, detail)


def audit_s1_2_motion_probe(probe_path: str | Path) -> S12MotionProbeAudit:
    path = Path(probe_path).resolve()
    gates: list[S12ProbeGate] = []
    if not path.is_file():
        return S12MotionProbeAudit(
            probe_path=path,
            gates=(S12ProbeGate("file", False, "probe JSON missing"),),
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return S12MotionProbeAudit(
            probe_path=path,
            gates=(S12ProbeGate("json", False, f"invalid JSON: {exc}"),),
        )

    gates.append(
        S12ProbeGate(
            "schema",
            data.get("schema") == "panthera-project-motion-probe-v1",
            f"schema={data.get('schema')!r}",
        )
    )
    gates.append(
        S12ProbeGate(
            "planner-class",
            data.get("planner_class") == "CuroboPlanner",
            f"planner_class={data.get('planner_class')!r}",
        )
    )
    gates.append(
        S12ProbeGate(
            "actual-qpos",
            _finite_numeric_sequence(data.get("actual_joint_positions_rad"), length=6),
            "requires six finite measured joint values",
        )
    )
    gates.append(_joint_limit_gate(data))
    gates.append(_probe_gate(data, "translation"))
    gates.append(_probe_gate(data, "orientation"))
    return S12MotionProbeAudit(probe_path=path, gates=tuple(gates))
