"""Offline audit helpers for an extracted S1.1 native-Ubuntu evidence directory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PANTHERA_SHA = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
ROBOTWIN_TREE_SHA = "3b0e2538a956c4c10c61c9311646a8d38e576598"
CUROBO_SHA = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"


@dataclass(frozen=True, slots=True)
class EvidenceGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class S11EvidenceAudit:
    root: Path
    gates: tuple[EvidenceGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[EvidenceGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _exact_file(root: Path, relative: str, expected: str) -> EvidenceGate:
    path = root / relative
    if not path.is_file():
        return EvidenceGate(relative, False, "missing")
    actual = _read(path)
    return EvidenceGate(relative, actual == expected, f"actual={actual}")


def _observation_gate(root: Path) -> EvidenceGate:
    path = root / "manifest" / "panthera-observation.json"
    if not path.is_file():
        return EvidenceGate("observation", False, "manifest/panthera-observation.json missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cameras = {item["camera_id"]: item for item in data["cameras"]}
        good = (
            data["schema"] == "panthera-observation-probe-v2"
            and data["planner_initialization"] == "bypassed-observation-only"
            and data["joint_names"] == [f"joint{i}" for i in range(1, 7)]
            and len(data["actual_joint_positions_rad"]) == 6
            and 0.0 <= float(data["actual_gripper_normalized"]) <= 1.0
            and data["proprioception_source"] == "Robot.get_left_arm_real_jointState"
            and set(cameras) == {"head_camera", "wrist_camera"}
            and all(
                int(item["width_px"]) > 0 and int(item["height_px"]) > 0
                for item in cameras.values()
            )
            and data["single_arm_setup"]["robot_articulation_count"] == 1
            and data["single_arm_setup"]["wrist_camera_count"] == 1
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return EvidenceGate("observation", False, f"invalid observation manifest: {exc}")
    return EvidenceGate("observation", good, "typed camera + actual-qpos contract")


def _torch_gate(root: Path) -> EvidenceGate:
    path = root / "manifest" / "torch-cuda.txt"
    if not path.is_file():
        return EvidenceGate("torch-cuda", False, "manifest/torch-cuda.txt missing")
    text = _read(path)
    required = ("torch_cuda= 12.8", "cuda_available= True", "capability= (7, 5)")
    return EvidenceGate("torch-cuda", all(item in text for item in required), text)


def _video_gate(root: Path) -> EvidenceGate:
    path = root / "manifest" / "video-probe.txt"
    if not path.is_file():
        return EvidenceGate("videos", False, "manifest/video-probe.txt missing")
    text = _read(path)
    blocks = text.count("--- ")
    widths = [int(value) for value in re.findall(r"^width=(\d+)$", text, flags=re.MULTILINE)]
    heights = [int(value) for value in re.findall(r"^height=(\d+)$", text, flags=re.MULTILINE)]
    frames = re.findall(r"^nb_frames=(.+)$", text, flags=re.MULTILINE)
    good = (
        blocks >= 2
        and len(widths) >= 2
        and len(heights) >= 2
        and all(widths)
        and all(heights)
        and len(frames) >= 2
    )
    return EvidenceGate("videos", good, f"streams={blocks}, widths={widths}, heights={heights}")


def _resource_gate(root: Path) -> EvidenceGate:
    path = root / "manifest" / "resource-summary.json"
    if not path.is_file():
        return EvidenceGate("resources", False, "manifest/resource-summary.json missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        good = (
            int(data["gpu_samples"]) >= 1
            and float(data["peak_vram_used_mib"]) >= 0
            and float(data["vram_total_mib"]) > 0
            and data.get("max_host_rss_kib") is not None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return EvidenceGate("resources", False, f"invalid resource summary: {exc}")
    return EvidenceGate("resources", good, json.dumps(data, sort_keys=True))


def _checksums_gate(root: Path) -> EvidenceGate:
    manifest = root / "checksums" / "SHA256SUMS.txt"
    if not manifest.is_file():
        return EvidenceGate("checksums", False, "checksums/SHA256SUMS.txt missing")
    checked = 0
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
        if match is None:
            failures.append(f"malformed:{line}")
            continue
        expected, raw_name = match.groups()
        relative = raw_name.removeprefix("*").removeprefix("./")
        path = root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual.lower() != expected.lower():
            failures.append(f"mismatch:{relative}")
    good = checked > 0 and not failures
    return EvidenceGate("checksums", good, f"checked={checked}, failures={failures}")


def audit_s1_1_evidence(root: Path) -> S11EvidenceAudit:
    """Audit one extracted evidence directory without assuming a Linux path."""

    root = root.resolve()
    gates = [
        _exact_file(root, "PASS", "PASS"),
        EvidenceGate(
            "no-failed-marker",
            not (root / "FAILED").exists(),
            "FAILED marker must be absent",
        ),
        _exact_file(root, "manifest/panthera-robotwin.sha", PANTHERA_SHA),
        _exact_file(root, "manifest/vendored-robotwin-tree.sha", ROBOTWIN_TREE_SHA),
        _exact_file(root, "manifest/curobo.sha", CUROBO_SHA),
        _torch_gate(root),
        _observation_gate(root),
        _video_gate(root),
        _resource_gate(root),
        _checksums_gate(root),
    ]
    return S11EvidenceAudit(root=root, gates=tuple(gates))
