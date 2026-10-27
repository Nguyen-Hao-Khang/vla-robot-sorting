import hashlib
import json
from pathlib import Path

from vla_robot_sorting.evidence.s1_1 import (
    CUROBO_SHA,
    PANTHERA_SHA,
    ROBOTWIN_TREE_SHA,
    audit_s1_1_evidence,
)


def write_text(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_valid_evidence(root: Path) -> None:
    write_text(root, "PASS", "PASS\n")
    write_text(root, "manifest/panthera-robotwin.sha", PANTHERA_SHA + "\n")
    write_text(root, "manifest/vendored-robotwin-tree.sha", ROBOTWIN_TREE_SHA + "\n")
    write_text(root, "manifest/curobo.sha", CUROBO_SHA + "\n")
    write_text(
        root,
        "manifest/torch-cuda.txt",
        "torch_cuda= 12.8\ncuda_available= True\ncapability= (7, 5)\n",
    )

    observation = {
        "schema": "panthera-observation-probe-v2",
        "planner_initialization": "bypassed-observation-only",
        "joint_names": [f"joint{i}" for i in range(1, 7)],
        "actual_joint_positions_rad": [0.0] * 6,
        "actual_gripper_normalized": 0.5,
        "proprioception_source": "Robot.get_left_arm_real_jointState",
        "cameras": [
            {"camera_id": "head_camera", "width_px": 640, "height_px": 480},
            {"camera_id": "wrist_camera", "width_px": 320, "height_px": 240},
        ],
        "single_arm_setup": {
            "robot_articulation_count": 1,
            "wrist_camera_count": 1,
        },
    }
    write_text(root, "manifest/panthera-observation.json", json.dumps(observation))
    write_text(
        root,
        "manifest/video-probe.txt",
        "--- head.mp4\nwidth=640\nheight=480\nnb_frames=10\n"
        "--- wrist.mp4\nwidth=320\nheight=240\nnb_frames=10\n",
    )

    resource = {
        "gpu_samples": 3,
        "peak_vram_used_mib": 2048,
        "vram_total_mib": 6144,
        "max_host_rss_kib": 1024,
    }
    write_text(root, "manifest/resource-summary.json", json.dumps(resource))

    checksum_path = root / "checksums" / "SHA256SUMS.txt"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  ./{path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_complete_s1_1_evidence_passes(tmp_path: Path) -> None:
    make_valid_evidence(tmp_path)

    result = audit_s1_1_evidence(tmp_path)

    assert result.passed
    assert result.failures == ()


def test_checksum_corruption_fails_audit(tmp_path: Path) -> None:
    make_valid_evidence(tmp_path)
    write_text(tmp_path, "manifest/curobo.sha", "corrupted\n")

    result = audit_s1_1_evidence(tmp_path)

    assert not result.passed
    failed_names = {gate.name for gate in result.failures}
    assert "manifest/curobo.sha" in failed_names
    assert "checksums" in failed_names
