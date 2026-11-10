"""Replay real RGB-D into the unified control gate without forging object identity."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from vla_robot_sorting.contracts.grasping import ParallelJawGripper
from vla_robot_sorting.contracts.semantic import SourceLanguage
from vla_robot_sorting.orchestration.unified_sprint1 import UnifiedSprint1Pipeline
from vla_robot_sorting.perception.native_depth_foreground_proposal import (
    evaluate_native_depth_foreground,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-rgbd-dir", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--language", choices=("vi", "en"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = args.native_rgbd_dir.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("refuse overwrite or source mutation")
    evidence = evaluate_native_depth_foreground(source)
    # The native foreground module deliberately returns no identified SceneSnapshot.
    # Therefore interpreting this RGB-D as an object/collision model is prohibited.
    if (
        evidence.get("controller_eligible") is not False
        or evidence.get("object_identity_verified") is not False
        or evidence.get("scene_snapshot_created") is not False
    ):
        raise ValueError("unexpected control claims from RGBD evidence")
    report = UnifiedSprint1Pipeline().run(
        command=args.command,
        language=SourceLanguage(args.language),
        scene=None,
        semantic=None,
        semantic_command_sha256=sha256(args.command.encode("utf-8")).hexdigest(),
        gripper=ParallelJawGripper(0.01, 0.09, 0.01, 0.02, 0.08),
    )
    payload = {
        "schema": "sprint1-native-rgbd-unified-preflight-v1",
        "source_dir": str(source),
        "rgbd_report_sha256": evidence["report_sha256"],
        "camera_count": evidence["camera_count"],
        "observation": "native_RGBD_two_camera_foreground_no_object_identity",
        "command": args.command,
        "language": args.language,
        "run": {**asdict(report), "status": report.status.value},
        "execution_attempted": False,
        "physics_steps": 0,
        "claim": "AUTHENTIC_RGBD_OBSERVATION_NOT_END_TO_END_NATIVE_VLA",
    }
    if report.executed or report.task_success_final:
        raise ValueError("observation-only gate cannot claim native execution")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("REAL_RGBD_UNIFIED_PREFLIGHT=" + report.status.value)
    print("camera_count=" + str(evidence["camera_count"]))
    print("native_execution_attempted=False")
    print("output=" + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
