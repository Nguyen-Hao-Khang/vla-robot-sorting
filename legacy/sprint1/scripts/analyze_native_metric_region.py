"""Inspect one operator-selected rectangular RGB-D region (not object pose).

Requires BOTH source-verified SAPIEN head/wrist camera artifacts.
Does not perform object detection or authorize robot motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Never silently import an editable install from a DIFFERENT DAMH worktree.
# Each checkout must run its own source; the source revision is a safety gate.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.perception.metric_pixel import MetricPixelError
from vla_robot_sorting.perception.metric_region import project_metric_region
from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--camera", required=True, choices=("head_camera", "wrist_camera"))
    parser.add_argument(
        "--bbox", required=True, nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument(
        "--output", type=Path, help="Optional NEW JSON path outside native evidence"
    )
    args = parser.parse_args()

    try:
        region = project_metric_region(
            args.evidence_dir, camera_id=args.camera, bbox_xyxy_px=tuple(args.bbox)
        )
        data = {
            "classification": "ROI_DEPTH_ONLY_NOT_OBJECT_POSE_NOT_TASK_ACCEPTANCE",
            "source_scope": "operator_selected_unsegmented_rectangle",
            "native_execution_performed_by_this_command": False,
            "object_identity_verified": False,
            "object_pose_verified": False,
            "controller_eligible": False,
            "region": asdict(region),
        }
        encoded = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            output = args.output.resolve()
            if output.is_relative_to(args.evidence_dir.resolve()):
                raise MetricPixelError("refuse writing inside immutable native evidence")
            if output.exists():
                raise MetricPixelError("refuse overwrite of existing analysis")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (MetricPixelError, DepthEvidenceError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"classification": "ROI_DEPTH_REJECTED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
