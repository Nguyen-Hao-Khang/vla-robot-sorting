"""Read-only offline RGB rectangle proposals on source-audited native RGB-D.

A proposal from THIS capture has native manifest/depth provenance, unlike a
manual HDF5 annotation. This script never labels objects or projects pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError
from vla_robot_sorting.perception.native_metric_quality import (
    NativeMetricQualityError,
)
from vla_robot_sorting.perception.native_rgbd_proposals import (
    generate_native_rgbd_proposals,
)
from vla_robot_sorting.perception.region_proposals import RegionProposalError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_depth_dir", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file OUTSIDE immutable native export")
    parser.add_argument("--max-proposals", type=int, default=500)
    parser.add_argument("--min-area-px", type=int, default=64)
    args = parser.parse_args()
    try:
        directory = args.native_depth_dir.resolve()
        output = args.output.resolve() if args.output else None
        if output is not None:
            if output == directory or directory in output.parents:
                raise RegionProposalError("output cannot mutate original native RGB-D evidence")
            if output.exists():
                raise RegionProposalError("refuse to overwrite existing analysis")
        # Each call checks the independent TWO-camera metric quality before
        # using any color; no fake episode ID or implicit sensor-frame fusion.
        cameras = [
            generate_native_rgbd_proposals(
                directory,
                camera_id=name,
                max_proposals=args.max_proposals,
                min_area_px=args.min_area_px,
            )
            for name in ("head_camera", "wrist_camera")
        ]
        value = {
            "schema": "damh.native-rgbd-unlabelled-region-proposals.v1",
            "status": "READ_ONLY_SOURCE_ALIGNED_REGION_PROPOSALS_NOT_OBJECT_POSE",
            "native_depth_dir": str(directory),
            "native_capture_performed_by_this_command": False,
            "pretrained_weights_used": False,
            "automatic_object_detection": False,
            "object_identity_verified": False,
            "object_pose_verified": False,
            "grasp_validated": False,
            "controller_eligible": False,
            "claim_boundary": (
                "2D proposals share the manifest of THIS native RGB/depth capture; "
                "no identity, pose, segmentation, or control."
            ),
            "cameras": [asdict(item) for item in cameras],
        }
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as f:
                f.write(text)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "source_cameras": len(cameras),
                    "proposal_counts": {cam.camera_id: len(cam.proposals) for cam in cameras},
                    "output": str(output) if output else None,
                    "controller_eligible": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        RegionProposalError,
        NativeMetricQualityError,
        DepthEvidenceError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(json.dumps({"status": "NATIVE_RGBD_PROPOSALS_REJECTED", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
