#!/usr/bin/env python3
"""Read-only native RGB-D proposal-to-observed-pixel replay.

Converts checked SAME native capture unlabelled 2D rectangles into auditable
observed pixel points. This NEVER estimates an object pose or grasp goal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.perception.metric_pixel import MetricPixelError
from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError
from vla_robot_sorting.perception.native_metric_quality import (
    NativeMetricQualityError,
)
from vla_robot_sorting.perception.native_region_depth_association import (
    associate_native_regions_with_observed_depth,
)
from vla_robot_sorting.perception.native_rgbd_proposals import NativeRGBDProposalSet
from vla_robot_sorting.perception.region_proposals import (
    RegionProposal,
    RegionProposalError,
)

EXPECTED_SCHEMA = "damh.native-rgbd-unlabelled-region-proposals.v1"


def analyze_source(native_dir: Path, proposal_json: Path, *, max_regions: int) -> dict:
    """Validate all source identities and independently replay bounded depth."""
    if type(max_regions) is not int or not 1 <= max_regions <= 32:
        raise RegionProposalError("max_regions must be in [1, 32]")
    proposal_bytes = proposal_json.read_bytes()
    payload = json.loads(proposal_bytes)
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_SCHEMA:
        raise RegionProposalError("proposal JSON schema mismatch")
    if payload.get("status") != "READ_ONLY_SOURCE_ALIGNED_REGION_PROPOSALS_NOT_OBJECT_POSE":
        raise RegionProposalError("proposal status/claim boundary mismatch")
    for key in (
        "native_capture_performed_by_this_command",
        "pretrained_weights_used",
        "automatic_object_detection",
        "object_identity_verified",
        "object_pose_verified",
        "grasp_validated",
        "controller_eligible",
    ):
        if payload.get(key) is not False:
            raise RegionProposalError(f"proposal falsely claims {key}")
    frames = payload.get("cameras")
    if not isinstance(frames, list) or len(frames) != 2:
        raise RegionProposalError("proposal JSON must contain exactly two cameras")
    results: list[dict] = []
    for index, camera_id in enumerate(("head_camera", "wrist_camera")):
        raw = frames[index]
        if not isinstance(raw, dict) or raw.get("camera_id") != camera_id:
            raise RegionProposalError("camera order or identity mismatch")
        items = raw.get("proposals")
        if not isinstance(items, list) or not items:
            raise RegionProposalError("proposal list absent or empty")
        source = dict(raw)
        source["proposals"] = tuple(RegionProposal(**item) for item in items)
        typed = NativeRGBDProposalSet(**source)
        measured = associate_native_regions_with_observed_depth(
            typed, native_dir, max_regions=max_regions
        )
        results.append(
            {
                "camera_id": camera_id,
                "source_manifest_sha256": typed.native_depth_manifest_sha256,
                "source_rgb_sha256": typed.rgb_sha256,
                "source_depth_sha256": typed.depth_sha256,
                "proposal_count": len(typed.proposals),
                "sampled_count": min(len(typed.proposals), max_regions),
                "observed_depth_point_count": len(measured),
                "measurements": [asdict(item) for item in measured],
            }
        )
    return {
        "schema": "damh.native-rgbd-observed-pixel-association.v1",
        "classification": "NATIVE_SOURCE_BOUND_PIXEL_DEPTH_ONLY_NOT_OBJECT_POSE",
        "proposal_json_sha256": sha256(proposal_bytes).hexdigest(),
        "cameras": results,
        "source_is_native_simulator_observation": True,
        "automatic_object_detection": False,
        "object_identity_verified": False,
        "object_pose_verified": False,
        "grasp_validated": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_depth_dir", type=Path)
    parser.add_argument("proposals", type=Path)
    parser.add_argument("--max-regions", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        native = args.native_depth_dir.resolve()
        proposal = args.proposals.resolve()
        if args.output is not None:
            output = args.output.resolve()
            if output in (native, proposal) or native in output.parents or output.exists():
                raise RegionProposalError("output would overwrite source or immutable evidence")
            if not output.parent.is_dir():
                raise RegionProposalError("output parent directory is absent")
        report = analyze_source(native, proposal, max_regions=args.max_regions)
        if args.output is not None:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "classification": report["classification"],
                    "proposal_json_sha256": report["proposal_json_sha256"],
                    "camera_summary": [
                        {
                            key: item[key]
                            for key in (
                                "camera_id",
                                "proposal_count",
                                "sampled_count",
                                "observed_depth_point_count",
                            )
                        }
                        for item in report["cameras"]
                    ],
                    "output": str(output) if args.output is not None else None,
                    "native_acceptance": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        RegionProposalError,
        MetricPixelError,
        NativeMetricQualityError,
        DepthEvidenceError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        print(
            json.dumps(
                {
                    "classification": "NATIVE_PIXEL_ASSOCIATION_REJECTED",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "controller_eligible": False,
                    "native_acceptance": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
