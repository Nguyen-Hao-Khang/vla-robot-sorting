#!/usr/bin/env python3
"""Audit canonical color landmarks in one same-frame native RGB-D export.

The labels mean "orange/blue visible surface", NEVER a verified object class.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from math import dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_robot_sorting.perception.metric_pixel import MetricPixelError
from vla_robot_sorting.perception.native_color_landmarks import find_native_color_landmarks
from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError
from vla_robot_sorting.perception.native_metric_quality import NativeMetricQualityError
from vla_robot_sorting.perception.region_proposals import RegionProposalError


def run(source: Path, *, min_mask_px: int) -> dict:
    frames = {}
    by_kind = {}
    for cam in ("head_camera", "wrist_camera"):
        items = find_native_color_landmarks(source, camera_id=cam, min_mask_px=min_mask_px)
        frames[cam] = [asdict(item) for item in items]
        for item in items:
            by_kind.setdefault(item.landmark_kind, {})[cam] = item
    cross_camera = []
    for kind in sorted(by_kind):
        views = by_kind[kind]
        if len(views) != 2:
            continue
        first = views["head_camera"]
        second = views["wrist_camera"]
        cross_camera.append(
            {
                "kind": kind,
                "observed_pixel_world_delta_m": dist(
                    first.observed_pixel.world_point_m,
                    second.observed_pixel.world_point_m,
                ),
                "meaning": "two surface pixels, NOT the same object point or pose",
            }
        )
    digests = {}
    for cam in ("head_camera", "wrist_camera"):
        p = source / f"{cam}_frame_000000_depth-manifest.json"
        digests[cam] = sha256(p.read_bytes()).hexdigest()
    return {
        "schema": "damh.native-canonical-color-landmarks.v1",
        "classification": "COLOR_SURFACE_LANDMARKS_NOT_OBJECT_IDENTITY_OR_POSE",
        "source_native_manifest_sha256": digests,
        "camera_landmarks": frames,
        "cross_camera_observation_only": cross_camera,
        "semantic_mapping_is_canonical_color_prior": True,
        "automatic_object_identity_verified": False,
        "object_pose_verified": False,
        "grasp_validated": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_depth_dir", type=Path)
    parser.add_argument("--min-mask-px", type=int, default=30)
    parser.add_argument("--output", type=Path, help="new JSON outside immutable native export")
    args = parser.parse_args()
    try:
        source = args.native_depth_dir.resolve()
        output = args.output.resolve() if args.output else None
        if output is not None:
            if output == source or source in output.parents or output.exists():
                raise RegionProposalError("refuse to overwrite or mutate native source")
            if not output.parent.is_dir():
                raise RegionProposalError("output parent does not exist")
        report = run(source, min_mask_px=args.min_mask_px)
        if output is not None:
            with output.open("x", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "classification": report["classification"],
                    "cameras": {
                        cam: [row["landmark_kind"] for row in data]
                        for cam, data in report["camera_landmarks"].items()
                    },
                    "cross_camera": report["cross_camera_observation_only"],
                    "output": str(output) if output else None,
                    "controller_eligible": False,
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
                    "classification": "COLOR_LANDMARKS_REJECTED",
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
