#!/usr/bin/env python3
# Scoped imported native evidence source; no semantic changes.
"""Recheck genuine RoboTwin RGB-D files and produce a non-control evidence bridge."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from vla_robot_sorting.perception.native_color_landmarks import find_native_color_landmarks
from vla_robot_sorting.perception.native_region_depth_association import (
    associate_native_regions_with_observed_depth,
)
from vla_robot_sorting.perception.native_rgbd_proposals import (
    generate_native_rgbd_proposals,
)
from vla_robot_sorting.perception.native_scene_evidence_bridge import (
    CAMERAS,
    build_unidentified_native_evidence,
    seal_visual_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-proposals", type=int, default=32)
    parser.add_argument("--max-depth-regions", type=int, default=16)
    args = parser.parse_args()

    source_dir = args.native_dir.resolve(strict=True)
    output = args.output.resolve()
    if source_dir == output or source_dir in output.parents:
        parser.error("output must be outside the source-native evidence directory")
    if output.exists():
        parser.error("output already exists; no evidence overwrite")

    rows = []
    for camera in CAMERAS:
        proposal_set = generate_native_rgbd_proposals(
            source_dir, camera_id=camera, max_proposals=args.max_proposals
        )
        depth = associate_native_regions_with_observed_depth(
            proposal_set, source_dir, max_regions=args.max_depth_regions
        )
        landmarks = find_native_color_landmarks(source_dir, camera_id=camera)
        rows.append(build_unidentified_native_evidence(proposal_set, depth, landmarks))

    project_root = Path(__file__).resolve().parents[1]
    source_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    report = seal_visual_evidence(rows, project_git_head=source_head)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("S1I_NATIVE_EVIDENCE_BRIDGE_PASS_NON_CONTROL")
    print(f"output={output}")
    print("cameras=2 scene_snapshot_created=False controller_eligible=False")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
