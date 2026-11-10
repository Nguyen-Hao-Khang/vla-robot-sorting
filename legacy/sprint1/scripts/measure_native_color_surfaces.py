#!/usr/bin/env python3
# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""Read-only source-bound geometric measurement of two native RGB-D color patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_robot_sorting.perception.native_multiview_surface_geometry import (
    evaluate_native_color_surfaces,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.native_dir.resolve(strict=True)
    output = args.output.resolve()
    if root == output or root in output.parents:
        parser.error("output must be outside source-native evidence")
    if output.exists():
        parser.error("refusing to overwrite evidence")
    try:
        result = evaluate_native_color_surfaces(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"S1I_MULTIVIEW_COLOR_SURFACE_FAIL: {type(exc).__name__}: {exc}")
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print("S1I_MULTIVIEW_COLOR_SURFACES_PASS_NON_CONTROL")
    print(f"output={output}")
    print(f"cameras={result['camera_count']}")
    for row in result["cameras"]:
        for surface in row["surfaces"]:
            print(f"SURFACE={row['camera_id']} {surface['kind']} pixels={surface['masked_pixel_count']} centroid_world_m={surface['visible_surface_centroid_world_m']}")
    for compare in result["same_color_diagnostics"]:
        print(f"DIAGNOSTIC_SAME_COLOR={compare['kind']} delta_m={compare['unpaired_color_centroid_distance_m']:.6f} correspondence_verified=False")
    print("SCENE_SNAPSHOT=False OBJECT_IDENTITY=False CONTROLLER_ELIGIBLE=False")
    print(f"REPORT_SHA256={result['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
